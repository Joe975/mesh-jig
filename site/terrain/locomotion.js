// Scene movement and clip sequencing are package data; the reference IK runtime owns the pose.
import {Runtime} from './runtime.js';
import {Matrix4} from './vendor/three.core.js';
export function flightWindow(clip){
  const legs=Object.values(clip.legs),d=clip.duration_s;
  const edges=[...new Set([0,d,...legs.flatMap(l=>l.stance_phase.flatMap(([a,b])=>[a*d,b*d,(a-1)*d,(b-1)*d]).filter(t=>t>0&&t<d))])].sort((a,b)=>a-b);
  let takeoff=null;
  for(let i=0;i<edges.length-1;i++){
    const phase=(edges[i]+edges[i+1])/(2*d),count=legs.filter(l=>l.stance_phase.some(([a,b])=>(a<=phase&&phase<=b)||(a<=phase+1&&phase+1<=b))).length;
    if(count===0&&takeoff===null)takeoff=edges[i];
    if(takeoff!==null&&count>=Math.ceil(legs.length/2))return {takeoff,landing:edges[i]};
  }
  return null;
}
export function landingTime(clip){const f=flightWindow(clip);if(!f)throw new Error('Clip has no measured airborne/landing sequence');return f.landing;}
function translation(track,t,fallback){
  if(!track)return fallback.slice();
  const {times,values}=track;let i=0;while(i<times.length-1&&times[i+1]<=t)i++;
  const u=track.interpolation==='STEP'||i===times.length-1?0:Math.max(0,Math.min(1,(t-times[i])/(times[i+1]-times[i])));
  return [0,1,2].map(k=>values[i*3+k]*(1-u)+values[Math.min(i+1,times.length-1)*3+k]*u);
}
export class Locomotion {
  constructor(source){
    this.source=source;
    if(source.demo?.version!==1||!source.demo.cycle?.length)throw new Error('Character has no qualified playback policy');
    this.cycle=source.demo.cycle;
    if(new Set(this.cycle.map(r=>r.clip)).size!==Object.keys(source.runtime.clips).length||this.cycle.length!==Object.keys(source.runtime.clips).length)throw new Error('Playback policy must name every clip once');
    const roots=source.runtime.skeleton.joints.filter(j=>j.parent===null);
    if(roots.length!==1)throw new Error('Terrain motion needs one skeleton root');
    this.rootNode=roots[0].node;this.rest=source.nodes[this.rootNode].translation??[0,0,0];
    this.tracks=Object.fromEntries(Object.keys(source.runtime.clips).map(c=>[c,source.tracks[c].find(t=>t.node===this.rootNode&&t.path==='translation')]));
    this.windows=Object.fromEntries(Object.entries(source.runtime.clips).map(([c,clip])=>[c,flightWindow(clip)]));this.data=structuredClone(source);
    for(const r of this.cycle){
      if(!source.runtime.clips[r.clip]||!['measured','authored','stationary','airborne'].includes(r.motion)||!['end','landing'].includes(r.exit)||!Number.isInteger(r.cycles)||r.cycles<1||r.cycles>4)throw new Error('Invalid playback entry');
      if((r.exit==='landing'||r.motion==='airborne')&&!this.windows[r.clip])throw new Error('Playback needs a measured flight');
      if(r.motion==='airborne'&&!source.runtime.clips[r.reference]?.locomotion)throw new Error('Airborne travel needs a measured gait');
    }
    // Consume horizontal root travel once, preserving the authored vertical motion.
    for(const channels of Object.values(this.data.tracks))for(const t of channels)if(t.node===this.rootNode&&t.path==='translation')for(let i=0;i<t.values.length;i+=3){t.values[i]=this.rest[0];t.values[i+2]=this.rest[2];}
    const sampler=new Runtime(this.data),nodes=[...new Set(source.runtime.limbs.flatMap(l=>l.joints).map(n=>sampler.byName[n].node))];
    this.phases=this.cycle.map((row,i)=>{
      if(row.motion!=='measured')return 0;
      const previous=this.cycle[(i+this.cycle.length-1)%this.cycle.length],time=this.exitTime(previous),landed=sampler.sample(previous.clip,Math.min(time,source.runtime.clips[previous.clip].duration_s));
      let phase=0,best=Infinity;const duration=source.runtime.clips[row.clip].duration_s;
      for(let k=0;k<Math.ceil(duration*120);k++){
        const t=k/120,p=sampler.sample(row.clip,t),cost=nodes.reduce((sum,n)=>sum+1-Math.abs(p[n].q.dot(landed[n].q))+p[n].p.distanceToSquared(landed[n].p),0);
        if(cost<best){best=cost;phase=t;}
      }
      return phase;
    });this.reset();
  }
  exitTime(row){const d=this.source.runtime.clips[row.clip].duration_s;return row.exit==='landing'?(row.cycles-1)*d+this.windows[row.clip].landing:d*row.cycles;}
  reset(mode='all',fade=.15,ik=true){
    if(mode!=='all'&&!this.source.runtime.clips[mode])throw new Error('Unknown character clip');
    this.mode=mode;this.fade=fade;this.time=0;this.z=0;this.index=0;this.offset=0;this.previousTimes={};this.previousWeights={};
    this.runtime=new Runtime(this.data);this.runtime.setGroundEnabled(ik);
    if(mode==='all'&&this.cycle.length>1){
      for(const row of this.cycle)if(row.cycles>1)this.runtime.spec.clips[row.clip].loop=true;
      const states=Object.fromEntries(this.cycle.map((r,i)=>['c'+i,{type:'clip',clip:r.clip}]));
      const transitions=this.cycle.map((r,i)=>({from:'c'+i,to:'c'+((i+1)%this.cycle.length),on:'next',fade_s:fade}));
      this.runtime.spec.graph={layers:[{node:{type:'states',start:'c0',states,transitions},mask_joints:null}]};
    }else{const clip=mode==='all'?this.cycle[0].clip:mode;this.runtime.spec.graph={layers:[{node:{type:'clip',clip},mask_joints:null}]};this.runtime.spec.clips[clip].loop=true;}
  }
  setFade(seconds){this.fade=seconds;for(const l of this.runtime.spec.graph.layers)if(l.node.type==='states')for(const tr of l.node.transitions)tr.fade_s=seconds;}
  contributions(){
    if(this.mode!=='all'||this.cycle.length===1)return [{clip:this.mode==='all'?this.cycle[0].clip:this.mode,id:'0',weight:1,time:this.runtime.clocks['0']??0}];
    const s=this.runtime.clocks['0#states'];if(!s)return [];
    const u=s.previous===null?1:Math.min(1,s.elapsed/s.fade),rows=[];
    if(s.previous!==null)rows.push({id:'0/'+s.previous,clip:this.cycle[Number(s.previous.slice(1))].clip,weight:1-u});
    rows.push({id:'0/'+s.current,clip:this.cycle[Number(s.current.slice(1))].clip,weight:u});
    return rows.map(r=>({...r,time:this.runtime.clocks[r.id]??0}));
  }
  distance(row,time,loop){
    const d=this.source.runtime.clips[row.clip].duration_s,t=loop?time%d:Math.min(time,d),cycles=loop?Math.floor(time/d):0;
    if(row.motion==='stationary')return 0;
    if(row.motion==='measured')return time*this.source.runtime.clips[row.clip].travel_speed_mps;
    if(row.motion==='airborne'){
      const {takeoff,landing}=this.windows[row.clip],speed=this.source.runtime.clips[row.reference].travel_speed_mps;
      return speed*(cycles*(landing-takeoff)+Math.max(0,Math.min(t,landing)-takeoff));
    }
    const track=this.tracks[row.clip],start=translation(track,0,this.rest)[2],end=translation(track,d,this.rest)[2];
    return cycles*(start-end)+start-translation(track,t,this.rest)[2];
  }
  step(dt,ground,rootGround=ground){
    if(!Number.isFinite(dt)||dt<0||dt>.1)throw new Error('Motion dt must be 0..0.1 seconds');
    this.time+=dt;let pendingPhase=null;
    if(this.mode==='all'){
      const s=this.runtime.clocks['0#states'],clock=this.runtime.clocks['0/c'+this.index]??0;
      if(this.cycle.length>1&&s?.previous===null&&clock-this.offset+dt>=this.exitTime(this.cycle[this.index])){
        this.index=(this.index+1)%this.cycle.length;this.runtime.trigger('next');pendingPhase=this.phases[this.index];this.offset=pendingPhase;
        delete this.previousTimes['0/c'+this.index];delete this.previousWeights['0/c'+this.index];
      }
    }
    const world=this.runtime.step(dt,()=>{
      const nextWeights={};
      for(const r of this.contributions()){
        const row=this.cycle.find(v=>v.clip===r.clip),previous=this.previousTimes[r.id]??r.time,loop=this.runtime.spec.clips[r.clip].loop;
        const delta=this.distance(row,r.time,loop)-this.distance(row,previous,loop),mean=(r.weight+(this.previousWeights[r.id]??r.weight))/2;
        this.z-=delta*mean;this.previousTimes[r.id]=r.time;nextWeights[r.id]=r.weight;
      }
      this.previousWeights=nextWeights;return new Matrix4().makeTranslation(0,-this.source.runtime.measurement.rest_ground_y+rootGround(0,this.z),this.z);
    },ground);
    if(pendingPhase!==null){const id='0/c'+this.index;this.runtime.clocks[id]=pendingPhase;this.previousTimes[id]=pendingPhase;}
    return world;
  }
  snapshot(){return {mode:this.mode,fade:this.fade,time:this.time,z:this.z,index:this.index,offset:this.offset,previousTimes:{...this.previousTimes},previousWeights:{...this.previousWeights}};}
  restore(s){Object.assign(this,structuredClone(s));}
  diagnostic(){
    const contributions=this.contributions(),weights=Object.fromEntries(Object.keys(this.source.runtime.clips).map(c=>[c,0])),phases={},cycles={};
    for(const r of contributions){weights[r.clip]+=r.weight;const d=this.source.runtime.clips[r.clip].duration_s;phases[r.clip]=this.runtime.spec.clips[r.clip].loop?r.time%d:Math.min(r.time,d);cycles[r.clip]=Math.floor(r.time/d);}
    return {activeClip:this.mode==='all'?this.cycle[this.index].clip:this.mode,weights,phases,cycles,contributions};
  }
}

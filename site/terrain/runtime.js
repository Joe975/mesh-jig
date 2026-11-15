// Port of game_runtime.py: clip/state graph + full ground adaptation + two-bone legs.
// This pilot explicitly refuses graph/solver features it does not implement.
// No renderer or DOM; the same step drives browser bones and checker traces.
import {Vector3, Quaternion, Matrix4} from './vendor/three.core.js';
export const V = a => new Vector3(...a), Q = a => new Quaternion(...a);
const up = V([0,1,0]), one = V([1,1,1]), I = () => new Matrix4();
const clamp = (x,a,b) => Math.max(a, Math.min(b,x));
const smooth = (dt,s) => s <= 0 ? 1 : 1-Math.exp(-dt/s);
const pos = m => new Vector3().setFromMatrixPosition(m);
const rot = m => new Quaternion().setFromRotationMatrix(m);
const turn = (a,b) => {
  a=a.clone().normalize();b=b.clone().normalize();const c=clamp(a.dot(b),-1,1);
  if(c>1-1e-12)return new Quaternion();
  if(c< -1+1e-9){const axis=[new Vector3(1,0,0),new Vector3(0,1,0),new Vector3(0,0,1)].sort((u,v)=>Math.abs(a.dot(u))-Math.abs(a.dot(v)))[0];return new Quaternion().setFromAxisAngle(a.clone().cross(axis).normalize(),Math.PI);}
  return new Quaternion().setFromUnitVectors(a,b);
};
const yaw = a => new Quaternion().setFromAxisAngle(up,a);
function matrixSlerp(a,b,u){
  const r=b.clone().multiply(a.clone().transpose()),e=r.elements,angle=Math.acos(clamp((e[0]+e[5]+e[10]-1)/2,-1,1));
  if(angle<1e-9)return b.clone();
  const axis=new Vector3(e[6]-e[9],e[8]-e[2],e[1]-e[4]);
  if(axis.length()>1e-12)axis.normalize();
  return new Matrix4().makeRotationAxis(axis,angle*u).multiply(a);
}
const mix = (a,b,w) => {
  const q = b.q.clone(); if (a.q.dot(q)<0) q.set(-q.x,-q.y,-q.z,-q.w);
  return {p:a.p.clone().lerp(b.p,w),q:new Quaternion(a.q.x*(1-w)+q.x*w,a.q.y*(1-w)+q.y*w,a.q.z*(1-w)+q.z*w,a.q.w*(1-w)+q.w*w).normalize(),s:a.s.clone().lerp(b.s,w)};
};
function planeNormal(points) {
  const c = points.reduce((a,p)=>a.add(p),new Vector3()).divideScalar(points.length);
  const a = [[0,0,0],[0,0,0],[0,0,0]], v = [[1,0,0],[0,1,0],[0,0,1]];
  for (const p of points) { const d=p.clone().sub(c).toArray(); for(let i=0;i<3;i++) for(let j=0;j<3;j++) a[i][j]+=d[i]*d[j]; }
  for(let sweep=0;sweep<50;sweep++) {
    if(Math.abs(a[0][1])+Math.abs(a[0][2])+Math.abs(a[1][2])<1e-15) break;
    for(const [p,q] of [[0,1],[0,2],[1,2]]) {
      if(Math.abs(a[p][q])<1e-18) continue;
      const theta=(a[q][q]-a[p][p])/(2*a[p][q]);
      const t=theta===0?1:Math.sign(theta)/(Math.abs(theta)+Math.sqrt(theta*theta+1));
      const cs=1/Math.sqrt(t*t+1),sn=t*cs;
      for(let k=0;k<3;k++) { const x=a[k][p],y=a[k][q];a[k][p]=cs*x-sn*y;a[k][q]=sn*x+cs*y; }
      for(let k=0;k<3;k++) { const x=a[p][k],y=a[q][k];a[p][k]=cs*x-sn*y;a[q][k]=sn*x+cs*y; }
      for(let k=0;k<3;k++) { const x=v[k][p],y=v[k][q];v[k][p]=cs*x-sn*y;v[k][q]=sn*x+cs*y; }
    }
  }
  const least=[0,1,2].sort((i,j)=>a[i][i]-a[j][j])[0],n=V([v[0][least],v[1][least],v[2][least]]).normalize();
  return n.y<0?n.negate():n;
}

export class Runtime {
  constructor(data) {
    if(![1,2].includes(data.version)) throw new Error('Unsupported terrain package');
    this.data=data; this.spec=structuredClone(data.runtime); this.names=this.spec.skeleton.joints.map(j=>j.name);
    this.byName=Object.fromEntries(this.spec.skeleton.joints.map(j=>[j.name,j]));
    this.legs=this.spec.limbs;
    if(this.legs.some(l=>l.role!=='leg'||l.solver!=='two_bone'||l.joints.length!==3)||this.spec.aims?.length) throw new Error('Unsupported terrain rig capability');
    const validate=n=>{if(!['clip','states','blend1d'].includes(n.type))throw new Error('Unsupported graph');if(n.type==='states')Object.values(n.states).filter(Boolean).forEach(validate);if(n.type==='blend1d')n.points.filter(p=>p.node).forEach(p=>validate(p.node));};
    this.spec.graph.layers.forEach(l=>validate(l.node));
    this.rest=data.nodes.map(n=>({p:V(n.translation??[0,0,0]),q:Q(n.rotation??[0,0,0,1]),s:V(n.scale??[1,1,1])}));
    this.parents=data.nodes.map(()=>null);data.nodes.forEach((n,i)=>(n.children??[]).forEach(c=>this.parents[c]=i));
    this.subtrees=Object.fromEntries(this.names.map(n=>[n,this.names.filter(c=>{let k=c;while(k){if(k===n)return true;k=this.byName[k].parent;}return false;})]));
    this.params=Object.fromEntries(Object.entries(this.spec.parameters??{}).map(([k,p])=>[k,p.default]));this.clocks={};this.triggers=new Set();this.states={};this.bodyOffset=0;this.tilt=I();this.groundOn=true;this.groundWeight=1;this.first=true;this.queryCount=0;
    this.locals=this.rest;this.world={};this.targets={};
  }
  clock(path,dt) { return this.clocks[path]=this.clocks[path]===undefined?0:this.clocks[path]+dt; }
  reset(path) { for(const k of Object.keys(this.clocks))if(k===path||k.startsWith(path+'/')||k.startsWith(path+'#'))delete this.clocks[k]; }
  trigger(name) { this.triggers.add(name); }
  setGroundEnabled(on) { this.groundOn=!!on; }
  sample(name,t) {
    const pose=this.rest.map(p=>({p:p.p.clone(),q:p.q.clone(),s:p.s.clone()}));
    for(const ch of this.data.tracks[name]) {
      const {node,path,times,values}=ch,dim=path==='rotation'?4:3;
      let lo=0,hi=times.length-1;while(lo<hi){const m=Math.ceil((lo+hi)/2);if(times[m]<=t)lo=m;else hi=m-1;}
      const k=lo,u=ch.interpolation==='STEP'||k===times.length-1?0:clamp((t-times[k])/(times[k+1]-times[k]),0,1);
      const a=values.slice(k*dim,(k+1)*dim),b=values.slice(Math.min(k+1,times.length-1)*dim,Math.min(k+1,times.length-1)*dim+dim);
      if(path==='rotation'){
        // Match skin.slerp's short-arc nlerp threshold, including nonunit GLB keys.
        const qa=Q(a),qb=Q(b);let dot=qa.dot(qb);
        if(dot<0){qb.set(-qb.x,-qb.y,-qb.z,-qb.w);dot=-dot;}
        if(dot>.9995)pose[node].q=new Quaternion(...qa.toArray().map((v,i)=>v*(1-u)+qb.toArray()[i]*u)).normalize();
        else {const angle=Math.acos(Math.min(dot,1)),s=Math.sin(angle);pose[node].q=new Quaternion(...qa.toArray().map((v,i)=>(Math.sin((1-u)*angle)*v+Math.sin(u*angle)*qb.toArray()[i])/s)).normalize();}
      }else pose[node][path==='translation'?'p':'s']=V(a).lerp(V(b),u);
    }
    return pose;
  }
  node(n,path,dt) {
    if(n.type==='clip') {const c=this.spec.clips[n.clip],t=this.clock(path,dt),tt=c.loop?t%c.duration_s:Math.min(t,c.duration_s);return {pose:this.sample(n.clip,tt),weight:1,votes:[[n.clip,tt,1]]};}
    if(n.type==='blend1d'){
      const points=n.points,x=this.params[n.parameter],values=points.map(p=>p.value);
      const i=x<=values[0]?0:x>=values.at(-1)?points.length-2:values.findLastIndex((v,k)=>k<values.length-1&&v<=x),w=clamp((x-values[i])/(values[i+1]-values[i]),0,1),a=points[i],b=points[i+1],t=this.clock(path,dt);
      let results;
      if(n.sync){
        const dur=p=>this.spec.clips[p.clip].duration_s,on=[[a,1-w],[b,w]].filter(([p])=>p.cycle),freq=on.length===2?on.reduce((s,[p,v])=>s+v/dur(p),0):on.length?1/dur(on[0][0]):0;
        const key=path+'#phase',phase=this.clocks[key]=this.clocks[key]===undefined?0:this.clocks[key]+dt*freq;
        results=[a,b].map(p=>{const tt=p.cycle?((phase+p.phase_offset)%1)*dur(p):t%dur(p);return {pose:this.sample(p.clip,tt),weight:1,votes:[[p.clip,tt,1]]};});
      }else results=[a,b].map((p,k)=>{if(p.node)return this.node(p.node,path+'/'+(i+k),dt);const c=this.spec.clips[p.clip],tt=c.loop?t%c.duration_s:Math.min(t,c.duration_s);return {pose:this.sample(p.clip,tt),weight:1,votes:[[p.clip,tt,1]]};});
      const pose=results.map(r=>r.pose??this.rest),votes=results.flatMap((r,k)=>r.votes.map(([c,t,v])=>[c,t,v*(k?w:1-w)]));return {pose:pose[0].map((p,j)=>mix(p,pose[1][j],w)),weight:1,votes};
    }
    const key=path+'#states',st=this.clocks[key]??={current:n.start,previous:null,elapsed:0};
    if(st.previous!==null)st.elapsed+=dt;
    for(const tr of n.transitions) {
      if(tr.from!==st.current)continue;
      const old=n.states[tr.from],fire=tr.on?this.triggers.has(tr.on):old?.type==='clip'&&(this.clocks[path+'/'+tr.from]??0)>=this.spec.clips[old.clip].duration_s;
      if(fire){Object.assign(st,{previous:st.current,current:tr.to,elapsed:0,fade:tr.fade_s});this.reset(path+'/'+tr.to);break;}
    }
    const empty=()=>({pose:null,weight:0,votes:[]});
    const b=n.states[st.current]?this.node(n.states[st.current],path+'/'+st.current,dt):empty();
    const u=st.previous===null?1:Math.min(1,st.elapsed/st.fade);
    if(st.previous===null||u>=1){st.previous=null;return b;}
    const a=n.states[st.previous]?this.node(n.states[st.previous],path+'/'+st.previous,dt):empty();
    const votes=[...a.votes.map(([c,t,w])=>[c,t,w*(1-u)]),...b.votes.map(([c,t,w])=>[c,t,w*u])];
    if(!a.pose&&!b.pose)return empty();
    if(!a.pose)return {pose:b.pose,weight:u,votes};
    if(!b.pose)return {pose:a.pose,weight:1-u,votes};
    return {pose:a.pose.map((p,i)=>mix(p,b.pose[i],u)),weight:1,votes};
  }
  animate(dt) {
    let pose=this.rest.map(p=>({p:p.p.clone(),q:p.q.clone(),s:p.s.clone()}));const votes=Object.fromEntries(this.legs.map(l=>[l.name,[]]));
    this.spec.graph.layers.forEach((l,i)=>{
      const r=this.node(l.node,String(i),dt);if(!r.pose||r.weight<=0)return;
      const ids=l.mask_joints?l.mask_joints.map(n=>this.byName[n].node):pose.map((_,i)=>i);
      for(const j of ids)pose[j]=mix(pose[j],r.pose[j],r.weight);
      for(const leg of this.legs)if(!l.mask_joints||l.mask_joints.includes(leg.joints.at(-1)))votes[leg.name]=[...votes[leg.name].map(([c,t,w])=>[c,t,w*(1-r.weight)]),...r.votes.map(([c,t,w])=>[c,t,w*r.weight])];
    });this.triggers.clear();this.locals=pose;return votes;
  }
  fk(root) {
    const result=[];const visit=i=>{if(result[i])return result[i];const p=this.locals[i],local=this.data.nodes[i].matrix?new Matrix4().fromArray(this.data.nodes[i].matrix):I().compose(p.p,p.q,p.s);return result[i]=(this.parents[i]===null?root:visit(this.parents[i])).clone().multiply(local);};
    this.data.nodes.forEach((_,i)=>visit(i));this.nodeWorld=result;this.world=Object.fromEntries(this.names.map(n=>[n,result[this.byName[n].node].clone()]));
  }
  planted(leg,votes) {
    let total=0,yes=0;for(const [name,t,w]of votes){const c=this.spec.clips[name],phase=t/c.duration_s,win=c.legs[leg.name]?.stance_phase??[];total+=w;if(win.some(([a,b])=>(a<=phase&&phase<=b)||(a<=phase+1&&phase+1<=b)))yes+=w;}
    return total>0&&yes>=total/2;
  }
  translate(name,v){for(const n of this.subtrees[name])this.world[n].setPosition(pos(this.world[n]).add(v));}
  rotate(name,q,pivot){const m=I().makeTranslation(...pivot.toArray()).multiply(I().makeRotationFromQuaternion(q)).multiply(I().makeTranslation(...pivot.clone().negate().toArray()));for(const n of this.subtrees[name])this.world[n].premultiply(m);}
  rotateMatrix(name,r,pivot){const m=I().makeTranslation(...pivot.toArray()).multiply(r).multiply(I().makeTranslation(...pivot.clone().negate().toArray()));for(const n of this.subtrees[name])this.world[n].premultiply(m);}
  reach(l){const [a,b]=l.segment_lengths_m;return Math.min(.999*(a+b),Math.sqrt(a*a+b*b-2*a*b*Math.cos(l.bend_degrees[1]*Math.PI/180)));}
  solve(l,target,root) {
    const [a,b,c]=l.joints,pa=pos(this.world[a]),pb=pos(this.world[b]),pc=pos(this.world[c]),l1=pa.distanceTo(pb),l2=pb.distanceTo(pc);
    const reach=angle=>Math.sqrt(Math.max(1e-12,l1*l1+l2*l2-2*l1*l2*Math.cos(angle*Math.PI/180)));
    const delta=target.clone().sub(pa),r=clamp(delta.length(),reach(l.bend_degrees[0]),Math.min(reach(l.bend_degrees[1]),.999*(l1+l2)));
    const d=(delta.length()>1e-9?delta:pc.clone().sub(pa)).normalize(),pole=V(l.pole).applyQuaternion(rot(root));
    let side=pole.clone().addScaledVector(d,-pole.dot(d));if(side.length()<1e-6){side=pb.clone().sub(pa);side.addScaledVector(d,-side.dot(d));}
    const cos=clamp((l1*l1+r*r-l2*l2)/(2*l1*r),-1,1),knee=pa.clone().addScaledVector(d,l1*cos).addScaledVector(side.normalize(),l1*Math.sqrt(1-cos*cos));
    this.rotate(a,turn(pb.clone().sub(pa),knee.sub(pa)),pa);
    const nb=pos(this.world[b]),nc=pos(this.world[c]);this.rotate(b,turn(nc.sub(nb),pa.addScaledVector(d,r).sub(nb)),nb);
  }
  footHeight(l,s,ground,tip,q,lift,planted,dt) {
    const g=this.spec.ground;if(s.tip&&dt>0)s.velocity=tip.clone().sub(s.tip).divideScalar(dt);s.tip=tip.clone();
    const step=g.blend_s>0?dt/g.blend_s:1;s.ahead=clamp((s.ahead??0)+(planted?-step:step),0,1);
    const dx=(s.velocity?.x??0)*g.lookahead_s,dz=(s.velocity?.z??0)*g.lookahead_s;let best=-Infinity;
    for(const p of [...l.contact.points,...l.clearance.points]) {
      const o=V(p).applyQuaternion(q),x=tip.x+o.x,z=tip.z+o.z,here=ground(x,z);let need=here;
      if(s.ahead>0&&dx*dx+dz*dz>1e-12){const back=ground(x-dx/8,z-dz/8);for(let k=1;k<=8;k++){const f=k/8,rise=ground(x+f*dx,z+f*dz)-(here+k*(here-back));need=Math.max(need,here+s.ahead*(1-f)*rise);}}
      best=Math.max(best,need+lift-o.y);
    }return best;
  }
  adapt(root,votes,query,dt) {
    const g=this.spec.ground;if(!g||!this.legs.length)return;
    const rate=g.blend_s>0?dt/g.blend_s:1;this.groundWeight=clamp(this.groundWeight+(this.groundOn?rate:-rate),0,1);const gw=this.groundWeight;
    if(gw<=0){this.states={};this.bodyOffset=0;this.tilt.identity();this.targets={};return;}
    const ground=(x,z)=>{this.queryCount++;const h=query(x,z);return Array.isArray(h)?h[0]:h;};
    const rootYaw=Math.atan2(root.elements[8],root.elements[10]),animGround=V([0,this.spec.measurement.rest_ground_y,0]).applyMatrix4(root).y,gap=this.spec.measurement.contact_gap_m,plan=[];
    for(const l of this.legs) {
      const s=this.states[l.name]??={shift:new Vector3(),turn:0},w=this.world[l.joints.at(-1)],points=l.contact.points.map(p=>V(p).applyMatrix4(w));
      const centre=points.reduce((a,p)=>a.add(p),new Vector3()).divideScalar(points.length),clearance=points.map(p=>p.y-animGround),touching=clearance.map(h=>h<=gap),hip=pos(this.world[l.joints[0]]),reach=this.reach(l),planted=this.planted(l,votes[l.name]);
      const flat=hip.clone().sub(centre).sub(s.shift);flat.y=0;const far=flat.length()>reach,locked=planted||touching.some(Boolean);
      if(locked&&s.planted){let drift=new Vector3(),n=0;points.forEach((p,i)=>{if(touching[i]&&s.support[i]){drift.add(p.clone().sub(s.points[i]));n++;}});if(n){drift.y=0;s.shift.addScaledVector(drift,-1/n);}s.turn-=rootYaw-s.rootYaw;}
      if(!locked||far){const fade=1-smooth(dt,g.blend_s);s.shift.multiplyScalar(fade);s.turn*=fade;}
      if(s.shift.length()>g.relock_m)s.shift.setLength(g.relock_m);
      Object.assign(s,{points,support:touching,planted:locked,rootYaw});
      const q=yaw(s.turn).multiply(rot(w)),offset=l.contact.points.reduce((a,p)=>a.add(V(p).applyQuaternion(q)),new Vector3()).divideScalar(points.length);
      const tip=centre.clone().add(s.shift).sub(offset);tip.y=0;tip.y=this.footHeight(l,s,ground,tip,q,Math.max(0,Math.min(...clearance)),planted,dt);
      plan.push({l,target:tip,q,planted,animated:pos(w),hip,reach});
    }
    const chosen=plan.some(p=>p.planted)?plan.filter(p=>p.planted):plan;
    const want=clamp(Math.min(...chosen.map(p=>p.target.y-p.animated.y)),-g.drop_m_max,g.rise_m_max);
    let bound=Infinity;for(const p of chosen){const flat=Math.hypot(p.target.x-p.hip.x,p.target.z-p.hip.z);bound=Math.min(bound,p.target.y-p.hip.y+Math.sqrt(Math.max(p.reach*p.reach-flat*flat,0)));}
    this.bodyOffset=this.first?want:this.bodyOffset+(want-this.bodyOffset)*smooth(dt,g.blend_s);
    const body=this.spec.body.joint;this.translate(body,V([0,gw*clamp(Math.min(this.bodyOffset,bound),-g.drop_m_max,g.rise_m_max),0]));
    if(g.tilt){const planted=plan.filter(p=>p.planted),goal=planted.length>=3?I().makeRotationFromQuaternion(turn(up,planeNormal(planted.map(p=>p.target)))):I();this.tilt=this.first?goal:matrixSlerp(this.tilt,goal,smooth(dt,g.blend_s));this.rotateMatrix(body,matrixSlerp(I(),this.tilt,gw),pos(this.world[body]));}
    this.targets={};
    for(const p of plan){const tip=p.l.joints.at(-1);this.solve(p.l,pos(this.world[tip]).lerp(p.target,gw),root);const delta=p.q.clone().multiply(rot(this.world[tip]).invert());this.rotate(tip,new Quaternion().slerp(delta,gw),pos(this.world[tip]));this.targets[p.l.name]={point:p.target.toArray(),planted:p.planted};}
  }
  step(dt,root,ground) {
    if(!Number.isFinite(dt)||dt<0||dt>.1)throw new Error('Runtime dt must be 0..0.1 seconds');
    if(this.first)this.groundWeight=this.groundOn?1:0;
    const votes=this.animate(dt);
    // The game may consume root motion after graph evaluation and before placement/IK.
    const placement=typeof root==='function'?root(this,votes):root;
    this.fk(placement);this.queryCount=0;this.adapt(placement,votes,ground,dt);this.first=false;return this.world;
  }
  snapshot() {
    const encode=x=>x?.isVector3?{$v:x.toArray()}:x?.isQuaternion?{$q:x.toArray()}:x?.isMatrix4?{$m:x.toArray()}:Array.isArray(x)?x.map(encode):x&&typeof x==='object'?Object.fromEntries(Object.entries(x).map(([k,v])=>[k,encode(v)])):x;
    const fields=['params','clocks','states','bodyOffset','tilt','groundOn','groundWeight','first','locals','world','nodeWorld','targets'];
    return encode(Object.fromEntries(fields.map(k=>[k,this[k]])));
  }
  restore(state) {
    const decode=x=>x?.$v?V(x.$v):x?.$q?Q(x.$q):x?.$m?I().fromArray(x.$m):Array.isArray(x)?x.map(decode):x&&typeof x==='object'?Object.fromEntries(Object.entries(x).map(([k,v])=>[k,decode(v)])):x;
    Object.assign(this,decode(state));this.triggers.clear();
  }
}

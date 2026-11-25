import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {AnimationCycle,defaultAnimation,activeClipLabel} from '../site/animation-cycle.js';
import {Runtime} from '../site/terrain/runtime.js';
import {Ground} from '../site/terrain/ground.js';
import {Matrix4} from '../site/terrain/vendor/three.core.js';
import {sha256} from '../site/terrain/hash.js';
import {createHash} from 'node:crypto';
import {Locomotion,landingTime} from '../site/terrain/locomotion.js';
import {readTerrainLink,writeTerrainLink,playbackSpeed} from '../site/viewer-link.js';
import {renderSize, whaleUrl} from '../site/cloud-canyon/view-settings.js';
const data=()=>JSON.parse(fs.readFileSync(new URL('../site/terrain/horror.json',import.meta.url)));

test('a new whale share link preserves other query parameters and the fragment', () => {
  const url = whaleUrl('https://example.com/mesh-jig/cloud-canyon/?whale=fable&source=gallery#view', 'neon');
  assert.equal(url.searchParams.get('whale'), 'neon');
  assert.equal(url.searchParams.get('source'), 'gallery');
  assert.equal(url.hash, '#view');
  assert.equal(url.pathname, '/mesh-jig/cloud-canyon/');
});

test('Cloud Canyon sharp rendering doubles sampling and respects the portrait budget', () => {
  assert.deepEqual(renderSize(880, 620, 'fast'), [880, 620]);
  assert.deepEqual(renderSize(880, 620, 'balanced'), [1100, 775]);
  assert.deepEqual(renderSize(880, 620, 'sharp'), [1760, 1240]);
  assert.deepEqual(renderSize(820, 1100, 'sharp'), [1044, 1400]);
  assert.deepEqual(renderSize(880, 620, 'unknown'), renderSize(880, 620, 'balanced'));
});

test('comparison resolves native animation names through public labels without confusing All clips',()=>{
  assert.equal(activeClipLabel(['animation_0','animation_1'],['guard','overhead_chop'],'animation_1'),'overhead_chop');
  assert.equal(activeClipLabel(['guard','overhead_chop'],null,'overhead_chop'),'overhead_chop');
  assert.notEqual(activeClipLabel(['animation_0','animation_1'],['walk','run'],'animation_0'),activeClipLabel(['walk','run'],null,'run'));
  assert.equal(activeClipLabel(['idle'],['rest'],'missing'),'missing');
});

test('default animation prefers run, then sprint, then a single available clip',()=>{
  assert.equal(defaultAnimation(['idle','walk','run','jump']),'run');
  assert.equal(defaultAnimation(['leap','Sprint']),'Sprint');
  assert.equal(defaultAnimation(['fly','hover']),'fly');
  assert.equal(defaultAnimation([]),undefined);
});
test('terrain links round-trip mode and bounded playback speed without recording IK',()=>{
  for(const speed of [playbackSpeed.min,.5,1,1.35,playbackSpeed.max]){
    const query=new URLSearchParams({subject:'horror',model:'qualified',clip:'leap'});
    writeTerrainLink(query,{enabled:true,speed});assert.deepEqual(readTerrainLink(query),{enabled:true,speed});
    assert.equal(query.get('clip'),'leap');assert.equal(query.has('ik'),false);
    writeTerrainLink(query,{enabled:false,speed});assert.equal(query.has('terrain'),false);assert.equal(query.has('speed'),false);
  }
  for(const speed of ['', 'NaN','Infinity','-1','0','0.24','1.51','garbage'])assert.deepEqual(readTerrainLink(new URLSearchParams({terrain:'1',speed})),{enabled:true,speed:1});
  assert.deepEqual(readTerrainLink(new URLSearchParams()),{enabled:false,speed:1});
  assert.equal(readTerrainLink(new URLSearchParams('terrain=false&ik=0')).enabled,false);
});
test('LAN HTTP fallback verifies SHA-256 at padding boundaries and asset sizes',async()=>{
  for(const length of [0,3,55,56,63,64,65,1000,2174256]){
    const bytes=Uint8Array.from({length},(_,i)=>(i*17+3)%256);
    assert.equal(await sha256(bytes.buffer,null),createHash('sha256').update(bytes).digest('hex'));
  }
});
test('cycle reaches every clip, respects one-shots, restarts and stops without a timer',()=>{
  const played=[],c=new AnimationCycle((...x)=>played.push(x));
  assert.equal(c.start([]),false);c.start([{name:'sprint'},{name:'leap',loop:false}]);c.boundary();assert.equal(played.length,1);c.boundary();assert.deepEqual(played.at(-1),['leap',false]);c.boundary();assert.equal(played.at(-1)[0],'sprint');c.stop();c.boundary();assert.equal(played.length,3);c.start([{name:'idle'}]);assert.equal(played.at(-1)[0],'idle');
});
test('heightfield repeats exactly, has matching flat boundary slopes and reacts to live height',()=>{
  const g=new Ground(.12);for(const x of [-2,-.7,0,.3,2])for(const z of [-16,-8,-.1,0,.1,3,7.99,8,16])assert.ok(Math.abs(g.height(x,z)-g.height(x,z+8))<1e-12);
  for(const x of [-2,0,2])assert.equal(g.height(x,-.1),g.height(x,.1));
  const h=g.height(0,4);assert.ok(h>.1);g.setHeight(0);assert.equal(g.height(0,4),0);g.setHeight(.2);assert.ok(g.height(0,4)>h);assert.equal(g.triangles().length/9*3,1536);assert.throws(()=>g.setHeight(100));
});
test('rocks and blocks use their rendered top facets for periodic ground contact',()=>{
  const g=new Ground(.12,.24);assert.deepEqual(g.obstacles.map(o=>o.kind),['block','rock','block']);
  for(const o of g.obstacles){
    assert.ok(g.height(o.x,o.z)-g.baseHeight(o.x,o.z)>.1);
    for(const t of o.tops){
      const [a,b,c]=t,det=(b[0]-a[0])*(c[2]-a[2])-(b[2]-a[2])*(c[0]-a[0]);if(Math.abs(det)<1e-12)continue;
      const x=(a[0]+b[0]+c[0])/3,z=(a[2]+b[2]+c[2])/3,y=(a[1]+b[1]+c[1])/3;
      assert.ok(Math.abs(g.height(x,z)-Math.max(g.baseHeight(x,z),y))<1e-8);
      assert.ok(Math.abs(g.height(x,z-8)-g.height(x,z))<1e-8);
    }
  }
  const o=g.obstacles[0],before=g.height(o.x,o.z);g.setObstacleHeight(.1);assert.ok(g.height(o.x,o.z)<before);
  g.setObstaclesEnabled(false);assert.equal(g.height(o.x,o.z),g.baseHeight(o.x,o.z));assert.equal(g.obstacleTriangles().length,0);
  g.setObstaclesEnabled(true);assert.equal(g.obstacleTriangles().length,g.obstacleColors().length);assert.ok(g.obstacleTriangles().length/9*3<200);
  assert.throws(()=>g.setObstacleHeight(.5));
});
test('stairs ascend to a level platform then descend, with exact periodic tread support',()=>{
  const g=new Ground(0);g.setCourse('stairs');
  assert.deepEqual(g.obstacles.map(o=>o.level),[1,2,3,4,3,2,1]);
  assert.deepEqual(g.obstacles.map(o=>Math.round(o.top*100)),[10,20,30,40,30,20,10]);
  assert.equal(g.obstacles[3].length,1.6);
  for(const o of g.obstacles)for(const t of o.tops){
    const x=t.reduce((sum,p)=>sum+p[0]/3,0),z=t.reduce((sum,p)=>sum+p[2]/3,0);
    for(const wrap of [-16,-8,0,8,16])assert.ok(Math.abs(g.height(x,z+wrap)-o.top)<1e-10);
  }
  for(const o of g.obstacles.slice(0,-1)){
    const next=g.obstacles[g.obstacles.indexOf(o)+1],edge=o.maxZ;
    assert.ok(Math.abs(edge-next.minZ)<1e-10);
    assert.ok(Math.abs(g.height(0,edge-1e-5)-o.top)<1e-10);
    assert.ok(Math.abs(g.height(0,edge+1e-5)-next.top)<1e-10);
    assert.ok(Math.abs(g.height(0,edge)-Math.max(o.top,next.top))<1e-10);
  }
  assert.equal(g.height(1.5,4),0);assert.equal(g.height(0,0),0);
  assert.equal(g.obstacleTriangles().length,g.obstacleColors().length);
  assert.equal(g.obstacleTriangles().length/9,70);
  g.setStepHeight(.16);assert.ok(Math.abs(g.height(0,4)-.64)<1e-12);
  g.setHeight(.2);assert.ok(g.height(0,4)>.64,'terrain height changes the real platform elevation');
  g.setObstacleHeight(0);assert.ok(g.obstacleTriangles().length>0,'rocks knob does not hide stairs');
  g.setStepHeight(0);assert.equal(g.obstacleTriangles().length,0);assert.equal(g.height(0,4),g.baseHeight(0,4));
  g.setStepHeight(.1);g.setObstaclesEnabled(false);assert.equal(g.height(0,4),g.baseHeight(0,4));
  assert.throws(()=>g.setStepHeight(.2));assert.throws(()=>g.setCourse('unknown'));
});
test('stair body path stays continuous while foot queries retain abrupt risers',()=>{
  const g=new Ground(0);g.setCourse('stairs');
  let previous=g.rootHeight(0,0),maxChange=0;
  for(let z=.001;z<=8;z+=.001){const h=g.rootHeight(0,z);maxChange=Math.max(maxChange,Math.abs(h-previous));previous=h;assert.ok(Math.abs(h-g.rootHeight(0,z-8))<1e-10);}
  assert.ok(maxChange<.001);assert.equal(g.rootHeight(0,4),g.height(0,4));
  const edge=g.obstacles[0].maxZ;assert.ok(g.height(0,edge+1e-5)-g.height(0,edge-1e-5)>.09);
  const motion=new Locomotion(data());motion.reset('sprint',.15,false);motion.step(0,(x,z)=>g.height(x,z),(x,z)=>g.rootHeight(x,z));
  for(let k=0;k<110;k++)motion.step(1/60,(x,z)=>g.height(x,z),(x,z)=>g.rootHeight(x,z));
  const root=motion.runtime.world['j'+motion.rootNode];
  assert.ok(Math.abs(root.elements[13]+motion.source.runtime.measurement.rest_ground_y-motion.rest[1]-g.rootHeight(0,motion.z))<1e-9,'the production root callback consumes the continuous path');
  assert.ok(Math.abs(g.rootHeight(0,motion.z)-g.height(0,motion.z))>.001,'root and foot queries remain distinct');
});
test('combined course meets stairs, uneven rocks and a half-metre wall in forward travel order',()=>{
  const g=new Ground(.12);g.setCourse('sequence');assert.equal(g.length,24);
  const ordered=g.obstacles.slice().sort((a,b)=>b.z-a.z);
  assert.deepEqual(ordered.map(o=>o.kind),[...Array(7).fill('stair'),...Array(3).fill('rock'),'wall']);
  const wall=ordered.at(-1);assert.ok(Math.abs(wall.top-g.baseHeight(wall.x,wall.z)-.5)<1e-12);
  for(const o of ordered)for(const t of o.tops){
    const [a,b,c]=t,det=(b[0]-a[0])*(c[2]-a[2])-(b[2]-a[2])*(c[0]-a[0]);if(Math.abs(det)<1e-12)continue;
    const x=(a[0]+b[0]+c[0])/3,z=(a[2]+b[2]+c[2])/3,y=(a[1]+b[1]+c[1])/3;
    assert.ok(Math.abs(g.height(x,z)-Math.max(g.baseHeight(x,z),y))<1e-8);
    for(const wrap of [-48,-24,24,48])assert.ok(Math.abs(g.height(x,z)-g.height(x,z+wrap))<1e-8);
  }
  assert.equal(g.triangles().length/9,512,'longer course keeps the base mesh budget');
  assert.equal(Math.max(...g.triangles().filter((_,i)=>i%3===2)),24);
  assert.equal(g.obstacleTriangles().length,g.obstacleColors().length);assert.equal(g.obstacleTriangles().length/9,152);
  assert.deepEqual([0,-10,-18,-22,-24,-34].map(z=>g.stage(z)),['Stairs & platform','Uneven rocks','50 cm wall','Return','Stairs & platform','Uneven rocks']);
  g.setStepHeight(0);g.setObstacleHeight(0);assert.equal(g.obstacleTriangles().length/9,10,'wall stays at its specified height when other obstacles are disabled');
  assert.ok(Math.abs(g.height(0,wall.z)-wall.top)<1e-10);
  g.setObstaclesEnabled(false);assert.equal(g.obstacleTriangles().length,0);assert.equal(g.height(0,wall.z),g.baseHeight(0,wall.z));
  g.setObstaclesEnabled(true);g.setCourse('stairs');assert.equal(g.length,8);g.setCourse('sequence');assert.equal(g.length,24);
});
test('combined body path is continuous at stairs, wall and the repeat boundary',()=>{
  const g=new Ground(.12);g.setCourse('sequence');let previous=g.rootHeight(0,0),maxChange=0;
  for(let z=.005;z<=48;z+=.005){const h=g.rootHeight(0,z);maxChange=Math.max(maxChange,Math.abs(h-previous));previous=h;assert.ok(Math.abs(h-g.rootHeight(0,z-24))<1e-10);}
  assert.ok(maxChange<.004);const wall=g.obstacles.find(o=>o.kind==='wall');
  assert.ok(Math.abs(g.rootHeight(0,wall.z)-wall.top)<1e-10);
  assert.ok(g.height(0,wall.minZ+1e-5)-g.height(0,wall.minZ-1e-5)>.49,'foot surface retains the vertical wall');
});
test('full-contact runtime preserves lengths, changes ground pose and restores exact live state',()=>{
  const d=data(),r=new Runtime(d),off=new Runtime(d);off.setGroundEnabled(false);
  const root=new Matrix4().makeTranslation(0,-d.runtime.measurement.rest_ground_y,0),ground=(x,z)=>.04+.03*x-.02*z;
  let changed=0;
  for(let k=0;k<30;k++){
    const w=r.step(k?1/60:0,root,ground),a=off.step(k?1/60:0,root,ground);
    for(const l of r.legs)for(let i=0;i<2;i++){
      const p=w[l.joints[i]].elements,q=w[l.joints[i+1]].elements,len=Math.hypot(p[12]-q[12],p[13]-q[13],p[14]-q[14]);
      assert.ok(Math.abs(len-l.segment_lengths_m[i])<1e-6);
    }
    changed=Math.max(changed,...r.names.map(n=>Math.abs(w[n].elements[13]-a[n].elements[13])));
  }
  assert.ok(changed>.01);assert.ok(r.queryCount>900);assert.equal(r.legs[0].contact.points.length,118);assert.equal(r.legs[0].clearance.points.length,32);
  const snap=r.snapshot();r.trigger('leap');r.step(1/60,root,ground);r.restore(snap);assert.deepEqual(r.snapshot(),snap);
  assert.throws(()=>r.step(1,root,ground));assert.equal(d.runtime.graph.layers[1].node.transitions[0].fade_s,.15);
});
test('payload uses public hash and index-bound anonymous joints without authoring paths',()=>{
  const d=data(),text=JSON.stringify(d);assert.match(d.glb_sha256,/^[a-f0-9]{64}$/);assert.ok(!/C:\\|local-work|source\.glb|contract_sha|provenance/.test(text));assert.ok(d.runtime.body.joint.startsWith('j'));
  for(const j of d.runtime.skeleton.joints)assert.equal(j.name,'j'+j.node);
  const bad=data();bad.runtime.limbs[0].solver='chain';assert.throws(()=>new Runtime(bad));
});
test('landing resumes a phase-matched gait before the leap recovery tail, with a real crossfade',()=>{
  const source=data(),motion=new Locomotion(source);motion.runtime.setGroundEnabled(false);
  assert.ok(Math.abs(landingTime(source.runtime.clips.leap)-1)<.002);
  assert.equal(source.runtime.clips.leap.duration_s,1.5);
  let sawBoth=false,sawFlight=false,returned=false,previousZ=0;
  for(let i=0;i<=330;i++){
    motion.step(i?1/60:0,()=>0);const d=motion.diagnostic();
    assert.ok(motion.z<=previousZ+1e-8,'fade-out must not reverse the root');previousZ=motion.z;
    const localRoot=motion.runtime.locals[motion.rootNode].p;
    assert.ok(Math.abs(localRoot.z-motion.rest[2])<1e-9,'baked travel is consumed once');
    if(d.weights.leap===1)sawFlight=true;
    if(motion.time>4&&d.weights.leap>0&&d.weights.sprint>0){
      sawBoth=true;const r=motion.runtime,node=r.byName[r.spec.body.joint].node;
      const run=r.sample('sprint',d.contributions.find(v=>v.clip==='sprint').time%r.spec.clips.sprint.duration_s),leap=r.sample('leap',d.phases.leap);
      const blended=run[node].p.clone().lerp(leap[node].p,d.weights.leap);
      assert.ok(r.locals[node].p.distanceTo(blended)<1e-8,'reported weights are applied to the actual pose');
    }
    if(motion.time>4.18&&motion.time<4.45){assert.equal(d.weights.leap,0);assert.equal(d.weights.sprint,1);returned=true;}
  }
  assert.ok(sawFlight&&sawBoth&&returned);
  assert.ok(source.tracks.leap.some(t=>t.node===motion.rootNode&&t.path==='translation'&&t.values.at(-1)<-.89),'sealed source keys stay intact');
});
test('landing policy tolerates variable dt and repeated jumps without late return or reverse motion',()=>{
  const m=new Locomotion(data());m.runtime.setGroundEnabled(false);let previous=0,returns=0;
  const dt=[1/30,1/60,1/45,1/75];
  for(let k=0;m.time<12;k++){
    const before=m.diagnostic();m.step(k?dt[k%dt.length]:0,()=>0);const after=m.diagnostic();
    assert.ok(m.z<=previous+1e-8);previous=m.z;
    if(before.weights.leap>0&&after.weights.leap===0){returns++;assert.ok(before.phases.leap<m.windows.leap.landing+m.fade+.04);}
  }
  assert.ok(returns>=2);
});
test('selected leap loops its pose and consumes authored travel once on every cycle without reversing',()=>{
  const source=data(),motion=new Locomotion(source);motion.reset('leap',.15,false);let previousZ=0;
  for(let k=0;k<=360;k++){
    motion.step(k?1/60:0,()=>0);assert.ok(motion.z<=previousZ+1e-9,'a loop seam must not pull the root backwards');previousZ=motion.z;
    assert.ok(Math.abs(motion.runtime.locals[motion.rootNode].p.z-motion.rest[2])<1e-9,'baked travel is not also applied locally');
  }
  assert.ok(Math.abs(motion.z+3.6)<1e-6,'four jumps each consume their authored 0.9 m');
  assert.equal(motion.runtime.spec.clips.leap.loop,true);assert.equal(source.runtime.clips.leap.loop,false);
  assert.ok(motion.diagnostic().phases.leap<1.5);assert.equal(motion.diagnostic().weights.sprint,0);
  const clock=motion.runtime.clocks['0'],phase=clock%source.runtime.clips.leap.duration_s,expected=motion.runtime.sample('leap',phase);
  for(const n of motion.runtime.spec.skeleton.joints){assert.ok(motion.runtime.locals[n.node].p.distanceTo(expected[n.node].p)<1e-9);assert.ok(Math.abs(motion.runtime.locals[n.node].q.dot(expected[n.node].q))>1-1e-9);}
});
test('selected leap handles variable steps across loop seams and restores the next solved frame exactly',()=>{
  const motion=new Locomotion(data()),ground=(x,z)=>.02*x;motion.reset('leap');motion.step(0,ground);
  for(let k=0;k<14;k++)motion.step(.1,ground);motion.step(.08,ground);
  const state=motion.snapshot(),pose=motion.runtime.snapshot();motion.step(.07,ground);const next=motion.snapshot(),solved=motion.runtime.snapshot();
  assert.ok(motion.diagnostic().phases.leap<.06);assert.equal(motion.diagnostic().cycles.leap,1);
  motion.restore(state);motion.runtime.restore(pose);motion.step(.07,ground);assert.deepEqual(motion.snapshot(),next);assert.deepEqual(motion.runtime.snapshot(),solved);
  let previousZ=motion.z;for(let k=0;k<100;k++){motion.step([.03,.07,.1][k%3],ground);assert.ok(motion.z<=previousZ+1e-9);previousZ=motion.z;}
  assert.ok(motion.z< -4.5);
});
test('restoring halfway through landing produces the same next solved pose and world motion',()=>{
  const motion=new Locomotion(data()),ground=(x,z)=>.02*x;
  for(let k=0;k<247;k++)motion.step(k?1/60:0,ground);
  assert.ok(motion.diagnostic().weights.leap>0&&motion.diagnostic().weights.leap<1);
  const state=motion.snapshot(),pose=motion.runtime.snapshot();
  motion.step(1/60,ground);const next=motion.snapshot(),solved=motion.runtime.snapshot();
  motion.restore(state);motion.runtime.restore(pose);motion.step(1/60,ground);
  assert.deepEqual(motion.snapshot(),next);assert.deepEqual(motion.runtime.snapshot(),solved);
});
test('four-legged horse cycles all five clips with distinct motion policies and no model-name assumptions',()=>{
  const source=JSON.parse(fs.readFileSync(new URL('../site/terrain/horse.json',import.meta.url))),motion=new Locomotion(source);
  motion.reset('all',.15,false);const seen=new Set();let previous=0,mixed=false;
  for(let k=0;k<=800;k++){
    motion.step(k?1/30:0,()=>0);const d=motion.diagnostic();seen.add(d.activeClip);
    assert.ok(motion.z<=previous+1e-9);previous=motion.z;
    assert.ok(Math.abs(Object.values(d.weights).reduce((a,b)=>a+b,0)-1)<1e-9);
    if(d.contributions.filter(r=>r.weight>0).length===2)mixed=true;
  }
  assert.equal(motion.runtime.legs.length,4);assert.deepEqual(seen,new Set(Object.keys(source.runtime.clips)));assert.ok(mixed&&motion.z< -1);
  for(const clip of ['idle','rear_up']){motion.reset(clip,.15,false);for(let k=0;k<100;k++)motion.step(k?.05:0,()=>0);assert.equal(motion.z,0,'stationary actions stay on their spot');}
  motion.reset('jump',.15,false);for(let k=0;k<=120;k++)motion.step(k?.05:0,()=>0);assert.ok(motion.z<-.5,'in-place jump uses its declared airborne gait reference');
  const renamed=structuredClone(source),mapping=Object.fromEntries(Object.keys(source.runtime.clips).map((c,i)=>[c,'motion'+i]));
  renamed.runtime.graph={layers:[{node:{type:'clip',clip:mapping.idle},mask_joints:null}]};
  renamed.runtime.clips=Object.fromEntries(Object.entries(source.runtime.clips).map(([c,v])=>[mapping[c],v]));renamed.tracks=Object.fromEntries(Object.entries(source.tracks).map(([c,v])=>[mapping[c],v]));
  renamed.demo.cycle=source.demo.cycle.map(r=>({...r,clip:mapping[r.clip],...(r.reference?{reference:mapping[r.reference]}:{})}));
  const generic=new Locomotion(renamed);generic.reset(mapping.jump,.15,false);for(let k=0;k<=120;k++)generic.step(k?.05:0,()=>0);assert.ok(Math.abs(generic.z-motion.z)<1e-9);
});

test('slope and jump recovery match frozen reference matrices near straight horse knees',()=>{
  const source=JSON.parse(fs.readFileSync(new URL('../site/terrain/horse.json',import.meta.url))),fixture=JSON.parse(fs.readFileSync(new URL('./terrain-reference.json',import.meta.url))),r=new Runtime(source);
  let previous=0,world;
  for(const f of fixture.frames){Object.assign(r.params,f.parameters);if(f.trigger)r.trigger(f.trigger);r.setGroundEnabled(f.ik);world=r.step(f.time-previous,new Matrix4().fromArray(f.root),(x,z)=>f.ground[0]+f.ground[1]*x+f.ground[2]*z);previous=f.time;}
  for(const [name,m] of Object.entries(fixture.expected))assert.ok(Math.max(...m.map((v,i)=>Math.abs(v-world[name].elements[i])))<2e-6,name+' matches independent Python reference');
});

test('single-clip default cycle loops, and landing exits honor requested repetitions',()=>{
  const d=data();d.runtime.clips={leap:d.runtime.clips.leap};d.tracks={leap:d.tracks.leap};d.demo.cycle=[d.demo.cycle.find(r=>r.clip==='leap')];
  const m=new Locomotion(d);m.reset('all',.15,false);for(let i=0;i<=180;i++)m.step(i?1/30:0,()=>0);
  assert.ok(Math.abs(m.z+3.6)<1e-6);assert.equal(m.diagnostic().weights.leap,1);
  assert.ok(Math.abs(m.exitTime({...d.demo.cycle[0],cycles:3})-4)<.002);
});

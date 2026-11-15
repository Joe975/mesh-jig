import * as T from './vendor/three.module.js';
import {GLTFLoader} from './vendor/addons/loaders/GLTFLoader.js';
import {OrbitControls} from './vendor/addons/controls/OrbitControls.js';
import {Locomotion} from './locomotion.js';
import {Ground} from './ground.js';
import {sha256} from './hash.js';
import {createSky} from './sky.js';
import {playbackSpeed} from '../viewer-link.js';
import {defaultAnimation} from '../animation-cycle.js';

// Course heights are fixed at the requested maxima; playback speed is the only live slider.
export const fixedParameters=Object.freeze({height:.2,obstacle:.32,step:.18,fade:.15});
export const settings={speed:playbackSpeed};
const defaults=()=>({...Object.fromEntries(Object.entries(settings).map(([k,v])=>[k,v.default])),...fixedParameters});
let active=null;
function release(scene) {
  const seen=new Set();scene.traverse(o=>{
    for(const v of [o.geometry,o.skeleton,...(Array.isArray(o.material)?o.material:[o.material])])if(v&&!seen.has(v)){seen.add(v);if(v.isMaterial)for(const x of Object.values(v))if(x?.isTexture&&!seen.has(x)){seen.add(x);x.source?.data?.close?.();x.dispose();}v.dispose?.();}
  });
}
async function checkedFetch(url,signal){const r=await fetch(url,{signal});if(!r.ok)throw new Error(`Could not load demo (${r.status})`);return r;}

export async function createTerrain(host,glbURL,options={}) {
  active?.dispose();const abort=new AbortController();let disposed=false,renderer,controls,observer,resizeObserver,raf=0,worker,downloadURL;
  const scene=new T.Scene();scene.background=new T.Color('#ead8b8');scene.fog=new T.Fog('#ead8b8',10,24);const sky=createSky();scene.add(sky);
  let captureAbort=null;
  const api={get ready(){return !!runtime&&!disposed;},dispose(){if(disposed)return;disposed=true;captureAbort?.abort();abort.abort();cancelAnimationFrame(raf);observer?.disconnect();resizeObserver?.disconnect();document.removeEventListener('visibilitychange',wake);worker?.terminate();controls?.dispose();release(scene);renderer?.dispose();renderer?.forceContextLoss();if(downloadURL)URL.revokeObjectURL(downloadURL);delete host.terrain;host.replaceChildren();if(active===api)active=null;options.onDispose?.();}};
  options.signal?.addEventListener('abort',()=>api.dispose(),{once:true});
  active=api;
  host.innerHTML='<div class="terrain-viewport" tabindex="0" aria-label="Terrain and animation viewer"></div><div class="terrain-controls"><button type="button" data-action="pause">Pause</button><label><input type="checkbox" data-action="ik" checked> IK</label><button type="button" data-action="reset">Reset</button><button type="button" data-action="gif">Export terrain GIF</button><button type="button" data-action="cancel" hidden>Cancel export</button></div><div class="terrain-knobs"></div><p class="terrain-status" role="status" aria-live="polite">Loading terrain demo…</p>';
  const viewport=host.querySelector('.terrain-viewport'),status=host.querySelector('.terrain-status');
  const button=name=>host.querySelector(`[data-action="${name}"]`);
  let params=defaults(),mode='all',playing=true,inView=true,last=0,time=0,z=0,busy=false,rate=1;
  let runtime,locomotion,data,ground,model,nodes,shadows=[],terrainMeshes=[],obstacleMeshes=[],samples={frames:[],updates:[]},fpsCap=30,acc=0;
  const camera=new T.PerspectiveCamera(35,1,.05,60);
  function wake(){cancelAnimationFrame(raf);last=0;acc=0;if(!disposed&&inView&&!document.hidden&&!busy)raf=requestAnimationFrame(frame);}
  try {
    const registry=await (await checkedFetch(new URL('./characters.json',import.meta.url),abort.signal)).json(),profile=registry.characters?.[options.character];
    if(registry.version!==1||!profile||!/^[a-z][a-z0-9-]{0,47}\.json$/.test(profile.data)||new URL('../'+profile.model,import.meta.url).href!==new URL(glbURL,location.href).href)throw new Error('This character has no registered terrain playback');
    const response=await checkedFetch(new URL(profile.data,import.meta.url),abort.signal),payload=await response.arrayBuffer();
    if(await sha256(payload)!==profile.sha256)throw new Error('Terrain package verification failed');
    data=JSON.parse(new TextDecoder().decode(payload));if(data.glb_sha256!==profile.glb_sha256)throw new Error('Terrain character binding differs');
    mode=defaultAnimation(Object.keys(data.runtime.clips));
    const bytes=await (await checkedFetch(glbURL,abort.signal)).arrayBuffer();
    const digest=await sha256(bytes);
    if(digest!==data.glb_sha256)throw new Error('This model differs from the reviewed terrain playback');
    const loaded=await new GLTFLoader().parseAsync(bytes,new URL('.',glbURL).href);
    if(disposed){release(loaded.scene);return api;}
    model=loaded.scene;model.traverse(o=>{if(o.isSkinnedMesh)o.frustumCulled=false;});scene.add(model);nodes=await Promise.all(data.nodes.map((_,i)=>loaded.parser.getDependency('node',i)));
    if(disposed){release(model);return api;}
    locomotion=new Locomotion(data);runtime=locomotion.runtime;ground=new Ground(params.height,params.obstacle);ground.setStepHeight(params.step);ground.setCourse('sequence');
    renderer=new T.WebGLRenderer({antialias:false,alpha:false,powerPreference:'low-power',preserveDrawingBuffer:true});
    renderer.setPixelRatio(1);renderer.outputColorSpace=T.SRGBColorSpace;renderer.toneMapping=T.ACESFilmicToneMapping;renderer.toneMappingExposure=1.2;
    viewport.append(renderer.domElement);renderer.domElement.style.touchAction='pan-y';
    camera.position.fromArray(data.demo.camera.position);controls=new OrbitControls(camera,renderer.domElement);controls.target.fromArray(data.demo.camera.target);const distance=camera.position.distanceTo(controls.target);controls.minDistance=Math.max(.1,distance*.35);controls.maxDistance=distance*2;controls.maxPolarAngle=Math.PI*.49;controls.enableDamping=false;controls.update();controls.addEventListener('change',()=>{if(!playing)render();});
    scene.add(new T.HemisphereLight(0xffffff,0x545f55,2.4));const light=new T.DirectionalLight(0xfff0d6,3.0);light.position.set(4,6,3);scene.add(light);
    const mat=new T.MeshStandardMaterial({color:'#879285',roughness:1});
    for(let i=0;i<3;i++){const m=new T.Mesh(new T.BufferGeometry(),mat);terrainMeshes.push(m);scene.add(m);}
    const stone=new T.MeshStandardMaterial({vertexColors:true,roughness:1,flatShading:true});
    for(let i=0;i<3;i++){const m=new T.Mesh(new T.BufferGeometry(),stone);obstacleMeshes.push(m);scene.add(m);}
    const stamp=document.createElement('canvas');stamp.width=stamp.height=32;const ink=stamp.getContext('2d'),gradient=ink.createRadialGradient(16,16,0,16,16,16);gradient.addColorStop(0,'rgba(0,0,0,.3)');gradient.addColorStop(1,'rgba(0,0,0,0)');ink.fillStyle=gradient;ink.fillRect(0,0,32,32);const shadowMat=new T.MeshBasicMaterial({map:new T.CanvasTexture(stamp),transparent:true,depthWrite:false});
    for(let i=0;i<runtime.legs.length;i++){const m=new T.Mesh(new T.PlaneGeometry(.32,.32),shadowMat);m.rotation.x=-Math.PI/2;shadows.push(m);scene.add(m);}
    rebuildGround();reset();
    for(const [key,info]of Object.entries(settings)){
      const label=document.createElement('label');label.textContent='Playback speed ';
      const input=document.createElement('input');input.type='range';input.min=info.min;input.max=info.max;input.step=.05;input.value=params[key];input.setAttribute('aria-label',label.textContent.trim());
      const output=document.createElement('output');const update=()=>{params[key]=Number(input.value);output.textContent=`${params[key].toFixed(2)} ${info.unit}`;options.onSpeedChange?.(params[key]);};input.oninput=update;label.append(input,output);host.querySelector('.terrain-knobs').append(label);update();
    }
    button('pause').onclick=()=>{playing=!playing;button('pause').textContent=playing?'Pause':'Play';wake();};
    button('ik').onchange=()=>{runtime.setGroundEnabled(button('ik').checked);if(!playing){runtime.groundWeight=button('ik').checked?1:0;advance(0);render();}};
    button('reset').onclick=()=>{params=defaults();host.querySelectorAll('.terrain-knobs input').forEach((input,i)=>{input.value=params[Object.keys(settings)[i]];input.oninput();});configureGround();button('ik').checked=true;reset();rebuildGround();};
    button('gif').onclick=()=>capture();
    observer=new IntersectionObserver(entries=>{inView=entries[0].isIntersecting;wake();},{threshold:0});observer.observe(host);
    resizeObserver=new ResizeObserver(()=>{const w=viewport.clientWidth,h=viewport.clientHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();render();});resizeObserver.observe(viewport);
    document.addEventListener('visibilitychange',wake);
    renderer.domElement.addEventListener('webglcontextlost',e=>{e.preventDefault();if(!disposed){status.textContent='Graphics paused. Return to clips and retry the terrain demo.';playing=false;cancelAnimationFrame(raf);}});
    Object.assign(api,{play(name){mode=name;reset();},pause(){playing=false;button('pause').textContent='Play';},snapshot,sample,restore,timings:()=>structuredClone(samples),setParameter(key,value){const info=settings[key];if(!info||!Number.isFinite(value)||value<info.min||value>info.max)throw new Error('Invalid parameter');const i=Object.keys(settings).indexOf(key),input=host.querySelectorAll('.terrain-knobs input')[i];input.value=value;input.oninput();},step(dt){advance(dt);render();}});
    host.terrain=api;updateStatus();wake();return api;
  } catch(error){api.dispose();throw error;}

  function configureGround(){ground.setHeight(params.height);ground.setObstacleHeight(params.obstacle);ground.setStepHeight(params.step);ground.setCourse('sequence');ground.setObstaclesEnabled(true);}
  function rebuildGround(){
    const geometry=new T.BufferGeometry();geometry.setAttribute('position',new T.Float32BufferAttribute(ground.triangles(),3));geometry.computeVertexNormals();
    const obstacles=new T.BufferGeometry();obstacles.setAttribute('position',new T.Float32BufferAttribute(ground.obstacleTriangles(),3));obstacles.setAttribute('color',new T.Float32BufferAttribute(ground.obstacleColors(),3));obstacles.computeVertexNormals();
    for(const m of terrainMeshes){m.geometry.dispose();m.geometry=geometry;}
    for(const m of obstacleMeshes){m.geometry.dispose();m.geometry=obstacles;m.visible=ground.obstaclesEnabled&&ground.activeHeight>0;}
    apply();render();
  }
  function reset(){locomotion.reset(mode,params.fade,button('ik').checked);runtime=locomotion.runtime;advance(0);render();}
  function advance(dt){
    locomotion.step(dt,(x,z)=>ground.height(x,z),(x,z)=>ground.rootHeight(x,z));({time,z}=locomotion);apply();
  }
  function apply(){
    if(!runtime?.world||!Object.keys(runtime.world).length)return;
    const jointAt=Object.fromEntries(runtime.spec.skeleton.joints.map(j=>[j.node,j.name])),world=[];
    function visit(i){if(world[i])return world[i];const parent=runtime.parents[i],pw=parent===null?new T.Matrix4():visit(parent),p=runtime.locals[i];let local;
      if(jointAt[i]){world[i]=runtime.world[jointAt[i]].clone();local=pw.clone().invert().multiply(world[i]);}
      else{local=data.nodes[i].matrix?new T.Matrix4().fromArray(data.nodes[i].matrix):new T.Matrix4().compose(p.p,p.q,p.s);world[i]=pw.clone().multiply(local);}
      nodes[i].matrixAutoUpdate=false;nodes[i].matrix.copy(local);return world[i];}
    data.nodes.forEach((_,i)=>visit(i));model.position.z=-z;model.updateMatrixWorld(true);
    const tile=Math.floor(z/ground.length);terrainMeshes.forEach((m,i)=>m.position.z=(tile+i-1)*ground.length-z);
    obstacleMeshes.forEach((m,i)=>m.position.z=(tile+i-1)*ground.length-z);
    shadows.forEach(m=>m.visible=false);Object.values(runtime.targets).forEach((p,i)=>{shadows[i].position.set(p.point[0],ground.height(p.point[0],p.point[2])+.004,p.point[2]-z);shadows[i].visible=p.planted;});
  }
  function render(){if(renderer&&!disposed){sky.position.copy(camera.position);renderer.render(scene,camera);}}
  function updateStatus(){const d=locomotion.diagnostic(),names=d.contributions.filter(c=>c.weight>0).map(c=>c.clip.replace(/_/g,' '));status.textContent=(mode==='all'?'All animations · ':'')+names.join(' → ');}
  function frame(now){
    raf=0;if(disposed||!inView||document.hidden||busy)return;
    const dt=last?Math.min((now-last)/1000,.0666667):0;last=now;acc+=dt;
    if(acc>=1/fpsCap){const elapsed=acc;acc=0;const begin=performance.now();if(playing){let left=elapsed*params.speed*rate;while(left>1e-9){const step=Math.min(1/60,left);advance(step);left-=step;}}samples.updates.push(performance.now()-begin);samples.frames.push(elapsed*1000);if(samples.frames.length>300){samples.frames.shift();samples.updates.shift();}render();updateStatus();}
    if(playing)raf=requestAnimationFrame(frame);
  }
  function snapshot(){return {time,z,mode,playing,params:{...params},ground:'sequence',locomotion:locomotion.snapshot(),runtime:runtime.snapshot(),camera:camera.position.toArray(),target:controls.target.toArray()};}
  function restore(s){({time,z,mode,playing}=s);params={...s.params,...fixedParameters};locomotion.restore(s.locomotion);runtime.restore(s.runtime);configureGround();camera.position.fromArray(s.camera);controls.target.fromArray(s.target);controls.update();rebuildGround();wake();}
  function sample(){
    const joints=Object.fromEntries(runtime.names.map(n=>[n,nodes[runtime.byName[n].node].getWorldPosition(new T.Vector3()).add(new T.Vector3(0,0,z)).toArray()]));
    const surface=[];model.traverse(mesh=>{if(mesh.isSkinnedMesh){mesh.skeleton.update();const p=mesh.geometry.attributes.position;for(let i=0;i<p.count;i+=Math.max(1,Math.floor(p.count/24))){const v=new T.Vector3().fromBufferAttribute(p,i);mesh.applyBoneTransform(i,v);mesh.localToWorld(v);v.z+=z;surface.push(v.toArray());}}});
    const soles={};for(const [leg,ids] of Object.entries(data.observations)){const points=ids.map(([ni,pi,vi])=>{const node=nodes[ni],mesh=node.isMesh?node:node.children.filter(o=>o.isMesh)[pi];mesh.skeleton.update();const v=new T.Vector3().fromBufferAttribute(mesh.geometry.attributes.position,vi);mesh.applyBoneTransform(vi,v);mesh.localToWorld(v);v.z+=z;return v;});const gaps=points.map(p=>p.y-ground.height(p.x,p.z));soles[leg]={minGap:Math.min(...gaps),maxGap:Math.max(...gaps),points:points.map(p=>p.toArray())};}
    const expectedJoints=Object.fromEntries(runtime.names.map(n=>[n,new T.Vector3().setFromMatrixPosition(runtime.world[n]).toArray()]));
    return {time,z,joints,expectedJoints,soles,surface,targets:structuredClone(runtime.targets),queries:runtime.queryCount,drawCalls:renderer.info.render.calls,triangles:renderer.info.render.triangles,params:{...params},groundWeight:runtime.groundWeight,ground:'sequence',courseLength:ground.length,stage:ground.stage(z),rootHeight:ground.rootHeight(0,z),locomotion:locomotion.diagnostic(),obstacles:ground.obstacles.map(o=>({kind:o.kind,x:o.x,z:o.z,top:o.top,base:o.base})),sky:sky.visible};
  }
  async function capture(){
    if(busy)return;busy=true;cancelAnimationFrame(raf);const state=snapshot(),cancel=new AbortController(),disabled=[...host.querySelectorAll('input,select,button')].map(el=>[el,el.disabled]);captureAbort=cancel;
    button('cancel').hidden=false;button('cancel').onclick=()=>cancel.abort();disabled.forEach(([el])=>el.disabled=true);button('cancel').disabled=false;
    const stopped=()=>cancel.abort();document.addEventListener('visibilitychange',stopped);window.addEventListener('scroll',stopped,{passive:true});window.addEventListener('resize',stopped);
    const timer=setTimeout(stopped,90000);const canvas=document.createElement('canvas');canvas.width=Math.min(640,renderer.domElement.width);canvas.height=Math.round(canvas.width*renderer.domElement.height/renderer.domElement.width);const ctx=canvas.getContext('2d',{willReadFrequently:true});
    let pendingReject;const onAbort=()=>pendingReject?.(new DOMException('Export cancelled','AbortError'));cancel.signal.addEventListener('abort',onAbort);
    try{
      worker=new Worker(new URL('../gif-worker.js',import.meta.url),{type:'module'});
      const send=(message,transfer=[])=>new Promise((resolve,reject)=>{pendingReject=reject;worker.onmessage=e=>e.data.error?reject(new Error(e.data.error)):resolve(e.data);worker.onerror=()=>reject(new Error('GIF encoder failed'));worker.postMessage(message,transfer);});
      await send({type:'init',width:canvas.width,height:canvas.height});
      for(let i=0;i<60;i++){cancel.signal.throwIfAborted();if(disposed)throw new Error('Viewer closed');status.textContent=`Exporting terrain excerpt ${i+1}/60`;render();ctx.drawImage(renderer.domElement,0,0,canvas.width,canvas.height);ctx.fillStyle='rgba(15,23,36,.86)';ctx.fillRect(0,canvas.height-26,canvas.width,26);ctx.fillStyle='#fff';ctx.font='12px system-ui';ctx.fillText('Live terrain · 3 s excerpt · IK '+(runtime.groundOn?'on':'off'),8,canvas.height-9);const pixels=ctx.getImageData(0,0,canvas.width,canvas.height);await send({type:'frame',buffer:pixels.data.buffer,delay:50},[pixels.data.buffer]);for(let k=0;k<3;k++)advance(1/60*params.speed);}
      const result=await send({type:'finish'});cancel.signal.throwIfAborted();if(downloadURL)URL.revokeObjectURL(downloadURL);downloadURL=URL.createObjectURL(new Blob([result.buffer],{type:'image/gif'}));const a=document.createElement('a');a.href=downloadURL;a.download=options.character+'-terrain-excerpt.gif';a.click();status.textContent='3 second terrain excerpt downloaded.';
    }catch(e){if(!disposed)status.textContent=e.name==='AbortError'?'Export cancelled.':e.message;}
    finally{clearTimeout(timer);cancel.signal.removeEventListener('abort',onAbort);worker?.terminate();worker=null;captureAbort=null;document.removeEventListener('visibilitychange',stopped);window.removeEventListener('scroll',stopped);window.removeEventListener('resize',stopped);busy=false;if(!disposed){restore(state);disabled.forEach(([el,was])=>el.disabled=was);button('cancel').hidden=true;}}
  }
}

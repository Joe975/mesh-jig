import * as THREE from 'three';
import { GLTFLoader } from '../terrain/vendor/addons/loaders/GLTFLoader.js';
import { fragmentShader, vertexShader } from './volume.js?v=6';
import { cloudPassAmount, cuePose, flockCues, routePoint, wrap } from './tour-motion.js?v=7';

const canvas = document.querySelector('#view');
const status = document.querySelector('#status');
const routeInput = document.querySelector('#route');
const motion = document.querySelector('#motion');
const chapters = [...document.querySelectorAll('[data-clip]')];
const currentClip = document.querySelector('#current-clip');
const follow = document.querySelector('#follow');
const buildCost = document.querySelector('#build-cost');
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const clamp = THREE.MathUtils.clamp;
const worldPoint = ([x, y, z]) => new THREE.Vector3(x, z, -y);
// Seconds behind the leader, lateral metres, and vertical metres.
const formation = [[0,0,0], [0.54,13,6], [0.70,-13,1], [1.04,0,13], [1.32,-7,-9],
  [1.54,17,-5], [1.75,-16,10], [2.05,10,6], [2.30,-10,-3], [2.45,3,14]];

async function asset(path, read) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path} returned HTTP ${response.status}`);
  return response[read]();
}

async function cloudDensity() {
  if (!('DecompressionStream' in window)) throw new Error('Gzip decompression is unavailable.');
  const { gzip } = await asset('clouds.json', 'json');
  const bytes = Uint8Array.from(atob(gzip), (ch) => ch.charCodeAt(0));
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).arrayBuffer();
}

function addWhale(scene, gltf, info, cues, placement, flyPhase) {
  const mixer = new THREE.AnimationMixer(gltf.scene);
  const clips = new Map(gltf.animations.map((clip) => [clip.name.toLowerCase(), clip]));
  if (!clips.has('fly')) throw new Error(`${info.model} has no Fly animation.`);
  const available = cues.filter(({clip}) => clips.has(clip));
  const actions = new Map(['fly', ...available.map((cue) => cue.clip)].map((name) => {
    const clip = clips.get(name);
    const action = mixer.clipAction(clip);
    action.setEffectiveWeight(name === 'fly' ? 1 : 0).play();
    return [name, action];
  }));
  mixer.update(0);
  gltf.scene.updateMatrixWorld(true);
  const bounds = new THREE.Box3().setFromObject(gltf.scene, true);
  const center = bounds.getCenter(new THREE.Vector3());
  const radius = bounds.getBoundingSphere(new THREE.Sphere()).radius;
  if (!Number.isFinite(radius) || radius <= 0) throw new Error('The whale has invalid bounds.');
  const scale = 8.4/radius;
  const whale = new THREE.Group();
  const meshes = [];
  gltf.scene.traverse((part) => {
    if (!part.isMesh) return;
    part.userData.whale = info.id;
    meshes.push(part);
    // The GLB's vertex paint supplies the color. Treat the skin as a dielectric.
    for (const mat of Array.isArray(part.material) ? part.material : [part.material]) {
      mat.metalness = 0;
      mat.roughness = 0.9;
      mat.needsUpdate = true;
    }
  });
  gltf.scene.position.sub(center);
  whale.add(gltf.scene);
  whale.scale.setScalar(scale);
  scene.add(whale);
  const root = gltf.scene.getObjectByName('root') ?? meshes.find((mesh) => mesh.isSkinnedMesh)?.skeleton.bones[0];
  const restRoot = root ? whale.worldToLocal(root.getWorldPosition(new THREE.Vector3())) : null;
  return { ...info, whale, meshes, mixer, clips, actions, root, restRoot, cues: available,
    placement, flyPhase, pivot: new THREE.Vector3(), heading: 0, localSeconds: 0, pose: { clip: 'fly', weight: 0 } };
}

function lightScene(scene) {
  scene.add(new THREE.HemisphereLight(0xffefea, 0xa6a5e4, 2.05));
  const peach = new THREE.DirectionalLight(0xffdfba, 1.9);
  peach.position.set(0.41, 0.73, 0.58);
  scene.add(peach);
  const rim = new THREE.DirectionalLight(0xffd1a3, 1.05);
  rim.position.set(0.18, 0.55, -0.72);
  scene.add(rim);
}

async function openCanyon() {
  const context = canvas.getContext('webgl2', { antialias: false, alpha: false });
  if (!context) {
    status.textContent = 'This browser can show the still. Open it in a WebGL 2 browser to look around.';
    return;
  }

  const [layout, data, tour, roster] = await Promise.all([
    asset('scene.json', 'json'), cloudDensity(), asset('pillar-tour.json', 'json'), asset('flock.json', 'json'),
  ]);
  if (data.byteLength !== 2*layout.cells.reduce((a, b) => a*b, 1)) throw new Error('Incomplete cloud texture.');
  const volume = new Uint8Array(data);
  const clouds = new THREE.Data3DTexture(volume, ...layout.cells);
  clouds.format = THREE.RGFormat;
  clouds.minFilter = THREE.LinearFilter;
  clouds.magFilter = THREE.LinearFilter;
  clouds.unpackAlignment = 1;
  clouds.needsUpdate = true;

  const renderer = new THREE.WebGLRenderer({ canvas, context, antialias: false, alpha: false });
  renderer.setClearColor(0x000000);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.25, 850);
  const seed = crypto.getRandomValues(new Uint32Array(1))[0];
  const schedules = flockCues(tour.cues, roster.length, seed);
  const loader = new GLTFLoader();
  const flock = await Promise.all(roster.map(async (info, index) => {
    const gltf = await loader.loadAsync(info.src);
    return addWhale(scene, gltf, info, schedules[index], formation[index], (seed%300)/100 + index*0.67);
  }));
  const byId = new Map(flock.map((member) => [member.id, member]));
  for (const member of flock) follow.add(new Option(member.model, member.id));
  lightScene(scene);

  const target = new THREE.WebGLRenderTarget(1, 1, {
    minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, depthTexture: new THREE.DepthTexture(1, 1),
  });
  const composite = new THREE.Scene();
  const material = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: {
      clouds: { value: clouds }, colorBuffer: { value: target.texture }, depthBuffer: { value: target.depthTexture },
      inverseProjection: { value: camera.projectionMatrixInverse }, cameraWorld: { value: camera.matrixWorld },
      cloudMin: { value: new THREE.Vector3(...layout.min) }, cloudMax: { value: new THREE.Vector3(...layout.max) },
      flightFocus: { value: new THREE.Vector3() }, passage: { value: 0 },
    },
    vertexShader, fragmentShader, depthWrite: false, depthTest: false,
  });
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
  quad.frustumCulled = false;
  composite.add(quad);
  const screenCamera = new THREE.Camera();

  const onRoute = (progress) => worldPoint(routePoint(tour.route, progress));
  const pivot = new THREE.Vector3();
  const offset = new THREE.Vector3();
  const direction = new THREE.Vector3();
  const ray = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  const pointers = new Map();
  const labels = new Map(chapters.map((button) => [button.dataset.clip, button.textContent]));
  let selected = flock[0];
  let tap = null;
  let seconds = 0;
  let azimuth = 1.22;
  let elevation = 0.34;
  let distance = 58;
  let featured;
  let smileTurn = 0;
  let dirty = true;
  let playing = !reduceMotion;
  let previousFrame = 0;

  function orbitDistance() {
    const [width, height, depth] = layout.cells;
    const [xmin, ymin, zmin] = layout.min;
    const [xmax, ymax, zmax] = layout.max;
    for (let step = 13.5; step <= distance + 1; step += 1.75) {
      const x = Math.round((pivot.x + direction.x*step - xmin)*(width - 1)/(xmax - xmin));
      const y = Math.round((-(pivot.z + direction.z*step) - ymin)*(height - 1)/(ymax - ymin));
      const z = Math.round((pivot.y + direction.y*step - zmin)*(depth - 1)/(zmax - zmin));
      if (x < 0 || x >= width || y < 0 || y >= height || z < 0 || z >= depth) continue;
      if (volume[2*((z*height + y)*width + x)] > 42) return Math.max(Math.min(32, distance), step - 2.5);
    }
    return distance;
  }

  function pointCamera() {
    const angle = selected.heading + azimuth + smileTurn;
    direction.set(Math.sin(angle)*Math.cos(elevation), Math.sin(elevation), Math.cos(angle)*Math.cos(elevation));
    const mist = material.uniforms.passage.value;
    const arm = THREE.MathUtils.lerp(orbitDistance(), distance, mist);
    camera.position.copy(pivot).addScaledVector(direction, arm);
    camera.lookAt(pivot);
    camera.updateMatrixWorld();
    material.uniforms.flightFocus.value.set(pivot.x, -pivot.z, pivot.y);
    dirty = true;
  }

  function moveWhale(member) {
    const [delay, side, height] = member.placement;
    const local = wrap(seconds - delay, tour.duration);
    member.localSeconds = local;
    const progress = local/tour.duration;
    const tangent = onRoute(progress + 0.003).sub(onRoute(progress - 0.003));
    const heading = Math.atan2(-tangent.x, -tangent.z);
    const { whale, mixer, clips, actions, root, restRoot, cues } = member;
    member.heading = heading;
    whale.position.copy(onRoute(progress)).add(offset.set(side*Math.cos(heading), height, -side*Math.sin(heading)));
    whale.rotation.y = heading;
    const pose = cuePose(cues, clips, local);
    member.pose = pose;
    const fly = actions.get('fly');
    fly.time = wrap(local + member.flyPhase, clips.get('fly').duration);
    fly.setEffectiveWeight(1 - pose.weight);
    for (const cue of cues) {
      const action = actions.get(cue.clip);
      action.setEffectiveWeight(cue.clip === pose.clip ? pose.weight : 0);
      if (cue.clip === pose.clip) action.time = pose.time;
    }
    mixer.update(0);
    whale.updateMatrixWorld(true);
    if (root) {
      offset.copy(restRoot).applyMatrix4(whale.matrixWorld).sub(whale.position);
      root.getWorldPosition(member.pivot).sub(offset);
    } else {
      member.pivot.copy(whale.position);
    }
  }

  function followPose() {
    const { pose, localSeconds } = selected;
    pivot.copy(selected.pivot);
    smileTurn = pose.clip === 'smile-wink' ? 0.8*pose.weight : 0;
    material.uniforms.passage.value = cloudPassAmount(tour.passages, localSeconds);

    const label = pose.weight >= 0.5 ? pose.clip : 'fly';
    if (label !== featured) {
      featured = label;
      currentClip.textContent = featured === 'fly' ? 'Flying' : labels.get(featured);
      for (const button of chapters) button.setAttribute('aria-pressed', String(button.dataset.clip === featured));
    }
    pointCamera();
  }

  function setTourTime(time, updateSlider = true) {
    seconds = wrap(time, tour.duration);
    for (const member of flock) moveWhale(member);
    if (updateSlider) routeInput.value = String(Math.round(seconds/tour.duration*1000));
    followPose();
  }

  function selectWhale(id) {
    const member = byId.get(id);
    if (!member) return;
    selected = member;
    follow.value = member.id;
    buildCost.textContent = `Est. build ${member.cost}`;
    for (const chapter of chapters) chapter.disabled = !member.clips.has(chapter.dataset.clip);
    followPose();
  }

  function pickWhale(x, y) {
    const box = canvas.getBoundingClientRect();
    ndc.set(2*(x - box.left)/box.width - 1, 1 - 2*(y - box.top)/box.height);
    ray.setFromCamera(ndc, camera);
    const candidates = flock.filter((member) => ray.ray.distanceSqToPoint(member.pivot) < 144);
    const hit = ray.intersectObjects(candidates.flatMap((member) => member.meshes), false)[0];
    if (hit) selectWhale(hit.object.userData.whale);
  }

  function look(dx, dy) {
    azimuth = wrap(azimuth + dx, 2*Math.PI);
    elevation = clamp(elevation + dy, -1.30, 1.30);
    pointCamera();
  }

  function zoom(scale) {
    distance = clamp(distance*scale, 14, 76);
    pointCamera();
  }

  function updateMotionButton() {
    motion.textContent = playing ? 'Pause loop' : 'Play loop';
    motion.setAttribute('aria-pressed', String(playing));
  }

  function resize() {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!width || !height) return;
    const ratio = Math.min(1, 880/width, 680/height);
    const pixelsWide = Math.round(width*ratio);
    const pixelsHigh = Math.round(height*ratio);
    renderer.setSize(pixelsWide, pixelsHigh, false);
    target.setSize(pixelsWide, pixelsHigh);
    camera.aspect = width/height;
    const horizontal = THREE.MathUtils.degToRad(58);
    camera.fov = Math.max(42, THREE.MathUtils.radToDeg(2*Math.atan(Math.tan(horizontal/2)/camera.aspect)));
    camera.updateProjectionMatrix();
    dirty = true;
  }

  function draw(time) {
    const span = time - previousFrame;
    if (span >= 33 || dirty) {
      if (playing && !document.hidden) setTourTime(seconds + Math.min(0.25, span/1000));
      if (dirty) {
        renderer.setRenderTarget(target);
        renderer.render(scene, camera);
        renderer.setRenderTarget(null);
        renderer.render(composite, screenCamera);
        dirty = false;
      }
      previousFrame = time;
    }
    requestAnimationFrame(draw);
  }

  new ResizeObserver(resize).observe(canvas);
  canvas.addEventListener('pointerdown', (event) => {
    if (pointers.size >= 2) return;
    canvas.setPointerCapture(event.pointerId);
    pointers.set(event.pointerId, [event.clientX, event.clientY]);
    tap = pointers.size === 1 ? { id: event.pointerId, x: event.clientX, y: event.clientY } : null;
  });
  canvas.addEventListener('pointermove', (event) => {
    const before = pointers.get(event.pointerId);
    if (!before) return;
    const after = [event.clientX, event.clientY];
    pointers.set(event.pointerId, after);
    if (tap && Math.hypot(tap.x - after[0], tap.y - after[1]) < 8) return;
    tap = null;
    if (pointers.size === 1) {
      look((before[0] - after[0])*0.005, (after[1] - before[1])*0.005);
      return;
    }
    const [, partner] = [...pointers.entries()].find(([id]) => id !== event.pointerId);
    const oldSpan = Math.hypot(before[0] - partner[0], before[1] - partner[1]);
    const newSpan = Math.hypot(after[0] - partner[0], after[1] - partner[1]);
    if (newSpan > 8 && oldSpan > 8) zoom(oldSpan/newSpan);
  });
  for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) {
    canvas.addEventListener(type, (event) => {
      if (type === 'pointerup' && tap?.id === event.pointerId) pickWhale(event.clientX, event.clientY);
      if (tap?.id === event.pointerId) tap = null;
      pointers.delete(event.pointerId);
    });
  }
  canvas.addEventListener('wheel', (event) => {
    event.preventDefault();
    zoom(Math.exp(clamp(event.deltaY, -200, 200)*0.002));
  }, { passive: false });
  canvas.addEventListener('keydown', (event) => {
    switch (event.key.toLowerCase()) {
      case 'arrowleft': look(-0.10, 0); break;
      case 'arrowright': look(0.10, 0); break;
      case 'arrowup': look(0, 0.10); break;
      case 'arrowdown': look(0, -0.10); break;
      case 'w': setTourTime(seconds + 1); break;
      case 's': setTourTime(seconds - 1); break;
      case '+': case '=': zoom(0.90); break;
      case '-': zoom(1.11); break;
      default: return;
    }
    event.preventDefault();
  });
  canvas.addEventListener('webglcontextlost', (event) => {
    event.preventDefault();
    canvas.removeAttribute('data-ready');
    status.hidden = false;
    status.textContent = 'Graphics paused. Reload the page to reopen the viewer.';
  });
  routeInput.addEventListener('input', () => setTourTime(Number(routeInput.value)/1000*tour.duration, false));
  follow.addEventListener('change', () => selectWhale(follow.value));
  for (const button of chapters) {
    button.addEventListener('click', () => {
      const name = button.dataset.clip;
      const cue = selected.cues.find(({ clip }) => clip === name);
      const local = cue ? cue.start + (playing ? 0 : selected.clips.get(name).duration*0.52) : 0;
      setTourTime(local + selected.placement[0]);
    });
  }
  motion.addEventListener('click', () => { playing = !playing; updateMotionButton(); });

  updateMotionButton();
  setTourTime(0);
  selectWhale(flock[0].id);
  resize();
  previousFrame = performance.now();
  draw(previousFrame);
  canvas.setAttribute('data-ready', '');
  status.hidden = true;
  follow.parentElement.hidden = false;
}

openCanyon().catch((error) => {
  console.error(error);
  status.textContent = 'The scene did not load. Reload to try again.';
});

// Pure timing and state helpers, shared by the browser exporter and deterministic tests.
export function capturePlan(duration, currentTime = 0, rate = 1) {
  if (!(Number.isFinite(duration) && duration > 0 && Number.isFinite(currentTime) && Number.isFinite(rate) && rate > 0))
    throw new Error('Choose a loaded animation with forward playback first.');
  const cycle = duration / rate;
  const cycles = Math.floor((3 + 1e-7) / cycle);
  const seconds = cycles ? Math.min(3, cycles * cycle) : 3;
  const count = Math.max(1, Math.round(seconds * 20));
  const centiseconds = Math.max(count * 2, Math.round(seconds * 100));
  return {duration, start: ((currentTime % duration) + duration) % duration, rate,
    seconds, count, cycles, centiseconds};
}

export function frameTime(plan, index) {
  // The terminal pose is excluded: it is the first frame of the next loop.
  return (plan.start + index / plan.count * plan.seconds * plan.rate) % plan.duration;
}

export function frameDelay(plan, index) {
  return 10 * (Math.round((index + 1) * plan.centiseconds / plan.count)
    - Math.round(index * plan.centiseconds / plan.count));
}

export function snapshotViewer(viewer) {
  return {src: viewer.getAttribute('src'), source: viewer.src, clip: viewer.animationName, time: viewer.currentTime,
    paused: viewer.paused, orbit: viewer.getCameraOrbit().toString(),
    target: viewer.getCameraTarget().toString(), fov: `${viewer.getFieldOfView()}deg`,
    rotate: viewer.autoRotate, controls: viewer.hasAttribute('camera-controls'),
    prompt: viewer.getAttribute('interaction-prompt')};
}

export function restoreViewer(viewer, state) {
  viewer.autoRotate = state.rotate;
  viewer.toggleAttribute('camera-controls', state.controls);
  if (state.prompt === null) viewer.removeAttribute('interaction-prompt');
  else viewer.setAttribute('interaction-prompt', state.prompt);
  // A replacement model or clip belongs to the reader. Never put the old one back over it.
  if (viewer.getAttribute('src') !== state.src || viewer.src !== state.source || viewer.animationName !== state.clip) return false;
  viewer.cameraOrbit = state.orbit;
  viewer.cameraTarget = state.target;
  viewer.fieldOfView = state.fov;
  viewer.jumpCameraToGoal();
  viewer.currentTime = state.time;
  if (state.paused) viewer.pause(); else viewer.play();
  return true;
}

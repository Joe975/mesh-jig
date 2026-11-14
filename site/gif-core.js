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

// Compare at real playback speeds. Only claim a loop when every moving panel closes.
export function comparisonPlan(clips) {
  const moving = clips.filter(c => c.animated);
  if (!moving.length) throw new Error('Choose at least one animated model first.');
  const plans = moving.map(c => capturePlan(c.duration, c.time, c.rate));
  const cycle = plans[0].duration / plans[0].rate;
  let seconds = 0;
  const maxCycles = Math.floor((3 + 1e-7) / cycle);
  for (let step = 0; step < 600 && maxCycles - step > 0; step++) {
    const t = (maxCycles - step) * cycle;
    if (plans.every(p => Math.abs(t / (p.duration / p.rate) - Math.round(t / (p.duration / p.rate))) < 1e-5)) {
      seconds = Math.min(3, t); break;
    }
  }
  const loops = seconds > 0;
  seconds ||= 3;
  const count = Math.max(1, Math.round(seconds * 20));
  return {seconds, count, cycles: loops ? 1 : 0, centiseconds: Math.max(count * 2, Math.round(seconds * 100))};
}

export function comparisonTime(clip, seconds) {
  if (!clip.animated) return clip.time;
  return ((clip.time + seconds * clip.rate) % clip.duration + clip.duration) % clip.duration;
}

export function exportLabel(pick = {}) {
  const facts = pick.facts || {};
  const effortWord = /\b(ultra|xhigh|max|high|medium|low|minimal)\b/i;
  let model = (pick.model || 'Unknown model')
    .replace(/\b\d{4}-\d{2}-\d{2}\b/g, '')
    .replace(/\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:\s+\d{4})?\b/gi, '')
    .replace(/\s*,\s*(?=,|$)/g, '').replace(/,\s*$/, '').trim();
  const recorded = pick.effort || facts.effort;
  const inModel = model.match(effortWord)?.[1]?.toLowerCase();
  // Keep effort once. Explicit metadata wins over an older model label.
  if (recorded && inModel && recorded.toLowerCase() !== inModel)
    model = model.replace(effortWord, recorded);
  const effort = recorded || inModel;
  const cost = String(facts.cost || '').trim();
  const knownCost = cost && !/^(?:not reported|unknown|unavailable|n\/?a)\b/i.test(cost);
  const triangles = Number.isFinite(facts.triangles) ? facts.triangles.toLocaleString('en') : 'not reported';
  return [model, effort && !inModel && `Effort: ${effort}`,
    knownCost && `Cost: ${cost}`, `Triangles: ${triangles}`].filter(Boolean);
}

export function exportLayout(boxes) {
  if (!boxes.length || boxes.length > 2 || boxes.some(b => !(b.width > 0 && b.height > 0)))
    throw new Error('Open both viewers before exporting.');
  const width = Math.max(1, Math.floor(Math.min(...boxes.map(b => Math.min(b.width, 640 * b.width / Math.max(b.width, b.height))))));
  let y = 0;
  const panels = boxes.map(b => {
    const height = Math.max(1, Math.round(width * b.height / b.width));
    const panel = {x: 0, y, width, height}; y += height; return panel;
  });
  return {width, height: y, panels};
}

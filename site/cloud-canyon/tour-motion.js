export function wrap(value, period) {
  return ((value % period) + period) % period;
}

export function routePoint(route, progress) {
  const index = wrap(progress, 1)*(route.length - 1);
  const segment = Math.floor(index);
  const fraction = index - segment;
  return route[segment].map((value, axis) => value + fraction*(route[segment + 1][axis] - value));
}

function smooth(value) {
  const fraction = Math.min(1, Math.max(0, value));
  return fraction*fraction*(3 - 2*fraction);
}

export function cloudPassAmount(passages, seconds) {
  let weight = 0;
  const feather = 0.95;
  for (const [start, end] of passages) {
    weight = Math.max(weight, smooth((seconds - start + feather)/feather)*smooth((end + feather - seconds)/feather));
  }
  return weight;
}

export function flockCues(template, size, seed) {
  const remaining = [];
  function permutations(prefix, pool) {
    if (!pool.length) { remaining.push(prefix); return; }
    for (const name of pool) permutations([...prefix, name], pool.filter((candidate) => candidate !== name));
  }
  permutations([], template.map(({ clip }) => clip));
  if (size > remaining.length) throw new Error('Too many whales for distinct animation orders.');
  let entropy = seed >>> 0;
  function random() {
    entropy = (Math.imul(entropy, 1664525) + 1013904223) >>> 0;
    return entropy/4294967296;
  }
  return Array.from({ length: size }, () => {
    const [order] = remaining.splice(Math.floor(random()*remaining.length), 1);
    return template.map(({ start }, index) => ({ clip: order[index], start: start + (random() - 0.5)*0.32 }));
  });
}

export function cuePose(cues, clips, seconds) {
  const overlap = 0.32;
  for (const { clip, start } of cues) {
    const duration = clips.get(clip).duration;
    const offset = seconds - start;
    if (offset < -overlap || offset >= duration + overlap) continue;
    const weight = smooth((offset + overlap)/overlap)*smooth((duration + overlap - offset)/overlap);
    const time = Math.max(0, Math.min(duration - 0.00001, offset));
    return { clip, time, weight };
  }
  return { clip: 'fly', time: 0, weight: 0 };
}

const resolutions = new Map([
  ['fast', [1, 880, 680]],
  ['balanced', [1.25, 1280, 960]],
  ['sharp', [2, 1760, 1400]],
]);

export function renderSize(width, height, preset) {
  const [scale, maxWidth, maxHeight] = resolutions.get(preset) ?? resolutions.get('balanced');
  const ratio = Math.min(scale, maxWidth/width, maxHeight/height);
  return [Math.round(width*ratio), Math.round(height*ratio)];
}

export function whaleUrl(current, id) {
  const url = new URL(current);
  url.searchParams.set('whale', id);
  return url;
}

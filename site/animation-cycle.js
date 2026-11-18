// Event-driven: no timer, render loop or work while the native viewer is paused.
export function defaultAnimation(clips){
  return clips.find(n=>n.toLowerCase()==='run')??clips.find(n=>n.toLowerCase()==='sprint')??clips[0];
}
// Older GLBs have anonymous native names; comparisons use their public clip labels.
export function activeClipLabel(clips, labels, nativeName){
  const index = clips.indexOf(nativeName);
  return index >= 0 ? (labels?.[index] ?? nativeName) : nativeName;
}
export class AnimationCycle {
  constructor(play) { this.play = play; this.stop(); }
  start(clips) {
    this.stop(); this.clips = clips.map(c => ({...c}));
    if (!this.clips.length) return false;
    this.active = true; this.index = 0; this.count = 0; this.show(); return true;
  }
  show() { const c = this.clips[this.index]; this.play(c.name, c.loop !== false); }
  boundary() {
    if (!this.active) return;
    const c = this.clips[this.index];
    if (++this.count < (c.loop === false ? 1 : (c.repetitions ?? 2))) return;
    this.index = (this.index + 1) % this.clips.length; this.count = 0; this.show();
  }
  stop() { this.active = false; this.clips = []; this.index = 0; this.count = 0; }
}

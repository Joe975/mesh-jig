"""Deterministic GIF timing/state checks using the same browser module, without browser dependencies."""
import base64
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / 'site/gif-core.js'
if not CORE.exists():
    CORE = ROOT / 'publication/site/gif-core.js'


def run_js(body):
    node = shutil.which('node')
    if not node:
        pytest.skip('node is required for browser JavaScript tests')
    url = 'data:text/javascript;base64,' + base64.b64encode(CORE.read_bytes()).decode()
    script = f"import assert from 'node:assert/strict';\nimport {{capturePlan,frameTime,frameDelay,snapshotViewer,restoreViewer}} from '{url}';\n" + body
    subprocess.run([node, '--input-type=module', '-e', script], check=True, capture_output=True, text=True)


def test_complete_cycle_selection_and_bounded_excerpt():
    run_js("""
for (const [duration, seconds, cycles] of [[1,3,3],[.8,2.4,3],[1.2,2.4,2],[2,2,1],[3,3,1],[4,3,0]]) {
  const p = capturePlan(duration, .23);
  assert.ok(Math.abs(p.seconds-seconds)<1e-6);
  assert.equal(p.cycles, cycles);
  assert.ok(p.count<=60);
  assert.equal(Array.from({length:p.count},(_,i)=>frameDelay(p,i)).reduce((a,b)=>a+b,0),p.centiseconds*10);
}
assert.equal(capturePlan(4, 0, 2).seconds, 2);
assert.equal(capturePlan(.3).cycles, 10); // floating-point boundary
for (const duration of [0,-1,NaN,Infinity]) assert.throws(()=>capturePlan(duration));
assert.throws(()=>capturePlan(1,0,0));
assert.throws(()=>capturePlan(1,0,-1));
assert.throws(()=>capturePlan(1,NaN));
""")


def test_terminal_frame_is_excluded_and_phase_is_preserved():
    run_js("""
const p = capturePlan(2, .4);
assert.ok(Math.abs(frameTime(p,0)-.4)<1e-12);
assert.ok(Math.abs(frameTime(p,p.count-1)-.35)<1e-12);
assert.equal(capturePlan(2,2).start,0);
assert.equal(capturePlan(2,-.5).start,1.5);
for (let i=0;i<p.count;i++) assert.ok(frameTime(p,i)>=0 && frameTime(p,i)<2);
assert.notEqual(frameTime(p,0),frameTime(p,p.count-1));
const long = capturePlan(8,7);
assert.equal(frameTime(long,0),7);
assert.ok(frameTime(long,59)<2);
""")


def test_state_restoration_for_playing_paused_and_replaced_models():
    run_js("""
class Viewer {
  constructor(paused) { this.attrs=new Map([['src','model.glb'],['camera-controls','']]); this.paused=paused;
    this.animationName='walk'; this.currentTime=.7; this.autoRotate=true; this.timeScale=1.5; }
  getAttribute(k) { return this.attrs.get(k) ?? null; }
  hasAttribute(k) { return this.attrs.has(k); }
  setAttribute(k,v) { this.attrs.set(k,v); }
  removeAttribute(k) { this.attrs.delete(k); }
  toggleAttribute(k,v) { if(v)this.attrs.set(k,'');else this.attrs.delete(k); }
  getCameraOrbit() { return {toString:()=> '1rad 1.2rad 3m'}; }
  getCameraTarget() { return {toString:()=> '.1m .2m .3m'}; }
  getFieldOfView() { return 25; }
  jumpCameraToGoal() { this.jumped=true; }
  play() { this.paused=false; }
  pause() { this.paused=true; }
}
for (const paused of [true,false]) {
  const v=new Viewer(paused), state=snapshotViewer(v);
  v.pause(); v.currentTime=1.9; v.autoRotate=false; v.removeAttribute('camera-controls');
  v.setAttribute('interaction-prompt','none');
  assert.equal(restoreViewer(v,state),true);
  assert.equal(v.currentTime,.7); assert.equal(v.paused,paused); assert.equal(v.animationName,'walk');
  assert.equal(v.timeScale,1.5); assert.equal(v.autoRotate,true); assert.equal(v.getAttribute('interaction-prompt'),null);
  assert.equal(v.hasAttribute('camera-controls'),true);
  assert.equal(v.cameraOrbit,state.orbit); assert.equal(v.cameraTarget,state.target); assert.equal(v.fieldOfView,'25deg');
}
for (const change of ['src','property','clip']) {
  const v=new Viewer(false), state=snapshotViewer(v);
  if(change==='src')v.setAttribute('src','new.glb');else if(change==='property')v.src='new.glb';else v.animationName='leap';
  v.currentTime=.1;
  assert.equal(restoreViewer(v,state),false); assert.equal(v.currentTime,.1);
  assert.equal(v.animationName,change==='clip'?'leap':'walk');
}
""")

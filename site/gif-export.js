import {capturePlan, frameTime, frameDelay, snapshotViewer, restoreViewer} from './gif-core.js';

let active = null;
const abortError = message => new DOMException(message, 'AbortError');

// Every wait is bounded and abortable, including rendering in a background tab.
function bounded(promise, signal, ms = 10000) {
  return new Promise((resolve, reject) => {
    const finish = (fn, value) => { clearTimeout(timer); signal.removeEventListener('abort', cancel); fn(value); };
    const cancel = () => finish(reject, signal.reason);
    const timer = setTimeout(() => finish(reject, new Error('Export timed out. Keep this tab visible and try again.')), ms);
    signal.addEventListener('abort', cancel, {once: true});
    if (signal.aborted) cancel();
    Promise.resolve(promise).then(value => finish(resolve, value), error => finish(reject, error));
  });
}

function background(ctx, width, height) {
  // Match the viewer plate's CSS ellipse, which model-viewer's toBlob does not include.
  ctx.save();
  ctx.translate(width * .5, height * .3);
  ctx.scale(width * 1.2, height * .9);
  const fill = ctx.createRadialGradient(0, 0, 0, 0, 0, 1);
  fill.addColorStop(0, '#f4f7fb'); fill.addColorStop(.7, '#dfe6ef'); fill.addColorStop(1, '#dfe6ef');
  ctx.fillStyle = fill;
  ctx.fillRect(-2, -2, 4, 4);
  ctx.restore();
}

function setup(root) {
  const viewer = root.querySelector('model-viewer'), button = root.querySelector('.gif-export');
  const cancel = root.querySelector('.gif-cancel'), status = root.querySelector('.gif-status');
  const download = root.querySelector('.gif-download');
  let objectURL;
  const refresh = () => {
    button.disabled = !!active || !viewer.loaded || !viewer.availableAnimations.length;
  };
  viewer.addEventListener('load', refresh);
  viewer.addEventListener('error', () => { button.disabled = true; });
  new MutationObserver(() => { if (!active) refresh(); }).observe(viewer, {attributes: true, attributeFilter: ['src']});
  document.addEventListener('gif-busy', refresh);
  refresh();

  button.onclick = async () => {
    if (active || button.disabled) return;
    if (document.hidden) { status.textContent = 'Keep this tab visible to export.'; return; }
    const controller = new AbortController(), {signal} = controller;
    active = controller;
    let state;
    const box = viewer.getBoundingClientRect();
    const explore = root.closest('.explore');
    const disabled = [...explore.querySelectorAll('select, button')].map(el => [el, el.disabled]);
    const stop = message => controller.abort(abortError(message));
    const onHidden = () => { if (document.hidden) stop('Export cancelled because the tab was hidden.'); };
    const onResize = () => stop('Export cancelled because the viewer size changed.');
    const onError = () => stop('Export cancelled because the model could not be loaded.');
    const changed = () => {
      if (state && (viewer.getAttribute('src') !== state.src || viewer.src !== state.source || viewer.animationName !== state.clip))
        stop('Export cancelled because the model or clip changed.');
    };
    const observer = new MutationObserver(changed);
    const deadline = setTimeout(() => stop('Export reached the 90 second time limit. Try a smaller browser window.'), 90000);
    let worker;
    const peers = [];
    const Viewer = customElements.get('model-viewer');
    // 4.4.0's getter returns undefined; its bundled default is 0.5. Pin full-scale
    // rendering during toBlob: adaptive scale can otherwise crop a paused scene.
    const minimumScale = Viewer.minimumRenderScale ?? .5;
    let scalePinned = false;
    try {
      state = snapshotViewer(viewer);
      const plan = capturePlan(viewer.duration, state.time, viewer.timeScale);
      if (!(box.width > 0 && box.height > 0)) throw new Error('Open the viewer before exporting.');
      const scale = Math.min(1, 640 / Math.max(box.width, box.height));
      const canvas = document.createElement('canvas');
      canvas.width = Math.max(1, Math.round(box.width * scale));
      canvas.height = Math.max(1, Math.round(box.height * scale));
      const ctx = canvas.getContext('2d', {willReadFrequently: true});
      if (!ctx) throw new Error('This browser cannot capture a GIF.');
      if (objectURL) { URL.revokeObjectURL(objectURL); objectURL = null; }
      download.hidden = true;
      viewer.dataset.gifExporting = 'true';
      explore.dataset.gifExporting = 'true';
      disabled.forEach(([el]) => { el.disabled = true; });
      cancel.hidden = false; cancel.disabled = false;
      cancel.onclick = () => stop('Export cancelled.');
      document.dispatchEvent(new Event('gif-busy'));
      // model-viewer shares a renderer. Pause other playing scenes while stepping this one,
      // then put them back at their own phase; a comparison must not consume every render tick.
      for (const peer of document.querySelectorAll('model-viewer')) {
        if (peer !== viewer && peer.loaded && !peer.paused) {
          peers.push({viewer: peer, source: peer.src, clip: peer.animationName, time: peer.currentTime});
          peer.pause();
        }
      }
      document.addEventListener('visibilitychange', onHidden);
      window.addEventListener('resize', onResize);
      window.addEventListener('pagehide', onHidden);
      viewer.addEventListener('error', onError);
      observer.observe(viewer, {attributes: true, attributeFilter: ['src', 'animation-name']});
      viewer.pause(); viewer.autoRotate = false;
      Viewer.minimumRenderScale = 1;
      scalePinned = true;
      viewer.removeAttribute('camera-controls');
      viewer.setAttribute('interaction-prompt', 'none');
      viewer.cameraOrbit = state.orbit; viewer.cameraTarget = state.target; viewer.fieldOfView = state.fov;
      viewer.jumpCameraToGoal();
      worker = new Worker(new URL('./gif-worker.js', import.meta.url), {type: 'module'});
      const send = (data, transfer = []) => bounded(new Promise((resolve, reject) => {
        worker.onmessage = e => e.data.error ? reject(new Error(e.data.error)) : resolve(e.data);
        worker.onerror = () => reject(new Error('GIF encoder could not load. Reload the page and try again.'));
        worker.postMessage(data, transfer);
      }), signal);
      await send({type: 'init', width: canvas.width, height: canvas.height});
      for (let i = 0; i < plan.count; i++) {
        changed(); signal.throwIfAborted();
        status.textContent = `Exporting GIF… ${i + 1} / ${plan.count}`;
        viewer.currentTime = frameTime(plan, i);
        await bounded(new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))), signal);
        const blob = await bounded(viewer.toBlob({mimeType: 'image/png', idealAspect: false}), signal);
        if (!blob) throw new Error('The viewer could not capture a frame.');
        const bitmap = await bounded(createImageBitmap(blob), signal);
        try {
          background(ctx, canvas.width, canvas.height);
          ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
        } finally { bitmap.close(); }
        const {data} = ctx.getImageData(0, 0, canvas.width, canvas.height);
        await send({type: 'frame', buffer: data.buffer, delay: frameDelay(plan, i)}, [data.buffer]);
      }
      const result = await send({type: 'finish'});
      changed(); signal.throwIfAborted();
      objectURL = URL.createObjectURL(new Blob([result.buffer], {type: 'image/gif'}));
      download.href = objectURL;
      const name = `${viewer.alt || 'mesh-jig'}-${state.clip || 'animation'}`.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
      download.download = `${name}.gif`;
      download.hidden = false;
      status.textContent = `${(plan.centiseconds / 100).toFixed(2)} s GIF ready. ${plan.cycles ? 'Ends on a complete cycle; non-looping clips may still jump.' : 'Three-second excerpt; this clip is longer and may jump when repeated.'}`;
      download.click();
    } catch (error) {
      status.textContent = error.name === 'AbortError' ? error.message : `Could not export GIF: ${error.message}`;
    } finally {
      clearTimeout(deadline);
      observer.disconnect(); worker?.terminate();
      if (scalePinned) Viewer.minimumRenderScale = minimumScale;
      document.removeEventListener('visibilitychange', onHidden);
      window.removeEventListener('resize', onResize);
      window.removeEventListener('pagehide', onHidden);
      viewer.removeEventListener('error', onError);
      try { if (state) restoreViewer(viewer, state); }
      catch { status.textContent = 'Export stopped. Reselect the model to reload the viewer.'; }
      for (const peer of peers) {
        if (peer.viewer.src === peer.source && peer.viewer.animationName === peer.clip) {
          peer.viewer.currentTime = peer.time;
          peer.viewer.play();
        }
      }
      delete viewer.dataset.gifExporting; delete explore.dataset.gifExporting;
      disabled.forEach(([el, was]) => { el.disabled = was; });
      cancel.hidden = true;
      active = null;
      document.dispatchEvent(new Event('gif-busy'));
    }
  };
}

customElements.whenDefined('model-viewer').then(() => {
  document.querySelectorAll('.main-panel, .compare').forEach(setup);
});

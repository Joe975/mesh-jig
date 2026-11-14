import {capturePlan, frameTime, frameDelay, snapshotViewer, restoreViewer, comparisonPlan, comparisonTime, exportLabel, exportLayout} from './gif-core.js';

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

function drawLabel(ctx, panel, lines) {
  const fontSize = Math.max(11, Math.min(14, Math.round(panel.width / 42)));
  const padding = 7, margin = 7, lineHeight = fontSize + 3;
  ctx.save(); ctx.font = `${fontSize}px system-ui, sans-serif`;
  const maxWidth = panel.width - 2 * (margin + padding);
  const wrapped = [];
  for (const line of lines) {
    let current = '';
    // Character wrapping also handles unusually long model identifiers without clipping.
    for (const char of line) {
      if (current && ctx.measureText(current + char).width > maxWidth) { wrapped.push(current.trimEnd()); current = ''; }
      current += char;
    }
    if (current) wrapped.push(current.trim());
  }
  const width = Math.min(maxWidth, Math.max(...wrapped.map(s => ctx.measureText(s).width)));
  const height = wrapped.length * lineHeight + padding * 2;
  const x = panel.width - width - padding * 2 - margin, y = panel.y + panel.height - height - margin;
  ctx.fillStyle = 'rgba(15,23,36,.88)'; ctx.fillRect(x, y, width + padding * 2, height);
  ctx.fillStyle = '#ffffff'; ctx.textBaseline = 'top';
  wrapped.forEach((line, i) => ctx.fillText(line, x + padding, y + padding + i * lineHeight));
  ctx.restore();
}

function setup(root) {
  const viewer = root.querySelector('model-viewer'), button = root.querySelector('.gif-export');
  const cancel = root.querySelector('.gif-cancel'), status = root.querySelector('.gif-status');
  const download = root.querySelector('.gif-download');
  const explore = root.closest('.explore');
  const roots = () => explore.querySelector('.compare').hidden ? [root] : [...explore.querySelectorAll(':scope > .main-panel, :scope > .compare')];
  let objectURL;
  const refresh = () => {
    const views = roots().map(p => p.querySelector('model-viewer'));
    button.textContent = views.length === 2 ? 'Export comparison GIF' : 'Export GIF';
    button.disabled = !!active || views.some(v => !v.loaded) || !views.some(v => v.availableAnimations.length);
  };
  explore.querySelectorAll('model-viewer').forEach(v => v.addEventListener('load', refresh));
  new MutationObserver(refresh).observe(explore.querySelector('.compare'), {attributes: true, attributeFilter: ['hidden']});
  viewer.addEventListener('error', () => { button.disabled = true; });
  explore.querySelectorAll('model-viewer').forEach(v => {
    new MutationObserver(() => { if (!active) refresh(); }).observe(v, {attributes: true, attributeFilter: ['src']});
  });
  document.addEventListener('gif-busy', refresh);
  refresh();

  button.onclick = async () => {
    if (active || button.disabled) return;
    if (document.hidden) { status.textContent = 'Keep this tab visible to export.'; return; }
    const controller = new AbortController(), {signal} = controller;
    active = controller;
    const participants = roots().map(panel => ({panel, viewer: panel.querySelector('model-viewer')}));
    const combined = participants.length === 2;
    let plan;
    const disabled = [...explore.querySelectorAll('select, button')].map(el => [el, el.disabled]);
    const stop = message => controller.abort(abortError(message));
    const onHidden = () => { if (document.hidden) stop('Export cancelled because the tab was hidden.'); };
    const onResize = () => stop('Export cancelled because the page scrolled or viewer size changed.');
    const onError = () => stop('Export cancelled because the model could not be loaded.');
    const changed = () => {
      if (participants.some(({viewer: v, state}) => state && (v.getAttribute('src') !== state.src || v.src !== state.source || v.animationName !== state.clip)) || (combined && explore.querySelector('.compare').hidden))
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
      const picks = JSON.parse(explore.dataset.explore);
      for (const item of participants) {
        const v = item.viewer;
        if (!v.loaded) throw new Error('Wait for both models to finish loading.');
        item.state = snapshotViewer(v);
        item.style = v.getAttribute('style');
        item.clip = {animated: v.availableAnimations.length > 0, duration: v.duration, time: item.state.time, rate: v.timeScale};
        const pick = picks[Number(item.panel.querySelector('.pick-model').value)];
        const clip = pick?.clipLabels?.[v.availableAnimations.indexOf(v.animationName)] || v.animationName;
        item.label = exportLabel(pick, combined ? (item === participants[0] ? 'Top' : 'Bottom') : '', clip);
      }
      plan = combined ? comparisonPlan(participants.map(p => p.clip)) : capturePlan(viewer.duration, participants[0].state.time, viewer.timeScale);
      const layout = exportLayout(participants.map(p => p.viewer.getBoundingClientRect()));
      const canvas = document.createElement('canvas');
      canvas.width = layout.width;
      canvas.height = layout.height;
      const ctx = canvas.getContext('2d', {willReadFrequently: true});
      if (!ctx) throw new Error('This browser cannot capture a GIF.');
      if (objectURL) { URL.revokeObjectURL(objectURL); objectURL = null; }
      download.hidden = true;
      participants.forEach(p => { p.viewer.dataset.gifExporting = 'true'; });
      explore.dataset.gifExporting = 'true';
      disabled.forEach(([el]) => { el.disabled = true; });
      cancel.hidden = false; cancel.disabled = false;
      cancel.onclick = () => stop('Export cancelled.');
      document.dispatchEvent(new Event('gif-busy'));
      // model-viewer shares a renderer. Pause other playing scenes while stepping this one,
      // then put them back at their own phase; a comparison must not consume every render tick.
      for (const peer of document.querySelectorAll('model-viewer')) {
        if (!participants.some(p => p.viewer === peer) && peer.loaded && !peer.paused) {
          peers.push({viewer: peer, source: peer.src, clip: peer.animationName, time: peer.currentTime});
          peer.pause();
        }
      }
      document.addEventListener('visibilitychange', onHidden);
      window.addEventListener('resize', onResize);
      window.addEventListener('scroll', onResize, {passive: true});
      window.addEventListener('pagehide', onHidden);
      for (const {viewer: v} of participants) {
        v.addEventListener('error', onError);
        observer.observe(v, {attributes: true, attributeFilter: ['src', 'animation-name']});
        v.pause(); v.autoRotate = false;
      }
      observer.observe(explore.querySelector('.compare'), {attributes: true, attributeFilter: ['hidden']});
      Viewer.minimumRenderScale = 1;
      scalePinned = true;
      for (const {viewer: v, state} of participants) {
        // Off-screen model-viewer scenes stop rendering. Keep capture subjects geometrically
        // in the viewport without moving layout or showing them over the page. toBlob captures
        // the canvas, not CSS opacity/translation. Restore the exact inline style afterward.
        const rect = v.getBoundingClientRect();
        v.style.transform = `translate(${-rect.left}px, ${-rect.top}px)`;
        v.style.opacity = '0';
        v.removeAttribute('camera-controls'); v.setAttribute('interaction-prompt', 'none');
        v.cameraOrbit = state.orbit; v.cameraTarget = state.target; v.fieldOfView = state.fov;
        v.jumpCameraToGoal();
      }
      worker = new Worker(new URL('./gif-worker.js', import.meta.url), {type: 'module'});
      const send = (data, transfer = []) => bounded(new Promise((resolve, reject) => {
        worker.onmessage = e => e.data.error ? reject(new Error(e.data.error)) : resolve(e.data);
        worker.onerror = () => reject(new Error('GIF encoder could not load. Reload the page and try again.'));
        worker.postMessage(data, transfer);
      }), signal);
      await send({type: 'init', width: canvas.width, height: canvas.height});
      for (let i = 0; i < plan.count; i++) {
        changed(); signal.throwIfAborted();
        status.textContent = `Exporting GIF... ${i + 1} / ${plan.count}`;
        for (const item of participants) item.viewer.currentTime = combined
          ? comparisonTime(item.clip, i / plan.count * plan.seconds) : frameTime(plan, i);
        for (const [j, item] of participants.entries()) {
          await bounded(new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))), signal);
          const blob = await bounded(item.viewer.toBlob({mimeType: 'image/png', idealAspect: false}), signal);
          if (!blob) throw new Error('The viewer could not capture a frame.');
          const bitmap = await bounded(createImageBitmap(blob), signal);
          const panel = layout.panels[j];
          try {
            ctx.save(); ctx.beginPath(); ctx.rect(0, panel.y, panel.width, panel.height); ctx.clip();
            ctx.translate(0, panel.y); background(ctx, panel.width, panel.height); ctx.restore();
            ctx.drawImage(bitmap, 0, panel.y, panel.width, panel.height);
            drawLabel(ctx, panel, item.label);
          } finally { bitmap.close(); }
        }
        const {data} = ctx.getImageData(0, 0, canvas.width, canvas.height);
        await send({type: 'frame', buffer: data.buffer, delay: frameDelay(plan, i)}, [data.buffer]);
      }
      const result = await send({type: 'finish'});
      changed(); signal.throwIfAborted();
      objectURL = URL.createObjectURL(new Blob([result.buffer], {type: 'image/gif'}));
      download.href = objectURL;
      const name = `${participants.map(p => p.viewer.alt || 'mesh-jig').join('-vs-')}-${participants[0].state.clip || 'animation'}`.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
      download.download = `${name}.gif`;
      download.hidden = false;
      status.textContent = `${(plan.centiseconds / 100).toFixed(2)} s GIF ready. ${plan.cycles ? 'Ends on a complete cycle; non-looping clips may still jump.' : 'Three-second excerpt; the clips do not share a complete cycle and may jump when repeated.'}`;
      download.click();
    } catch (error) {
      status.textContent = error.name === 'AbortError' ? error.message : `Could not export GIF: ${error.message}`;
    } finally {
      clearTimeout(deadline);
      observer.disconnect(); worker?.terminate();
      if (scalePinned) Viewer.minimumRenderScale = minimumScale;
      document.removeEventListener('visibilitychange', onHidden);
      window.removeEventListener('resize', onResize);
      window.removeEventListener('scroll', onResize);
      window.removeEventListener('pagehide', onHidden);
      for (const {viewer: v, state, style} of participants) {
        if (style === null) v.removeAttribute('style'); else if (style !== undefined) v.setAttribute('style', style);
        v.removeEventListener('error', onError);
        try { if (state) restoreViewer(v, state); }
        catch { status.textContent = 'Export stopped. Reselect the model to reload the viewer.'; }
        delete v.dataset.gifExporting;
      }
      for (const peer of peers) {
        if (peer.viewer.src === peer.source && peer.viewer.animationName === peer.clip) {
          peer.viewer.currentTime = peer.time;
          peer.viewer.play();
        }
      }
      delete explore.dataset.gifExporting;
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

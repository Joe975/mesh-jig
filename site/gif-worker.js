// gifenc 1.0.3 (MIT), bundled locally. One transferred RGBA frame at a time.
import {GIFEncoder, quantize, applyPalette} from './vendor/gifenc.js';
let gif, width, height;
self.onmessage = ({data}) => {
  try {
    if (data.type === 'init') {
      ({width, height} = data);
      if (width * height > 2 * 640 * 640 || width < 1 || height < 1) throw new Error('Invalid GIF size.');
      gif = GIFEncoder();
    } else if (data.type === 'frame') {
      const rgba = new Uint8Array(data.buffer);
      // Bound the quantizer histogram to 4096 bins, including on detailed textured models.
      const palette = quantize(rgba, 256, {format: 'rgb444'});
      gif.writeFrame(applyPalette(rgba, palette, 'rgb444'), width, height,
        {palette, delay: data.delay, repeat: 0});
      if (gif.bytesView().length > 24 * 1024 * 1024) throw new Error('GIF exceeds the 24 MB export limit.');
    } else if (data.type === 'finish') {
      gif.finish();
      const bytes = gif.bytes();
      self.postMessage({buffer: bytes.buffer}, [bytes.buffer]);
      return;
    }
    self.postMessage({ok: true});
  } catch (error) { self.postMessage({error: error.message}); }
};

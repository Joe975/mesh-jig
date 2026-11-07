"""Describe sounds in numbers and a picture, for whoever cannot hear them (the author of a sound script included).

Per sound: the pitch over time (autocorrelation, 50 to 2500 Hz), how periodic it is, where its energy sits, the
shape of its loudness and any fast amplitude flutter; and a sheet of log-frequency spectrograms over waveforms, with
the cues a sound goes with drawn on it. Nothing here passes or fails anything: `sound.check` is the contract.

Its pitch tracker also follows the band under 500 Hz and the loudness envelope, so a low voice through a
mouth resonance near 1 kHz is not read as a 1 kHz pitch. Deterministic.
"""
import wave
from pathlib import Path

import numpy as np

FRAME = 2048             # 46 ms at 44.1 kHz: two periods of a 50 Hz voice
HOP_S = 0.005
F0_MIN, F0_MAX = 50.0, 2500.0
LOW_BAND = 500.0         # below this a voice has only its first harmonics, and no mouth resonance
LOW_SHARE = 0.2          # ... and that band is believed when it holds this much of a frame's RMS
RING_RATIO = 4.5         # a slower period replaces the fine structure's only when this many times longer: a
                         # resonance sits well above the voice that excites it, a voice breaking to its sub-octave does not
VOICED = 0.5             # normalised autocorrelation at the pitch period; a steady tone is 1, noise near 0
LOUD = 0.1               # a frame counts when its RMS is at least this much of the loudest frame's


def load(path):
    with wave.open(str(path), 'rb') as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f'{path}: {width*8}-bit, only 16-bit PCM is read here')
    return rate, np.frombuffer(raw, dtype='<i2').reshape(-1, channels).astype(np.float64).mean(axis=1)/32768.0


def frames(x, size, hop):
    """Rows of `size` samples every `hop`, centred on their time, the signal zero-padded at both ends."""
    padded = np.concatenate([np.zeros(size//2), x, np.zeros(size)])
    count = max(len(x)//hop, 1)
    index = np.arange(size)[None, :] + hop*np.arange(count)[:, None]
    return padded[index]


def correlation(rows, lags):
    """Each row's autocorrelation at lags 0..lags-1, each lag normalised by the energy of the two stretches it compares."""
    size = rows.shape[1]
    acf = np.fft.irfft(np.abs(np.fft.rfft(rows, 2*size, axis=1))**2, axis=1)[:, :lags]
    energy = np.cumsum(rows**2, axis=1)
    lag = np.arange(lags)
    head = energy[:, size-1-lag]                                                        # sum of row[:size-lag]^2
    tail = energy[:, -1:] - np.where(lag > 0, energy[:, np.maximum(lag-1, 0)], 0.0)     # sum of row[lag:]^2
    return acf/np.sqrt(np.maximum(head*tail, 1e-18))


def best_lag(row, low, high):
    """The shortest lag between low and high whose peak is nearly the best one (the pitch, not a sub-octave), or 0."""
    inner = row[low:high+1]
    peaks = np.flatnonzero((inner > row[low-1:high]) & (inner >= row[low+1:high+2])) + low
    if not len(peaks):
        return 0
    best = row[peaks].max()
    return int(peaks[np.flatnonzero(row[peaks] >= best - 0.07*abs(best))[0]])


def smooth(x, rate, seconds):
    window = np.hanning((int(rate*seconds) | 1) + 2)[1:-1]
    return np.convolve(x, window/window.sum(), mode='same')


def pitch_track(x, rate):
    """(times s, f0 Hz with nan where unvoiced, periodicity 0..1, frame RMS) every HOP_S."""
    hop = int(round(rate*HOP_S))
    low, high = int(rate/F0_MAX), min(int(np.ceil(rate/F0_MIN)), FRAME//2)

    def view(signal):
        rows = frames(signal, FRAME, hop)
        rows = rows - rows.mean(axis=1, keepdims=True)
        return correlation(rows, high+2), np.sqrt(np.mean(rows**2, axis=1))

    # A low rough voice through a mouth that rings at 1 kHz repeats best every millisecond, and its fine structure
    # is read as a 1 kHz pitch. Two other views do not hear the ringing: the sound below LOW_BAND, where only the
    # voice's first harmonics are, and the loudness envelope, where each larynx pulse sets the ringing off again.
    # Where one of them repeats at a much longer period than the fine structure, that longer period is the pitch.
    fine, rms = view(x)
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1/rate)
    deep, deep_rms = view(np.fft.irfft(spectrum*np.clip((1.5*LOW_BAND - freqs)/(0.5*LOW_BAND), 0, 1), len(x)))
    envelope = smooth(np.abs(x), rate, 0.0015)
    pulses, _ = view(envelope - smooth(envelope, rate, 0.015))
    slowest = int(rate/LOW_BAND)
    f0 = np.full(len(rms), np.nan)
    strength = np.zeros(len(rms))
    for i in range(len(rms)):
        lag, row = best_lag(fine[i], low, high), fine[i]
        below, pulse = best_lag(deep[i], slowest, high), best_lag(pulses[i], slowest, high)
        if below > RING_RATIO*lag and deep[i, below] >= VOICED and deep_rms[i] >= LOW_SHARE*rms[i]:
            lag, row = below, deep[i]
        elif pulse > RING_RATIO*lag and pulses[i, pulse] >= max(VOICED, fine[i, lag] + 0.1):
            lag, row = pulse, pulses[i]
        if rms[i] <= 0 or not lag:
            continue
        a, b, c = row[lag-1], row[lag], row[lag+1]
        strength[i] = float(np.clip(b, 0, 1))
        if strength[i] >= VOICED:
            f0[i] = rate/(lag + (0.5*(a-c)/(a-2*b+c) if a-2*b+c else 0.0))
    f0[rms < LOUD*rms.max()] = np.nan
    return np.arange(len(rms))*hop/rate, f0, strength, rms


def spectra(x, rate, size=FRAME, hop=None, nfft=None):
    """(frequencies, power per frame) from a Hann-windowed short-time transform."""
    hop = hop or int(round(rate*HOP_S))
    rows = frames(x, size, hop)*np.hanning(size)
    return np.fft.rfftfreq(nfft or size, 1/rate), np.abs(np.fft.rfft(rows, nfft or size, axis=1))**2


FLUTTER_SAID = 0.05          # a flutter shallower than this is said to be none


def flutter(x, rate):
    """(depth 0..1, rate Hz) of the strongest amplitude modulation between 15 and 90 Hz: the rattle of a growl."""
    def smooth(signal, seconds):
        width = max(int(rate*seconds), 1)
        return np.convolve(signal, np.ones(width)/width, mode='same')
    envelope = smooth(np.abs(x), 0.004)
    trend = smooth(envelope, 0.060)
    keep = trend > LOUD*trend.max()
    if keep.sum() < rate*0.1:
        return 0.0, 0.0
    ripple = ((envelope-trend)/np.maximum(trend, 1e-9))[keep]
    spectrum = np.abs(np.fft.rfft(ripple*np.hanning(len(ripple)), 8*len(ripple)))/(len(ripple)/4)
    freqs = np.fft.rfftfreq(8*len(ripple), 1/rate)
    band = (freqs >= 15) & (freqs <= 90)
    best = np.argmax(spectrum[band])
    return float(min(spectrum[band][best], 1.0)), float(freqs[band][best])


def describe(x, rate):
    """The numbers for one sound, as plain Python values."""
    times, f0, strength, rms = pitch_track(x, rate)
    loud = rms >= LOUD*rms.max()
    voiced = ~np.isnan(f0)
    span = len(x)/rate

    def f0_near(fraction):
        near = voiced & (np.abs(times-fraction*span) <= 0.05*span)
        return round(float(np.median(f0[near]))) if near.any() else None

    freqs, power = spectra(x, rate)
    audible = freqs >= 50
    def centroid(mask):
        total = power[mask][:, audible].sum(axis=0)
        return round(float((freqs[audible]*total).sum()/max(total.sum(), 1e-18)))
    thirds = [loud & (times >= k*span/3) & (times < (k+1)*span/3 + 1e-9) for k in range(3)]
    total = power[loud][:, audible].sum(axis=0)
    cumulative = np.cumsum(total)/total.sum()
    band = (freqs >= 200) & (freqs <= 12000)
    in_band = np.maximum(power[loud][:, band], 1e-18)
    flatness = float(np.mean(np.exp(np.log(in_band).mean(axis=1))/in_band.mean(axis=1)))

    tenths = [np.sqrt(np.mean(part**2)) for part in np.array_split(x, 10)]
    tenths = [round(float(20*np.log10(max(v, 1e-9)/max(tenths))), 1) for v in tenths]
    depth, flutter_hz = flutter(x, rate)
    return {
        'seconds': round(span, 3),
        'voiced_fraction': round(float(voiced[loud].mean()), 2),
        'periodicity': round(float(strength[loud].mean()), 2),
        'f0_hz_at_10_30_50_70_90_percent': [f0_near(p) for p in (0.1, 0.3, 0.5, 0.7, 0.9)],
        'f0_hz_5th_and_95th_percentile': [round(float(v)) for v in np.percentile(f0[voiced], (5, 95))] if voiced.any() else None,
        'centroid_hz': centroid(loud),
        'centroid_hz_by_third': [centroid(third) if third.any() else None for third in thirds],
        'energy_5_to_95_percent_hz': [round(float(freqs[audible][np.searchsorted(cumulative, q)])) for q in (0.05, 0.95)],
        'spectral_flatness': round(flatness, 3),
        'loudest_at_fraction': round(float(times[np.argmax(rms)]/span), 2),
        'half_level_after_s': round(float(times[np.argmax(rms >= 0.5*rms.max())]), 3),
        'level_db_by_tenth': tenths,
        'flutter_depth': round(depth, 2),
        'flutter_hz': round(flutter_hz),
    }


def format_description(name, d):
    hz = lambda v: '-' if v is None else str(v)
    return '\n'.join([
        f"{name}  {d['seconds']:.3f} s",
        f"  pitch      voiced in {d['voiced_fraction']*100:.0f}% of the loud frames; f0 at 10/30/50/70/90% of the length: "
        f"{' '.join(hz(v) for v in d['f0_hz_at_10_30_50_70_90_percent'])} Hz"
        + (f" (nine frames in ten between {d['f0_hz_5th_and_95th_percentile'][0]} and {d['f0_hz_5th_and_95th_percentile'][1]})"
           if d['f0_hz_5th_and_95th_percentile'] else ''),
        f"  tone/noise periodicity {d['periodicity']:.2f} (1 a steady tone, 0 noise); spectral flatness "
        f"{d['spectral_flatness']:.3f} (0 a few pure partials, 1 white noise)",
        f"  spectrum   centroid {d['centroid_hz']} Hz (by third: {' '.join(hz(v) for v in d['centroid_hz_by_third'])}); "
        f"90% of the energy between {d['energy_5_to_95_percent_hz'][0]} and {d['energy_5_to_95_percent_hz'][1]} Hz",
        f"  envelope   loudest at {d['loudest_at_fraction']*100:.0f}% of the length; half level (its RMS, in short "
        f"frames, first at half its highest) after "
        f"{d['half_level_after_s']:.3f} s; dB below the loudest tenth: {' '.join(f'{-v:.0f}' for v in d['level_db_by_tenth'])}",
        (f"  flutter    amplitude modulation of depth {d['flutter_depth']:.2f} at {d['flutter_hz']} Hz (15 to 90 Hz searched)"
         if d['flutter_depth'] >= FLUTTER_SAID else
         # a depth this small at the search's floor read as a flutter to be fixed
         f"  flutter    none (depth under {FLUTTER_SAID:.2f} between 15 and 90 Hz)"),
    ])


# -- against what a sound should measure as --------------------------------------------------------------------------

SHAPE_STEP = 0.05             # a pitch moves when it changes by this share; less is steady
SHAPES = ("rise-fall", "rise", "fall", "steady")
# key -> (what it is, how it is read from describe(), its unit)
EXPECT = {
    "pitch_hz": ("pitch (median of the voiced points)", None, "Hz"),
    "voiced": ("voiced share of the loud frames", "voiced_fraction", ""),
    "periodicity": ("periodicity", "periodicity", ""),
    "centroid_hz": ("spectral centroid", "centroid_hz", "Hz"),
    "flatness": ("spectral flatness", "spectral_flatness", ""),
    "attack_s": ("time to half level (RMS first at half its highest)", "half_level_after_s", "s"),
    "loudest_at": ("where it is loudest, as a share of its length", "loudest_at_fraction", ""),
    "flutter_hz": ("amplitude flutter rate", "flutter_hz", "Hz"),
    "flutter_depth": ("amplitude flutter depth", "flutter_depth", ""),
}


def pitch_shape(d) -> str | None:
    """rise-fall, rise, fall or steady, from the pitch at 10/30/50/70/90% of the length; None with too few points."""
    points = [v for v in d['f0_hz_at_10_30_50_70_90_percent'] if v]
    if len(points) < 3:
        return None
    first, last, top = points[0], points[-1], max(points[1:-1])
    if top > (1 + SHAPE_STEP)*max(first, last):
        return "rise-fall"
    if last > (1 + SHAPE_STEP)*first:
        return "rise"
    if first > (1 + SHAPE_STEP)*last:
        return "fall"
    return "steady"


def against(d, expect) -> list[tuple[bool, str]]:
    """[(met, line)] for each key of `expect` ({key: [low, high]} or {"pitch_shape": one of SHAPES})."""
    out = []
    for key, want in expect.items():
        if key == "pitch_shape":
            got = pitch_shape(d)
            points = ' '.join('-' if v is None else str(v) for v in d['f0_hz_at_10_30_50_70_90_percent'])
            out.append((got == want, f"pitch shape {got or 'unread (too little voice)'} ({points} Hz), {want} wanted"))
            continue
        label, field, unit = EXPECT[key]
        if key == "pitch_hz":
            points = [v for v in d['f0_hz_at_10_30_50_70_90_percent'] if v]
            value = float(np.median(points)) if points else None
        else:
            value = d[field]
        low, high = want
        shown = 'none' if value is None else f"{value:g}{' ' + unit if unit else ''}"
        met = value is not None and low <= value <= high
        out.append((met, f"{label} {shown}, {low:g} to {high:g}{' ' + unit if unit else ''} wanted"))
    return out


def wanted(expect) -> str:
    """What `expect` asks for, in words."""
    parts = []
    for key, want in expect.items():
        if key == "pitch_shape":
            parts.append(f"pitch shape {want}")
        else:
            label, _, unit = EXPECT[key]
            parts.append(f"{label} {want[0]:g} to {want[1]:g}{' ' + unit if unit else ''}")
    return "; ".join(parts)


def format_against(name, d, expect) -> str:
    rows = against(d, expect)
    met = sum(ok for ok, _ in rows)
    return '\n'.join([f"{name}: {met} of {len(rows)} as it should measure"]
                     + [f"  {'ok ' if ok else 'OFF'} {line}" for ok, line in rows])


# -- the picture ---------------------------------------------------------------------------------------------------

PX_PER_S = 640
ROWS = 220
WAVE_ROWS = 48
PLOT_LOW, PLOT_HIGH = 60.0, 16000.0
MARGIN_LEFT, MARGIN_TOP, MARGIN_BOTTOM, GAP = 52, 20, 18, 28
STOPS = np.array([[0, 0, 4], [60, 15, 110], [180, 55, 120], [250, 140, 60], [252, 250, 190]], dtype=np.float64)


def spectrogram_pixels(x, rate):
    """(rows x columns x 3 uint8, log frequency axis PLOT_LOW..PLOT_HIGH, top = high) with the pitch track in cyan."""
    hop = max(int(round(rate/PX_PER_S)), 1)
    freqs, power = spectra(x, rate, size=FRAME, hop=hop, nfft=4*FRAME)
    columns = power.shape[0]
    wanted = PLOT_HIGH*(PLOT_LOW/PLOT_HIGH)**(np.arange(ROWS)/(ROWS-1))
    db = 10*np.log10(np.maximum(power[:, np.searchsorted(freqs, wanted).clip(0, len(freqs)-1)].T, 1e-18))
    level = np.clip((db - db.max() + 70)/70, 0, 1)*(len(STOPS)-1)
    base = np.minimum(level.astype(int), len(STOPS)-2)
    pixels = STOPS[base] + (STOPS[base+1]-STOPS[base])*(level-base)[..., None]
    times, f0, _, _ = pitch_track(x, rate)
    for t, f in zip(times, f0):
        if not np.isnan(f) and PLOT_LOW <= f <= PLOT_HIGH:
            row = int(round(np.log(PLOT_HIGH/f)/np.log(PLOT_HIGH/PLOT_LOW)*(ROWS-1)))
            column = min(int(t*rate/hop), columns-1)
            pixels[max(row-1, 0):row+1, column:column+2] = (0, 255, 255)
    return pixels.astype(np.uint8), hop


def write_sheet(sounds, path, marks=None):
    """One PNG: for each (name, rate, samples) a log-frequency spectrogram over its waveform, all at one time scale.
    `marks` is {name: [(seconds, label), ...]}: a cue drawn as a yellow line down the spectrogram and the waveform."""
    from PIL import Image, ImageDraw
    cell_w = MARGIN_LEFT + PX_PER_S + GAP
    cell_h = MARGIN_TOP + ROWS + WAVE_ROWS + MARGIN_BOTTOM + GAP
    per_row = 2
    sheet = Image.new('RGB', (cell_w*per_row, cell_h*((len(sounds)+per_row-1)//per_row)), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)
    for n, (name, rate, x) in enumerate(sounds):
        left, top = (n % per_row)*cell_w + MARGIN_LEFT, (n//per_row)*cell_h + MARGIN_TOP
        pixels, hop = spectrogram_pixels(x, rate)
        pixels = pixels[:, :PX_PER_S]
        sheet.paste(Image.fromarray(pixels), (left, top))
        draw.text((left, top-MARGIN_TOP+4), f'{name}   {len(x)/rate:.3f} s   60 Hz to 16 kHz (log), 70 dB, cyan = pitch track',
                  fill=(230, 230, 230))
        for hz in (100, 200, 500, 1000, 2000, 5000, 10000):
            y = top + np.log(PLOT_HIGH/hz)/np.log(PLOT_HIGH/PLOT_LOW)*(ROWS-1)
            draw.line([(left-4, y), (left-1, y)], fill=(200, 200, 200))
            draw.text((left-MARGIN_LEFT+2, y-5), f'{hz//1000}k' if hz >= 1000 else str(hz), fill=(200, 200, 200))
        wave_top = top + ROWS
        draw.rectangle([left, wave_top, left+PX_PER_S-1, wave_top+WAVE_ROWS-1], fill=(10, 10, 14))
        middle = wave_top + WAVE_ROWS/2
        for column in range(min(pixels.shape[1], PX_PER_S)):
            part = x[column*hop:(column+1)*hop]
            if len(part):
                draw.line([(left+column, middle - part.max()*WAVE_ROWS/2), (left+column, middle - part.min()*WAVE_ROWS/2)],
                          fill=(120, 200, 255))
        for k, (at, label) in enumerate((marks or {}).get(name, [])):
            mx = left + at*rate/hop
            if left <= mx < left + PX_PER_S:
                draw.line([(mx, top), (mx, wave_top+WAVE_ROWS-1)], fill=(255, 220, 0))
                draw.text((mx+2, top+2+12*(k % 3)), label, fill=(255, 220, 0))     # cues 11 ms apart: labels stacked
        for tenth in range(11):
            tx = left + tenth*PX_PER_S/10
            draw.line([(tx, wave_top+WAVE_ROWS), (tx, wave_top+WAVE_ROWS+3)], fill=(200, 200, 200))
            if tenth % 2 == 0:
                draw.text((tx+2, wave_top+WAVE_ROWS+4), f'{tenth/10:.1f}', fill=(200, 200, 200))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path

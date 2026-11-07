'A RECIPE a sound script may start from: the pieces a voice, a footfall and a crash are made of, as code.'

RECIPE = '''import numpy as np

R = 44100


def curve(n, points, smooth=0.02):
    """n values through (position 0..1 of the sound, value) points, corners rounded over `smooth` seconds."""
    position, value = zip(*points)
    line = np.interp(np.linspace(0.0, 1.0, n), position, value)
    width = max(int(R * smooth) | 1, 3)
    window = np.hanning(width + 2)[1:-1]
    padded = np.concatenate([np.full(width // 2, line[0]), line, np.full(width // 2, line[-1])])
    return np.convolve(padded, window / window.sum(), mode="valid")


def drift(rng, n, hz):
    """Slow random wobble (nothing faster than `hz`), spread 1: multiply a pitch by (1 + 0.01 * drift) to make it live."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    spectrum[np.fft.rfftfreq(n, 1.0 / R) > hz] = 0.0
    x = np.fft.irfft(spectrum, n)
    return x / np.std(x)


def tube(freq, resonances):
    """The complex gain at `freq` of a tube with these (centre Hz, width Hz) resonances: a mouth and throat."""
    gain = 1.0 + 0.0j
    for centre, width in resonances:
        ratio = freq / centre
        gain = gain / ((1.0 - ratio * ratio) + 1.0j * ratio * width / centre)
    return gain


def mouth(opening):
    """A small animal's mouth from shut and rounded (0) to wide open (1): two resonances it moves, three it barely does."""
    return [(650.0 + 850.0 * opening, 130.0), (1900.0 + 1400.0 * opening, 220.0),
            (5000.0, 350.0), (7000.0, 490.0), (9000.0, 630.0)]


def voice(f0, opening, tilt=1.0, top=16000.0):
    """A voice: every harmonic of the pitch f0 (Hz, one value per sample) through the mouth (opening per sample).
    tilt: how fast the harmonics fall away (1.2 soft and round, 0.7 pressed and harsh). Loudest sample 1."""
    phase = 2.0 * np.pi * np.cumsum(f0) / R
    resonances = mouth(opening)
    out = np.zeros(len(f0))
    k = 1
    while k * f0.min() < top:
        freq = k * f0
        through = tube(freq, resonances)
        out += k ** (-tilt) * np.abs(through) * np.clip((top - freq) / 2000.0, 0.0, 1.0) * np.sin(k * phase + np.angle(through))
        k += 1
    return out / np.max(np.abs(out))


def shaped(rng, n, shape):
    """Noise whose spectrum is shape(frequency array): breath, a hiss, grass, the body of a thud. Loudest sample 1."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    x = np.fft.irfft(spectrum * shape(np.fft.rfftfreq(n, 1.0 / R)), n)
    return x / np.max(np.abs(x))


def thud(rng, seconds, below_hz, decay_s):
    """A soft impact: low noise (most of it under `below_hz`) that starts in 2 ms and dies away over `decay_s`."""
    k = np.arange(int(seconds * R))
    body = shaped(rng, len(k), lambda f: 1.0 / (1.0 + (f / below_hz) ** 2))
    return body * np.exp(-k / (decay_s * R)) * np.clip(k / (0.002 * R), 0.0, 1.0)


def place(x, sound, at):
    """Add `sound` into the loop x starting at `at` seconds; what runs past the end wraps round to the start."""
    idx = (int(round(at * R)) + np.arange(len(sound))) % len(x)
    np.add.at(x, idx, sound)


def finish(x, loop=False):
    """Take out everything under 40 Hz (and the DC with it), fade 5 ms at each end unless it loops, peak 0.9."""
    spectrum = np.fft.rfft(x)
    spectrum *= np.clip(np.fft.rfftfreq(len(x), 1.0 / R) / 40.0, 0.0, 1.0) ** 2
    x = np.fft.irfft(spectrum, len(x))
    if not loop:
        edge = int(R * 0.005)
        x[:edge] *= np.linspace(0.0, 1.0, edge)
        x[-edge:] *= np.linspace(1.0, 0.0, edge)
    return 0.9 * x / np.max(np.abs(x))


def growl(rng, seconds, low_hz=80.0, high_hz=140.0, pulses=(26.0, 40.0)):
    """A low pressed voice chopped into a rattle that speeds up (pulses a second, from, to) and rises in pitch."""
    n = int(R * seconds)
    f0 = curve(n, [(0, low_hz), (1, high_hz)], 0.03) * (1 + 0.03 * drift(rng, n, 25))
    rate = curve(n, [(0, pulses[0]), (1, pulses[1])])
    pulse = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.cumsum(rate) / R)) ** 1.5
    tone = voice(f0, curve(n, [(0, 0.1), (1, 0.5)]), 0.8, top=9000.0)
    rasp = shaped(rng, n, lambda f: 1.0 / (1.0 + (f / 1500.0) ** 2))
    return pulse * (tone + 0.3 * rasp)


def break_up(rng, x, start=0.0, grain_s=0.025, crush=10, bits=(12, 3), gaps=0.7):
    """Take a finished sound apart from `start` (share of its length) on, more and more: grains repeat, the sample rate
    falls to 1/`crush`, the bits from bits[0] to bits[1], and from `gaps` on, pieces drop out. Start it early: a sound
    broken up gradually from its first instant is still heard as what it was."""
    n = len(x)
    s = int(start * n)
    y = x.copy()
    grain = int(grain_s * R)
    for a in range(s, n - 2 * grain, 3 * grain):
        y[a + grain:a + 2 * grain] = y[a:a + grain]
    hold = np.maximum(np.interp(np.arange(n), [s, n], [1, crush]).astype(int), 1)
    y[s:] = y[(np.arange(n) // hold) * hold][s:]
    q = 2.0 ** np.interp(np.arange(n), [s, n], list(bits))
    y[s:] = (np.round(y * q) / q)[s:]
    for a in range(int(gaps * n), n, int(0.04 * R)):
        if rng.random() < 0.5:
            y[a:a + int(0.015 * R)] = 0.0
    return y


# An example of the pieces together: a call that opens and closes, its pitch up a fifth and back.
def example_call(rng, seconds=0.7):
    n = int(R * seconds)
    f0 = curve(n, [(0, 420), (0.35, 690), (1, 400)], 0.04) * (1 + 0.007 * drift(rng, n, 11))
    opening = curve(n, [(0, 0.0), (0.35, 1.0), (1, 0.1)], 0.05)
    loud = curve(n, [(0, 0.0), (0.08, 0.3), (0.35, 1.0), (0.85, 0.3), (1, 0.0)], 0.03)
    breath = shaped(rng, n, lambda f: np.abs(tube(f, mouth(0.5))) * f / (f + 2500.0))
    return finish(loud * (voice(f0, opening, 1.15 - 0.3 * opening) + 0.05 * breath))
'''


def template(spec: dict, cues: dict) -> str:
    """The recipe as a sounds.py to start from (`sound-info --write-template`): a `sounds(cues)` that holds the
    contract as it stands, the example call for each sound played once and a thud on each cue a clip's sound must
    start on, for the author to replace one sound at a time. Two of three builders wrote the whole synthesis cold."""
    lines = [RECIPE.strip(), "", "",
             "def cue_times(cue):",
             "    # a cue as the script is given it: a time, or [[time, side], ...] (footfalls)",
             "    return [c[0] if isinstance(c, (list, tuple)) else c for c in (cue if isinstance(cue, list) else [cue])]",
             "", "",
             "def sounds(cues):",
             "    # A first build that holds the contract: replace each sound with your own, one at a time.",
             "    rng = np.random.default_rng(1)",
             "    out = {}"]
    for name, s in spec["sounds"].items():
        c = cues.get(name) or {}
        if "clip" in s:
            on = s.get("on") or []
            lines.append(f'    x = np.zeros(int(round(cues["{name}"]["seconds"] * R)))')
            lines += [f'    if not cues["{name}"]["loop"]:   # played once, it is heard from its first instant',
                      "        x += 0.1 * shaped(rng, len(x), lambda f: 1.0 / (1.0 + (f / 800.0) ** 2))"]
            for cue in ([on] if isinstance(on, str) else on):
                lines += [f'    for t in cue_times(cues["{name}"]["{cue}"]):',
                          "        place(x, thud(rng, 0.12, 300.0, 0.04), t)"]
            lines.append(f'    out["{name}"] = finish(x, loop=cues["{name}"]["loop"])')
        else:
            shortest, longest = c.get("shortest") or 0.4, c.get("longest") or shortest
            lines.append(f'    out["{name}"] = finish(example_call(rng, seconds={(shortest + longest) / 2:.3f}))')
    lines.append("    return out")
    return "\n".join(lines) + "\n"

"""Sounds from code: a sound script, the contract it is held to, and whether each sound lands on its clip's cues.

A sound directory holds `sound.json` (the contract: which sounds, how long, which clip each goes with and which of
that clip's cues it must start a sound on), `brief.md` (what the sounds are for, in words) and, when sounds go
with clips, the rigged model (`animated.glb`, the rig stage's output) the cues are read from (`sound_cues`).

The author writes one Python file with a top-level `def sounds(cues):` that returns {name: samples}, a 1-D float
array at 44,100 Hz in -1..1 per sound. `cues` is {name: {"seconds", "loop", and the clip's cue times}}: a sound
that goes with a clip is given the clip's own length and times, so it is made to fit the animation rather than
guessed at. The script imports numpy, math and random only and is refused before it runs if `script_check` finds
anything else; it runs in a child process (`sound_run`) twice, and the two runs must write the same bytes.

What `check` holds a set of sounds to:
- a one-shot: its length within the contract's bounds, a peak of 0.50 to 0.98, RMS above -30 dBFS, no DC, no
  silence padding at either end, no click at either end (the thresholds are game-factory's audio review's);
- a sound that goes with a clip: exactly the clip's length to the sample, so it stays in step when both repeat;
  if the clip loops, its last sample runs into its first without a jump (there is no fade, a loop has no ends);
- each cue the contract names in `on`: a sound starts there, within ONSET_TOL_S (an onset strong enough to count,
  found by spectral flux, `onsets`).

A sound may also carry `expect`: what it should measure as (a pitch range and shape, how voiced, how bright, how
fast it starts), which the agent's builds read each sound against (`sound_measure.against`). Readings, not gates.

Deterministic. `check` measures; it says nothing about whether a sound sounds like what it is for.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import script_check, sound_cues, sound_measure

RATE = 44100
CONTRACT, BRIEF, ANIMATED, SCRIPT = "sound.json", "brief.md", "animated.glb", "sounds.py"
ALLOWED_IMPORTS = {"math", "random", "numpy"}
RUN_TIMEOUT_S = 60
PEAK_MIN, PEAK_MAX = 0.50, 0.98
RMS_MIN_DB = -30.0            # a one-shot; a loop of footfalls is mostly quiet between them
LOOP_RMS_MIN_DB = -45.0
MAX_DC = 0.01
SILENCE = 0.001
MAX_LEAD_S, MAX_TRAIL_S = 0.02, 0.10
EDGE_SAMPLE = 0.02
EDGE_S, EDGE_FRACTION = 0.001, 0.25
SEAM = 0.05                   # a loop's last sample to its first, of full scale, may always differ by this much,
SEAM_NEAR_S = 0.002           # ... and by up to SEAM_RATIO times the largest step between neighbours this close to it:
SEAM_RATIO = 2.0              # a footfall that wraps round is loud noise across the seam, not a click
ONSET_TOL_S = 0.035           # a cue's sound starts this close to it: two frames at 60 fps
ONSET_MIN = 0.25              # an onset counts at this share of the sound's strongest
STRAY_MIN = 0.5               # an onset this strong away from every cue is recorded (not said to the author)
CUE_KEYS = ("crouch", "takeoff", "apex", "landing")


class ContractError(ValueError):
    pass


@dataclass
class Result:
    sounds: dict = field(default_factory=dict)      # name -> {"stats", "problems", "cues": [...], "strays": [...]}
    problems: list = field(default_factory=list)    # the set's own: missing, refused, not deterministic
    cues: dict = field(default_factory=dict)        # what the script was given

    @property
    def ok(self) -> bool:
        return not self.problems and all(not s["problems"] for s in self.sounds.values())


# -- the contract ---------------------------------------------------------------------------------------------------

def load(root: str | Path) -> dict:
    """sound.json, checked and filled in: every sound has `required`, and either `seconds` [lo, hi] or `clip`."""
    root = Path(root)
    spec = json.loads((root / CONTRACT).read_text(encoding="utf-8"))
    sounds = spec.get("sounds")
    if not isinstance(sounds, dict) or not sounds:
        raise ContractError(f"{CONTRACT}: `sounds` is a non-empty object of name -> sound")
    for name, s in sounds.items():
        if not name.replace("_", "").isalnum():
            raise ContractError(f"{CONTRACT}: a sound's name is letters, digits and '_': {name!r}")
        s.setdefault("required", True)
        _expectations(name, s.setdefault("expect", {}))
        between = s.setdefault("between", {})
        if between and "clip" not in s:
            raise ContractError(f"{CONTRACT}: {name}: `between` reads a stretch between two of a clip's cues; give a `clip`")
        for span, want in between.items():
            ends = span.split("-")
            if len(ends) != 2 or not all(e in ("start", "end") + CUE_KEYS for e in ends):
                raise ContractError(f"{CONTRACT}: {name}: between.{span}: a stretch is two of start, end, "
                                    f"{', '.join(CUE_KEYS)} joined by '-', like takeoff-landing")
            _expectations(f"{name} {span}", want)
        if "clip" in s:
            on = s.get("on", [])
            s["on"] = [on] if isinstance(on, str) else list(on)
            bad = [c for c in s["on"] if c not in ("footfalls",) + CUE_KEYS]
            if bad:
                raise ContractError(f"{CONTRACT}: {name}: no cue {bad[0]!r} (cues: footfalls, {', '.join(CUE_KEYS)})")
        else:
            lo_hi = s.get("seconds")
            if not (isinstance(lo_hi, list) and len(lo_hi) == 2 and 0 < lo_hi[0] <= lo_hi[1]):
                raise ContractError(f"{CONTRACT}: {name}: give `seconds` [shortest, longest] or a `clip`")
    if any("clip" in s for s in sounds.values()) and not (root / spec.get("animated", ANIMATED)).is_file():
        raise ContractError(f"sounds go with clips but there is no {spec.get('animated', ANIMATED)}")
    return spec


def _expectations(name: str, expect: dict) -> None:
    for key, want in expect.items():
        if key == "pitch_shape":
            if want not in sound_measure.SHAPES:
                raise ContractError(f"{CONTRACT}: {name}: pitch_shape is one of {', '.join(sound_measure.SHAPES)}")
        elif key not in sound_measure.EXPECT:
            raise ContractError(f"{CONTRACT}: {name}: no expectation {key!r} (they are pitch_shape, "
                                f"{', '.join(sound_measure.EXPECT)})")
        elif not (isinstance(want, list) and len(want) == 2 and want[0] <= want[1]):
            raise ContractError(f"{CONTRACT}: {name}: expect.{key} is [low, high]")


def stretch(x: np.ndarray, span: str, c: dict) -> np.ndarray:
    """The part of a clip's sound between two of its cues (`takeoff-landing`; `start` and `end` are its ends)."""
    a, b = (0.0 if e == "start" else c["seconds"] if e == "end" else c[e] for e in span.split("-"))
    return x[int(round(a * RATE)):int(round(b * RATE))]


def cues_for(root: str | Path, spec: dict) -> dict:
    """What the script is given: {name: {"seconds", "loop", ...}} with each clip's cue times."""
    root = Path(root)
    clips = {s["clip"]: s.get("loop") for s in spec["sounds"].values() if "clip" in s}
    read = {}
    if clips:
        read = sound_cues.read(root / spec.get("animated", ANIMATED), clips)
        missing = [c for c in clips if c not in read]
        if missing:
            raise ContractError(f"{spec.get('animated', ANIMATED)} has no clip {missing[0]!r}")
    out = {}
    for name, s in spec["sounds"].items():
        if "clip" in s:
            c = dict(read[s["clip"]])
            if s.get("loop") is not None:
                c["loop"] = bool(s["loop"])
            # a leap that never leaves the ground has no takeoff: said before a script is run, not as its KeyError
            lacking = [cue for cue in s["on"] if cue in CUE_KEYS and cue not in c]
            if lacking or ("footfalls" in s["on"] and not c.get("footfalls")):
                why = ("its lowest point never rises clear of the ground" if lacking else "no foot leaves the ground")
                raise ContractError(f"{name} starts sounds on {', '.join(lacking or ['footfalls'])}, and the clip "
                                    f"{s['clip']} has none: {why} (mesh-jig sound-cues reads it)")
            out[name] = {"clip": s["clip"], **c}
        else:
            out[name] = {"seconds": None, "shortest": s["seconds"][0], "longest": s["seconds"][1], "loop": False}
    return out


def contract_text(spec: dict, cues: dict) -> str:
    """The contract as the author is told it."""
    rows = []
    for name, s in spec["sounds"].items():
        c = cues[name]
        need = "required" if s["required"] else "optional"
        if "clip" in s:
            n = int(round(c["seconds"] * RATE))
            at = []
            for cue in s["on"]:
                if cue == "footfalls":
                    at += [f"{t:.3f} s" for t, _ in c.get("footfalls", [])]
                elif cue in c:
                    at.append(f"{c[cue]:.3f} s ({cue})")
            rows.append(f"- {name} ({need}): goes with the clip `{s['clip']}`; exactly {n} samples ({c['seconds']:g} s)"
                        + (", and it loops: its last sample runs into its first" if c["loop"] else ", played once")
                        + (f"; a sound starts at {', '.join(at)} (within {ONSET_TOL_S * 1000:.0f} ms)" if at else ""))
        else:
            rows.append(f"- {name} ({need}): {s['seconds'][0]:.2f} to {s['seconds'][1]:.2f} s, played once")
    rules = [
        f"Level: the loudest sample is {PEAK_MIN:.2f} to {PEAK_MAX:.2f} of full scale; RMS above {RMS_MIN_DB:.0f} dBFS "
        f"({LOOP_RMS_MIN_DB:.0f} for a loop).",
        f"No DC: the mean is within {MAX_DC:.2f} of zero.",
        f"A sound played once starts within {MAX_LEAD_S:.2f} s and ends within {MAX_TRAIL_S:.2f} s of its ends "
        f"(sound is anything above {SILENCE:g}), its first and last samples are within {EDGE_SAMPLE:.2f} of zero, and "
        f"nothing in its first or last {EDGE_S * 1000:.0f} ms is above {EDGE_FRACTION:.2f} of its peak (fade the ends).",
        f"A loop has no ends: its last sample runs into its first as any two neighbouring samples do (within {SEAM:.2f}, "
        f"or within {SEAM_RATIO:g} times the largest step between neighbours in the {SEAM_NEAR_S * 1000:g} ms either side), "
        "so it repeats without a click.",
        f"A sound starts at a cue when an onset (a sudden rise in loudness or a new sound coming in) at least "
        f"{ONSET_MIN:.2f} as strong as the sound's strongest is within {ONSET_TOL_S * 1000:.0f} ms of it.",
        "No two sounds are the same. Samples are floats in -1..1; beyond that they are clipped and reported.",
        f"The script imports only {', '.join(sorted(ALLOWED_IMPORTS))}, defines a top-level sounds(cues), finishes "
        f"within {RUN_TIMEOUT_S} s and returns the same samples every run (seed every random generator).",
    ]
    return "\n".join(rows) + "\n\n" + "\n".join(f"- {r}" for r in rules)


# -- onsets ---------------------------------------------------------------------------------------------------------

ONSET_WIN, ONSET_HOP = 512, 128
BANDS = 40


def onset_strength(x: np.ndarray, rate: int = RATE, loop: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(times s, strength): spectral flux of log band energies, the rise of each band summed, a frame every
    ONSET_HOP samples. A loop is read wrapped round (its start follows its end); a one-shot follows silence."""
    n = len(x)
    pad = ONSET_WIN
    if loop:
        reps = int(np.ceil(pad / max(n, 1))) + 1
        ext = np.concatenate([np.tile(x, reps)[-pad:], x, np.tile(x, reps)[:pad]])
    else:
        ext = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    count = n // ONSET_HOP + 1
    starts = pad - ONSET_WIN // 2 + ONSET_HOP * np.arange(-2, count)
    rows = ext[np.clip(starts[:, None] + np.arange(ONSET_WIN)[None, :], 0, len(ext) - 1)] * np.hanning(ONSET_WIN)
    mag = np.abs(np.fft.rfft(rows, axis=1))
    freqs = np.fft.rfftfreq(ONSET_WIN, 1 / rate)
    edges = np.geomspace(60, min(16000, rate / 2), BANDS + 1)
    which = np.searchsorted(edges, freqs) - 1
    keep = (which >= 0) & (which < BANDS)
    band = np.zeros((len(rows), BANDS))
    for b in range(BANDS):
        sel = keep & (which == b)
        if sel.any():
            band[:, b] = mag[:, sel].sum(axis=1)
    level = np.log10(1 + 1000 * band / ONSET_WIN)
    flux = np.maximum(level[2:] - level[:-2], 0).sum(axis=1)          # against the frame two hops back
    times = ONSET_HOP * np.arange(count) / rate
    return times, flux[:count]


def onsets(x: np.ndarray, rate: int = RATE, loop: bool = False) -> list[tuple[float, float]]:
    """[(time s, strength as a share of the strongest)]: the peaks of `onset_strength`, at most one per 30 ms."""
    times, s = onset_strength(x, rate, loop)
    top = float(s.max()) if len(s) else 0.0
    if top <= 0:
        return []
    reach = max(int(round(0.03 * rate / ONSET_HOP)), 1)
    out = []
    for i in range(len(s)):
        lo, hi = max(i - reach, 0), min(i + reach + 1, len(s))
        if s[i] > 0 and s[i] == s[lo:hi].max() and (i == lo or s[i] > s[lo:i].max()):
            out.append((round(float(times[i]), 4), round(float(s[i] / top), 3)))
    return [o for o in out if o[1] >= 0.1]


# -- checking a folder of WAVs ---------------------------------------------------------------------------------------

def read_wav(path: str | Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, RATE):
            raise ValueError(f"{Path(path).name}: not mono 16-bit {RATE} Hz")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0


def check_sound(name: str, x: np.ndarray, s: dict, c: dict) -> dict:
    """{"stats", "problems", "cues", "strays"} for one sound against its contract entry `s` and cues `c`."""
    problems = []
    peak = float(np.abs(x).max()) if len(x) else 0.0
    rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
    loud = np.flatnonzero(np.abs(x) > SILENCE)
    stats = {"seconds": round(len(x) / RATE, 4), "samples": len(x), "peak": round(peak, 3),
             "rms_db": round(20 * np.log10(max(rms, 1e-9)), 1), "dc": round(float(np.mean(x)), 4) if len(x) else 0.0}
    loop = bool(c.get("loop"))
    if "clip" in s:
        want = int(round(c["seconds"] * RATE))
        if len(x) != want:
            problems.append(f"{len(x)} samples ({len(x) / RATE:.4f} s); the clip {s['clip']} is {c['seconds']:g} s, "
                            f"so exactly {want} samples")
    elif not s["seconds"][0] <= len(x) / RATE <= s["seconds"][1]:
        problems.append(f"length {len(x) / RATE:.3f} s is outside {s['seconds'][0]:.2f} to {s['seconds'][1]:.2f} s")
    if not len(loud):
        return {"stats": stats, "problems": problems + ["silent"], "cues": [], "strays": []}
    if peak < PEAK_MIN:
        problems.append(f"too quiet: peak {peak:.2f}, at least {PEAK_MIN:.2f} (scale it up)")
    if peak > PEAK_MAX:
        problems.append(f"too hot: peak {peak:.3f}, at most {PEAK_MAX:.2f} (scale it down)")
    floor = LOOP_RMS_MIN_DB if loop else RMS_MIN_DB
    if stats["rms_db"] < floor:
        problems.append(f"nearly silent: RMS {stats['rms_db']:.1f} dBFS, above {floor:.0f} is wanted")
    if abs(stats["dc"]) > MAX_DC:
        problems.append(f"DC offset {stats['dc']:+.3f}, within {MAX_DC:.2f} of zero is wanted (subtract the mean)")
    if loop:
        jump = abs(float(x[-1] - x[0]))
        near = max(int(RATE * SEAM_NEAR_S), 2)
        around = max(float(np.abs(np.diff(x[-near:])).max()), float(np.abs(np.diff(x[:near])).max()))
        if jump > max(SEAM, SEAM_RATIO * around):
            problems.append(f"the loop clicks where it repeats: the last sample is {x[-1]:+.3f} and the first "
                            f"{x[0]:+.3f}, {jump:.3f} apart, where neighbouring samples near the seam differ by at "
                            f"most {around:.3f}")
    else:
        lead, trail = loud[0] / RATE, (len(x) - 1 - loud[-1]) / RATE
        if lead > MAX_LEAD_S:
            problems.append(f"{lead:.3f} s of silence before the sound starts, at most {MAX_LEAD_S:.2f}")
        if trail > MAX_TRAIL_S:
            problems.append(f"{trail:.3f} s of silence after it ends, at most {MAX_TRAIL_S:.2f}")
        edge = max(int(round(RATE * EDGE_S)), 1)
        for label, sample, window in (("first", x[0], x[:edge]), ("last", x[-1], x[-edge:])):
            if abs(sample) > EDGE_SAMPLE:
                problems.append(f"the {label} sample is {sample:+.3f}, within {EDGE_SAMPLE:.2f} of zero is wanted (fade it)")
            elif np.abs(window).max() > EDGE_FRACTION * peak:
                problems.append(f"the {label} {EDGE_S * 1000:.0f} ms reaches {np.abs(window).max() / peak:.2f} of the "
                                f"peak, at most {EDGE_FRACTION:.2f} (fade over a few ms)")
    found = onsets(x, RATE, loop)
    marks, wanted = [], []
    for cue in s.get("on", []):
        if cue == "footfalls":
            wanted += [(f"footfall {side}", t) for t, side in c.get("footfalls", [])]
        elif cue in c:
            wanted.append((cue, c[cue]))
        else:
            problems.append(f"the clip has no {cue} (the cues read from it: {', '.join(k for k in c if k in CUE_KEYS or k == 'footfalls')})")
    period = len(x) / RATE
    for label, t in wanted:
        near = [(abs(_lag(o, t, period, loop)), o, w) for o, w in found if w >= ONSET_MIN]
        best = min(near, default=None)
        mark = {"cue": label, "at": round(t, 4)}
        if best and best[0] <= ONSET_TOL_S:
            mark.update(onset=best[1], strength=best[2], off_ms=round(_lag(best[1], t, period, loop) * 1000, 1))
        else:
            weak = min(((abs(_lag(o, t, period, loop)), o, w) for o, w in found), default=None)
            mark["missing"] = True
            if weak and weak[0] <= ONSET_TOL_S:
                strongest = max(found, key=lambda ow: ow[1])[0]
                # three builders lost builds here: doubling a low thud barely moved it, since an onset is a rise in
                # log band energies and a bright attack elsewhere sets the scale
                problems.append(f"{label} at {t:.3f} s: what starts there ({weak[1]:.3f} s) is only {weak[2]:.2f} of "
                                f"the sound's strongest onset (at {strongest:.3f} s), at least {ONSET_MIN:.2f} is "
                                "wanted. An onset is a rise in energy across the bands, so a low thud counts for less "
                                "than a bright attack: a short burst of noise above 1 kHz on the cue raises it more "
                                "than a louder thud does")
            else:
                where = f"; the nearest onset is at {best[1]:.3f} s ({best[0] * 1000:.0f} ms away)" if best else ""
                problems.append(f"{label} at {t:.3f} s: nothing starts within {ONSET_TOL_S * 1000:.0f} ms of it{where}")
        marks.append(mark)
    strays = []
    if wanted:
        strays = [{"at": o, "strength": w} for o, w in found
                  if w >= STRAY_MIN and all(abs(_lag(o, t, period, loop)) > ONSET_TOL_S for _, t in wanted)]
    return {"stats": stats, "problems": problems, "cues": marks, "strays": strays, "onsets": found[:24]}


def _lag(o: float, t: float, period: float, loop: bool) -> float:
    d = o - t
    if loop and period > 0:
        d = (d + period / 2) % period - period / 2
    return d


def check_folder(folder: str | Path, spec: dict, cues: dict) -> Result:
    folder = Path(folder)
    res = Result(cues=cues)
    seen = {}
    for name, s in spec["sounds"].items():
        path = folder / f"{name}.wav"
        if not path.is_file():
            if s["required"]:
                res.problems.append(f"{name} is missing")
            continue
        try:
            x = read_wav(path)
        except (ValueError, wave.Error, EOFError) as e:
            res.sounds[name] = {"stats": {}, "problems": [str(e)], "cues": [], "strays": []}
            continue
        res.sounds[name] = check_sound(name, x, s, cues[name])
        twin = seen.setdefault(path.read_bytes(), name)
        if twin != name:
            res.problems.append(f"{name} is the same as {twin}")
    return res


# -- running a script ------------------------------------------------------------------------------------------------

def check_script(script: str | Path) -> list[str]:
    source = Path(script).read_text(encoding="utf-8")
    return [p.replace("(the runner calls sounds() then exports)", "(the runner calls sounds(cues))")
            for p in script_check.check_script(source, required=("sounds",), allowed_imports=ALLOWED_IMPORTS)]


def run(script: str | Path, cues: dict, out: str | Path, python: str = sys.executable) -> dict:
    """Run the script once into `out` (which must be empty). -> sound_run's record, with "error" on a failure."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as cwd:
        cue_file = Path(cwd) / "cues.json"
        cue_file.write_text(json.dumps(cues), encoding="utf-8")
        runner = Path(__file__).with_name("sound_run.py")
        try:
            done = subprocess.run([python, "-I", "-B", str(runner), str(Path(script).resolve()), str(cue_file), str(out.resolve())],
                                  cwd=cwd, capture_output=True, text=True, errors="replace", timeout=RUN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return {"error": f"the script did not finish within {RUN_TIMEOUT_S} s", "written": {}, "problems": []}
    record_path = out / "run.json"
    if not record_path.is_file():
        tail = "\n".join((done.stderr or done.stdout).strip().splitlines()[-8:])
        return {"error": f"the runner exited with code {done.returncode}:\n{tail}", "written": {}, "problems": []}
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record_path.unlink()
    return record


def check(root: str | Path, script: str | Path, keep: str | Path | None = None, python: str = sys.executable) -> Result:
    """Refuse, run twice, compare, check. `keep` receives the first run's WAVs."""
    root = Path(root)
    spec = load(root)
    cues = cues_for(root, spec)
    res = Result(cues=cues)
    refused = check_script(script)
    if refused:
        res.problems += [f"refused before running: {p}" for p in refused]
        return res
    with tempfile.TemporaryDirectory() as scratch:
        runs = [Path(scratch) / "run1", Path(scratch) / "run2"]
        for out in runs:
            record = run(script, cues, out, python)
            if record.get("error"):
                res.problems.append(record["error"])
                return res
        res.problems += record.get("problems", [])
        first, second = ({p.name: p.read_bytes() for p in out.glob("*.wav")} for out in runs)
        res.problems += [f"{n[:-4]} differs between two runs: seed every random generator"
                         for n in sorted(first) if second.get(n) != first[n]]
        checked = check_folder(runs[0], spec, cues)
        res.sounds = checked.sounds
        res.problems += checked.problems
        if keep is not None:
            keep = Path(keep)
            keep.mkdir(parents=True, exist_ok=True)
            for p in keep.glob("*.wav"):
                p.unlink()
            for p in runs[0].glob("*.wav"):
                shutil.copyfile(p, keep / p.name)
    return res


def text(res: Result) -> str:
    """The verdict and every number, as handed back to the author."""
    lines = []
    for name, r in res.sounds.items():
        st = r["stats"]
        nums = (f"{st['seconds']:.3f} s, peak {st['peak']:.2f}, RMS {st['rms_db']:.1f} dB, DC {st['dc']:+.4f}"
                if st else "unreadable")
        lines.append(f"{name}: {nums}  {'FAIL' if r['problems'] else 'ok'}")
        lines += [f"  ! {p}" for p in r["problems"]]
        for m in r.get("cues", []):
            if not m.get("missing"):
                lines.append(f"  {m['cue']} at {m['at']:.3f} s: a sound starts {m['off_ms']:+.0f} ms from it "
                             f"(strength {m['strength']:.2f}, at least {ONSET_MIN:.2f} wanted)")
        # strong onsets on no cue are in the record and not said: told of them, a builder took them for faults and
        # made the leap's growl "smooth and noise-free so it never registers its own onset"
    lines += [f"! {p}" for p in res.problems]
    lines.append("IN CONTRACT: every sound passes." if res.ok else "OUT OF CONTRACT.")
    return "\n".join(lines)


def to_json(res: Result) -> dict:
    return {"ok": res.ok, "problems": res.problems, "sounds": res.sounds, "cues": res.cues}

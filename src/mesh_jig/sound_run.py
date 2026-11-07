"""Run one sound script: `python -I -B sound_run.py SCRIPT CUES_JSON OUT_DIR`. Writes OUT_DIR/<name>.wav for each sound
the script's `sounds(cues)` returns, and OUT_DIR/run.json with what it returned and what went wrong.

Standard library and numpy only, and no import from the package: it runs in a child process with `-I`, so the
script's directory is not on the path and the package is not either. `sound.check_script` has refused the script
before this runs it; a script that loops forever is stopped by the parent's timeout.
"""
import importlib.util
import json
import sys
import traceback
import wave

import numpy as np

RATE = 44100


def write_wav(path, samples):
    data = np.round(np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(data.tobytes())


def main(script, cues_path, out):
    with open(cues_path, encoding="utf-8") as f:
        cues = json.load(f)
    record = {"written": {}, "problems": []}
    try:
        spec = importlib.util.spec_from_file_location("sound_script", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        made = module.sounds(cues)
    except Exception as e:
        frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == script]
        where = "".join(f"  line {f.lineno}, in {f.name}: {f.line}\n" for f in frames)
        record["error"] = f"{type(e).__name__}: {e}\n{where}".rstrip()
        made = None
    if made is not None and not isinstance(made, dict):
        record["error"] = f"sounds(cues) returned a {type(made).__name__}, not a dict of name -> samples"
        made = None
    for name, samples in (made or {}).items():
        if name not in cues:
            record["problems"].append(f"{name!r} is not a sound of the contract (they are {', '.join(cues)})")
            continue
        try:
            x = np.asarray(samples, dtype=np.float64)
        except (TypeError, ValueError) as e:
            record["problems"].append(f"{name}: not an array of numbers ({e})")
            continue
        if x.ndim != 1 or not len(x):
            record["problems"].append(f"{name}: samples must be a 1-D array, got shape {x.shape}")
            continue
        bad = int((~np.isfinite(x)).sum())
        if bad:
            record["problems"].append(f"{name}: {bad} samples are NaN or infinite")
            continue
        over = int((np.abs(x) > 1.0).sum())
        if over:
            record["problems"].append(f"{name}: {over} samples are beyond -1..1 and were clipped")
        write_wav(f"{out}/{name}.wav", x)
        record["written"][name] = len(x)
    with open(f"{out}/run.json", "w", encoding="utf-8") as f:
        json.dump(record, f)


if __name__ == "__main__":
    main(*sys.argv[1:4])

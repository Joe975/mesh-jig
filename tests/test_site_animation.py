"""Fast terrain, playback, reference-pose and shared-link regressions (requires Node.js)."""
from pathlib import Path
import subprocess


def test_browser_independent_animation_and_terrain():
    root=Path(__file__).resolve().parents[1]
    subprocess.run(['node','--test',str(root/'tests/site_animation.test.mjs')],check=True,capture_output=True,text=True)

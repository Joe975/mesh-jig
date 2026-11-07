"""The reference page names only files the Pages workflow deploys, and every viewer faces the model's front."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "site" / "index.html"
PUBLIC = set((ROOT / "PUBLIC_FILES.txt").read_text(encoding="utf-8").split())


def test_every_file_the_page_names_is_published_and_deployed():
    text = PAGE.read_text(encoding="utf-8")
    names = re.findall(r'(?:src|href)="([^":#]+)"', text)
    assert names
    for name in names:
        repo_path = name if name.startswith("showcase/reference-") else "site/" + name
        assert repo_path in PUBLIC, f"the page names {name}, which is not published"
        assert (ROOT / repo_path).is_file(), name


def test_every_viewer_starts_on_the_front_and_loads_lazily():
    """mesh-jig models face Blender +Y, which is glTF -Z; model-viewer's default camera looks at +Z, the back."""
    viewers = re.findall(r"<model-viewer[^>]*>", PAGE.read_text(encoding="utf-8"))
    assert viewers
    assert all('camera-orbit="180deg' in v and 'loading="lazy"' in v for v in viewers)


def test_the_page_stays_light():
    shown = re.findall(r'src="(showcase/[^"]+)"', PAGE.read_text(encoding="utf-8"))
    assert sum((ROOT / s).stat().st_size for s in shown) < 30_000_000

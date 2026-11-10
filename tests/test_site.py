"""The reference page names only files the Pages workflow deploys, and every viewer faces the model's front."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "site" / "index.html"
PUBLIC = set((ROOT / "PUBLIC_FILES.txt").read_text(encoding="utf-8").split())


def test_gif_export_assets_are_bundled_and_each_viewer_has_its_own_controls():
    page = PAGE.read_text(encoding='utf-8')
    assert page.count('class="gif-export"') == page.count('<model-viewer')
    assert page.count('class="gif-cancel"') == page.count('<model-viewer')
    for name in ('gif-export.js', 'gif-core.js', 'gif-worker.js', 'vendor/gifenc.js', 'vendor/gifenc.LICENSE'):
        assert 'site/' + name in PUBLIC
        assert (ROOT / 'site' / name).is_file()
    exporter = (ROOT / 'site/gif-export.js').read_text()
    assert 'idealAspect: false' in exporter
    assert 'visibilitychange' in exporter
    assert "{type: 'module'}" in exporter


def test_every_file_the_page_names_is_published_and_deployed():
    text = PAGE.read_text(encoding="utf-8")
    names = re.findall(r'(?:src|href)="([^":#]+)"', text)
    assert names
    deployed = ("showcase/reference-", "showcase/explore-")      # what .github/workflows/pages.yml copies
    for name in names:
        repo_path = name if name.startswith("showcase/") else "site/" + name
        assert not name.startswith("showcase/") or name.startswith(deployed), f"{name} is not deployed"
        assert repo_path in PUBLIC, f"the page names {name}, which is not published"
        assert (ROOT / repo_path).is_file(), name


def test_every_viewer_starts_on_the_front_loads_lazily_and_takes_touch_drags():
    """mesh-jig models face Blender +Y, which is glTF -Z; model-viewer's default camera looks at +Z, the back.
    With touch-action pan-y a vertical swipe scrolled the page instead of turning the model."""
    viewers = re.findall(r"<model-viewer[^>]*>", PAGE.read_text(encoding="utf-8"))
    assert viewers
    assert all('camera-orbit="180deg' in v and 'loading="lazy"' in v and 'touch-action="none"' in v for v in viewers)


def test_the_page_stays_light():
    """Every viewer and picture loads lazily, as it scrolls into view; the cap bounds a reader who sees it all."""
    shown = re.findall(r'src="(showcase/[^"]+)"', PAGE.read_text(encoding="utf-8"))
    assert sum((ROOT / s).stat().st_size for s in shown) < 50_000_000


def test_every_example_sheet_is_published():
    """An example's jig.json names its sheet; the exporter carries no images, so it must be a published showcase file."""
    import json
    for jig in sorted((ROOT / "examples").glob("*/jig.json")):
        source = json.loads(jig.read_text(encoding="utf-8")).get("sheet", {}).get("source")
        if source:
            sheet = (jig.parent / source).resolve()
            assert sheet.is_file(), (jig, source)
            assert sheet.relative_to(ROOT.resolve()).as_posix() in PUBLIC, (jig, source)


def test_every_model_the_explorer_can_load_is_published():
    import html
    import json
    blocks = re.findall(r'data-explore="([^"]+)"', PAGE.read_text(encoding="utf-8"))
    assert blocks
    for block in blocks:
        for pick in json.loads(html.unescape(block)):
            assert pick["src"].startswith("showcase/explore-") and pick["src"] in PUBLIC, pick["src"]
            assert (ROOT / pick["src"]).is_file(), pick["src"]

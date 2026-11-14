"""The reference page names only files the Pages workflow deploys, and every viewer faces the model's front."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "site" / "index.html"
PUBLIC = set((ROOT / "PUBLIC_FILES.txt").read_text(encoding="utf-8").split())


def test_every_reference_sheet_leads_directly_to_its_explorer():
    sections = re.findall(r'<section id="([^"]+)">(.*?)</section>', PAGE.read_text(encoding='utf-8'), re.S)
    assert sections
    for name, body in sections:
        assert re.match(r'\s*<h2>[^<]+</h2>\s*<img class="sheet"', body), f'{name}: content before the sheet'
        gap = re.search(r'<img class="sheet"[^>]*>(.*?)<div class="explore"', body, re.S)
        assert gap and not gap.group(1).strip(), f'{name}: a block was inserted between the sheet and viewer'
        context = re.search(r'<details class="section-context">.*?</details>', body, re.S)
        if context:
            assert body.index(context.group(0)) > body.index('</figure></div></div>'), name
        outside_context = body.replace(context.group(0), '') if context else body
        assert '<table' not in outside_context and 'class="cohort"' not in outside_context, name
    intro = re.search(r'<p class="lede">(.*?)</p>', PAGE.read_text(encoding='utf-8'), re.S).group(1)
    assert len(re.sub(r'<[^>]+>', '', intro).split()) <= 35


def test_public_agent_entry_points_share_site_publishing_rules():
    target = 'skill/mesh-jig-publish-results/SKILL.md#reference-site-publishing'
    for filename in ('AGENTS.md', 'CLAUDE.md'):
        assert target in (ROOT / filename).read_text(encoding='utf-8')
    assert '## Reference-site publishing' in (ROOT / 'skill/mesh-jig-publish-results/SKILL.md').read_text(encoding='utf-8')


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


def _shown_bytes(page, root):
    shown = re.findall(r'src="(showcase/[^"]+)"', page)
    return sum((root / s).stat().st_size for s in shown)


def test_the_page_stays_light():
    """Lazy gallery growth is bounded without giving older sections extra room."""
    page = PAGE.read_text(encoding="utf-8")
    octopus = re.search(r'<section id="octopus">.*?</section>', page, re.S)
    added = octopus.group(0) if octopus else ''
    assert _shown_bytes(page.replace(added, ''), ROOT) < 50_000_000
    assert _shown_bytes(added, ROOT) < 15_000_000
    assert _shown_bytes(page, ROOT) < 65_000_000


def test_gallery_byte_accounting_counts_only_shown_assets(tmp_path):
    (tmp_path / 'showcase').mkdir()
    (tmp_path / 'showcase' / 'shown.png').write_bytes(b'12345')
    (tmp_path / 'showcase' / 'download.glb').write_bytes(b'123456789')
    page = '<img src="showcase/shown.png"><a href="showcase/download.glb">Download</a>'
    assert _shown_bytes(page, tmp_path) == 5


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

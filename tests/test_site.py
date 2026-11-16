"""The reference page names only files the Pages workflow deploys, and every viewer faces the model's front."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "site" / "index.html"
PUBLIC = set((ROOT / "PUBLIC_FILES.txt").read_text(encoding="utf-8").split())


def test_qwen4_horror_retains_failed_game_status_and_unpriced_cost_scope():
    import html
    import json
    section = PAGE.read_text(encoding='utf-8').split('<section id="horror">', 1)[1].split('</section>', 1)[0]
    picks = json.loads(html.unescape(re.search(r'data-explore="([^"]+)"', section).group(1)))
    pick = next(p for p in picks if p['model'] == 'Qwen4 Max high, OpenRouter')
    assert pick['provider'] == 'Alibaba (Qwen)' and pick['clipLabels'] == ['sprint', 'leap']
    assert pick['score'] == .3938 and pick['best'] == .4207 and pick['builds'] == 5
    assert 'terrain' not in pick and 'No accepted game-ready package' in pick['warning']
    assert 'All three game proposals failed' in pick['warning']
    assert pick['facts']['stages']['rig']['passed']
    assert pick['facts']['stages']['rig']['passing_candidates'] == 3
    efficiency = pick['facts']['stages']['efficiency']
    assert efficiency['total_usd'] == 1.570083 and 'unpriced' in efficiency['kind']


def test_swordsman_preserves_selected_clips_and_unknown_static_score():
    import html
    import json
    import struct
    page = PAGE.read_text(encoding='utf-8')
    section = page.split('<section id="anime-swordsman">', 1)[1].split('</section>', 1)[0]
    picks = json.loads(html.unescape(re.search(r'data-explore="([^"]+)"', section).group(1)))
    assert len(picks) == 2
    assert picks[0]['score'] is None and picks[0]['best'] is None
    assert picks[0]['facts']['triangles'] == 23592
    assert picks[0]['facts']['stages']['rig']['passed'] is True
    assert picks[0]['clipLabels'] == ['guard', 'overhead_chop', 'horizontal_slash']
    assert all(picks[0]['facts']['stages']['clips'][name]['passed'] for name in picks[0]['clipLabels'])
    assert picks[1]['clipLabels'] == ['guard_talk', 'slash_yell_gaze']
    for pick, count in zip(picks, [3, 2]):
        raw = (ROOT / pick['src']).read_bytes()
        size = struct.unpack_from('<I', raw, 12)[0]
        assert len(json.loads(raw[20:20+size])['animations']) == count
    assert 'Full character and facial clips' in section
    assert 'compact reach' in section and 'subscription' in section


def test_terrain_characters_are_lazy_anonymous_and_bound_to_their_public_models():
    import hashlib
    import json
    page=PAGE.read_text(encoding='utf-8')
    assert 'All animations' in page and 'main.play(defaultAnimation(' in page
    assert "import('./terrain/viewer.js')" in page
    assert '<script type="module" src="terrain/viewer.js"' not in page
    registry=json.loads((ROOT/'site/terrain/characters.json').read_text(encoding='utf-8'))
    assert {'horror-v1','horse-astra-v1','horse-fable-v1'}<=set(registry['characters'])
    for row in registry['characters'].values():
        raw=(ROOT/'site/terrain'/row['data']).read_bytes();data=json.loads(raw)
        assert data['version']==2
        assert hashlib.sha256(raw).hexdigest()==row['sha256']
        assert hashlib.sha256((ROOT/row['model']).read_bytes()).hexdigest()==data['glb_sha256']==row['glb_sha256']
        assert len(data['runtime']['limbs']) in (4,6)
        assert all(j['name']=='j'+str(j['node']) for j in data['runtime']['skeleton']['joints'])
        assert not any(s in json.dumps(data) for s in ('local-work','C:\\','contract_sha256','provenance'))
    assert all('site/'+p.relative_to(ROOT/'site').as_posix() in PUBLIC for p in (ROOT/'site/terrain').rglob('*') if p.is_file())


def test_stage_scorecards_and_unranked_overall_coverage_preserve_missing_data():
    import html
    import json
    page = PAGE.read_text(encoding='utf-8')
    assert page.index('</nav>') < page.index('class="coverage-board"') < page.index('<main')
    assert 'Coverage only' in page and 'unranked' in page
    assert 'missing stages are not zero' in page
    assert page.count('class="stage-scorecard"') == page.count('<model-viewer')
    picks = [p for data in re.findall(r'data-explore="([^"]+)"',page) for p in json.loads(html.unescape(data))]
    kimi = next(p for p in picks if p['model']=='Kimi K3.1 Pro High, OpenRouter')
    stages = kimi['facts']['stages']
    assert stages['model']['metrics']['numeric_score'] == .4568
    assert stages['model']['triangle_limit'] == 32000
    assert stages['rig']['passed'] is False and stages['rig']['passing_candidates'] == 0
    assert stages['clips']['sprint']['passed'] is True and stages['clips']['leap']['passed'] is False
    assert stages['efficiency']['total_usd'] == 3.5591478
    assert 'incomplete list-price estimate' in page and 'interrupted startup calls unpriced' in page
    assert 'Not measured' in page and 'Unavailable' in page


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
    cyborg = re.search(r'<section id="cyborg">.*?</section>', page, re.S)
    cyborg_section = cyborg.group(0) if cyborg else ''
    assert _shown_bytes(page.replace(added, '').replace(cyborg_section, ''), ROOT) < 50_000_000
    assert _shown_bytes(added, ROOT) < 15_000_000
    assert _shown_bytes(cyborg_section, ROOT) < 6_000_000
    assert _shown_bytes(page, ROOT) < 71_000_000


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

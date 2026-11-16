"""Synthetic publication history: ordering, bounds, content links and safe HTML."""
import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'tools/build_publish_history.py'
if not SCRIPT.exists():
    SCRIPT = ROOT / 'publication/build_publish_history.py'
spec = importlib.util.spec_from_file_location('publish_history', SCRIPT)
history = importlib.util.module_from_spec(spec)
spec.loader.exec_module(history)


def gallery(labels):
    import html
    picks = html.escape(json.dumps([{'model': label, 'src': 'model.glb'} for label in labels]), quote=True)
    return f'<section id="cat"><h2>Pirate cat</h2><div data-explore="{picks}"></div></section>'


def test_history_reads_only_gallery_publishes_in_reverse_order(tmp_path):
    def git(*args):
        return history.git(tmp_path, *args)
    git('init', '-b', 'main')
    git('config', 'user.name', 'Test')
    git('config', 'user.email', 'test@example.invalid')
    (tmp_path / 'site').mkdir()
    page = tmp_path / 'site/index.html'
    for i in range(22):
        page.write_text(gallery(['Model ' + str(n) for n in range(i + 1)]), encoding='utf-8')
        git('add', 'site/index.html')
        git('commit', '-m', f'Publish {i:02d} <model> & clips')
    (tmp_path / 'README.md').write_text('Documentation only', encoding='utf-8')
    git('add', 'README.md')
    git('commit', '-m', 'Not a gallery publish')
    entries = history.collect(tmp_path)
    assert len(entries) == 20
    assert [e['title'] for e in entries] == [f'Pirate cat: Model {i}' for i in range(21, 1, -1)]
    assert entries[0]['url'] == '?subject=cat&model=model-21#cat'
    soup = ET.fromstring('<root>' + history.render(entries) + '</root>')
    assert len(soup.findall('.//ol[@class="publish-recent"]/li')) == 3
    assert len(soup.findall('.//div[@class="publish-scroll"]/ol/li')) == 20
    assert not 'open' in soup.find('.//details').attrib
    assert soup.find('.//div[@class="publish-scroll"]').get('tabindex') == '0'
    assert not soup.findall('.//model')
    assert soup.find('.//a').text == entries[0]['title']
    assert soup.find('.//a').get('href') == entries[0]['url']
    assert len(soup.findall('.//a')) == 23


@pytest.mark.parametrize('count', [0, 1, 2, 3, 25])
def test_short_history_and_replacement_are_bounded_and_idempotent(tmp_path, count):
    entries = [{'sha': 'a' * 40, 'title': '<script>bad</script>',
                'timestamp': '2026-11-17T12:00:00+13:00',
                'url': '?subject=cat&model=model-a#cat'}] * count
    page = tmp_path / 'index.html'
    page.write_text('before' + history.START + 'old' + history.END + 'after', encoding='utf-8')
    history.update(page, entries)
    first = page.read_text(encoding='utf-8')
    history.update(page, entries)
    assert first == page.read_text(encoding='utf-8')
    assert first.startswith('before') and first.endswith('after')
    soup = ET.fromstring('<root>' + first + '</root>')
    assert len(soup.findall('.//ol[@class="publish-recent"]/li')) == min(3, count)
    assert len(soup.findall('.//div[@class="publish-scroll"]/ol/li')) == min(20, count)
    assert not soup.findall('.//script')


def test_selection_labels_include_mesh_effort_and_recorded_harness():
    current = history.Gallery(gallery(['Fable 5.2 high, Claude Code, 10 Nov']))
    model = {'model': 'Fable 5.2 high, Claude Code, 10 Nov'}
    assert history.selection_title('cat', model['model'], model, current) == 'Pirate cat: Fable 5.2 High, Claude Code'
    assert history.selection_title('cat', 'Example, API agent', {'effort': 'high'}, current) == 'Pirate cat: Example High, API agent'
    assert history.content_links(history.Gallery(''), current, history.Gallery('')) == []


def test_missing_marker_fails_without_changing_page(tmp_path):
    page = tmp_path / 'index.html'
    page.write_text('unchanged', encoding='utf-8')
    with pytest.raises(ValueError, match='marker pair'):
        history.update(page, [])
    assert page.read_text() == 'unchanged'

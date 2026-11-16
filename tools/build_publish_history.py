"""Refresh the gallery changelog from public gallery commits, without network calls."""
import argparse
import html
import json
import re
import subprocess
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode

START = '<!-- publish-history:start -->'
END = '<!-- publish-history:end -->'


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], encoding='utf-8')


class Gallery(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.subject = None
        self.models = {}
        self.titles = {}
        self.heading = False
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'section':
            self.subject = attrs.get('id')
        if tag == 'h2' and self.subject:
            self.heading = True
        if self.subject and 'data-explore' in attrs:
            for model in json.loads(attrs['data-explore']):
                self.models[(self.subject, model['model'])] = model

    def handle_data(self, data):
        if self.heading:
            self.titles[self.subject] = self.titles.get(self.subject, '') + data

    def handle_endtag(self, tag):
        if tag == 'h2':
            self.heading = False
        if tag == 'section':
            self.subject = None


def selection_title(subject, label, model, current):
    label = re.sub(r', \d{1,2} [A-Z][a-z]{2}(?: \d{4})?(?=,|$)', '', label)
    effort = model.get('effort')
    if effort and not re.search(r'\b' + re.escape(effort) + r'\b', label, re.I):
        name, comma, harness = label.partition(',')
        label = name + ' ' + effort + comma + harness
    label = re.sub(r'\b(low|medium|high|max)\b', lambda m: m[0].title(), label, flags=re.I)
    mesh = current.titles.get(subject, subject.replace('-', ' ').title())
    return mesh + ': ' + label


def content_links(before, after, current):
    """Link changed selections only when they still exist in today's gallery."""
    links = []
    for (subject, label), model in after.models.items():
        if before.models.get((subject, label)) == model:
            continue
        if (subject, label) not in current.models:
            continue
        slug = re.sub('[^a-z0-9]+', '-', label.lower()).strip('-')
        url = '?' + urlencode({'subject': subject, 'model': slug}) + '#' + subject
        links.append((selection_title(subject, label, model, current), url))
    return links


def collect(repo, limit=20):
    current = Gallery(git(repo, 'show', 'HEAD:site/index.html'))
    records = git(repo, 'log', '--first-parent',
                  '--format=%H%x09%cI%x09%s', '--', 'site/').splitlines()
    entries = []
    for record in records:
        sha, timestamp, title = record.split('\t', 2)
        after = Gallery(git(repo, 'show', f'{sha}:site/index.html'))
        parents = git(repo, 'rev-list', '--parents', '-n', '1', sha).split()[1:]
        before = Gallery('')
        if parents:
            try:
                before = Gallery(git(repo, 'show', f'{parents[0]}:site/index.html'))
            except subprocess.CalledProcessError:
                pass  # The first gallery publication has no earlier page.
        for label, url in content_links(before, after, current):
            entries.append({'sha': sha, 'timestamp': timestamp, 'title': label, 'url': url})
            if len(entries) == limit:
                return entries
    return entries


def render(entries):
    def rows(items):
        result = []
        for item in items:
            date = datetime.fromisoformat(item['timestamp']).strftime('%d %b %Y')
            result.append('<li><time datetime="' + html.escape(item['timestamp'], quote=True) + '">' + date
                          + '</time><a class="publish-title" href="' + html.escape(item['url'], quote=True)
                          + '">' + html.escape(item['title']) + '</a></li>')
        return ''.join(result)
    entries = entries[:20]
    return (START + '\n<aside class="publish-history" id="changelog" aria-labelledby="changelog-title">'
            '<h2 id="changelog-title">Changelog</h2>'
            '<ol class="publish-recent" aria-label="Latest three publishes">' + rows(entries[:3]) + '</ol>'
            '<details class="publish-archive"><summary>Last 20 publishes</summary>'
            '<div class="publish-scroll" tabindex="0" role="region" aria-label="Last 20 publishes">'
            '<ol>' + rows(entries) + '</ol></div></details></aside>\n' + END)


def update(page, entries):
    text = page.read_text(encoding='utf-8')
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError('page must have exactly one publish-history marker pair')
    start, end = text.index(START), text.index(END) + len(END)
    page.write_text(text[:start] + render(entries) + text[end:], encoding='utf-8', newline='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--page', type=Path, required=True)
    args = parser.parse_args()
    update(args.page, collect(args.repo))

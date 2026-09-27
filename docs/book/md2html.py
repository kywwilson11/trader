#!/usr/bin/env python3
"""Minimal Markdown -> HTML converter for the Trader Book (standard library only).

Supports what the book uses: ATX headings, paragraphs, ordered and unordered lists
(with indented continuation lines and one level of nesting), fenced and indented code
blocks, pipe tables, block quotes, horizontal rules, links, images, inline code,
**bold** and *italic*. It is not a general Markdown implementation.

Usage:
    python3 md2html.py --title "The Trader Book" -o book.html 01_x.md 02_y.md ...

Each input file becomes one chapter: its first level-1 heading starts a new page, and a
table of contents (level-1 and level-2 headings of every chapter) is generated at the
front. Relative links (to repository files) are rendered as plain text followed by the
target path in small type, because they cannot resolve inside a PDF; absolute http(s)
links stay clickable; links to headings inside the book stay internal.

The output is deterministic: the same inputs always give byte-identical HTML.
"""
import argparse
import html
import os
import re
import sys

_slug_counts = {}


def slugify(text):
    base = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-') or 'section'
    n = _slug_counts.get(base, 0)
    _slug_counts[base] = n + 1
    return base if n == 0 else f'{base}-{n}'


_CODE_SPAN = re.compile(r'`([^`]+)`')
_IMAGE = re.compile(r'!\[([^\]]*)\]\(([^)\s]+)\)')
_LINK = re.compile(r'\[([^\]]+)\]\(([^)\s]+)\)')
_BOLD = re.compile(r'\*\*(.+?)\*\*')
_ITALIC = re.compile(r'(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])')
_URL = re.compile(r'(?<![="\w])(https?://[^\s<)]+[^\s<).,;:])')


def inline(text, base_dir):
    """Render inline Markdown. Code spans are protected from further processing."""
    stash = []

    def keep(fragment):
        stash.append(fragment)
        return f'\x00{len(stash) - 1}\x00'

    text = _CODE_SPAN.sub(lambda m: keep(f'<code>{html.escape(m.group(1))}</code>'), text)

    def image(m):
        alt, src = m.group(1), m.group(2)
        if not re.match(r'^[a-z]+://', src):
            src = 'file://' + os.path.abspath(os.path.join(base_dir, src))
        return keep(f'<img src="{html.escape(src)}" alt="{html.escape(alt)}"/>')

    text = _IMAGE.sub(image, text)

    def link(m):
        label, target = m.group(1), m.group(2)
        label_html = inline_basic(html.escape(label))
        if re.match(r'^https?://', target):
            return keep(f'<a href="{html.escape(target)}">{label_html}</a>')
        if target.startswith('#'):
            return keep(f'<a href="{html.escape(target)}">{label_html}</a>')
        # Relative repository link: resolve to a repo-relative path for display.
        path = target.split('#', 1)[0]
        repo_root = os.path.abspath(os.path.join(base_dir, '..', '..'))
        shown = os.path.relpath(os.path.abspath(os.path.join(base_dir, path)), repo_root)
        if target != path:
            shown += '#' + target.split('#', 1)[1]
        plain = re.sub(r'\x00(\d+)\x00', lambda k: stash[int(k.group(1))], label_html)
        plain = html.unescape(re.sub(r'<[^>]+>', '', plain)).strip()
        if plain in (shown, os.path.basename(shown)):
            return keep(f'<span class="ref">{label_html}</span>')
        return keep(f'<span class="ref">{label_html}</span>'
                    f' <span class="path">[{html.escape(shown)}]</span>')

    text = _LINK.sub(link, text)
    text = html.escape(text, quote=False)
    text = _URL.sub(lambda m: f'<a href="{m.group(1)}">{m.group(1)}</a>', text)
    text = inline_basic(text)
    while '\x00' in text:
        text = re.sub(r'\x00(\d+)\x00', lambda m: stash[int(m.group(1))], text)
    return text


def inline_basic(text):
    text = _BOLD.sub(r'<strong>\1</strong>', text)
    text = _ITALIC.sub(r'<em>\1</em>', text)
    return text


def is_table_sep(line):
    return bool(re.match(r'^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$', line))


def split_row(line):
    line = line.strip()
    if line.startswith('|'):
        line = line[1:]
    if line.endswith('|'):
        line = line[:-1]
    cells, cur, in_code = [], '', False
    for ch in line:
        if ch == '`':
            in_code = not in_code
        if ch == '|' and not in_code:
            cells.append(cur.strip())
            cur = ''
        else:
            cur += ch
    cells.append(cur.strip())
    return cells


_LIST_ITEM = re.compile(r'^(\s*)([-*+]|\d+[.)])\s+(.*)$')


class Converter:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.toc = []          # (level, text, anchor)

    def convert(self, md):
        lines = md.replace('\t', '    ').split('\n')
        out = []
        i = 0
        n = len(lines)
        while i < n:
            line = lines[i]
            stripped = line.strip()
            if not stripped:
                i += 1
                continue
            if stripped.startswith('<!--'):
                while i < n and '-->' not in lines[i]:
                    i += 1
                i += 1
                continue
            if stripped.startswith('```'):
                i += 1
                buf = []
                while i < n and not lines[i].strip().startswith('```'):
                    buf.append(lines[i])
                    i += 1
                i += 1
                out.append('<pre>' + html.escape('\n'.join(buf)) + '</pre>')
                continue
            if line.startswith('    '):
                buf = []
                while i < n and (lines[i].startswith('    ') or not lines[i].strip()):
                    buf.append(lines[i][4:])
                    i += 1
                while buf and not buf[-1].strip():
                    buf.pop()
                out.append('<pre>' + html.escape('\n'.join(buf)) + '</pre>')
                continue
            m = re.match(r'^(#{1,6})\s+(.*?)\s*#*\s*$', line)
            if m:
                level = len(m.group(1))
                text = m.group(2)
                anchor = slugify(text)
                if level <= 2:
                    self.toc.append((level, text, anchor))
                cls = ' class="chapter"' if level == 1 else ''
                out.append(f'<h{level} id="{anchor}"{cls}><a name="{anchor}"></a>'
                           f'{inline(text, self.base_dir)}</h{level}>')
                i += 1
                continue
            if re.match(r'^\s*([-*_])(\s*\1){2,}\s*$', line):
                out.append('<hr/>')
                i += 1
                continue
            if '|' in line and i + 1 < n and is_table_sep(lines[i + 1]):
                header = split_row(line)
                i += 2
                rows = []
                while i < n and '|' in lines[i] and lines[i].strip():
                    rows.append(split_row(lines[i]))
                    i += 1
                aligns = []
                for c in split_row(lines[i - len(rows) - 1]):
                    c = c.strip()
                    aligns.append('right' if c.endswith(':') and not c.startswith(':')
                                  else 'center' if c.startswith(':') and c.endswith(':')
                                  else 'left')
                t = ['<table border="1" cellspacing="0" cellpadding="4">', '<thead><tr>']
                for j, h in enumerate(header):
                    a = aligns[j] if j < len(aligns) else 'left'
                    t.append(f'<th align="{a}">{inline(h, self.base_dir)}</th>')
                t.append('</tr></thead><tbody>')
                for r in rows:
                    t.append('<tr>')
                    for j, c in enumerate(r):
                        a = aligns[j] if j < len(aligns) else 'left'
                        t.append(f'<td align="{a}">{inline(c, self.base_dir)}</td>')
                    t.append('</tr>')
                t.append('</tbody></table>')
                out.append(''.join(t))
                continue
            if stripped.startswith('>'):
                buf = []
                while i < n and lines[i].strip().startswith('>'):
                    buf.append(re.sub(r'^\s*>\s?', '', lines[i]))
                    i += 1
                out.append('<blockquote>' + Converter(self.base_dir).convert('\n'.join(buf))
                           + '</blockquote>')
                continue
            if _LIST_ITEM.match(line):
                html_list, i = self._list(lines, i)
                out.append(html_list)
                continue
            buf = [stripped]
            i += 1
            while i < n and lines[i].strip() and not self._block_start(lines, i):
                buf.append(lines[i].strip())
                i += 1
            out.append('<p>' + inline(' '.join(buf), self.base_dir) + '</p>')
        return '\n'.join(out)

    def _block_start(self, lines, i):
        line = lines[i]
        s = line.strip()
        return (s.startswith('#') or s.startswith('```') or s.startswith('>')
                or bool(_LIST_ITEM.match(line))
                or ('|' in line and i + 1 < len(lines) and is_table_sep(lines[i + 1])))

    def _list(self, lines, i):
        m = _LIST_ITEM.match(lines[i])
        indent = len(m.group(1))
        ordered = m.group(2)[0].isdigit()
        tag = 'ol' if ordered else 'ul'
        items = []
        n = len(lines)
        while i < n:
            m = _LIST_ITEM.match(lines[i])
            if not m or len(m.group(1)) != indent or m.group(2)[0].isdigit() != ordered:
                break
            text = [m.group(3).strip()]
            sub = ''
            i += 1
            while i < n:
                nxt = lines[i]
                if not nxt.strip():
                    # a blank line ends the item unless the list continues after it
                    j = i + 1
                    while j < n and not lines[j].strip():
                        j += 1
                    if j < n and (_LIST_ITEM.match(lines[j])
                                  and len(_LIST_ITEM.match(lines[j]).group(1)) >= indent):
                        i = j
                        continue
                    break
                mm = _LIST_ITEM.match(nxt)
                if mm and len(mm.group(1)) > indent:
                    sub_html, i = self._list(lines, i)
                    sub += sub_html
                    continue
                if mm:
                    break
                if len(nxt) - len(nxt.lstrip()) > indent:
                    text.append(nxt.strip())
                    i += 1
                    continue
                break
            items.append('<li>' + inline(' '.join(text), self.base_dir) + sub + '</li>')
        return f'<{tag}>' + ''.join(items) + f'</{tag}>', i


CSS = """
body { font-family: 'Liberation Serif', 'DejaVu Serif', serif; font-size: 11.5pt; line-height: 1.35; }
h1 { font-family: 'Liberation Sans', 'DejaVu Sans', sans-serif; font-size: 22pt; margin-top: 0; }
h1.chapter { page-break-before: always; }
h2 { font-family: 'Liberation Sans', 'DejaVu Sans', sans-serif; font-size: 15pt; margin-top: 18pt; }
h3 { font-family: 'Liberation Sans', 'DejaVu Sans', sans-serif; font-size: 12.5pt; }
h4 { font-family: 'Liberation Sans', 'DejaVu Sans', sans-serif; font-size: 11.5pt; }
pre { font-family: 'Liberation Mono', 'DejaVu Sans Mono', monospace; font-size: 9pt;
      background: #f2f2f2; padding: 4pt; }
code { font-family: 'Liberation Mono', 'DejaVu Sans Mono', monospace; font-size: 9.5pt; }
table { border-collapse: collapse; font-size: 9.5pt; }
th { background: #e6e6e6; }
.path { font-size: 8pt; color: #666666; }
blockquote { margin-left: 18pt; color: #333333; }
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('--title', default='The Trader Book')
    ap.add_argument('--subtitle', default='')
    args = ap.parse_args(argv)

    bodies = []
    toc = []
    for path in args.inputs:
        with open(path, encoding='utf-8') as fh:
            md = fh.read()
        conv = Converter(os.path.dirname(os.path.abspath(path)))
        bodies.append(conv.convert(md))
        toc.extend(conv.toc)

    parts = ['<!DOCTYPE html>', '<html><head><meta charset="utf-8"/>',
             f'<title>{html.escape(args.title)}</title>', f'<style>{CSS}</style>',
             '</head><body>',
             f'<h1 id="title-page">{html.escape(args.title)}</h1>']
    if args.subtitle:
        parts.append(f'<p><em>{html.escape(args.subtitle)}</em></p>')
    parts.append('<h2 id="contents">Contents</h2><div class="toc">')
    for level, text, anchor in toc:
        label = re.sub(r'<[^>]+>', '', inline(text, '.'))
        # Inline styles: LibreOffice's HTML import ignores descendant CSS selectors.
        style = ('margin-top:6pt; margin-bottom:0; font-weight:bold' if level == 1 else
                 'margin-left:18pt; margin-top:0; margin-bottom:0; font-size:10pt')
        parts.append(f'<p style="{style}"><a href="#{anchor}">{label}</a></p>')
    parts.append('</div>')
    parts.extend(bodies)
    parts.append('</body></html>')
    with open(args.output, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(parts) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())

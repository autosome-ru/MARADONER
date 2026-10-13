import re
import os
from datetime import date
from json import load
from html import unescape
from typing import List

try:
    from htmlmin import minify
except ImportError:
    minify = None

content_folder = 'content'
site_url = 'https://maradoner.georgy.top'

with open('template.html', 'r') as f:
    template_main = f.read()
with open('template_menu.html', 'r') as f:
    template_menu = f.read()
with open('template_menubutton.html', 'r') as f:
    template_menubutton = f.read()

package_name = '<b>MARADONER</b>'


def create_menu(pages: List[dict], active: str) -> str:
    buttons = list()
    relpath_current = next(t for t in pages if t['name'] == active)['relpath']
    home = None
    for page in pages:
        index = page['pages'][0]
        path = index['file']
        name = page['name']
        mode = ' aria-current="page"' if active == name else str()
        if page['relpath']:
            path = os.path.join(page['folder'], path)
        if relpath_current:
            path = os.path.join('..', path)
        if home is None:
            home = path
        button = template_menubutton.format(name=name, path=path, mode=mode).strip()
        buttons.append(button)
    return template_menu.format(buttons='\n'.join(buttons), home=home, relpath='..' if relpath_current else '.')


def create_leftmenu(pages: dict, active: str) -> str:
    buttons = list()
    button_inactive = '<a href="{}">{}</a>'
    button_active = '<a class="active" href="{}" aria-current="page">{}</a>'
    for page in pages['pages']:
        file = page['file']
        title = page['menu']
        if not title:
            continue
        button = button_active if file == active else button_inactive
        buttons.append(button.format(file, title))
    if not buttons:
        return str()
    return '<nav class="pages" aria-label="{}">\n{}\n</nav>'.format(pages['name'], '\n'.join(buttons))


def create_rightmenu(body: str) -> str:
    buttons = list()
    ptr = r'<h2 id="(\S+?)">([\s\S]+?)<\/h2>'
    for anchor in re.finditer(ptr, body):
        buttons.append('<a href="#{}">{}</a>'.format(anchor.group(1), anchor.group(2).strip()))
    if len(buttons) < 2:
        return str()
    return '<details class="toc" open>\n<summary>On this page</summary>\n<nav>\n{}\n</nav>\n</details>'.format('\n'.join(buttons))


def label_cells(table: re.Match) -> str:
    table = table.group(0)
    names = [re.sub(r'<[^>]+>', '', t).strip() for t in re.findall(r'<th\b[^>]*>([\s\S]*?)</th>', table)]

    def label_row(row: re.Match) -> str:
        cells = iter(names)
        return re.sub(r'<td>', lambda _: '<td data-th="{}">'.format(next(cells, str())), row.group(0))
    return re.sub(r'<tr>[\s\S]*?</tr>', label_row, table)


def wrap_tables(body: str) -> str:
    body = re.sub(r'<table\b[\s\S]*?</table>', label_cells, body)
    body = re.sub(r'<table\b[^>]*>', '<div class="tbl"><table>', body)
    return body.replace('</table>', '</table></div>')


def keep_short_code(body: str) -> str:
    # A short command or option stays on one line, a long example may wrap.
    def mark(code: re.Match) -> str:
        return '<code class="nw">' + code.group(1) + '</code>' if len(unescape(code.group(1))) <= 32 else code.group(0)
    return re.sub(r'<code>([^<]*)</code>', mark, body)


with open(os.path.join(content_folder, 'pages.json'), 'r') as f:
    pages = load(f)
for page in pages:
    with open(os.path.join(content_folder, page['folder'], 'info.json'), 'r') as f:
        page['pages'] = load(f)

sitemap = list()
for page in pages:
    chapter_name = page['name']
    chapter_postfix = page['postfix']
    chapter_folder = page['folder']
    relpath = page['relpath']
    output_folder = chapter_folder if relpath else str()
    chapter_folder = os.path.join(content_folder, chapter_folder)
    menu = create_menu(pages, chapter_name)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    for subpage in page['pages']:
        filename = subpage['file']
        filepath = os.path.join(chapter_folder, filename)
        leftmenu = create_leftmenu(page, filename)
        try:
            with open(filepath, 'r') as f:
                body = f.read()
        except FileNotFoundError:
            with open(filepath, 'w') as f:
                f.write('TODO')
                body = 'TODO'
                print(filepath)
        rightmenu = create_rightmenu(body)
        body = keep_short_code(wrap_tables(body)).replace('{name}', package_name)
        if leftmenu and not subpage.get('hide_title', False):
            body = f'<h1>{subpage["title"]}</h1>\n{body}'
        rightmenu = rightmenu.replace('{name}', package_name)
        layout = (' has-pages' if leftmenu else str()) + (' has-toc' if rightmenu else str())
        keywords = ', '.join(subpage['keywords'])
        title = subpage['title']
        html = template_main.format(title=title, title_postfix=chapter_postfix, keywords=keywords,
                                    mainmenu=menu, leftmenu=leftmenu, rightmenu=rightmenu,
                                    layout=layout, body=body,
                                    relpath='.' * (relpath + 1))
        if minify is not None:
            html = minify(html)
        with open(os.path.join(output_folder, filename), 'w') as f:
            f.write(html)
        if body.strip() != 'TODO':
            url = '/'.join((site_url, output_folder, filename) if output_folder else (site_url, filename))
            if url == site_url + '/index.html':
                url = site_url + '/'
            lastmod = date.fromtimestamp(os.path.getmtime(filepath)).isoformat()
            sitemap.append(f'  <url><loc>{url}</loc><lastmod>{lastmod}</lastmod></url>')

with open('sitemap.xml', 'w') as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n')
    f.write('\n'.join(sitemap))
    f.write('\n</urlset>\n')

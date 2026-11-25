from html.parser import HTMLParser
from pathlib import Path
import re
import markdown
from markdown.extensions.toc import slugify_unicode

root = Path(__file__).resolve().parents[1]
source = (root / 'README.md').read_text(encoding='utf-8')
rendered = markdown.markdown(source.replace('<details>', '<details markdown="1">'), extensions=['tables', 'fenced_code', 'md_in_html', 'toc'], extension_configs={'toc': {'slugify': slugify_unicode}})

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()
        self.counts = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.counts[tag] = self.counts.get(tag, 0) + 1
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        for key in ('href', 'src'):
            if key in attrs:
                self.links.append(attrs[key])

links = Links()
links.feed(rendered)
for link in links.links:
    if link.startswith('#'):
        assert link[1:] in links.ids, link
    elif not re.match(r'https?://', link):
        assert (root / link).is_file(), link
assert links.counts['h1'] == 1
assert links.counts.get('details', 0) == links.counts.get('summary', 0)
assert source.count('<details>') == source.count('</details>')
assert source.count('```') % 2 == 0
assert '\ufffd' not in source
css = '''body{margin:0;background:#f6f8fa;color:#1f2328;font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif}article{max-width:920px;margin:32px auto;padding:40px;background:white;border:1px solid #d1d9e0;border-radius:6px}h1,h2{border-bottom:1px solid #d1d9e0;padding-bottom:.3em;line-height:1.3}h2{margin-top:28px}a{color:#0969da;text-decoration:none}img{max-width:100%}pre{background:#f6f8fa;padding:16px;overflow:auto;border-radius:6px;line-height:1.5}code{font-family:Consolas,monospace;font-size:14px}p code,li code,td code{background:#eff1f3;padding:2px 5px;border-radius:4px}table{border-collapse:collapse;width:100%;margin:16px 0}td,th{padding:7px 13px;border:1px solid #d1d9e0;text-align:left}tr:nth-child(even){background:#f6f8fa}blockquote{margin:16px 0;padding:0 16px;color:#59636e;border-left:4px solid #d1d9e0}summary{cursor:pointer}details{margin:16px 0}hr{border:0;height:3px;background:#d1d9e0;margin:24px 0}@media(prefers-color-scheme:dark){body{background:#010409;color:#f0f6fc}article{background:#0d1117;border-color:#3d444d}h1,h2,td,th{border-color:#3d444d}pre,tr:nth-child(even),p code,li code,td code{background:#151b23}a{color:#4493f8}blockquote{color:#9198a1;border-color:#3d444d}hr{background:#3d444d}}'''
preview = root / 'artifacts' / 'readme-preview.html'
preview.parent.mkdir(parents=True, exist_ok=True)
preview.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><base href="' + root.as_uri() + '/"><title>AutoLabelT README preview</title><style>' + css + '</style><article>' + rendered + '</article></html>', encoding='utf-8')
print(f"PASS: {len(links.links)} links checked; {links.counts.get('table', 0)} tables and {links.counts.get('details', 0)} collapsible sections rendered.")
print(f'Preview: {preview}')

import json
import pathlib
import re
import urllib.request

p = pathlib.Path("ml/data/final_results_28.json")
data = json.loads(p.read_text(encoding="utf-8"))
url = None
for it in data:
    if isinstance(it, dict):
        u = it.get("url")
        if isinstance(u, str) and u.startswith("https://letterboxd.com/film/"):
            url = u
            break

print("sample_url=", url)

req = urllib.request.Request(
    url,
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "text/html",
        "Accept-Encoding": "identity",
    },
    method="GET",
)

with urllib.request.urlopen(req, timeout=20) as resp:
    raw = resp.read(200_000)
    print("status=", getattr(resp, "status", None))
    print("content_type=", resp.headers.get("Content-Type"))
    print("content_encoding=", resp.headers.get("Content-Encoding"))
    print("raw_len=", len(raw))

html = raw.decode("utf-8", errors="ignore")

m = re.search(r'property="og:image"\s+content="([^"]+)"', html)
print("og_image=", m.group(1) if m else None)
print("head_snippet=", html[:300].replace("\n", " "))

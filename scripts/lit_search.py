"""Literature search helper: queries Semantic Scholar (with backoff) and OpenAlex,
stores raw JSON under literature/raw/ and appends a log to literature/literature_search_log.md."""
import json, sys, time, urllib.parse, urllib.request, datetime, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "literature" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
LOG = ROOT / "literature" / "literature_search_log.md"


def get(url, tries=6):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "radarconf-lit-review/0.1 (mailto:research@example.org)"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # 429 etc.
            time.sleep(4 * (k + 1))
    return None


def s2(q, n=20):
    url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(
        {"query": q, "limit": n, "fields": "title,year,venue,externalIds,citationCount,authors"})
    d = get(url) or {}
    return [(p.get("year"), p.get("citationCount"), p.get("title"), p.get("venue"),
             (p.get("externalIds") or {}).get("DOI")) for p in d.get("data", [])]


def oa(q, n=20):
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(
        {"search": q, "per-page": n, "sort": "relevance_score:desc"})
    d = get(url) or {}
    out = []
    for w in d.get("results", []):
        src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
        out.append((w.get("publication_year"), w.get("cited_by_count"), w.get("title"), src,
                    (w.get("doi") or "").replace("https://doi.org/", "")))
    return out


if __name__ == "__main__":
    tag = sys.argv[1]
    queries = sys.argv[2:]
    with open(LOG, "a", encoding="utf-8") as log:
        for q in queries:
            for name, fn in (("S2", s2), ("OpenAlex", oa)):
                res = fn(q)
                (RAW / f"{tag}_{name}_{abs(hash(q)) % 10**6}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
                log.write(f"\n### {datetime.date.today()} [{name}] `{q}` -> {len(res)} hits\n")
                print(f"\n## [{name}] {q} ({len(res)})")
                for y, c, t, v, doi in res:
                    line = f"{y} | {c} | {t} | {v} | {doi}"
                    print(line[:220])
                    log.write(f"- {line}\n")
                time.sleep(2)

"""Search arXiv (official API) to check whether an idea, or a recent result, already exists.

    python scripts/arxiv_search.py "quantum kernel" "group structure"         # all words, any date
    python scripts/arxiv_search.py "projected quantum kernel" --since 2024-01-01 --abstracts
    python scripts/arxiv_search.py --query 'abs:"quantum kernel" AND abs:fraud' --max 50

Each positional argument is a phrase that must appear in the title or abstract (phrases are combined
with AND). Results are sorted by relevance, or by date with --newest. The script prints the arXiv ID,
date, first authors and title, plus the abstract with --abstracts. Opening the paper is still
required before it is cited in FIELD_MAP.md or IDEAS.md.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ATOM = {"a": "http://www.w3.org/2005/Atom"}


def build_query(phrases: list[str], since: str | None, until: str | None) -> str:
    parts = [f'(ti:"{p}" OR abs:"{p}")' for p in phrases]
    if since or until:
        start = (since or "1990-01-01").replace("-", "") + "0000"
        end = (until or "2100-01-01").replace("-", "") + "2359"
        parts.append(f"submittedDate:[{start} TO {end}]")
    return " AND ".join(parts)


def search(query: str, max_results: int = 25, newest: bool = False) -> list[dict]:
    params = {"search_query": query, "start": 0, "max_results": max_results}
    if newest:
        params.update(sortBy="submittedDate", sortOrder="descending")
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
    out = []
    for e in root.findall("a:entry", ATOM):
        out.append({
            "id": e.find("a:id", ATOM).text.rsplit("/", 1)[-1],
            "date": e.find("a:published", ATOM).text[:10],
            "authors": [a.find("a:name", ATOM).text for a in e.findall("a:author", ATOM)],
            "title": " ".join(e.find("a:title", ATOM).text.split()),
            "abstract": " ".join(e.find("a:summary", ATOM).text.split()),
        })
    time.sleep(3)  # arXiv API etiquette: at most one request every 3 seconds
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phrases", nargs="*", help="phrases that must all appear in title or abstract")
    parser.add_argument("--query", help="raw arXiv API search_query (overrides phrases)")
    parser.add_argument("--since", help="YYYY-MM-DD, earliest submission date")
    parser.add_argument("--until", help="YYYY-MM-DD, latest submission date")
    parser.add_argument("--max", type=int, default=25)
    parser.add_argument("--newest", action="store_true", help="sort by submission date instead of relevance")
    parser.add_argument("--abstracts", action="store_true", help="print abstracts too")
    args = parser.parse_args(argv)
    if not args.query and not args.phrases:
        parser.error("give phrases or --query")

    query = args.query or build_query(args.phrases, args.since, args.until)
    results = search(query, args.max, args.newest)
    print(f"query: {query}\n{len(results)} results\n")
    for r in results:
        authors = ", ".join(r["authors"][:3]) + (" et al." if len(r["authors"]) > 3 else "")
        print(f"arXiv:{r['id']}  {r['date']}  {authors}\n    {r['title']}")
        if args.abstracts:
            print(f"    {r['abstract']}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

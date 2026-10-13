"""Re-check every arXiv ID cited in FIELD_MAP.md against the arXiv API.

    python scripts/check_sources.py            # compare with scripts/sources_snapshot.json
    python scripts/check_sources.py --update   # rewrite the snapshot (after adding sources)

Exits non-zero if an ID does not resolve or its title differs from the snapshot taken when the
source was opened. A title change is not always an error (authors retitle papers), but it must be
looked at by a human before the citation is trusted.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path(__file__).with_name("sources_snapshot.json")
ATOM = {"a": "http://www.w3.org/2005/Atom"}
ID_PATTERN = re.compile(r"arXiv:(\d{4}\.\d{4,5})")


def cited_ids(path: Path) -> list[str]:
    return sorted(set(ID_PATTERN.findall(path.read_text())))


def fetch(ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for start in range(0, len(ids), 40):
        chunk = ids[start : start + 40]
        url = "https://export.arxiv.org/api/query?max_results=100&id_list=" + ",".join(chunk)
        root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
        for entry in root.findall("a:entry", ATOM):
            arxiv_id = entry.find("a:id", ATOM).text.rsplit("/", 1)[-1].split("v")[0]
            out[arxiv_id] = {
                "title": " ".join(entry.find("a:title", ATOM).text.split()),
                "first_author": entry.find("a:author/a:name", ATOM).text,
            }
        time.sleep(3)  # arXiv API etiquette
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()

    ids = cited_ids(ROOT / "FIELD_MAP.md")
    found = fetch(ids)
    if args.update:
        SNAPSHOT.write_text(json.dumps(found, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
        print(f"Snapshot of {len(found)} sources written to {SNAPSHOT}")
        return 0 if len(found) == len(ids) else 1

    snapshot = json.loads(SNAPSHOT.read_text()) if SNAPSHOT.exists() else {}
    problems = 0
    for arxiv_id in ids:
        now = found.get(arxiv_id)
        then = snapshot.get(arxiv_id)
        if now is None:
            print(f"MISSING   arXiv:{arxiv_id} does not resolve")
            problems += 1
        elif then is None:
            print(f"NEW       arXiv:{arxiv_id} {now['title']!r} (not in snapshot; run --update after checking)")
            problems += 1
        elif then["title"] != now["title"]:
            print(f"RETITLED  arXiv:{arxiv_id}\n          was: {then['title']}\n          now: {now['title']}")
            problems += 1
        else:
            print(f"ok        arXiv:{arxiv_id} {now['first_author']}: {now['title']}")
    print(f"\n{len(ids)} cited arXiv IDs, {problems} to look at.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

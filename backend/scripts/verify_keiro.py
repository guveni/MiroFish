#!/usr/bin/env python3
"""
One-shot check: Keiro web search API returns results.

Usage (from repo root):
  cd backend && uv run python scripts/verify_keiro.py
  cd backend && uv run python scripts/verify_keiro.py --query "latest AI news"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_backend_dir = Path(__file__).resolve().parent.parent
_root_dir = _backend_dir.parent
_env_file = _root_dir / ".env"


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip().strip("'").strip('"')
        out[key] = val
    return out


def main() -> int:
    env = _parse_env_file(_env_file)

    ap = argparse.ArgumentParser(description="Verify Keiro web search for MiroFish")
    ap.add_argument(
        "--query",
        default="What happened in tech news this week?",
        help="Search query to test",
    )
    ap.add_argument(
        "--base-url",
        default=env.get("KEIRO_API_BASE_URL", "https://kierolabs.space/api/v2"),
        help="Keiro API base URL",
    )
    ap.add_argument(
        "--endpoint",
        default=env.get("KEIRO_SEARCH_ENDPOINT", "/search/content"),
        help="Search endpoint path",
    )
    args = ap.parse_args()

    api_key = env.get("KEIRO_API_KEY", "")
    if not api_key:
        print("KEIRO_API_KEY not set in .env", file=sys.stderr)
        return 1

    url = args.base_url.rstrip("/") + args.endpoint
    print(f"POST {url}")
    print(f"  query: {args.query!r}")

    import urllib.request
    import urllib.error

    payload = json.dumps({"query": args.query, "maxResults": 5}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:500]
        print(f"FAILED: HTTP {e.code} — {body}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1

    results = data.get("results") or []
    print(f"  Got {len(results)} results")

    for i, r in enumerate(results[:3]):
        title = r.get("title", "")
        uri = r.get("url", "")
        content = (r.get("content") or r.get("snippet") or r.get("description") or "")[:120]
        print(f"  [{i+1}] {title}")
        print(f"      {uri}")
        if content:
            print(f"      {content}...")

    if results:
        print("\n  PASS: Keiro search is working")
        return 0
    else:
        print("\n  WARN: no results returned — check your API key and quota", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

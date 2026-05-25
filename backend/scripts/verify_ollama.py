#!/usr/bin/env python3
"""
One-shot check: Ollama server is reachable and JSON mode works.

Usage (from repo root):
  cd backend && uv run python scripts/verify_ollama.py
  cd backend && uv run python scripts/verify_ollama.py --model mistral-small:24b
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

if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))


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

    ap = argparse.ArgumentParser(description="Verify Ollama for MiroFish")
    ap.add_argument(
        "--model",
        default=env.get("LLM_MODEL_NAME", "mistral-small:24b"),
        help="Model tag matching `ollama list` (default: LLM_MODEL_NAME from .env)",
    )
    ap.add_argument(
        "--base-url",
        default=env.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        help="Ollama OpenAI-compatible base URL",
    )
    args = ap.parse_args()

    base_url = args.base_url.rstrip("/")
    model = args.model

    # 1. Check server connectivity
    import urllib.request
    import urllib.error

    version_url = base_url.replace("/v1", "") + "/api/version"
    print(f"Checking Ollama server at {version_url} ...")
    try:
        with urllib.request.urlopen(version_url, timeout=5) as resp:
            data = json.loads(resp.read())
            print(f"  Server version: {data.get('version', 'unknown')}")
    except Exception as e:
        print(f"FAILED: cannot reach Ollama server: {e}", file=sys.stderr)
        print("  Start with: ollama serve  or  brew services start ollama", file=sys.stderr)
        return 1

    # 2. List models
    models_url = base_url.replace("/v1", "") + "/api/tags"
    try:
        with urllib.request.urlopen(models_url, timeout=5) as resp:
            models_data = json.loads(resp.read())
            names = [m.get("name", "") for m in models_data.get("models", [])]
            print(f"  Available models: {', '.join(names) or '(none)'}")
            if model not in names:
                bare = model.split(":")[0]
                if not any(bare in n for n in names):
                    print(f"  WARNING: {model!r} not found — run: ollama pull {model}", file=sys.stderr)
    except Exception:
        pass

    # 3. JSON mode smoke test via OpenAI SDK
    print(f"\nTesting JSON mode with model={model} at {base_url} ...")
    try:
        from openai import OpenAI

        client = OpenAI(api_key="ollama", base_url=base_url)
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": 'Reply with exactly {"ok": true} and nothing else.'}],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=64,
        )
        message = r.choices[0].message
        text = (message.content or "").strip() if message else ""
        print(f"  Raw response: {text}")

        parsed = json.loads(text)
        if parsed.get("ok") is True:
            print("  PASS: JSON mode works correctly")
        else:
            print(f"  WARN: valid JSON but unexpected content: {parsed}")

        usage = getattr(r, "usage", None)
        if usage:
            dump = getattr(usage, "model_dump", None)
            print(f"  Usage: {dump() if callable(dump) else usage}")

        return 0
    except json.JSONDecodeError:
        print(f"  WARN: response is not valid JSON — MiroFish has built-in repair logic.", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

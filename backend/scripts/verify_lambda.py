#!/usr/bin/env python3
"""
One-shot check: Lambda AI endpoint is reachable and JSON mode works.

Usage (from repo root):
  # 0) Start SSH tunnel first (keep this terminal open):
  # ssh -i "/Users/hgc/Documents/workspace/keys/guven-local.pem" -N -L 18000:127.0.0.1:8000 ubuntu@129.146.124.41
  cd backend && uv run python scripts/verify_lambda.py
  cd backend && uv run python scripts/verify_lambda.py --model hermes-3-llama-3.1-405b-fp8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import dotenv_values

_backend_dir = Path(__file__).resolve().parent.parent
_root_dir = _backend_dir.parent
_env_file = _root_dir / ".env"
_default_base_url = "http://127.0.0.1:18000/v1"

if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))


def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return {k: v for k, v in dotenv_values(path).items() if v is not None}


def main() -> int:
    env = _parse_env_file(_env_file)

    ap = argparse.ArgumentParser(description="Verify Lambda AI endpoint for MiroFish")
    ap.add_argument(
        "--model",
        default=env.get("LAMBDA_MODEL_NAME", "Qwen/Qwen3-32B"),
        help="Model id available on your Lambda endpoint (default: LAMBDA_MODEL_NAME from .env)",
    )
    ap.add_argument(
        "--base-url",
        default=env.get("LAMBDA_BASE_URL", _default_base_url),
        help="Lambda OpenAI-compatible base URL (defaults to LAMBDA_BASE_URL, then local SSH tunnel)",
    )
    ap.add_argument(
        "--api-key",
        default=env.get("LAMBDA_API_KEY", env.get("LLM_API_KEY", "")),
        help="Lambda API key (defaults to LAMBDA_API_KEY, then LLM_API_KEY from .env)",
    )
    args = ap.parse_args()

    base_url = args.base_url.rstrip("/")
    model = args.model
    api_key = (args.api_key or "").strip()

    if not api_key:
        # Many self-hosted OpenAI-compatible endpoints behind SSH tunnels do not
        # enforce API key auth. The OpenAI SDK still requires a non-empty value.
        api_key = "local-tunnel"
        print("No API key provided; using dummy key for local tunnel endpoint.")

    print(f"Testing Lambda endpoint with model={model} at {base_url} ...")
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
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
        print("  WARN: response is not valid JSON; MiroFish has built-in JSON repair logic.", file=sys.stderr)
        return 2
    except Exception as e:
        err_text = str(e)
        if "does not exist" in err_text and "model" in err_text:
            print(f"FAILED: {e}", file=sys.stderr)
            print("Hint: list available models at /v1/models and pass --model with one of them.", file=sys.stderr)
            print("Example: python scripts/verify_lambda.py --model Qwen/Qwen3-32B", file=sys.stderr)
            return 1
        print(f"FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

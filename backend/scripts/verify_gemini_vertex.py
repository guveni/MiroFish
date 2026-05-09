#!/usr/bin/env python3
"""
One-shot check: Gemini on Vertex responds (OpenAI-compatible endpoint + ADC).

Reads repo-root .env for LLM_USE_VERTEX_AI / VERTEX_AI_PROJECT_ID / defaults.
CLI flags override .env for region/model without load_dotenv(override=True) issues.

Usage (from repo root):
  cd backend && uv run python scripts/verify_gemini_vertex.py
  cd backend && uv run python scripts/verify_gemini_vertex.py --region us-central1 --model google/gemini-2.0-flash-001
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from openai import OpenAI

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
    ap = argparse.ArgumentParser(description="Verify Vertex Gemini chat completion")
    ap.add_argument("--region", default=None, help="Vertex region (default: from .env or us-central1)")
    ap.add_argument("--model", default=None, help="Model id with google/ prefix")
    ap.add_argument("--project", default=None, help="GCP project id (default: .env VERTEX_AI_PROJECT_ID or GOOGLE_CLOUD_PROJECT)")
    args = ap.parse_args()

    env = _parse_env_file(_env_file)
    if env.get("LLM_USE_VERTEX_AI", "").strip().lower() not in ("1", "true", "yes", "on"):
        print("Set LLM_USE_VERTEX_AI=true in .env for this script.", file=sys.stderr)
        return 2

    project = (args.project or env.get("VERTEX_AI_PROJECT_ID") or env.get("GOOGLE_CLOUD_PROJECT") or "").strip()
    if not project or project == "your-gcp-project-id":
        print("Set VERTEX_AI_PROJECT_ID (or pass --project) to a real GCP project id.", file=sys.stderr)
        return 2

    region = (args.region or env.get("VERTEX_AI_LOCATION") or "us-central1").strip()
    model = (args.model or env.get("LLM_MODEL_NAME") or "google/gemini-2.5-flash").strip()

    api_version = (env.get("VERTEX_AI_OPENAI_API_VERSION") or "v1").strip().lstrip("/")
    if api_version not in ("v1", "v1beta1"):
        api_version = "v1"

    rl = region.lower()
    if rl == "global":
        base_url = (
            f"https://aiplatform.googleapis.com/{api_version}/projects/"
            f"{project}/locations/global/endpoints/openapi"
        )
    else:
        base_url = (
            f"https://{region}-aiplatform.googleapis.com/{api_version}/projects/"
            f"{project}/locations/{region}/endpoints/openapi"
        )

    from google.auth import default
    from google.auth.transport.requests import Request

    creds, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(Request())
    if not creds.token:
        print("ADC refresh returned no token. Run: gcloud auth application-default login", file=sys.stderr)
        return 2

    client = OpenAI(api_key=creds.token, base_url=base_url)
    print(f"POST {base_url}")
    print(f"model={model}")

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": 'Reply with exactly "OK" and nothing else.'}],
            temperature=0,
            max_tokens=16,
        )
        text = (r.choices[0].message.content or "").strip()
        print("response:", repr(text))
        return 0
    except Exception as e:
        print("FAILED:", e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

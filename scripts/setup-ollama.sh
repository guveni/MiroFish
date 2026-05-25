#!/usr/bin/env bash
set -euo pipefail

# ── Colour helpers (honours NO_COLOR) ──────────────────────────────
if [[ -z "${NO_COLOR:-}" ]]; then
  BOLD="\033[1m"  GREEN="\033[32m"  YELLOW="\033[33m"
  RED="\033[31m"  CYAN="\033[36m"   RESET="\033[0m"
else
  BOLD=""  GREEN=""  YELLOW=""  RED=""  CYAN=""  RESET=""
fi

step()  { printf "\n${BOLD}${CYAN}▶ %s${RESET}\n" "$1"; }
ok()    { printf "  ${GREEN}✔ %s${RESET}\n" "$1"; }
warn()  { printf "  ${YELLOW}⚠ %s${RESET}\n" "$1"; }
fail()  { printf "  ${RED}✘ %s${RESET}\n" "$1"; }

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
ENV_EXAMPLE="$REPO_ROOT/.env.example"
OLLAMA_BASE="http://localhost:11434"

# ──────────────────────────────────────────────
# 1. Check / install Ollama
# ──────────────────────────────────────────────
step "Checking for Ollama"

if command -v ollama &>/dev/null; then
  ok "ollama found: $(ollama --version 2>/dev/null || echo 'unknown version')"
else
  warn "ollama not found on \$PATH"
  OS="$(uname -s)"
  if [[ "$OS" == "Darwin" ]]; then
    if command -v brew &>/dev/null; then
      printf "  Install via Homebrew? [Y/n] "
      read -r yn
      if [[ "${yn:-Y}" =~ ^[Yy]$ ]]; then
        brew install ollama
        ok "Installed via Homebrew"
      else
        fail "Skipped — install manually from https://ollama.com/download"
        exit 1
      fi
    else
      fail "Homebrew not found. Install Ollama from https://ollama.com/download"
      exit 1
    fi
  elif [[ "$OS" == "Linux" ]]; then
    printf "  Install via official installer (curl)? [Y/n] "
    read -r yn
    if [[ "${yn:-Y}" =~ ^[Yy]$ ]]; then
      curl -fsSL https://ollama.com/install.sh | sh
      ok "Installed via official installer"
    else
      fail "Skipped — install manually from https://ollama.com/download"
      exit 1
    fi
  else
    fail "Unsupported OS: $OS. Install Ollama manually."
    exit 1
  fi
fi

# ──────────────────────────────────────────────
# 2. Ensure the Ollama server is running
# ──────────────────────────────────────────────
step "Checking Ollama server at $OLLAMA_BASE"

if curl -sf "$OLLAMA_BASE/api/version" &>/dev/null; then
  ok "Server is running"
else
  warn "Server not reachable — starting ollama serve in the background"
  nohup ollama serve &>/dev/null &
  OLLAMA_PID=$!
  for i in {1..20}; do
    sleep 0.5
    if curl -sf "$OLLAMA_BASE/api/version" &>/dev/null; then
      ok "Server started (pid $OLLAMA_PID)"
      break
    fi
    if [[ $i -eq 20 ]]; then
      fail "Timed out waiting for Ollama server."
      fail "Try running 'ollama serve' manually in another terminal."
      exit 1
    fi
  done
  warn "For persistence run: brew services start ollama"
fi

# ──────────────────────────────────────────────
# 3. Choose and pull a model
# ──────────────────────────────────────────────
step "Choose a model"

printf "  1) mistral-small:24b  ${GREEN}(recommended, ~14 GB, fast JSON mode)${RESET}\n"
printf "  2) qwen2.5:32b-instruct-q4_K_M  (stronger reasoning, ~20 GB)\n"
printf "  3) Custom — enter an Ollama model tag\n"
printf "  Select [1]: "
read -r choice
choice="${choice:-1}"

case "$choice" in
  1) MODEL="mistral-small:24b" ;;
  2) MODEL="qwen2.5:32b-instruct-q4_K_M" ;;
  3)
    printf "  Enter model tag: "
    read -r MODEL
    if [[ -z "$MODEL" ]]; then
      fail "No model specified."
      exit 1
    fi
    ;;
  *)
    fail "Invalid choice: $choice"
    exit 1
    ;;
esac

printf "  Pulling %s (this may take a while on first run)...\n" "$MODEL"
ollama pull "$MODEL"
ok "Model $MODEL is ready"

# ──────────────────────────────────────────────
# 4. Smoke-test JSON mode
# ──────────────────────────────────────────────
step "Smoke-testing JSON mode via OpenAI-compatible endpoint"

JSON_RESP=$(curl -sf "$OLLAMA_BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Reply with exactly {\\\"ok\\\": true}\"}],
    \"response_format\": {\"type\": \"json_object\"}
  }" 2>&1) || true

if echo "$JSON_RESP" | grep -q '"ok"'; then
  ok "JSON mode works"
else
  warn "JSON mode smoke test did not return expected output."
  warn "Response: ${JSON_RESP:0:200}"
  warn "MiroFish may still work — LLMClient has built-in JSON repair."
fi

# ──────────────────────────────────────────────
# 5. Set recommended Ollama env vars in shell profile
# ──────────────────────────────────────────────
step "Configuring Ollama environment variables"

MARKER="# [MiroFish Ollama tuning]"
SHELL_RC=""
if [[ -n "${ZSH_VERSION:-}" ]] || [[ "$SHELL" == */zsh ]]; then
  SHELL_RC="$HOME/.zshrc"
elif [[ -n "${BASH_VERSION:-}" ]] || [[ "$SHELL" == */bash ]]; then
  SHELL_RC="$HOME/.bashrc"
fi

if [[ -n "$SHELL_RC" ]]; then
  if [[ -f "$SHELL_RC" ]] && grep -qF "$MARKER" "$SHELL_RC"; then
    ok "Ollama env vars already present in $SHELL_RC"
  else
    printf "  Add OLLAMA_NUM_PARALLEL / OLLAMA_KEEP_ALIVE / OLLAMA_CONTEXT_LENGTH to %s? [Y/n] " "$SHELL_RC"
    read -r yn
    if [[ "${yn:-Y}" =~ ^[Yy]$ ]]; then
      cat >> "$SHELL_RC" <<EOF

$MARKER
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_KEEP_ALIVE=30m
export OLLAMA_CONTEXT_LENGTH=8192
EOF
      ok "Appended to $SHELL_RC (source it or open a new shell)"
    else
      warn "Skipped — you can add them manually later."
    fi
  fi
else
  warn "Could not detect shell config file. Set these env vars manually:"
  warn "  OLLAMA_NUM_PARALLEL=4  OLLAMA_KEEP_ALIVE=30m  OLLAMA_CONTEXT_LENGTH=8192"
fi

# ──────────────────────────────────────────────
# 6. Patch .env for MiroFish
# ──────────────────────────────────────────────
step "Configuring MiroFish .env"

if [[ ! -f "$ENV_FILE" ]]; then
  if [[ -f "$ENV_EXAMPLE" ]]; then
    cp "$ENV_EXAMPLE" "$ENV_FILE"
    ok "Created .env from .env.example"
  else
    fail ".env.example not found at $ENV_EXAMPLE — cannot create .env"
    exit 1
  fi
fi

_env_set() {
  local key="$1" value="$2"
  if grep -qE "^${key}=" "$ENV_FILE"; then
    sed -i.bak "s|^${key}=.*|${key}=${value}|" "$ENV_FILE"
  elif grep -qE "^# *${key}=" "$ENV_FILE"; then
    sed -i.bak "s|^# *${key}=.*|${key}=${value}|" "$ENV_FILE"
  else
    echo "${key}=${value}" >> "$ENV_FILE"
  fi
}

_env_comment_out() {
  local key="$1"
  if grep -qE "^${key}=" "$ENV_FILE"; then
    sed -i.bak "s|^${key}=|# [setup-ollama] ${key}=|" "$ENV_FILE"
  fi
}

_env_set "LLM_PROVIDER" "ollama"
_env_set "LLM_MODEL_NAME" "$MODEL"
_env_comment_out "LLM_USE_VERTEX_AI"
_env_comment_out "VERTEX_AI_PROJECT_ID"
_env_comment_out "VERTEX_AI_LOCATION"

rm -f "${ENV_FILE}.bak"
ok "Set LLM_PROVIDER=ollama, LLM_MODEL_NAME=$MODEL"

# ──────────────────────────────────────────────
# 7. Run verify:ollama
# ──────────────────────────────────────────────
step "Running npm run verify:ollama"

if command -v npm &>/dev/null && [[ -f "$REPO_ROOT/package.json" ]]; then
  cd "$REPO_ROOT"
  npm run verify:ollama && ok "Verification passed" || warn "Verification script returned non-zero — check output above."
else
  warn "npm or package.json not found — skipping verification."
fi

# ──────────────────────────────────────────────
# Done
# ──────────────────────────────────────────────
printf "\n${BOLD}${GREEN}🎉 Ollama setup complete!${RESET}\n"
printf "  Run ${CYAN}npm run dev${RESET} to start MiroFish with local Ollama.\n"
printf "  To switch back: set ${CYAN}LLM_PROVIDER=vertex${RESET} in .env and restart.\n\n"

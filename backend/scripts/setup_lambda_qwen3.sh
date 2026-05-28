#!/usr/bin/env bash
# connect to the instance
#ssh -vv -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -N \
#   -L 18000:127.0.0.1:8000 \
#   -i "/Users/hgc/Documents/workspace/keys/guven-local.pem" \
#   ubuntu@192.222.51.57
#

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-32B}"
PORT="${PORT:-8000}"
VENV_DIR="${VENV_DIR:-$HOME/llm-env}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

export DEBIAN_FRONTEND=noninteractive
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$HF_HOME"

if ! command -v python3 >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y python3 python3-venv python3-pip git curl
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install -U "transformers>=4.51.0" "vllm>=0.8.0" "huggingface-hub>=0.23.0" "torch>=2.3.0" "numpy>=1.26.0"

if [[ -n "${HF_TOKEN:-}" ]]; then
  huggingface-cli login --token "$HF_TOKEN"
elif [[ ! -f "$HF_HOME/token" ]]; then
  echo "HF token not found. Run: huggingface-cli login"
fi

cat > "$HOME/start-vllm-qwen3.sh" <<STARTSCRIPT
#!/usr/bin/env bash
set -euo pipefail
source "$VENV_DIR/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python3 -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port "$PORT" \
  --model "$MODEL" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --enforce-eager
STARTSCRIPT
chmod +x "$HOME/start-vllm-qwen3.sh"

cat > "$HOME/test-vllm-qwen3.sh" <<TESTSCRIPT
#!/usr/bin/env bash
set -euo pipefail
PORT="${PORT:-8000}"
curl "http://127.0.0.1:${PORT}/v1/models"
echo
echo
echo '---'
curl "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-32B",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "temperature": 0.2,
    "max_tokens": 64
  }'
echo
TESTSCRIPT
chmod +x "$HOME/test-vllm-qwen3.sh"

cat <<SUMMARY
Setup complete.

Next steps:
1. source "$VENV_DIR/bin/activate"
2. huggingface-cli login   # if you did not set HF_TOKEN
3. bash "$HOME/start-vllm-qwen3.sh"
4. In another terminal: bash "$HOME/test-vllm-qwen3.sh"

Remote endpoint:
  http://YOUR_INSTANCE_IP:${PORT}/v1
SUMMARY

# Lambda Qwen3 Fast Setup

This package sets up a fresh Lambda Cloud GPU instance for serving Qwen3 through vLLM's OpenAI-compatible API. Lambda instances are typically accessed over SSH, and by default port 22 is open; custom API access requires opening the serving port in your firewall rules. [cite:28][cite:135]

## What is included

- `setup_lambda_qwen3.sh` installs a Python virtual environment and recent versions of vLLM, Transformers, Hugging Face Hub, Torch, and NumPy that are new enough for Qwen3 support. Qwen3 requires a recent Transformers release, and vLLM exposes an OpenAI-compatible HTTP server for inference. [cite:61][cite:45]
- The script also creates `~/start-vllm-qwen3.sh` to launch the server and `~/test-vllm-qwen3.sh` to test it locally. vLLM serves OpenAI-style endpoints such as `/v1/models` and `/v1/chat/completions`. [cite:45]

## Recommended instance

Qwen3-32B does not fit comfortably on a 40 GB A100 in BF16, which matches the out-of-memory behavior seen earlier. A 96 GB GH200/H100-class instance is a much better fit for the model and leaves more room for runtime overhead and KV cache. [cite:93][cite:117]

## Launch choices

- Base image: prefer **GPU Base 24.04** for a cleaner environment, or **Lambda Stack 24.04** if preinstalled AI tooling is preferred. Lambda's current public cloud docs reference both 24.04 image families. [cite:135]
- Filesystem: skip it for the first launch unless persistent model storage is needed. Filesystems are billed separately per GiB used per month, even when not attached to an instance. [cite:22][cite:19]
- Firewall: keep SSH access, and add port `8000` if the API must be reachable from outside the instance. By default, only port 22 is open. [cite:135]

## Quick start

1. SSH into the instance using the public IP and your configured SSH key. Lambda documents standard SSH access for on-demand instances. [cite:28][cite:147]
2. Copy the setup script to the instance and run:

```bash
bash setup_lambda_qwen3.sh
```

3. If not already authenticated, log in to Hugging Face:

```bash
source ~/llm-env/bin/activate
huggingface-cli login
```

4. Start the server:

```bash
bash ~/start-vllm-qwen3.sh
```

5. Test locally on the box:

```bash
bash ~/test-vllm-qwen3.sh
```

## Endpoint usage

Once running, the local API base URL is:

```text
http://127.0.0.1:8000/v1
```

If port 8000 is opened in Lambda firewall rules, the remote API base URL becomes:

```text
http://YOUR_INSTANCE_IP:8000/v1
```

vLLM's server is OpenAI-compatible, so standard OpenAI SDKs and `curl` requests work against this base URL. [cite:45]

Example:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "temperature": 0.2,
    "max_tokens": 64
  }'
```

## Notes

- The generated launch script uses `--max-model-len 8192`, `--gpu-memory-utilization 0.85`, and `--enforce-eager` to reduce memory pressure, following vLLM's documented memory-conservation guidance. [cite:93]
- Billing for the instance continues while it is running, even if it is idle. [cite:19]
- If a persistent filesystem is created, billing for that filesystem continues as long as it exists. [cite:22][cite:19]

python3 -m venv llm-env
source llm-env/bin/activate
python3 -m pip install -U "transformers>=4.51.0" "vllm>=0.8.0" "huggingface-hub>=0.23.0" "torch>=2.3.0" "numpy>=1.26.0"
source llm-env/bin/activate
huggingface-cli login

python3 -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model Qwen/Qwen3-32B \
  --tensor-parallel-size 1



# ssh -i "/Users/hgc/Documents/workspace/keys/guven-local.pem" -N -L 18000:127.0.0.1:8000 ubuntu@129.146.124.41
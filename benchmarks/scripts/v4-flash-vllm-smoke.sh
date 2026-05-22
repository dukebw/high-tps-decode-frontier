#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-deepseek-ai/DeepSeek-V4-Flash}
REVISION=${REVISION:-6976c7ff1b30a1b2cb7805021b8ba4684041f136}
IMAGE=${IMAGE:-docker.io/vllm/vllm-openai@sha256:4ac9b7c6dabc3ec762c0edef4e9245abe98373844da91cc53ee42e5c58280c5b}
RUN_ROOT=${RUN_ROOT:-/home/ubuntu/shared/logs/vllm-v4-flash-smoke}
HF_CACHE=${HF_CACHE:-/home/ubuntu/.cache/huggingface}
PORT=${PORT:-18000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
GPU_DEVICE=${GPU_DEVICE:-nvidia.com/gpu=all}
WAIT_TIMEOUT_S=${WAIT_TIMEOUT_S:-3600}
KEEP_SERVER=${KEEP_SERVER:-0}
DOCKER=${DOCKER:-docker}
SUDO=${SUDO:-sudo}

docker_cmd() {
  if [ -n "$SUDO" ]; then
    "$SUDO" "$DOCKER" "$@"
  else
    "$DOCKER" "$@"
  fi
}

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$RUN_ROOT/$timestamp"
name="vllm-v4-flash-smoke-$timestamp"
attention_config='{"use_fp4_indexer_cache":true}'

mkdir -p "$run_dir"
printf "%s\n" "$name" > "$run_dir/container.name"
printf "%s\n" "$PORT" > "$run_dir/port"
printf "%s\n" "$IMAGE" > "$run_dir/image"
printf "%s\n" "$MODEL@$REVISION" > "$run_dir/model"

stop_server() {
  if [ "$KEEP_SERVER" != "1" ]; then
    docker_cmd stop "$name" >/dev/null 2>&1 || true
  fi
}
trap stop_server EXIT

docker_cmd run --rm \
  --name "$name" \
  --device "$GPU_DEVICE" \
  --privileged \
  --ipc=host \
  --network host \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -e TILELANG_CLEANUP_TEMP_FILES=1 \
  -e VLLM_DISABLE_COMPILE_CACHE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_RPC_TIMEOUT=600000 \
  -e VLLM_LOG_STATS_INTERVAL=1 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  "$IMAGE" \
  "$MODEL" \
  --revision "$REVISION" \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --host 0.0.0.0 \
  --port "$PORT" \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --attention-config "$attention_config" \
  --moe-backend deep_gemm_mega_moe \
  --disable-uvicorn-access-log \
  > "$run_dir/server.log" 2>&1 &

printf "%s\n" "$!" > "$run_dir/server.pid"

deadline=$((SECONDS + WAIT_TIMEOUT_S))
until curl -sf "http://127.0.0.1:$PORT/v1/models" > "$run_dir/models.json"; do
  if ! docker_cmd ps --filter "name=$name" --format '{{.Names}}' | grep -qx "$name"; then
    printf "vLLM smoke container exited before readiness. Log: %s\n" "$run_dir/server.log" >&2
    exit 1
  fi

  if [ "$SECONDS" -ge "$deadline" ]; then
    printf "Timed out waiting for vLLM readiness. Log: %s\n" "$run_dir/server.log" >&2
    exit 1
  fi

  sleep 15
done

cat > "$run_dir/nonthink-request.json" <<JSON
{"model":"$MODEL","messages":[{"role":"user","content":"What is 17*19? Return only the final integer."}],"max_tokens":32,"temperature":1.0,"top_p":1.0}
JSON

cat > "$run_dir/think-high-request.json" <<JSON
{"model":"$MODEL","messages":[{"role":"user","content":"What is 17*19? Return only the final integer."}],"max_tokens":64,"temperature":1.0,"top_p":1.0,"chat_template_kwargs":{"thinking":true,"reasoning_effort":"high"}}
JSON

curl -sS --max-time 300 \
  -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "@$run_dir/nonthink-request.json" \
  > "$run_dir/nonthink.json"

curl -sS --max-time 300 \
  -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "@$run_dir/think-high-request.json" \
  > "$run_dir/think-high.json"

python3 - "$run_dir/nonthink.json" "$run_dir/think-high.json" <<'PY'
import json
import sys

for path in sys.argv[1:]:
    with open(path) as handle:
        payload = json.load(handle)
    choice = payload.get("choices", [{}])[0]
    message = choice.get("message", {})
    print(path)
    print("content=", repr(message.get("content")))
    print("reasoning=", repr(message.get("reasoning_content")))
    print("finish=", choice.get("finish_reason"))
    print("usage=", payload.get("usage"))
PY

printf "run_dir=%s\n" "$run_dir"
printf "container=%s\n" "$name"
printf "port=%s\n" "$PORT"

if [ "$KEEP_SERVER" = "1" ]; then
  trap - EXIT
  printf "KEEP_SERVER=1; leaving container running. Stop with: %s %s stop %s\n" "$SUDO" "$DOCKER" "$name"
fi

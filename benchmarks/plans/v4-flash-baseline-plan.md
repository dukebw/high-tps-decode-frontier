# V4 Flash Baseline Benchmark Plan

Status: draft, vLLM smoke passed on `b200-aws2`; blocked on SGLang smoke testing and benchmark harness selection

## Purpose

Define the first concrete benchmark plan for real `deepseek-ai/DeepSeek-V4-Flash` serving. This plan turns the [V4 Flash InferenceX-style frontier investigation](../../wiki/investigations/v4-flash-inferencex-frontier.md) into a reproducible run matrix for selecting the strongest public baseline across vLLM and SGLang.

## Source Provenance

- Investigation: `../../wiki/investigations/v4-flash-inferencex-frontier.md`
- Kernel map: `../../wiki/artifacts/v4-flash-kernel-map.md`
- vLLM recipe source: `../../sources/github/vllm-recipes-deepseek-v4-flash.yaml`
- vLLM recipe page snapshot: `../../sources/html/vllm-recipes-deepseek-v4-flash.html`
- SGLang cookbook snapshot: `../../sources/html/sglang-deepseek-v4-cookbook.html`
- DeepSeek V4 Flash HF model-card snapshot: `../../sources/huggingface/deepseek-ai-deepseek-v4-flash-readme.md`

## Model Pin

- Hugging Face repo: `deepseek-ai/DeepSeek-V4-Flash`
- Revision commit SHA: `6976c7ff1b30a1b2cb7805021b8ba4684041f136`
- Resolution date: 2026-05-22

Pinned file paths to record in benchmark results:

- `README.md`
- `config.json`
- `generation_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `encoding/README.md`
- `encoding/encoding_dsv4.py`
- `encoding/test_encoding_dsv4.py`
- `encoding/tests/test_input_1.json`
- `encoding/tests/test_input_2.json`
- `encoding/tests/test_input_3.json`
- `encoding/tests/test_input_4.json`
- `encoding/tests/test_output_1.txt`
- `encoding/tests/test_output_2.txt`
- `encoding/tests/test_output_3.txt`
- `encoding/tests/test_output_4.txt`
- `inference/README.md`
- `inference/config.json`
- `inference/convert.py`
- `inference/generate.py`
- `inference/kernel.py`
- `inference/model.py`
- `inference/requirements.txt`
- `model.safetensors.index.json`
- `model-00001-of-00046.safetensors` through `model-00046-of-00046.safetensors`

Do not use `main` as benchmark provenance. Do not silently bump this revision for any result record.

## Hardware Target

Run the first pass on one selected 4-GPU Blackwell topology that can serve V4 Flash without offload or workload distortion.

Candidate targets:

- `4xB200`
- `4xB300`
- `4xGB200`
- `4xGB300`

Rules:

- Do not compare points across hardware topologies as one baseline.
- If the accessible machine has 8 GPUs, constrain first-pass 4-GPU configs with `CUDA_VISIBLE_DEVICES` or equivalent runtime placement.
- If only H200 is available, record it as a separate non-Blackwell reproduction path, not the first Blackwell baseline.

## Remote Bootstrap State

`b200-aws2` status as of 2026-05-22:

- GPUs: 4x NVIDIA B200, 183359 MiB each.
- Topology: all GPU pairs report `NV18`; NICs are `ibp115s0f0` and `ibp116s0f0`.
- Driver: `590.48.01`.
- Container runtime: Docker CLI backed by Podman `4.6.2`.
- Disk: `/home` has about 2.0T total capacity; about 883G free before image pulls and model download.
- Mutagen sync: local repo synced to `b200-aws2.coder:/home/ubuntu/work/high-tps-decode-frontier`.
- Shared logs: remote `/home/ubuntu/shared/logs` mirrored to local `~/shared/b200-aws2/logs`.
- Model cache: pinned HF snapshot downloaded into `/home/ubuntu/.cache/huggingface`; final cache size observed as about `844G`.
- Download log: `~/shared/b200-aws2/logs/v4-flash-download/20260522T1800Z/download.log`.

## Runtime Image Pinning

The imported recipes use moving tags such as `vllm/vllm-openai:latest` and `lmsysorg/sglang:latest`. Before any run, replace those with immutable image digests or exact build SHAs and record them in the result.

Candidate runtime pins resolved on `b200-aws2` on 2026-05-22:

- vLLM: `docker.io/vllm/vllm-openai@sha256:4ac9b7c6dabc3ec762c0edef4e9245abe98373844da91cc53ee42e5c58280c5b`
- SGLang: `docker.io/lmsysorg/sglang@sha256:015f39a45844be5a7b35270c56dc4d9ebcfe9b0c21a3b4f877a4ee22e795bd7a`

Pulled image IDs on `b200-aws2`:

- vLLM: `2497255b1272`, version `0.21.0`, `VLLM_BUILD_COMMIT=ad7125a431e176d4161099480a66f0169609a690`, `VLLM_IMAGE_TAG=vllm/vllm-openai:v0.21.0`
- SGLang: `79d577610f50`, version `0.5.12`, `SGLANG_BUILD_COMMIT=127b9e3283f7c2a43234b852ff5c9f1796d53624`, `SGLANG_IMAGE_TAG=lmsysorg/sglang:v0.5.12`

Container GPU note for `b200-aws2`: the Docker CLI is backed by Podman. `--gpus all` did not expose CUDA inside containers. Generated `/etc/cdi/nvidia.yaml` with `nvidia-ctk`; use `--device nvidia.com/gpu=all` for GPU-visible runs.

## vLLM Smoke Test

Reproduction script: `../scripts/v4-flash-vllm-smoke.sh`. From the local synced workspace, run it on `b200-aws2` with:

```bash
rexec --flush bash benchmarks/scripts/v4-flash-vllm-smoke.sh
```

The script launches the pinned vLLM image and pinned HF revision from the local HF cache with `HF_HUB_OFFLINE=1`, waits for `/v1/models`, sends one Non-think and one Think High chat request, writes artifacts under `/home/ubuntu/shared/logs/vllm-v4-flash-smoke/<timestamp>/`, and stops the container unless `KEEP_SERVER=1`.

The script itself was verified on `b200-aws2` at `/home/ubuntu/shared/logs/vllm-v4-flash-smoke/20260522T184409Z/` after fixing an early readiness check that could exit before Podman registered the container name.

Completed smoke run on 2026-05-22:

- Run directory: `/home/ubuntu/shared/logs/vllm-v4-flash-smoke/20260522T181009Z/`.
- Local mirrored log directory: `~/shared/b200-aws2/logs/vllm-v4-flash-smoke/20260522T181009Z/`.
- Command shape: 4 visible B200s, `--device nvidia.com/gpu=all`, `--kv-cache-dtype fp8`, `--block-size 256`, `--data-parallel-size 4`, `--enable-expert-parallel`, `--tokenizer-mode deepseek_v4`, `--reasoning-parser deepseek_v4`, `--attention-config '{"use_fp4_indexer_cache":true}'`, `--moe-backend deep_gemm_mega_moe`.
- Smoke-only limits: `--max-model-len 8192`, `--max-num-seqs 4`, `--max-num-batched-tokens 8192`. This validates launch and request protocol only; it is not a 1M-context baseline.
- Readiness: `/v1/models` returned `deepseek-ai/DeepSeek-V4-Flash` with `max_model_len=8192` after about 8 minutes on a cold start with kernel compilation.
- Log-confirmed runtime behavior: resolved `DeepseekV4ForCausalLM`, loaded the pinned local HF snapshot, enabled DeepGEMM E8M0, used `fp8_ds_mla` KV cache format, used MXFP4 indexer cache, and initialized `world_size=4`.
- Non-think request: returned content `323`, `finish_reason=stop`, `completion_tokens=2`.
- Think High request: returned content `323`, `finish_reason=stop`, `completion_tokens=26`.
- Container stopped after the smoke test; all four B200s returned to `0 MiB` used memory.

Required runtime metadata:

- Pulled image ID for each pinned digest.
- Runtime version reported inside each container.
- CUDA, driver, NCCL, DeepGEMM, FlashInfer, and any runtime plugin SHAs or versions.
- Host kernel, container runtime, and GPU clock/power state when known.

## Runtime Config Matrix

### vLLM

Run this required variant:

| ID | Recipe source | Hardware scope | Key settings | Inclusion |
| --- | --- | --- | --- | --- |
| `vllm-single-node-tep-blackwell` | vLLM recipe default strategy `single_node_tep` | 4-GPU Blackwell | `--trust-remote-code`, `--kv-cache-dtype fp8`, `--block-size 256`, `--data-parallel-size 4`, `--enable-expert-parallel`, `--tokenizer-mode deepseek_v4`, `--reasoning-parser deepseek_v4`, Blackwell `--attention_config.use_fp4_indexer_cache=True`, Blackwell `--moe-backend deep_gemm_mega_moe` | Required |

vLLM variants deferred from the first matrix:

- `single_node_tp`: latency-oriented TP-only path, uses different GPU count/parallelism assumptions.
- `single_node_dep` and `multi_node_dep`: defer until the first 4-GPU DP+EP baseline exists.
- `pd_cluster`, Mooncake, and NIXL PD-disaggregation paths: defer because they use separate prefill/decode pools and should form a distinct topology/regime.

### SGLang

Run these required candidate variants when supported by the selected Blackwell hardware:

| ID | Recipe | Key settings | Inclusion |
| --- | --- | --- | --- |
| `sglang-low-latency-fp4-deepep` | Flash FP4 low-latency | Tensor parallelism 4, MTP steps 3, draft tokens 4, DeepEP backend, no MegaMoE | Required |
| `sglang-balanced-fp4-deepep` | Flash FP4 balanced | Tensor parallelism 4, MTP steps 1, draft tokens 2, DeepEP backend | Required |
| `sglang-max-throughput-fp4-deepep` | Flash FP4 max-throughput | Tensor parallelism 4, MTP disabled, DeepEP backend | Required |
| `sglang-balanced-fp4-megamoe-w4a8` | Flash FP4 balanced with MegaMoE W4A8 | `--moe-a2a-backend megamoe`, recommended `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096`, no manual `--moe-runner-backend` | Required if generator supports it on selected Blackwell target |
| `sglang-balanced-fp4-megamoe-w4a4` | Flash FP4 balanced with MegaMoE W4A4 | W4A8 settings plus `SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS=1` and `SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND=1` | Required if generator supports it on selected Blackwell target |
| `sglang-max-throughput-fp4-megamoe-w4a8` | Flash FP4 max-throughput with MegaMoE W4A8 | `--moe-a2a-backend megamoe`, recommended `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320`, no manual `--moe-runner-backend` | Required if generator supports it on selected Blackwell target |
| `sglang-max-throughput-fp4-megamoe-w4a4` | Flash FP4 max-throughput with MegaMoE W4A4 | W4A8 settings plus `SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS=1` and `SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND=1` | Required if generator supports it on selected Blackwell target |

SGLang variants deferred from the first matrix:

- `cp`: context-parallel prefill path; separate long-context topology/regime.
- `pd-disagg`: prefill/decode disaggregation; separate topology/regime.
- H100/H200 FP8-converted checkpoints; separate non-Blackwell or fallback reproduction path.
- HiCache CPU/GPU hierarchy; separate cache-system regime unless needed to implement the first agentic prefix-cache workload.

## Workload Matrix

Run both workload regimes before declaring the strongest baseline.

### No-Cache InferenceX Regime

Use public InferenceX random request scenarios unchanged. Disable/avoid prefix caching and disable warmup.

| ID | ISL target | OSL target | Input distribution | Prefix cache | Warmup |
| --- | ---: | ---: | --- | --- | --- |
| `inferencex-chat-1024-1024-no-cache` | 1024 | 1024 | 80-100% of ISL target | disabled/avoided | disabled |
| `inferencex-reasoning-1024-8192-no-cache` | 1024 | 8192 | 80-100% of ISL target | disabled/avoided | disabled |
| `inferencex-summarization-8192-1024-no-cache` | 8192 | 1024 | 80-100% of ISL target | disabled/avoided | disabled |

### Agentic Prefix-Cache Regime

Use a separate frontier for high-ISL multi-turn coding/agent traffic. Enable runtime prefix caching and explicitly warm the cache before timing starts.

| ID | Total ISL | Cached prefix target | Uncached delta | OSL target | Prefix cache | Warmup |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `agentic-prefix-cache-128k-95p-1k` | 128K | 95% | about 6.4K tokens | 1024 | enabled, measured hit rate required | explicit cache warmup |

Defer 512K and 1M total-ISL prefix-cache variants until the harness is stable.

## Sweep Matrix

- Request policy: closed-loop infinite-rate.
- Served-concurrency grid: `1, 2, 4, 8, 16, 32, 64, 128, 256`.
- Repetitions: 3 per point unless hardware availability forces an explicit exception.
- First-pass profiling: not required for every point; collect profiler traces manually later on selected frontier/knee points.

Maximum first-pass performance run count, if all optional MegaMoE variants launch:

- Runtime configs: 8.
- Workload scenarios: 4.
- Concurrency points: 9.
- Repetitions: 3.
- Total: 864 benchmark runs.

Minimum first-pass performance run count, if only the required non-MegaMoE runtime configs launch:

- Runtime configs: 4.
- Workload scenarios: 4.
- Concurrency points: 9.
- Repetitions: 3.
- Total: 432 benchmark runs.

## Quality Gate Matrix

Quality gate mode: report-only for the first baseline.

Run before interpreting performance results for any runtime config that will be considered as a baseline:

- OpenAI MRCR, 2-needle only, all 100 samples in each selected bin: 128K, 512K, and 1M.
- Entire LongBench v2 `long` subset.
- Reasoning modes: Non-think and Think High.

Compare scores to external references where settings are comparable, especially the DeepSeek V4 technical report and public MRCR/LongBench v2 references. Do not impose pass/fail thresholds until local baseline distributions exist.

## Serving Correctness Checks

Before performance runs, verify each runtime config:

- Serves `deepseek-ai/DeepSeek-V4-Flash` from revision `6976c7ff1b30a1b2cb7805021b8ba4684041f136`.
- Uses the intended tokenizer, parser, and DeepSeek V4 encoding path.
- Can produce both Non-think and Think High outputs using the configured request protocol.
- Honors requested max output lengths for 1K and 8K OSL scenarios.
- Records whether MTP/speculative decoding is enabled and how accepted tokens are counted.
- Records whether MegaMoE, DeepEP, DeepGEMM, FlashInfer, or other specialized backends are active.
- For no-cache runs, verifies prefix reuse is disabled or avoided by randomization.
- For prefix-cache runs, verifies target cached-prefix percentage and measured hit rate.

## First Result Records To Produce

Create one benchmark result record per runtime config and workload scenario after all repetitions finish. Each result record should aggregate the fixed concurrency sweep and list the non-dominated frontier points.

Recommended result-record path pattern:

- `benchmarks/results/v4-flash/<runtime-config>/<workload-id>.md`

Do not claim a global baseline until vLLM and SGLang have both run on the same hardware topology and both workload regimes.

## Open Setup Items

- Generate exact SGLang launch commands from the cookbook for the selected hardware and every included recipe variant.
- Decide the benchmark harness implementation path for InferenceX-style no-cache and agentic prefix-cache traffic.
- Decide how to compute tokens/s/GPU when a runtime uses fewer visible GPUs than the physical node contains.
- Confirm whether the full vLLM `1M` context setting launches cleanly before using it for quality or performance runs.

# Cronus-9B

**Sub-50 ms Zero-Generation Pre-Execution Security Guardrail on Qwen3.5-9B (`262,144` Native Context)**

Cronus-9B is a single-forward-pass, zero-generation pre-execution safety and authorization guardrail built on `Qwen/Qwen3.5-9B`. Instead of generating verbose text rationales, Cronus scores candidate action labels directly from single-token (`A..Z`) log-probabilities emitted by a frozen FP32 readout projection (`W0`, sliced to 64 rows at serving time), combined with a folded Stage-1 backbone adapter and two role-routed all-linear Punica LoRAs (`role_decision_r16`, `role_scope_r8`).

- **Code & Training Pipeline**: `FishCodeTech/cronus` (this repository)
- **Model Weights & Quantized Bundle**: [`Sn1waR/Cronus-9B`](https://huggingface.co/Sn1waR/Cronus-9B)

---

## Architecture Overview

1. **Zero-Generation Single-Token Readout (`W0`)**:
   - Deterministic prompt packing formats `(state, question)` into Qwen3.5 non-thinking chat prompts with fixed `A..Z` option labels.
   - A frozen 26×4096 FP32 slice (`W0_readout/readout.safetensors`) extracts option logits in 1 forward step (`max_tokens=1`, zero autoregressive decoding).
   - At vLLM boot, `lm_head` is sliced from `248,320` rows to `64` rows with self-verified `<= 1e-6` probability parity.
2. **Stage-1 Default Backbone + Stage-2 Two-Role Routing**:
   - **Rule-governed task router** (`cronus.router.route`, `cronus.task_heads.head_for`) routes each evaluation query to one of four heads: `W0` / `permission` (uses pre-folded `default`), `decision` (`default + role_decision_r16`), or `scope` (`default + role_scope_r8`).
   - In deployment, the Stage-1 `default` adapter (`r=8, alpha=16`) is pre-folded in FP32 and quantized to official Qwen3.5 `128x128` block-wise FP8 (`float8_e4m3fn`), leaving zero LoRA overhead on `W0`/`permission` calls and lightweight residual Punica LoRAs (`r=16` / `r=8`) on `decision` / `scope`.
3. **Full `262,144` Context & Hardware-Safe Execution**:
   - Enforces full native `262,144` tokens with explicit `413 ContextLengthExceeded` rejection on overflow (never silently truncates).
   - Chunked prefill is capped at `2,048` tokens, hybrid attention/Mamba state uses `mamba_cache_mode="align"`, and standalone Transformers execution pins `sm_121` Triton `Autotuner` kernels with `torch.cuda.synchronize()` loop throttling.

---

## Repository Structure

```text
cronus/
├── docker/
│   ├── Dockerfile.runtime          # PyTorch + Transformers + PEFT + FLA training/native image
│   ├── Dockerfile.vllm             # vLLM >= 0.30.0 serving image
│   └── run_vllm_gb10.sh            # Bounded container launcher for DGX Spark (GB10)
├── scripts/
│   └── check_gb10_safety.py        # Static AST check for unified-memory GPU invariants
├── src/cronus/
│   ├── configs/                    # Bundled schemas, router allowlist, pinned manifest & calibration
│   ├── kernels/                    # Triton FLA autotuner key pinning (inference + training)
│   ├── train/                      # W0 extraction, Stage-1 LoRA, Stage-2 roles, WiSE-FT selection
│   ├── deploy/                     # FP8/LoRA exporters, vLLM server, native static server
│   ├── eval/                       # DEV/sealed/ScopeJudge/ATBench-500/parity/latency evaluators
│   ├── formatting.py               # Deterministic Qwen3.5 non-thinking prompt & group packing
│   ├── paths.py                    # HF hub / local bundle layout resolution
│   ├── role_model.py               # Multi-role candidate loader with SHA-256 verification
│   ├── router.py                   # Criteria- & task-type-based role router
│   ├── runtime.py                  # Core Classifier wrapper for training & short-context parity
│   ├── static_engine.py            # Pre-merged static Transformers + FLA chunked-prefill runner
│   └── task_heads.py               # Canonical 4-head classifier helper (`W0`/`permission`/`decision`/`scope`)
└── pyproject.toml
```

---

## Quickstart

### 1. Download Weights from Hugging Face

```bash
hf download Sn1waR/Cronus-9B --local-dir ./Cronus-9B
```

`Sn1waR/Cronus-9B` contains:
- Root directory: pre-folded `128x128` block-wise FP8 `Cronus-9B` checkpoint + tokenizer (`config.json`, shards, `tokenizer.json`)
- `loras/role_decision_r16/`: residual `decision` Punica LoRA (`r=16, alpha=32`)
- `loras/role_scope_r8/`: residual `scope` Punica LoRA (`r=8, alpha=16`)
- `guard_config/`: `cronus_manifest.json`, `calibration_result.json`, `W0_readout/readout.safetensors`
- `source_bf16/`: BF16 `default` adapter and modular role checkpoints for retraining/re-export

### 2. Install Package

```bash
pip install -e ".[serve,eval]"
```

### 3. Serve with vLLM (Recommended)

Via Docker (with bounded unified-memory limits):

```bash
CRONUS_BUNDLE_DIR="$(pwd)/Cronus-9B" CRONUS_VLLM_IMAGE=cronus-vllm:latest ./docker/run_vllm_gb10.sh
```

Or directly via CLI / module:

```bash
python3 -m cronus.deploy.serve_vllm --bundle-dir ./Cronus-9B --host 0.0.0.0 --port 18971
```

### 4. Query `/evaluate` and `/healthz`

```bash
curl -s http://127.0.0.1:18971/healthz | jq .
```

Single-question or 4-question continuous batch request:

```bash
curl -s http://127.0.0.1:18971/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "id": "demo-001",
    "state": {
      "tool": "bash",
      "args": {"cmd": "ls -la /tmp/work"},
      "trusted": {"scope": "authorized /tmp/work workspace only"}
    },
    "questions": [
      {"type": "noul", "instructions": "Is production harm expected?", "criteria": {"true": "Harm expected.", "false": "No harm."}},
      {"type": "noul", "instructions": "Is this tool call out of scope?", "criteria": {"true": "Out of scope.", "false": "In scope."}},
      {"type": "choice", "instructions": "Select pre-execution action.", "criteria": {"approve": "Allow.", "escalate": "Ask human.", "reject": "Deny."}}
    ]
  }' | jq .
```

---

## Training & Export Pipeline

Reproduce the full Cronus-9B build from `Qwen/Qwen3.5-9B`:

1. `python3 -m cronus.train.extract_w0_readout` — extract the frozen 26×4096 FP32 `W0` projection for tokens `A..Z`.
2. `python3 -m cronus.train.train_stage1_default` — train the Stage-1 shared `default` adapter (`r=8, alpha=16`).
3. `python3 -m cronus.train.build_role_dataset` & `python3 -m cronus.train.cache_teacher` — build role splits (`decision` / `scope`) and cache Stage-1 teacher distributions.
4. `python3 -m cronus.train.train_stage2_role` — train Stage-2 role LoRAs with bounded KL anchor + family macro NLL.
5. `python3 -m cronus.train.wiseft_select` — interpolate and select Stage-2 checkpoints (`w=0.40` for `decision`, `epoch-3` for `scope`).
6. `python3 -m cronus.deploy.export_fp8_bundle` — fold `default` into FP32, quantize to `128x128` FP8, and export residual LoRAs.

### Pinned Candidate Artifact SHA-256 Checksums

| Artifact | Details | SHA-256 |
| --- | --- | --- |
| Base model lock | `Qwen/Qwen3.5-9B` (BF16, `262,144` native context) | `ce6771a6a7862390f37179d427133b24515f167d6c5a159a9acc0d073302c745` |
| `W0` readout | `readout.safetensors` (`26x4096` FP32 `W0` + `label_ids`) | `5433b51444d95c06db35ca5aaede155699e5bb92e08247d4b7ae78d970f4940c` |
| Stage-1 `default` | `adapter_model.safetensors` (176 tensors, `r=8`, `alpha=16`) | `6d55a22ff6b2b80aa61e5db56d365cb8141d01575688413939b4cd86c8ce4d2a` |
| `role_decision_r16` | `guard.safetensors` (496 tensors, `r=16`, `alpha=32`, WiSE-FT `w=0.40`) | `7f9dbcd86b7911ad9fed50043ee3c11ee49b33d7e3a74f20507c62f223717ada` |
| `role_scope_r8` | `guard.safetensors` (496 tensors, `r=8`, `alpha=16`, `epoch-3`) | `39ae42197acb684232c0cdba5385b82d6c43ad665e2b6601d93447e954375338` |

---

## Verified Benchmarks

### Overall Capabilities Comparison

| Model | Scope BA | ATBench (`@0.5`) | General Decision Acc |
| --- | --- | --- | --- |
| **Cronus-9B** | **88.09%** | **58.80%** | **100.00%** |
| THX-02 | 93.99% | 64.60% | 100.00% |
| Official Jev 1.14.0 | 79.96% | 81.40% | 90.41% |
| Nimble 9B v2 | 78.44% | 84.20% | 63.37% |

### ScopeJudge Authorization Boundary

Evaluated on `exec_unanimous` (`n=799`) for `Cronus-9B` and `THX-02`:

| Model | Recall (Out-of-Scope) | Specificity (In-Scope) | Balanced Accuracy (BA) |
| --- | --- | --- | --- |
| **Cronus-9B** (`ROC-AUC 0.9737`) | **81.48%** | **94.69%** | **88.09%** |
| THX-02 (`ROC-AUC 0.9961`) | 88.89% | 99.09% | 93.99% |
| Official Jev 1.14.0 | 70.69% | 89.24% | 79.96% |
| Nimble 9B v2 | 67.24% | 89.64% | 78.44% |
| DeepSeek Distill 9B (No Boundary Training) | 58.62% | 95.63% | 77.13% |
| Laya Multilingual 322M | 93.10% | 12.28% | 52.69% |
| Needle3 | 27.59% | 73.81% | 50.70% |
| Laya English 421M | 27.59% | 73.81% | 50.70% |
| Laya Typed 421M | 17.24% | 83.65% | 50.45% |
| Community Jev-Gemma 3 270M | 0.00% | 99.59% | 49.80% |
| Binary Random Generator | 44.83% | 49.54% | 47.19% |
| Google Gemma 3 270M IT + Jevify | 36.21% | 57.36% | 46.78% |

### DEV 869 & ATBench-500 Breakdown (`Cronus-9B` vs `THX-02`)

| Metric | Cronus-9B | THX-02 |
| --- | --- | --- |
| Physical Harm BA | **94.91%** | 93.98% |
| Scope Boundary BA | 93.44% | **93.94%** |
| AgentApprove BA | **98.33%** | 96.67% |
| Normal False-Block Rate | **5.88%** | 58.82% |
| DEV Decision NLL | **0.1331** | 0.2060 |
| ATBench-500 ROC-AUC | 0.9721 | **0.9949** |
| ATBench-500 Best-Threshold Acc | 89.80% | **97.00%** |

### Single-Node HTTP Serving Latency (NVIDIA DGX Spark GB10, `vLLM 0.30.0`, `128x128` FP8)

| Workload | P50 Latency | P95 / Max Latency |
| --- | --- | --- |
| Warm 4-question `W0`/permission batch (per-question forward) | **49.80 ms** | **51.41 ms** (`51.412 ms` max) |
| Cold `<=512`-token `W0`/permission single-question client | **80.40 ms** | — |
| Cold `<=512`-token all-roles single-question client | **83.04 ms** | **95.18 ms** |
| Warm `650`-token prefix-cache hit (client / forward) | **65.18 ms** / **63.63 ms** | — |

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

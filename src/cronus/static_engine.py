"""Static pre-merged Transformers + FLA execution engine (zero runtime PEFT).

Used both by the native fallback HTTP server (`cronus.deploy.serve_native`) and
offline static checkpoint evaluation scripts (`cronus.eval.*`) without any
unmerged PEFT/LoRA adapter dispatch overhead.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import importlib
import json
import os
from pathlib import Path
import string
import time

from safetensors.torch import load_file
import torch
from torch.nn.attention.bias import CausalBias, causal_lower_right
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
import triton
from triton.runtime.autotuner import Autotuner

from cronus.formatting import qwen_prompt
from cronus.kernels.native_pin import install
from cronus.paths import default_config_path

CUDA_MEM_FRACTION = float(os.environ.get("CUDA_MEM_FRACTION", "0.33"))
assert 0.0 < CUDA_MEM_FRACTION <= 0.35, CUDA_MEM_FRACTION

CHUNK_SIZE = 2048
MAX_CONTEXT_TOKENS = int(os.environ.get("MAX_CONTEXT_TOKENS", "262144"))


def clone_dynamic_cache(src: DynamicCache, text_config) -> DynamicCache:
    """Deep-clone cached KV and recurrent states prior to tail continuation."""
    dst = DynamicCache(config=text_config)
    for src_layer, dst_layer in zip(src.layers, dst.layers):
        for key, value in vars(src_layer).items():
            setattr(dst_layer, key, value.clone() if isinstance(value, torch.Tensor) else value)
    return dst


class PreallocatedDynamicLayer(DynamicLayer):
    """Fixed-capacity contiguous KV buffer avoiding repeated allocations during chunked prefill."""

    def __init__(self, max_seq_len: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.cur_len = 0
        self.k_buf = None
        self.v_buf = None

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        batch, heads, _, head_dim = key_states.shape
        self.k_buf = torch.empty((batch, heads, self.max_seq_len, head_dim), dtype=self.dtype, device=self.device)
        self.v_buf = torch.empty((batch, heads, self.max_seq_len, head_dim), dtype=self.dtype, device=self.device)
        self.keys = self.k_buf[:, :, :0, :]
        self.values = self.v_buf[:, :, :0, :]
        self.cur_len = 0
        self.is_initialized = True

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        seq_step = key_states.shape[-2]
        end = self.cur_len + seq_step
        self.k_buf[:, :, self.cur_len : end, :].copy_(key_states)
        self.v_buf[:, :, self.cur_len : end, :].copy_(value_states)
        self.cur_len = end
        self.keys = self.k_buf[:, :, :end, :]
        self.values = self.v_buf[:, :, :end, :]
        return self.keys, self.values


def make_preallocated_cache(text_config, total_len: int) -> DynamicCache:
    """Allocate a DynamicCache with PreallocatedDynamicLayer for full-attention layers."""
    cache = DynamicCache(config=text_config)
    for idx, layer in enumerate(cache.layers):
        if type(layer) is DynamicLayer:
            cache.layers[idx] = PreallocatedDynamicLayer(total_len)
    return cache


def install_flash_chunked_sdpa() -> None:
    """Install causal-lower-right SDPA handling for cached chunked prefill."""
    orig_mask_fn = ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]
    orig_attn_fn = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def fast_sdpa_mask(
        batch_size: int,
        q_length: int,
        kv_length: int,
        q_offset: int = 0,
        kv_offset: int = 0,
        attention_mask=None,
        local_size=None,
        allow_is_causal_skip: bool = True,
        **kwargs,
    ):
        if (
            allow_is_causal_skip
            and attention_mask is None
            and local_size is None
            and kv_offset == 0
            and q_offset + q_length == kv_length
        ):
            if q_length == kv_length or q_length == 1:
                return None
            if 1 < q_length < kv_length:
                return causal_lower_right(q_length, kv_length)
        return orig_mask_fn(
            batch_size=batch_size,
            q_length=q_length,
            kv_length=kv_length,
            q_offset=q_offset,
            kv_offset=kv_offset,
            attention_mask=attention_mask,
            local_size=local_size,
            allow_is_causal_skip=allow_is_causal_skip,
            **kwargs,
        )

    def fast_sdpa_forward(
        module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask,
        dropout: float = 0.0,
        scaling=None,
        is_causal=None,
        **kwargs,
    ):
        q_len = query.shape[2]
        kv_len = key.shape[2]
        if (attention_mask is None and 1 < q_len < kv_len) or isinstance(attention_mask, CausalBias):
            bias = attention_mask if isinstance(attention_mask, CausalBias) else causal_lower_right(q_len, kv_len)
            out = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=bias,
                dropout_p=dropout,
                scale=scaling,
                is_causal=False,
                enable_gqa=hasattr(module, "num_key_value_groups"),
            )
            return out.transpose(1, 2).contiguous(), None
        return orig_attn_fn(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            **kwargs,
        )

    ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = fast_sdpa_mask
    ALL_ATTENTION_FUNCTIONS["sdpa"] = fast_sdpa_forward


def chunk_spans(total_len: int, chunk_size: int = CHUNK_SIZE) -> list[tuple[int, int]]:
    """Split a token sequence into <= chunk_size spans, avoiding tiny final chunks."""
    spans: list[tuple[int, int]] = []
    pos = 0
    while pos < total_len:
        rem = total_len - pos
        if rem <= chunk_size:
            spans.append((pos, total_len))
            break
        step = chunk_size if (rem - chunk_size) >= 128 else (rem - 128)
        spans.append((pos, pos + step))
        pos += step
    return spans


def install_pinned_kernels(profile_path: Path | None = None) -> dict:
    """Pin both uncached and chunked-cache Triton FLA Autotuner keys for sm_121."""
    orig_run = Autotuner.run
    ev = install(str(profile_path or default_config_path("profile_native.json")))
    guarded_run = Autotuner.run

    mod = importlib.import_module("fla.ops.common.chunk_delta_h")
    obj = getattr(mod, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64")
    while not isinstance(obj, Autotuner):
        obj = obj.fn
    base_key = (
        32,
        32,
        128,
        128,
        64,
        False,
        "torch.bfloat16",
        "torch.bfloat16",
        "torch.bfloat16",
        "torch.bfloat16",
        "torch.float32",
        "torch.bfloat16",
    )
    dtypes = ("torch.float32", "torch.bfloat16")
    extra_pinned_keys: set[tuple[int, tuple]] = set()
    for d1 in dtypes:
        k1 = base_key + (d1,)
        obj.cache[k1] = triton.Config(
            {"BV": 32}, num_warps=2, num_stages=1, num_ctas=1, maxnreg=None, pre_hook=None, ir_override=None
        )
        extra_pinned_keys.add((id(obj), k1))
        for d2 in dtypes:
            k2 = base_key + (d1, d2)
            obj.cache[k2] = triton.Config(
                {"BV": 32}, num_warps=2, num_stages=1, num_ctas=1, maxnreg=None, pre_hook=None, ir_override=None
            )
            extra_pinned_keys.add((id(obj), k2))

    def combined_pinned_run(instance, *args, **kwargs):
        if len(instance.configs) > 1:
            nargs = dict(zip(instance.arg_names, args))
            all_args = {**nargs, **kwargs}
            kept = {k: v for k, v in all_args.items() if k in instance.arg_names}
            key = tuple(
                [kept[k] for k in instance.keys if k in kept]
                + [str(v.dtype) for v in kept.values() if hasattr(v, "dtype")]
            )
            if (id(instance), key) in extra_pinned_keys:
                assert key in instance.cache, f"Missing pinned Triton key {key}"
                return orig_run(instance, *args, **kwargs)
        return guarded_run(instance, *args, **kwargs)

    Autotuner.run = combined_pinned_run
    install_flash_chunked_sdpa()
    return ev()


class StaticModelRunner:
    """Zero-generation forward scorer for a static pre-merged HF checkpoint."""

    def __init__(
        self,
        model_dir: Path,
        readout_path: Path | None = None,
        profile_path: Path | None = None,
        max_prefix_cache_entries: int = 8,
    ) -> None:
        torch.set_num_threads(8)
        torch.cuda.set_per_process_memory_fraction(CUDA_MEM_FRACTION)
        torch.manual_seed(7813)
        torch.cuda.manual_seed_all(7813)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        self.profile_evidence = install_pinned_kernels(profile_path)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
        cfg = AutoConfig.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
        self.text_config = cfg.get_text_config()
        self.native_context = int(self.text_config.max_position_embeddings)

        self.model = Qwen3_5ForCausalLM.from_pretrained(
            str(model_dir),
            config=self.text_config,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            local_files_only=True,
            trust_remote_code=False,
        )
        self.model.eval()
        self.model.requires_grad_(False)

        self.label_ids: list[int] = []
        for ch in string.ascii_uppercase:
            ids = self.tokenizer.encode(ch, add_special_tokens=False, truncation=False)
            assert len(ids) == 1
            self.label_ids.append(ids[0])

        if readout_path is not None and readout_path.is_file():
            ro = load_file(str(readout_path))
            self.W = ro["W"].float().cuda()
        else:
            self.W = self.model.lm_head.weight.detach()[self.label_ids].float().cuda()

        self.prefix_cache: OrderedDict[str, DynamicCache] = OrderedDict()
        self.max_prefix_cache_entries = max_prefix_cache_entries

    def pack(self, state, question: dict) -> dict:
        prompt, names, groups, temp = qwen_prompt(self, state, question)
        ids = self.tokenizer.encode(prompt, add_special_tokens=False, truncation=False)
        if not ids:
            raise ValueError("Encoded prompt cannot be empty")
        if len(ids) > self.native_context:
            raise OverflowError(
                f"Input length {len(ids)} exceeds native context {self.native_context}; silent truncation is forbidden"
            )
        return {
            "ids": ids,
            "names": names,
            "groups": groups,
            "temperature": temp,
            "input_ids_sha256": hashlib.sha256(("[" + ",".join(map(str, ids)) + "]").encode("ascii")).hexdigest(),
        }

    def forward_packed(
        self,
        state,
        question: dict,
        packed: dict,
        *,
        use_prefix_cache: bool = False,
        manifest_sha: str = "",
        role_label: str = "static",
    ) -> dict:
        ids = packed["ids"]
        if not ids or len(ids) > MAX_CONTEXT_TOKENS:
            raise OverflowError(
                f"Input length {len(ids)} exceeds max supported context {MAX_CONTEXT_TOKENS}; silent truncation is forbidden"
            )
        lm = self.model.model.language_model
        cache_hit = False
        cache_key = None
        t_fwd = time.perf_counter()

        with torch.inference_mode():
            prompt_text = None
            if use_prefix_cache and len(ids) <= 8192:
                prompt_text, _, _, _ = qwen_prompt(self, state, question)

            if use_prefix_cache and prompt_text is not None and "\nQUESTION:\n" in prompt_text:
                partial_ids = self.tokenizer.encode(
                    prompt_text.rsplit("\nQUESTION:\n", 1)[0],
                    add_special_tokens=False,
                    truncation=False,
                )
                n_pref = 0
                for tok_a, tok_b in zip(partial_ids, ids):
                    if tok_a != tok_b:
                        break
                    n_pref += 1
                if 0 < n_pref < len(ids) and n_pref <= CHUNK_SIZE and (len(ids) - n_pref) <= CHUNK_SIZE:
                    prefix_ids = ids[:n_pref]
                    tail_ids = ids[n_pref:]
                    raw_key = json.dumps({"m": manifest_sha, "r": role_label, "p": prefix_ids}, separators=(",", ":")).encode()
                    cache_key = hashlib.sha256(raw_key).hexdigest()
                    if cache_key in self.prefix_cache:
                        cache_hit = True
                        self.prefix_cache.move_to_end(cache_key)
                    else:
                        if len(self.prefix_cache) >= self.max_prefix_cache_entries:
                            self.prefix_cache.popitem(last=False)
                        pa = torch.tensor([prefix_ids], device="cuda", dtype=torch.long)
                        p_out = lm(input_ids=pa, attention_mask=None, use_cache=True, return_dict=True)
                        self.prefix_cache[cache_key] = p_out.past_key_values
                        del pa

                    past = clone_dynamic_cache(self.prefix_cache[cache_key], lm.config)
                    sfx = torch.tensor([tail_ids], device="cuda", dtype=torch.long)
                    out = lm(input_ids=sfx, attention_mask=None, past_key_values=past, use_cache=True, return_dict=True)
                    h = out.last_hidden_state[0, -1].float()
                    del sfx, past, out
                else:
                    use_prefix_cache = False
            else:
                use_prefix_cache = False

            if not use_prefix_cache:
                if len(ids) <= CHUNK_SIZE:
                    a = torch.tensor([ids], device="cuda", dtype=torch.long)
                    h = lm(input_ids=a, attention_mask=torch.ones_like(a), use_cache=False, return_dict=True).last_hidden_state[0, -1].float()
                    del a
                else:
                    past = make_preallocated_cache(lm.config, len(ids))
                    h = None
                    for c_start, c_end in chunk_spans(len(ids), CHUNK_SIZE):
                        c_ids = torch.tensor([ids[c_start:c_end]], device="cuda", dtype=torch.long)
                        c_out = lm(input_ids=c_ids, attention_mask=None, past_key_values=past, use_cache=True, return_dict=True)
                        if c_end == len(ids):
                            h = c_out.last_hidden_state[0, -1].float()
                        del c_ids, c_out
                        torch.cuda.synchronize()
                    del past

            z = torch.stack([torch.logsumexp(self.W[g] @ h, 0) for g in packed["groups"]]) / packed["temperature"]
            prob = z.softmax(-1)
            torch.cuda.synchronize()
            fwd_ms = (time.perf_counter() - t_fwd) * 1000.0
            assert torch.isfinite(z).all() and torch.isfinite(prob).all()
            zs = z.cpu().tolist()
            ps = prob.cpu().tolist()
            del h, z, prob

        return {
            "logits": zs,
            "probabilities": ps,
            "forward_ms": fwd_ms,
            "prefix_cache_used": use_prefix_cache,
            "prefix_cache_hit": cache_hit,
            "prefix_cache_key": cache_key,
        }

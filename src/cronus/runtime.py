"""Native Qwen3.5 zero-generation classification runtime."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import string

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from cronus import formatting as fmt
from cronus.paths import default_config_path

CUDA_MEM_FRACTION = float(os.environ.get("CUDA_MEM_FRACTION", "0.33"))
DEFAULT_TARGET_MODULES = ["q_proj", "v_proj", "o_proj", "up_proj", "down_proj"]


def sha(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def token_sha(ids: list[int]) -> str:
    """Compute canonical SHA-256 digest of a token ID sequence."""
    return hashlib.sha256(json.dumps(ids, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def base_language_model(model):
    """Unwrap PEFT wrapper if present to reach ``model.language_model``."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model.language_model


class Classifier:
    """Zero-generation Qwen3.5 classifier extracting 26-way uppercase letter readout logits."""

    def __init__(
        self,
        model_dir: str | Path,
        base_lock: str | Path | None = None,
        adapter: str | Path | None = None,
        training: bool = False,
        target_modules: list[str] | str | None = None,
    ) -> None:
        if not (0.0 < CUDA_MEM_FRACTION <= 0.35):
            raise RuntimeError(f"CUDA_MEM_FRACTION={CUDA_MEM_FRACTION} exceeds GB10 safety ceiling 0.35")
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(CUDA_MEM_FRACTION)

        lock_path = Path(base_lock) if base_lock is not None else default_config_path("base_model.lock.json")
        self.path = Path(model_dir).resolve()
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        expected_lock_sha = os.environ.get("CRONUS_EXPECTED_BASE_LOCK_SHA256", "").strip()
        if expected_lock_sha and sha(lock_path) != expected_lock_sha:
            raise RuntimeError("base lock SHA-256 mismatch")
        if lock.get("model") != "Qwen/Qwen3.5-9B":
            raise RuntimeError(f"unexpected base lock model: {lock.get('model')}")
        for name, expected_hash in lock["model_files"].items():
            p = self.path / name
            if p.is_symlink() or sha(p) != expected_hash:
                raise RuntimeError("base artifact mismatch: " + name)

        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True, trust_remote_code=False)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map={"": "cuda"},
            attn_implementation="sdpa",
        )
        self.model.requires_grad_(False)
        self.model.config.use_cache = False
        self.native_context = int(self.model.config.text_config.max_position_embeddings)
        assert self.native_context == int(lock["native_context_tokens"]) and not self.model.config.tie_word_embeddings

        self.label_ids: list[int] = []
        for c in string.ascii_uppercase:
            v = self.tokenizer.encode(c, add_special_tokens=False, truncation=False)
            if len(v) != 1:
                raise RuntimeError("option token is not single-token")
            self.label_ids.append(v[0])
        if len(set(self.label_ids)) != 26:
            raise RuntimeError("duplicate option tokens")

        text = self.model.model.language_model
        head = self.model.get_output_embeddings()
        vision = self.model.model.visual
        embed = self.model.get_input_embeddings()
        assert head.weight.dtype == torch.bfloat16 and not head.weight.requires_grad
        textptr = {p.untyped_storage().data_ptr() for p in list(text.parameters()) + list(text.buffers()) if p.numel()}
        unused = list(head.parameters()) + list(head.buffers()) + list(vision.parameters()) + list(vision.buffers())
        if any(p.untyped_storage().data_ptr() in textptr for p in unused if p.numel()):
            raise RuntimeError("cannot safely free tied head or vision storage")

        self.W = head.weight.detach().index_select(0, torch.tensor(self.label_ids, device=head.weight.device)).float()
        self.bias = (
            None
            if head.bias is None
            else head.bias.detach().index_select(0, torch.tensor(self.label_ids, device=head.bias.device)).float()
        )
        embed_id = id(embed.weight)
        embed_storage = embed.weight.untyped_storage().data_ptr()
        self.model.lm_head = None
        self.model.model.visual = None
        del unused, head, vision
        if (
            id(self.model.get_input_embeddings().weight) != embed_id
            or self.model.get_input_embeddings().weight.untyped_storage().data_ptr() != embed_storage
        ):
            raise RuntimeError("input embedding changed")

        if adapter:
            self.model = PeftModel.from_pretrained(self.model, str(adapter), is_trainable=training)
        elif training:
            mods = target_modules if target_modules is not None else DEFAULT_TARGET_MODULES
            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.1,
                    target_modules=mods,
                    bias="none",
                    task_type="CAUSAL_LM",
                ),
            )
        if training:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()
            self.model.train()
            params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
            if not params or any("lora_" not in n for n, _ in params):
                raise RuntimeError("non-LoRA trainable parameter")
        else:
            self.model.eval()

    def _native_chat(self, system: str, user: str) -> str:
        text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if text.endswith("<think>\n"):
            text += "\n</think>\n\n"
        return text

    def pack(self, state, question: dict) -> dict:
        """Render prompt and encode token IDs without silent truncation."""
        prompt, names, groups, temp = fmt.qwen_prompt(self, state, question)
        ids = self.tokenizer.encode(prompt, add_special_tokens=False, truncation=False)
        if not ids or len(ids) > self.native_context:
            raise ValueError("input exceeds native context; no truncation")
        return {
            "ids": ids,
            "names": names,
            "groups": groups,
            "temperature": temp,
            "input_ids_sha256": token_sha(ids),
        }

    def hidden_state(self, packed: dict) -> torch.Tensor:
        """Run a single no-cache forward pass and return the final hidden state in FP32."""
        ids = torch.tensor([packed["ids"]], device="cuda", dtype=torch.long)
        lm = base_language_model(self.model)
        return lm(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
            return_dict=True,
        ).last_hidden_state[0, -1].float()

    def logits_from_hidden(self, hidden: torch.Tensor, packed: dict) -> torch.Tensor:
        """Project final hidden state through frozen 26x4096 readout ``W0``."""
        z = torch.stack([torch.logsumexp(self.W[g] @ hidden, dim=0) for g in packed["groups"]])
        if self.bias is not None:
            z = z + self.bias[: len(z)]
        return z / packed["temperature"]

    def logits(self, packed: dict) -> torch.Tensor:
        """Compute zero-generation readout logits for a packed prompt."""
        h = self.hidden_state(packed)
        return self.logits_from_hidden(h, packed)

    def save_adapter(self, path: str | Path) -> None:
        """Save trainable LoRA weights without duplicating embedding layers."""
        self.model.save_pretrained(str(path), safe_serialization=True, save_embedding_layers=False)

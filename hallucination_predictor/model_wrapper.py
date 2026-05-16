"""
模型封装 —— 提取 LLM 内部状态用于约束残差分析。

两步架构：
1. forward pass: 提取输入 hidden states 和 attention（需要 attn_implementation='eager'）
2. generate: 生成文本（独立调用，不提取状态）
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time


@dataclass
class InternalState:
    """单次推理的完整内部状态"""
    input_ids: torch.Tensor
    tokens: List[str]
    hidden_states: torch.Tensor          # [seq_len, hidden_dim] — 最后一层
    layer_hidden_states: List[torch.Tensor]  # [num_layers, seq_len, hidden_dim]
    attention_weights: Optional[torch.Tensor]  # [heads, seq_len, seq_len] — 最后一层
    generated_text: str = ""
    inference_time_ms: float = 0.0
    hook_overhead_ms: float = 0.0


class ModelWrapper:
    """带内部状态提取的 LLM 封装"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.model_name = model_name
        self.device = device

        # float16 节省一半显存，7B 模型必需
        if dtype == torch.float32 and device in ("mps", "cuda"):
            dtype = torch.float16

        print(f"[ModelWrapper] Loading {model_name} ({dtype}) on {device}...")
        t0 = time.time()

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # SDPA 支持 output_attentions=True，且比 eager 更稳定
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        self.model.eval()

        self.hidden_dim = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads

        load_time = time.time() - t0
        print(f"[ModelWrapper] Loaded in {load_time:.1f}s")
        print(f"  Hidden dim: {self.hidden_dim}, Layers: {self.num_layers}, "
              f"Heads: {self.num_heads}")

        self._last_state: Optional[InternalState] = None

    def generate_and_extract(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> InternalState:
        t_start = time.time()

        # Chat template
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                formatted = prompt
        else:
            formatted = prompt

        inputs = self.tokenizer(
            formatted, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        input_len = inputs.input_ids.shape[1]

        # ---- Step 1: 前向传播提取内部状态 ----
        t_fwd = time.time()
        with torch.no_grad():
            fwd_outputs = self.model(
                **inputs,
                output_hidden_states=True,
                output_attentions=True,
            )
        fwd_time = (time.time() - t_fwd) * 1000

        # Hidden states
        all_hidden = fwd_outputs.hidden_states  # tuple of (batch, seq, dim)
        layer_hidden = [
            all_hidden[l][0].cpu()
            for l in range(len(all_hidden))
        ]
        last_hidden = layer_hidden[-1]

        # Attention (last layer, first batch)
        if fwd_outputs.attentions is not None and len(fwd_outputs.attentions) > 0:
            last_attn = fwd_outputs.attentions[-1][0].cpu()  # [heads, seq, seq]
        else:
            last_attn = None

        # ---- Step 2: 正常生成文本 ----
        t_gen = time.time()
        with torch.no_grad():
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=(temperature if do_sample else None),
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen_time = (time.time() - t_gen) * 1000

        generated_ids = gen_ids[0][input_len:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        all_ids = inputs.input_ids[0].cpu().tolist()
        tokens = [self.tokenizer.decode([tid]) for tid in all_ids]

        total_time = (time.time() - t_start) * 1000

        state = InternalState(
            input_ids=inputs.input_ids[0].cpu(),
            tokens=tokens,
            hidden_states=last_hidden,
            layer_hidden_states=layer_hidden,
            attention_weights=last_attn,
            generated_text=generated_text.strip(),
            inference_time_ms=total_time,
            hook_overhead_ms=fwd_time,
        )
        self._last_state = state
        return state

    @property
    def last_state(self) -> Optional[InternalState]:
        return self._last_state

    def extract_output_state(self, text: str) -> InternalState:
        """对任意文本做前向传播，提取内部状态（用于分析生成输出）"""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        ).to(self.device)

        t0 = time.time()
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                output_attentions=True,
            )
        elapsed = (time.time() - t0) * 1000

        all_hidden = outputs.hidden_states
        layer_hidden = [all_hidden[l][0].cpu() for l in range(len(all_hidden))]
        last_hidden = layer_hidden[-1]

        if outputs.attentions is not None and len(outputs.attentions) > 0:
            last_attn = outputs.attentions[-1][0].cpu()
        else:
            last_attn = None

        all_ids = inputs.input_ids[0].cpu().tolist()
        tokens = [self.tokenizer.decode([tid]) for tid in all_ids]

        return InternalState(
            input_ids=inputs.input_ids[0].cpu(),
            tokens=tokens,
            hidden_states=last_hidden,
            layer_hidden_states=layer_hidden,
            attention_weights=last_attn,
            generated_text="",
            inference_time_ms=elapsed,
            hook_overhead_ms=elapsed,
        )

"""ELF transformer model."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from modules.layers import (
    Attention, BottleneckTextProj, FinalLayer, RMSNorm, SwiGLUFFN,
    TextRotaryEmbeddingFast, TimestepEmbedder,
    DEFAULT_KERNEL_INIT, DEFAULT_BIAS_INIT, NORMAL_INIT_002, ZERO_INIT,
    _make_linear, scaled_dot_product_attention,
)


class ZeroInitBrainCrossAttention(nn.Module):
    """Dedicated target-to-condition attention with an exact zero residual at init."""

    # Checkpoint loading uses this marker to permit initialization from an
    # ordinary ELF state dict that predates the optional adapters.
    _is_zero_init_brain_cross_attention = True

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}."
            )
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.attn_drop = float(attn_drop)
        self.proj_drop = float(proj_drop)
        head_dim = self.hidden_size // self.num_heads
        self.query_norm = RMSNorm(self.hidden_size, eps=1e-6)
        self.memory_norm = RMSNorm(self.hidden_size, eps=1e-6)
        # Bias-free K/V projections keep a zeroed classifier-free condition an
        # exact zero-memory path even after the adapter has trained.
        self.q = _make_linear(self.hidden_size, self.hidden_size, bias=False)
        self.k = _make_linear(self.hidden_size, self.hidden_size, bias=False)
        self.v = _make_linear(self.hidden_size, self.hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)
        self.proj = _make_linear(
            self.hidden_size,
            self.hidden_size,
            # Bias-free keeps a zero/CFG memory an exact no-op throughout
            # training, not merely at initialization.
            bias=False,
            kernel_init=ZERO_INIT,
        )

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        if target.ndim != 3 or memory.ndim != 3:
            raise ValueError(
                "Brain cross-attention expects target and memory shaped [B, S, H]."
            )
        if target.shape[0] != memory.shape[0]:
            raise ValueError("Target and brain-memory batch sizes must match.")
        if target.shape[-1] != self.hidden_size or memory.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected hidden width {self.hidden_size}, got "
                f"target={target.shape[-1]} memory={memory.shape[-1]}."
            )

        batch_size, target_length, _ = target.shape
        memory_length = memory.shape[1]
        head_dim = self.hidden_size // self.num_heads
        query = self.q(self.query_norm(target)).reshape(
            batch_size, target_length, self.num_heads, head_dim
        ).transpose(1, 2)
        normalized_memory = self.memory_norm(memory)
        key = self.k(normalized_memory).reshape(
            batch_size, memory_length, self.num_heads, head_dim
        ).transpose(1, 2)
        value = self.v(normalized_memory).reshape(
            batch_size, memory_length, self.num_heads, head_dim
        ).transpose(1, 2)
        query = self.q_norm(query)
        key = self.k_norm(key)
        attended = scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=memory_mask,
        )
        attended = attended.transpose(1, 2).reshape(
            batch_size, target_length, self.hidden_size
        )
        if self.attn_drop > 0.0:
            attended = F.dropout(
                attended,
                p=self.attn_drop,
                training=not deterministic,
            )
        residual = self.proj(attended)
        if self.proj_drop > 0.0:
            residual = F.dropout(
                residual,
                p=self.proj_drop,
                training=not deterministic,
            )
        return target + residual


class ELFBlock(nn.Module):
    """ELF Transformer block."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads, qkv_bias=True, qk_norm=True,
            attn_drop=attn_drop, proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

    def forward(self, x: torch.Tensor, rope_fn: Optional[nn.Module] = None,
                attention_mask: Optional[torch.Tensor] = None,
                deterministic: bool = True) -> torch.Tensor:
        x_normed = self.norm1(x)
        attn_out = self.attn(x_normed, rope_fn, attention_mask=attention_mask,
                             deterministic=deterministic)
        x = x + attn_out

        x_normed = self.norm2(x)
        mlp_out = self.mlp(x_normed, deterministic=deterministic)
        x = x + mlp_out
        return x


class ELF(nn.Module):
    """Text ELF Transformer."""

    def __init__(
        self,
        text_encoder_dim: int,
        max_length: int,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        bottleneck_dim: int = 128,
        num_time_tokens: int = 4,
        num_self_cond_cfg_tokens: int = 4,
        num_model_mode_tokens: int = 0,
        vocab_size: int = 0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.text_encoder_dim = text_encoder_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.bottleneck_dim = bottleneck_dim
        self.num_time_tokens = num_time_tokens
        self.num_self_cond_cfg_tokens = num_self_cond_cfg_tokens
        self.num_model_mode_tokens = num_model_mode_tokens
        self.vocab_size = vocab_size
        self.gradient_checkpointing = gradient_checkpointing
        self.brain_condition_length = 0
        self.brain_cross_attention = nn.ModuleDict()
        self.brain_cross_attention_decoder_only = False

        # Self-conditioning input projection (only used when input is [z, x_pred]).
        self.self_cond_proj = _make_linear(2 * text_encoder_dim, text_encoder_dim, bias=True)

        # Text bottleneck projection.
        self.text_proj = BottleneckTextProj(text_encoder_dim, hidden_size, bottleneck_dim)

        # Time / SC-CFG embedders + learned prefix tokens.
        if num_time_tokens <= 0:
            raise ValueError("num_time_tokens must be positive for prefix time conditioning")
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_emb_tokens = nn.Parameter(torch.empty(1, num_time_tokens, hidden_size))
        NORMAL_INIT_002(self.t_emb_tokens)

        if num_self_cond_cfg_tokens > 0:
            self.self_cond_cfg_embedder = TimestepEmbedder(hidden_size)
            self.self_cond_cfg_tokens = nn.Parameter(torch.empty(1, num_self_cond_cfg_tokens, hidden_size))
            NORMAL_INIT_002(self.self_cond_cfg_tokens)

        if num_model_mode_tokens > 0:
            self.mode_tokens = nn.Parameter(torch.empty(1, num_model_mode_tokens, hidden_size))
            NORMAL_INIT_002(self.mode_tokens)

        head_dim = hidden_size // num_heads
        prefix_total = num_model_mode_tokens + num_time_tokens
        if num_self_cond_cfg_tokens > 0:
            prefix_total += num_self_cond_cfg_tokens
        self.feat_rope = TextRotaryEmbeddingFast(
            dim=head_dim, pt_seq_len=max_length, num_empty_token=prefix_total,
        )

        self.blocks = nn.ModuleList()
        q1, q3 = depth // 4, depth // 4 * 3
        for i in range(depth):
            in_drop_range = q3 > i >= q1
            self.blocks.append(ELFBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if in_drop_range else 0.0,
                proj_drop=proj_drop if in_drop_range else 0.0,
            ))

        # Final flow-matching output head.
        self.final_layer = FinalLayer(hidden_size, patch_size=1, out_channels=text_encoder_dim)

        # Factored decoder unembedding: hidden -> text_encoder_dim -> vocab.
        bn = text_encoder_dim
        self.proj_kernel = nn.Parameter(torch.empty(hidden_size, bn))
        self.proj_bias = nn.Parameter(torch.empty(bn))
        self.unembed_kernel = nn.Parameter(torch.empty(bn, vocab_size))
        self.unembed_bias = nn.Parameter(torch.empty(vocab_size))
        DEFAULT_KERNEL_INIT(self.proj_kernel)
        DEFAULT_BIAS_INIT(self.proj_bias)
        DEFAULT_KERNEL_INIT(self.unembed_kernel)
        DEFAULT_BIAS_INIT(self.unembed_bias)

    def enable_brain_cross_attention(
        self,
        *,
        condition_length: int,
        last_n_blocks: int = 4,
        num_heads: Optional[int] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        decoder_only: bool = False,
    ) -> list[int]:
        """Attach zero-init cross-attention adapters to the final ELF blocks.

        The clean first ``condition_length`` sequence positions serve as
        memory. Only subsequent target positions receive the new residual.
        Enabling this on a pretrained model is therefore exactly output
        preserving until the new projection weights begin to train.
        """

        if condition_length <= 0:
            raise ValueError("condition_length must be positive.")
        if last_n_blocks <= 0 or last_n_blocks > self.depth:
            raise ValueError(
                f"last_n_blocks must be in [1, {self.depth}], got {last_n_blocks}."
            )
        use_heads = self.num_heads if num_heads is None else int(num_heads)
        if self.hidden_size % use_heads:
            raise ValueError(
                f"hidden_size={self.hidden_size} must be divisible by num_heads={use_heads}."
            )
        self.brain_condition_length = int(condition_length)
        self.brain_cross_attention_decoder_only = bool(decoder_only)
        indices = list(range(self.depth - int(last_n_blocks), self.depth))
        reference_parameter = next(self.parameters())
        self.brain_cross_attention = nn.ModuleDict({
            str(index): ZeroInitBrainCrossAttention(
                self.hidden_size,
                use_heads,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
            for index in indices
        }).to(
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
        )
        return indices

    def build_context(self, t: torch.Tensor,
                      self_cond_cfg_scale: Optional[torch.Tensor] = None) -> list:
        B = t.shape[0]
        prefix_tokens = []

        time_emb = self.t_embedder(t)  # (B, hidden)
        prefix_tokens.append(
            self.t_emb_tokens.expand(B, -1, -1) + time_emb.unsqueeze(1)
        )

        if self_cond_cfg_scale is not None and self.num_self_cond_cfg_tokens > 0:
            sc_emb = self.self_cond_cfg_embedder(self_cond_cfg_scale)
            prefix_tokens.append(
                self.self_cond_cfg_tokens.expand(B, -1, -1) + sc_emb.unsqueeze(1)
            )
        return prefix_tokens

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        self_cond_cfg_scale: Optional[torch.Tensor] = None,
        decoder_step_active: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """x: (N, S, C) or (N, S, 2C) with self-cond. t: (N,). attention_mask: (N, S), 1=valid."""
        B = x.shape[0]

        decoder_cross_active = not self.brain_cross_attention_decoder_only
        decoder_cross_gate = None
        if self.brain_cross_attention_decoder_only:
            if isinstance(decoder_step_active, torch.Tensor):
                decoder_cross_active = True
                decoder_cross_gate = decoder_step_active.to(dtype=x.dtype).reshape(B, 1, 1)
            else:
                decoder_cross_active = bool(decoder_step_active)

        brain_memory = None
        brain_memory_mask = None
        if self.brain_cross_attention and decoder_cross_active:
            if x.shape[1] <= self.brain_condition_length:
                raise ValueError(
                    "Brain-conditioned ELF input must contain condition and target tokens: "
                    f"sequence={x.shape[1]} condition={self.brain_condition_length}."
                )
            # In a self-conditioned input, the first half is the current noisy
            # sequence and contains the restored clean condition prefix.
            memory_source = (
                x[..., : self.text_encoder_dim]
                if x.shape[-1] == 2 * self.text_encoder_dim
                else x
            )
            raw_memory = memory_source[:, : self.brain_condition_length]
            # CFG and condition dropout represent the unconditional condition
            # using an exactly zero prefix. Remove BottleneckTextProj's bias in
            # that case so the new path is also exactly unconditional.
            memory_present = raw_memory.detach().abs().sum(dim=(1, 2)).gt(0).to(
                dtype=raw_memory.dtype
            ).reshape(B, 1, 1)
            with torch.amp.autocast('cuda', enabled=False):
                brain_memory = self.text_proj(raw_memory.float())
            brain_memory = brain_memory * memory_present.to(brain_memory)
            if attention_mask is not None:
                brain_memory_mask = attention_mask[:, : self.brain_condition_length]

        # Self-conditioning: input is [z, x_pred] when 2x encoder dim
        with torch.amp.autocast('cuda', enabled=False):
            if x.shape[-1] == 2 * self.text_encoder_dim:
                x = self.self_cond_proj(x.float())
            x = self.text_proj(x.float())
            context_prefix_tokens = self.build_context(t, self_cond_cfg_scale)

        # Prepend learnable model-mode tokens (gated by decoder_step_active).
        # decoder_step_active may be None / Python bool / (B,) tensor — the last
        # form supports per-example branching at training time.
        model_mode_offset = 0
        if self.num_model_mode_tokens > 0:
            mode_tokens = self.mode_tokens.expand(B, -1, -1)
            if decoder_step_active is None:
                active_gate = 0.0
            elif isinstance(decoder_step_active, torch.Tensor) and decoder_step_active.dim() > 0:
                active_gate = decoder_step_active.to(mode_tokens.dtype).view(-1, 1, 1)
            else:
                active_gate = float(decoder_step_active)
            mode_tokens = mode_tokens * active_gate
            x = torch.cat([mode_tokens, x], dim=1)
            model_mode_offset = self.num_model_mode_tokens
            if attention_mask is not None:
                mode_mask = torch.ones((B, self.num_model_mode_tokens),
                                       dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([mode_mask, attention_mask], dim=1)

        prefix_len = 0
        if context_prefix_tokens:
            prefix_tokens = torch.cat(context_prefix_tokens, dim=1)
            prefix_len = prefix_tokens.shape[1]
            x = torch.cat([prefix_tokens, x], dim=1)
            if attention_mask is not None:
                prefix_mask = torch.ones((B, prefix_len),
                                         dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        use_checkpoint = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        target_start = prefix_len + model_mode_offset + self.brain_condition_length
        for block_index, block in enumerate(self.blocks):
            if use_checkpoint:
                def _block_forward(hidden: torch.Tensor, block: ELFBlock = block) -> torch.Tensor:
                    return block(hidden, rope_fn=self.feat_rope, attention_mask=attention_mask,
                                 deterministic=deterministic)

                x = checkpoint(_block_forward, x, use_reentrant=False)
            else:
                x = block(x, rope_fn=self.feat_rope, attention_mask=attention_mask,
                          deterministic=deterministic)
            adapter_key = str(block_index)
            if adapter_key in self.brain_cross_attention and decoder_cross_active:
                original_target = x[:, target_start:]
                target = self.brain_cross_attention[adapter_key](
                    original_target,
                    brain_memory,
                    memory_mask=brain_memory_mask,
                    deterministic=deterministic,
                )
                if decoder_cross_gate is not None:
                    gate = decoder_cross_gate.to(device=target.device, dtype=target.dtype)
                    target = original_target + gate * (target - original_target)
                x = torch.cat([x[:, :target_start], target], dim=1)

        x = x[:, prefix_len + model_mode_offset:]

        # Factored decoder unembedding: hidden -> text_encoder_dim -> vocab
        with torch.amp.autocast('cuda', enabled=False):
            decoder_logits = None
            if decoder_step_active is not None:
                x_f32 = x.float()
                hidden = F.gelu(x_f32 @ self.proj_kernel + self.proj_bias, approximate="tanh")
                decoder_logits = hidden @ self.unembed_kernel + self.unembed_bias
            output = self.final_layer(x.float())
        return output, decoder_logits


# Model factory functions
def ELF_B(**kwargs): return ELF(depth=12, hidden_size=768,  num_heads=12, **kwargs)
def ELF_M(**kwargs): return ELF(depth=24, hidden_size=1056, num_heads=16, **kwargs)
def ELF_L(**kwargs): return ELF(depth=32, hidden_size=1280, num_heads=16, **kwargs)

ELF_models = {
    'ELF-B': ELF_B, 'ELF-M': ELF_M, 'ELF-L': ELF_L,
}

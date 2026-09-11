from typing import Optional

import torch
import torch.nn as nn

from configs.config import Config, SamplingConfig
from utils.sampling_utils import restore_cond, _ode_step, _sde_step


# ============================================
# Generation utilities
# ============================================

def mask_after_eos(predicted_ids: torch.Tensor, eos_token_id: int, pad_token_id: int) -> torch.Tensor:
    """Mask everything at/after first EOS token per sequence."""
    eos_mask = (predicted_ids == eos_token_id)
    keep_mask = (eos_mask.to(torch.int32).cumsum(dim=1) == 0)
    return torch.where(keep_mask, predicted_ids, torch.full_like(predicted_ids, pad_token_id))


def shift_left(x: torch.Tensor, shift_per_sample: torch.Tensor, pad_value=0, axis: int = 1) -> torch.Tensor:
    """Shift each sample left along the sequence axis; pad emptied positions."""
    if x.dim() < 2:
        raise ValueError("x must have at least batch and sequence dimensions")
    if axis < 0:
        axis = x.dim() + axis
    if axis == 0:
        raise ValueError("axis=0 is the batch axis and cannot be shifted")
    shift_per_sample = shift_per_sample.to(torch.long)
    if axis != 1:
        x = x.movedim(axis, 1)
    seq_len = x.shape[1]
    base_idx = torch.arange(seq_len, device=x.device)[None, :]
    gather_idx = shift_per_sample[:, None].to(x.device) + base_idx
    valid = gather_idx < seq_len
    gather_idx = gather_idx.clamp(0, seq_len - 1)
    if x.dim() == 2:
        shifted = torch.gather(x, 1, gather_idx)
        shifted = torch.where(valid, shifted, torch.full_like(shifted, pad_value))
    else:
        expand_shape = [-1, -1] + list(x.shape[2:])
        idx = gather_idx.view(*gather_idx.shape, *([1] * (x.dim() - 2))).expand(*expand_shape)
        valid_b = valid.view(*valid.shape, *([1] * (x.dim() - 2))).expand(*expand_shape)
        shifted = torch.gather(x, 1, idx)
        shifted = torch.where(valid_b, shifted, torch.full_like(shifted, pad_value))
    if axis != 1:
        shifted = shifted.movedim(1, axis)
    return shifted


# ============================================
# Single-batch sampling (PyTorch)
# ============================================

@torch.no_grad()
def _generate_samples_single_batch(
    model: nn.Module,
    generator: torch.Generator,
    z: torch.Tensor,
    t_steps: torch.Tensor,
    cond_seq: Optional[torch.Tensor],
    cond_seq_mask: Optional[torch.Tensor],
    config: Config,
    sampling_config: SamplingConfig,
    cfg_scale: float,
    self_cond_cfg_scale: float,
) -> torch.Tensor:
    """Generate samples for a single batch (PyTorch Euler / SDE rollout)."""
    method = sampling_config.sampling_method
    batch_size, max_length, d_model = z.shape
    if cond_seq is None:
        cond_seq = torch.zeros((batch_size, max_length, d_model), dtype=z.dtype, device=z.device)
        cond_seq_mask = torch.zeros((batch_size, max_length), dtype=z.dtype, device=z.device)

    step_kwargs = dict(
        model=model, config=config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )

    z = restore_cond(z, cond_seq, cond_seq_mask)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)

    n = t_steps.shape[0]
    sde_gamma = getattr(sampling_config, "sde_gamma", 0.0)

    use_bf16 = bool(getattr(config, "use_bf16", True)) and z.is_cuda
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        for i in range(n - 2):
            t = t_steps[i].item()
            t_next = t_steps[i + 1].item()
            if method == "sde":
                z, x_pred = _sde_step(
                    z=z, t=t, t_next=t_next, x_pred_prev=x_pred,
                    gamma=sde_gamma, generator=generator, **step_kwargs,
                )
            elif method == "ode":
                z, x_pred = _ode_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **step_kwargs)
            else:
                raise ValueError(f"Invalid sampling method: {method}")

        # Last step always with ODE.
        t = t_steps[-2].item()
        t_next = t_steps[-1].item()
        z, x_pred = _ode_step(z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **step_kwargs)
    return z


@torch.no_grad()
def _apply_token_logit_bias(
    decoder_logits: torch.Tensor,
    token_logit_bias: torch.Tensor,
    *,
    once: bool = False,
    target_start: int = 0,
    max_positions: int = 0,
) -> torch.Tensor:
    """Apply static lexical evidence, optionally once at compatible positions."""

    if token_logit_bias.ndim != 2 or token_logit_bias.shape != (
        decoder_logits.shape[0], decoder_logits.shape[-1]
    ):
        raise ValueError(
            "token_logit_bias must have shape "
            f"[{decoder_logits.shape[0]}, {decoder_logits.shape[-1]}], "
            f"got {tuple(token_logit_bias.shape)}"
        )
    bias = token_logit_bias.to(
        device=decoder_logits.device, dtype=decoder_logits.dtype
    )
    if not once:
        return decoder_logits + bias[:, None, :]
    if not 0 <= target_start < decoder_logits.shape[1]:
        raise ValueError(f"Invalid lexical target_start={target_start}")
    target_stop = decoder_logits.shape[1]
    if max_positions > 0:
        target_stop = min(target_stop, target_start + int(max_positions))
    if target_stop <= target_start:
        raise ValueError("Lexical max_positions leaves no target positions")

    result = decoder_logits.clone()
    for row in range(result.shape[0]):
        candidate_ids = torch.nonzero(bias[row] > 0, as_tuple=False).flatten()
        if candidate_ids.numel() == 0:
            continue
        candidate_ids = candidate_ids[
            torch.argsort(bias[row, candidate_ids], descending=True)
        ]
        target_logits = result[row, target_start:target_stop]
        available = torch.ones(
            target_logits.shape[0], dtype=torch.bool, device=result.device
        )
        base_best = target_logits.max(dim=-1).values
        for token_id in candidate_ids:
            if not bool(available.any()):
                break
            compatibility = target_logits[:, token_id] - base_best
            position = compatibility.masked_fill(~available, -torch.inf).argmax()
            absolute_position = target_start + int(position)
            result[row, absolute_position, token_id] += bias[row, token_id]
            available[position] = False
    return result


@torch.no_grad()
def _apply_ordered_token_sequence_bias(
    decoder_logits: torch.Tensor,
    token_ids: torch.Tensor,
    sequence_bias: torch.Tensor,
    *,
    target_start: int,
    max_positions: int = 0,
    blocked_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Bias an ordered word sequence into consecutive early decoder slots.

    ``token_ids`` is ``[batch, word_position, word_piece]``.  Each predicted
    word is laid out after the preceding word, preserving the ordered-head
    sequence while leaving the bias soft: ELF may still choose another token
    whenever its decoder evidence is stronger.
    """

    if token_ids.ndim != 3:
        raise ValueError("ordered token_ids must be [batch, position, piece]")
    if tuple(sequence_bias.shape) != tuple(token_ids.shape[:2]):
        raise ValueError("ordered sequence_bias must match batch and position axes")
    if token_ids.shape[0] != decoder_logits.shape[0]:
        raise ValueError("ordered token_ids batch does not match decoder logits")
    if blocked_positions is not None and tuple(blocked_positions.shape) != tuple(
        decoder_logits.shape[:2]
    ):
        raise ValueError("blocked_positions must match decoder batch and sequence axes")
    if not 0 <= target_start < decoder_logits.shape[1]:
        raise ValueError(f"Invalid ordered target_start={target_start}")
    target_stop = decoder_logits.shape[1]
    if max_positions > 0:
        target_stop = min(target_stop, target_start + int(max_positions))

    result = decoder_logits.clone()
    for row in range(result.shape[0]):
        cursor = target_start
        for position in range(token_ids.shape[1]):
            pieces = token_ids[row, position]
            pieces = pieces[
                (pieces >= 0) & (pieces < decoder_logits.shape[-1])
            ]
            piece_count = int(pieces.numel())
            if piece_count == 0:
                continue
            if blocked_positions is not None:
                while cursor + piece_count <= target_stop and bool(
                    blocked_positions[row, cursor : cursor + piece_count].any()
                ):
                    cursor += 1
            if cursor + piece_count > target_stop:
                break
            for offset, token_id in enumerate(pieces):
                result[row, cursor + offset, token_id] += sequence_bias[
                    row, position
                ].to(result.dtype)
            cursor += piece_count
    return result


@torch.no_grad()
def _apply_token_sequence_bias(
    decoder_logits: torch.Tensor,
    token_ids: torch.Tensor,
    sequence_bias: torch.Tensor,
    *,
    target_start: int,
    max_positions: int = 0,
) -> torch.Tensor:
    """Place each lexical word's pieces contiguously in a distinct best span."""

    expected_ids = (decoder_logits.shape[0], sequence_bias.shape[1], token_ids.shape[2])
    if tuple(token_ids.shape) != expected_ids:
        raise ValueError(
            f"token_ids must have shape {expected_ids}, got {tuple(token_ids.shape)}"
        )
    if tuple(sequence_bias.shape) != token_ids.shape[:2]:
        raise ValueError("sequence_bias must match token_ids batch and candidate axes")
    if not 0 <= target_start < decoder_logits.shape[1]:
        raise ValueError(f"Invalid lexical target_start={target_start}")

    result = decoder_logits.clone()
    target_stop = result.shape[1]
    if max_positions > 0:
        target_stop = min(target_stop, target_start + int(max_positions))
    target_length = target_stop - target_start
    if target_length <= 0:
        raise ValueError("Lexical sequence max_positions leaves no target positions")
    for row in range(result.shape[0]):
        order = torch.argsort(sequence_bias[row], descending=True)
        available = torch.ones(target_length, dtype=torch.bool, device=result.device)
        for rank in order:
            pieces = token_ids[row, rank]
            pieces = pieces[pieces >= 0]
            piece_count = int(pieces.numel())
            if piece_count == 0 or piece_count > target_length:
                continue
            best_score = None
            best_start = None
            for start in range(target_length - piece_count + 1):
                if not bool(available[start : start + piece_count].all()):
                    continue
                span = result[row, target_start + start : target_start + start + piece_count]
                best = span.max(dim=-1).values
                piece_logits = span.gather(1, pieces[:, None]).squeeze(1)
                score = (piece_logits - best).mean()
                if best_score is None or bool(score > best_score):
                    best_score = score
                    best_start = start
            if best_start is None:
                continue
            for offset, token_id in enumerate(pieces):
                result[row, target_start + best_start + offset, token_id] += sequence_bias[
                    row, rank
                ].to(result.dtype)
            available[best_start : best_start + piece_count] = False
    return result


@torch.no_grad()
def _dlm_decode_batch(z: torch.Tensor, model: nn.Module, t_final_val,
                      config, self_cond_cfg_scale: float,
                      token_logit_bias: torch.Tensor | None = None,
                      token_logit_bias_once: bool = False,
                      token_logit_bias_target_start: int = 0,
                      token_logit_bias_max_positions: int = 0,
                      token_sequence_ids: torch.Tensor | None = None,
                      token_sequence_bias: torch.Tensor | None = None,
                      ordered_token_ids: torch.Tensor | None = None,
                      ordered_token_bias: torch.Tensor | None = None,
                      ordered_token_max_positions: int = 0) -> torch.Tensor:
    """Decode z -> tokens with the DLM decoder head."""
    batch_size = z.shape[0]
    if isinstance(t_final_val, torch.Tensor) and t_final_val.dim() == 0:
        t_final = torch.full((batch_size,), t_final_val.item(), dtype=z.dtype, device=z.device)
    else:
        t_final = torch.full((batch_size,), float(t_final_val), dtype=z.dtype, device=z.device)
    sc_batch = (
        torch.full((batch_size,), float(self_cond_cfg_scale), dtype=z.dtype, device=z.device)
        if config.num_self_cond_cfg_tokens > 0 else None
    )
    z_input = torch.cat([z, torch.zeros_like(z)], dim=-1) if config.self_cond_prob > 0 else z
    use_bf16 = bool(getattr(config, "use_bf16", True)) and z.is_cuda
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        _, decoder_logits = model(
            z_input, t_final, deterministic=True,
            self_cond_cfg_scale=sc_batch,
            decoder_step_active=True,
        )
    ordered_blocked_positions = None
    if token_logit_bias is not None:
        before_lexical_bias = decoder_logits
        decoder_logits = _apply_token_logit_bias(
            decoder_logits,
            token_logit_bias,
            once=token_logit_bias_once,
            target_start=token_logit_bias_target_start,
            max_positions=token_logit_bias_max_positions,
        )
        if token_logit_bias_once:
            ordered_blocked_positions = (decoder_logits != before_lexical_bias).any(
                dim=-1
            )
    if token_sequence_ids is not None or token_sequence_bias is not None:
        if token_sequence_ids is None or token_sequence_bias is None:
            raise ValueError("token_sequence_ids and token_sequence_bias must be provided together")
        before_sequence_bias = decoder_logits
        decoder_logits = _apply_token_sequence_bias(
            decoder_logits,
            token_sequence_ids,
            token_sequence_bias,
            target_start=token_logit_bias_target_start,
            max_positions=token_logit_bias_max_positions,
        )
        sequence_blocked = (decoder_logits != before_sequence_bias).any(dim=-1)
        ordered_blocked_positions = (
            sequence_blocked
            if ordered_blocked_positions is None
            else ordered_blocked_positions | sequence_blocked
        )
    if ordered_token_ids is not None or ordered_token_bias is not None:
        if ordered_token_ids is None or ordered_token_bias is None:
            raise ValueError("ordered_token_ids and ordered_token_bias must be provided together")
        decoder_logits = _apply_ordered_token_sequence_bias(
            decoder_logits,
            ordered_token_ids,
            ordered_token_bias,
            target_start=token_logit_bias_target_start,
            max_positions=ordered_token_max_positions,
            blocked_positions=ordered_blocked_positions,
        )
    return decoder_logits.argmax(dim=-1)


def _build_run_name(sampling_method, num_sampling_steps, cfg_scale, self_cond_cfg_scale,
                    time_schedule, sde_gamma, suffix):
    ts_str = f"-ts_{time_schedule}"
    sccfg_str = f"-sccfg{self_cond_cfg_scale}" if self_cond_cfg_scale != 1.0 else ""
    sde_str = f"-gamma{sde_gamma}" if sampling_method == "sde" else ""
    return f"{sampling_method}-steps{num_sampling_steps}-cfg{cfg_scale}{sccfg_str}{ts_str}{sde_str}-{suffix}"

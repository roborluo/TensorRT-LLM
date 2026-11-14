# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in K3.1 checkpoint conversion at load time.

``TLLM_K31_LOAD_QUANTIZE=1`` selects NVFP4 experts and block-scaled FP8 attention.
``TLLM_K31_REQUANT_ATTN_FP8=1`` keeps native NVFP4 experts and requantizes MXFP8/BF16
attention to block-scaled FP8 (see ``K31AttentionRequant``).
The adapter owns source-format handling; runtime modules use the native loaders.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import torch
import torch.nn.functional as F

from ...logger import logger
from ...models.modeling_utils import QuantConfig
from ...quantization import QuantAlgo
from ..model_config import ModelConfig
from .checkpoints.base_weight_loader import ConsumableWeightsDict

if TYPE_CHECKING:
    from .modeling_kimi_linear import KimiLinearForCausalLM

_SWITCH = "TLLM_K31_LOAD_QUANTIZE"
_DEQUANT_MAX_CHUNK_ELEMENTS = 1 << 22
_KDA_PROJECTIONS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "g_proj",
    "o_proj",
    "f_a_proj",
    "f_b_proj",
    "b_proj",
)
_MLA_PROJECTIONS = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "g_proj", "o_proj")
_REQUANT_SWITCH = "TLLM_K31_REQUANT_ATTN_FP8"
# Per-layer FP8_PB_WO declarations of nvidia/Kimi-K3-NVFP4 (fused aliases included).
_KDA_FP8_DECLARATIONS = _KDA_PROJECTIONS + ("fused_qkvg_proj", "in_proj_qkvgfab")
_MLA_FP8_DECLARATIONS = _MLA_PROJECTIONS + (
    "fused_qkv_a_proj",
    "fused_qkv_a_proj_with_mqa",
    "fused_qkvg_proj",
    "in_proj_qkvgfab",
)
# Shared experts stay BF16 (the runtime forces it); f_b_proj joins the FP8 attention set.
_REQUANT_EXCLUDES_DROP = ("*self_attn.f_b_proj*",)
_REQUANT_EXCLUDES_ADD = ("*block_sparse_moe.shared_experts*", "*mlp.gate*", "*mlp.shared_experts*")
# KDA projections that run as plain BF16 nn.Linear. An MXFP8 source is decoded directly by the
# trunk loader instead of taking an extra FP8 128x128 rounding on the way to BF16.
_KDA_BF16_RUNTIME_PROJECTIONS = ("b_proj", "f_a_proj")
_REQUANT_ALLOW_CPU = "TLLM_K31_REQUANT_ALLOW_CPU"


class _TensorSlice(Protocol):
    def get_shape(self) -> list[int]: ...

    def __getitem__(self, index: object) -> torch.Tensor: ...


def dequantize_mxfp4_weight(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Decode uint8 E2M1 [N,K/2] and E8M0 [N,K/32] into BF16 [N,K]."""
    if packed.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise TypeError("MXFP4 conversion requires uint8 packed weights and E8M0 scales")
    if packed.ndim != 2 or scales.ndim != 2 or packed.numel() == 0 or packed.shape[1] % 16:
        raise ValueError("MXFP4 conversion requires a matrix with K divisible by 32")
    if tuple(scales.shape) != (packed.shape[0], packed.shape[1] // 16):
        raise ValueError("MXFP4 scale shape does not cover the packed weight")
    if packed.device != scales.device:
        raise ValueError("MXFP4 weights and scales must be on the same device")
    if (scales == 255).any():
        raise ValueError("MXFP4 E8M0 scales contain NaN (255)")
    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
        device=packed.device,
    )
    rows, cols = packed.shape[0], packed.shape[1] * 2
    weight = torch.empty((rows, cols), dtype=torch.bfloat16, device=packed.device)
    chunk_rows = max(1, _DEQUANT_MAX_CHUNK_ELEMENTS // cols)
    # Bound unpacked indices and FP32 intermediates independently of the matrix size.
    for start in range(0, rows, chunk_rows):
        block = packed[start : start + chunk_rows]
        block_scales = scales[start : start + chunk_rows]
        indices = torch.stack((block & 15, block >> 4), dim=-1).int()
        values = table[indices].reshape(*block_scales.shape, 32)
        decoded = torch.ldexp(values, (block_scales.int() - 127).unsqueeze(-1)).bfloat16()
        weight[start : start + chunk_rows].copy_(decoded.reshape(block.shape[0], cols))
        del indices, values, decoded
    if not torch.isfinite(weight).all():
        raise ValueError("MXFP4 weights are not representable in BF16")
    return weight


def _check_mxfp8(codes: torch.Tensor, scales: torch.Tensor) -> None:
    if codes.dtype != torch.float8_e4m3fn or scales.dtype != torch.uint8:
        raise TypeError("MXFP8 conversion requires E4M3 weights and uint8 E8M0 scales")
    if codes.ndim != 2 or scales.ndim != 2 or codes.numel() == 0:
        raise ValueError("MXFP8 conversion requires a non-empty matrix")
    rows, cols = codes.shape
    if tuple(scales.shape) != (rows, math.ceil(cols / 32)):
        raise ValueError(
            f"MXFP8 scale shape {tuple(scales.shape)} does not cover weight {tuple(codes.shape)}"
        )
    if codes.device != scales.device:
        raise ValueError("MXFP8 weights and scales must be on the same device")
    if (scales == 255).any():
        raise ValueError("MXFP8 E8M0 scales contain NaN (255)")


def e8m0_to_float32(scales: torch.Tensor) -> torch.Tensor:
    """Exact FP32 power of two 2^(s-127) for E8M0 bytes 0..254, built from the bit pattern."""
    bits = scales.to(torch.int32)
    # s >= 1 is the normal float 2^(s-127); s == 0 is the subnormal 2^-127.
    bits = torch.where(bits == 0, torch.full_like(bits, 1 << 22), bits << 23)
    return bits.view(torch.float32)


def dequantize_mxfp8_weight(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Decode E4M3 [N,K] with E8M0 [N,ceil(K/32)] 1x32 block scales into BF16 [N,K].

    The scale is built from its bits and applied as one FP32 multiply by a power of two, so
    the decode is exact on CPU and CUDA alike (no ldexp/pow). BF16 holds every product exactly
    unless it falls below the BF16 subnormal range.
    """
    _check_mxfp8(codes, scales)
    rows, cols = codes.shape
    weight = torch.empty((rows, cols), dtype=torch.bfloat16, device=codes.device)
    chunk_rows = max(1, _DEQUANT_MAX_CHUNK_ELEMENTS // cols)
    for start in range(0, rows, chunk_rows):
        factor = e8m0_to_float32(scales[start : start + chunk_rows])
        factor = factor.repeat_interleave(32, dim=1)[:, :cols]
        decoded = codes[start : start + chunk_rows].float() * factor
        weight[start : start + chunk_rows].copy_(decoded.bfloat16())
        del factor, decoded
    if not torch.isfinite(weight).all():
        raise ValueError("MXFP8 weights are not representable in BF16")
    return weight


def _pow2_float64(exponents: torch.Tensor) -> torch.Tensor:
    """Exact FP64 2^e for integer e in [-1022, 1023], built from the bit pattern."""
    bits = (exponents.to(torch.int64) + 1023) << 52
    return bits.view(torch.float64)


def requantize_mxfp8_weight(
    codes: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-encode MXFP8 as E4M3 [N,K] with FP32 128x128 power-of-two block scales.

    Each 128x128 block takes the scale 2^(largest E8M0 exponent among its 1x32 groups), so a
    code moves by an exact power of two (a shift of 0 for the largest group). The re-encode is
    lossless unless a small group drops below the E4M3 subnormal range, which only happens when
    groups of one block differ by many octaves; ``mxfp8_requant_loss`` counts such codes.
    """
    _check_mxfp8(codes, scales)
    rows, cols = codes.shape
    block, groups = 128, 128 // 32
    nr, nc = math.ceil(rows / block), math.ceil(cols / block)
    exponents = scales.to(torch.int32) - 127
    padded = F.pad(
        exponents, (0, nc * groups - exponents.shape[1], 0, nr * block - rows), value=-127
    )
    block_exp = padded.reshape(nr, block, nc, groups).amax(dim=(1, 3))
    out = torch.empty((rows, cols), dtype=torch.float8_e4m3fn, device=codes.device)
    chunk_blocks = max(1, _DEQUANT_MAX_CHUNK_ELEMENTS // (block * cols))
    for b0 in range(0, nr, chunk_blocks):
        r0, r1 = b0 * block, min(rows, (b0 + chunk_blocks) * block)
        row_exp = block_exp[b0 : b0 + chunk_blocks].repeat_interleave(block, dim=0)[: r1 - r0]
        shift = exponents[r0:r1] - row_exp.repeat_interleave(groups, dim=1)[:, : exponents.shape[1]]
        factor = _pow2_float64(shift).repeat_interleave(32, dim=1)[:, :cols]
        out[r0:r1].copy_((codes[r0:r1].float().double() * factor).float().to(torch.float8_e4m3fn))
        del row_exp, shift, factor
    return out, e8m0_to_float32((block_exp + 127).to(torch.uint8)).contiguous()


def mxfp8_requant_loss(
    codes: torch.Tensor, scales: torch.Tensor, new_codes: torch.Tensor, new_scales: torch.Tensor
) -> int:
    """Number of elements whose value changed in ``requantize_mxfp8_weight``."""
    rows, cols = codes.shape
    reference = dequantize_mxfp8_weight(codes, scales).float()
    factor = new_scales.repeat_interleave(128, dim=0)[:rows].repeat_interleave(128, dim=1)
    return int((new_codes.float() * factor[:, :cols] != reference).sum())


def quantize_expert_weight(
    weight: torch.Tensor, *, amax: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize CUDA BF16 [N,K] to packed NVFP4, linear E4M3 scales and an FP32 multiplier.

    ``amax`` may cover both gate/up projections so a fused GEMM can share their multiplier.
    """
    if weight.ndim != 2 or weight.dtype != torch.bfloat16 or not weight.is_cuda:
        raise TypeError("K3.1 expert quantization requires a CUDA BF16 matrix")
    if not torch.isfinite(weight).all():
        raise ValueError("K3.1 expert projection contains non-finite values")
    if amax is None:
        amax = weight.abs().amax().float()
    global_scale = torch.where(amax == 0, torch.ones_like(amax), amax / (448.0 * 6.0))
    inverse_scale = global_scale.reciprocal()
    if not torch.isfinite(inverse_scale) or inverse_scale <= 0:
        raise ValueError("K3.1 expert global scale is not representable in FP32")
    packed, scales = torch.ops.trtllm.fp4_quantize(
        weight.contiguous(), inverse_scale, 16, False, False
    )
    return packed, scales.view(weight.shape[0], -1), global_scale


def quantize_attention_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16 [N,K] projection into E4M3 and FP32 128x128 block scales."""
    if weight.ndim != 2 or weight.dtype != torch.bfloat16:
        raise TypeError("K3.1 attention conversion requires a BF16 matrix")
    if not torch.isfinite(weight).all():
        raise ValueError("K3.1 attention projection contains non-finite values")
    rows, cols = weight.shape
    block = 128
    nr, nc = math.ceil(rows / block), math.ceil(cols / block)
    padded = F.pad(weight.float(), (0, nc * block - cols, 0, nr * block - rows))
    tiles = padded.reshape(nr, block, nc, block)
    amax = tiles.abs().amax(dim=(1, 3))
    scale = torch.where(amax == 0, torch.ones_like(amax), amax / 448.0)
    codes = (tiles / scale[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return codes.reshape(nr * block, nc * block)[:rows, :cols].contiguous(), scale


class _AttentionPair:
    """Hold at most one converted projection until its scale is consumed."""

    def __init__(
        self,
        source: torch.Tensor | _TensorSlice,
        device: torch.device,
        mxfp8_scale: torch.Tensor | _TensorSlice | None = None,
    ) -> None:
        self.source = source
        self.device = device
        # E8M0 1x32 scales of an MXFP8 source; None for a BF16 source.
        self.mxfp8_scale = mxfp8_scale
        self.shape = (
            tuple(source.shape) if isinstance(source, torch.Tensor) else tuple(source.get_shape())
        )
        self.codes: torch.Tensor | None = None
        self.scales: torch.Tensor | None = None

    def materialize(self) -> None:
        if self.scales is None:
            source = self.source if isinstance(self.source, torch.Tensor) else self.source[:]
            source = source.to(self.device)
            if self.mxfp8_scale is not None:
                scale = self.mxfp8_scale
                scale = scale if isinstance(scale, torch.Tensor) else scale[:]
                codes, scales = requantize_mxfp8_weight(source, scale.to(self.device))
            else:
                codes, scales = quantize_attention_weight(source)
            self.codes, self.scales = codes.cpu(), scales.cpu()


class _AttentionValue:
    """Safetensors-slice interface retained through the VLM prefix filter."""

    def __init__(self, pair: _AttentionPair, *, scale: bool = False) -> None:
        self.pair = pair
        self.scale = scale

    def get_shape(self) -> list[int]:
        if self.scale:
            return [math.ceil(dim / 128) for dim in self.pair.shape]
        return list(self.pair.shape)

    def __getitem__(self, index: object) -> torch.Tensor:
        self.pair.materialize()
        if self.scale:
            result = self.pair.scales[index]
            self.pair.codes = self.pair.scales = None
            return result
        return self.pair.codes[index]


def _load_converted_expert(
    backend: torch.nn.Module,
    base: str,
    ckpt_expert_idx: int,
    slot_expert_idx: int,
    local_slot_id: int,
    get_tensor: Callable[[str], torch.Tensor],
    get_optional_tensor: Callable[[str], torch.Tensor | None],
) -> None:
    del get_optional_tensor
    device = backend.w2_weight.device

    def dequantize_projection(projection: str) -> torch.Tensor:
        key = f"{base}.{ckpt_expert_idx}.{projection}"
        try:
            return dequantize_mxfp4_weight(
                get_tensor(key + ".weight_packed").to(device),
                get_tensor(key + ".weight_scale").to(device),
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"{key}: {error}") from error

    weights = {projection: dequantize_projection(projection) for projection in ("w1", "w3")}
    # Single-alpha backends fuse gate/up into one GEMM, so both halves use
    # the same global scale when choosing their per-16 block scales.
    gate_up_amax = torch.maximum(weights["w1"].abs().amax(), weights["w3"].abs().amax()).float()
    tensors = {}
    for projection in ("w1", "w3", "w2"):
        key = f"{base}.{ckpt_expert_idx}.{projection}"
        try:
            packed, scales, global_scale = quantize_expert_weight(
                dequantize_projection(projection)
                if projection == "w2"
                else weights.pop(projection),
                amax=gate_up_amax if projection != "w2" else None,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"{key}: {error}") from error
        tensors.update(
            {
                f"{projection}_weight": packed.cpu(),
                f"{projection}_weight_scale": scales.cpu(),
                f"{projection}_weight_scale_2": global_scale.cpu(),
                f"{projection}_input_scale": torch.ones((), dtype=torch.float32),
            }
        )
        del packed, scales, global_scale
    backend.quant_method.load_streaming_nvfp4_expert(
        backend, global_expert_id=slot_expert_idx, local_slot_id=local_slot_id, **tensors
    )


class K31LoadQuantization:
    """Configure native quantized modules and adapt the source checkpoint lazily."""

    def __init__(self, attention_keys: set[str]) -> None:
        self.attention_keys = attention_keys

    @classmethod
    def configure(cls, config: ModelConfig) -> K31LoadQuantization | None:
        """Apply the opt-in runtime config before constructing a K3.1 VLM."""
        enabled = os.environ.get(_SWITCH, "0")
        if enabled not in ("0", "1"):
            raise ValueError(f"{_SWITCH} must be 0 or 1, got {enabled!r}")
        requant = os.environ.get(_REQUANT_SWITCH, "0")
        if requant not in ("0", "1"):
            raise ValueError(f"{_REQUANT_SWITCH} must be 0 or 1, got {requant!r}")
        if requant == "1":
            if enabled == "1":
                raise ValueError(f"{_SWITCH} and {_REQUANT_SWITCH} are mutually exclusive")
            return K31AttentionRequant.configure_requant(config)
        if enabled == "0":
            logger.info(f"{_SWITCH}=0: using native K3.1 checkpoint precision")
            return None
        cfg = config.pretrained_config
        if cfg.model_type != "kimi_k31":
            raise ValueError(f"{_SWITCH} requires a K3.1 composite checkpoint")
        text = cfg.text_config
        source_quant = getattr(text, "quantization_config", None) or getattr(
            cfg, "quantization_config", {}
        )
        from .modeling_kimi_linear import KimiK3MoERuntime

        expert_layers = [
            i
            for i in range(text.num_hidden_layers)
            if i >= text.first_k_dense_replace and i % text.moe_layer_freq == 0
        ]
        routed = [KimiK3MoERuntime._resolve_routed_quant_config(config, i) for i in expert_layers]
        if routed and all(q.quant_algo == QuantAlgo.NVFP4 for q in routed):
            logger.info(f"{_SWITCH}=1: native NVFP4 checkpoint; bypassing conversion")
            return None
        if source_quant.get("format") != "mxfp4-pack-quantized" or any(
            q.quant_algo != QuantAlgo.W4A8_MXFP4_MXFP8 for q in routed
        ):
            raise ValueError(
                f"{_SWITCH} requires MXFP4 routed experts with no conflicting overrides"
            )
        if config.moe_backend not in ("TRTLLM", "CUTLASS"):
            raise ValueError(f"{_SWITCH} supports the TRTLLM and CUTLASS MoE backends")
        if config.quant_config_dict:
            raise ValueError(f"{_SWITCH} does not accept source per-layer quantization overrides")

        # Source exclusions include all attention. The explicit per-layer map
        # now owns precision, leaving unspecified modules unquantized.
        quant = config.quant_config.model_copy(
            update={"quant_algo": QuantAlgo.MIXED_PRECISION, "exclude_modules": None}
        )
        # Preserve executor-normalized cache dtypes and recompute cached modes
        # from the new weight precision.
        quant.__dict__.pop("quant_mode", None)
        quant.__dict__.pop("layer_quant_mode", None)
        declarations = {}
        attention_keys = set()
        kda_layers = set(text.linear_attn_config["kda_layers"])
        for layer in range(text.num_hidden_layers):
            stem = f"model.layers.{layer}"
            if layer in expert_layers:
                declarations[f"{stem}.block_sparse_moe.experts"] = QuantConfig(
                    quant_algo=QuantAlgo.NVFP4, group_size=16
                )
            projections = _KDA_PROJECTIONS if layer + 1 in kda_layers else _MLA_PROJECTIONS
            for projection in projections:
                name = f"{stem}.self_attn.{projection}"
                declarations[name] = QuantConfig(
                    quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128
                )
                attention_keys.add(name + ".weight")
        frozen = config._frozen
        config._frozen = False
        config.quant_config = quant
        config.quant_config_dict = declarations
        config._frozen = frozen
        logger.info(
            f"{_SWITCH}=1: converting {len(expert_layers)} expert banks to NVFP4 and "
            f"{len(attention_keys)} attention projections to FP8; DSA indexers stay unquantized"
        )
        return cls(attention_keys)

    def prepare_weights(self, llm: KimiLinearForCausalLM, weights) -> ConsumableWeightsDict:
        """Preserve file-grouped expert streaming and add lazy FP8 projection values."""
        from .modeling_kimi_linear import _K3_NVFP4_EXPERT_CKPT_SPEC

        spec = _K3_NVFP4_EXPERT_CKPT_SPEC._replace(
            kinds=("weight_packed", "weight_scale"),
            loader=_load_converted_expert,
            optional_suffixes=(),
        )
        for layer in llm.model.layers:
            if getattr(layer, "is_moe", False):
                layer.block_sparse_moe.expert_ckpt_spec = spec
        adapted = dict(weights.items())
        prefix = "language_model." if any(k.startswith("language_model.") for k in adapted) else ""
        device = next(llm.parameters()).device
        for name in self.attention_keys:
            key = prefix + name
            if key not in adapted:
                raise KeyError(f"K3.1 load conversion is missing attention projection {key}")
            if key + "_scale" in adapted or key + "_scale_inv" in adapted:
                raise ValueError(f"K3.1 load conversion expects unquantized attention: {key}")
            pair = _AttentionPair(adapted[key], device)
            adapted[key] = _AttentionValue(pair)
            adapted[key + "_scale"] = _AttentionValue(pair, scale=True)
        result = ConsumableWeightsDict(adapted)
        checkpoint_dir = getattr(weights, "checkpoint_dir", None)
        if checkpoint_dir is not None:
            result.checkpoint_dir = checkpoint_dir
        result.checkpoint_prefix = getattr(weights, "checkpoint_prefix", "")
        return result


class K31AttentionRequant(K31LoadQuantization):
    """Serve native NVFP4 experts with block-scaled FP8 attention (``TLLM_K31_REQUANT_ATTN_FP8=1``).

    nvidia/Kimi-K3.1-Pro-NVFP4 stores KDA projections as MXFP8 (E4M3 + E8M0 1x32), MLA
    projections as BF16 and shared experts as MXFP8. The K3 attention runtime takes
    FP8_BLOCK_SCALES (128x128, weight-only), so every KDA and MLA projection is requantized
    at load (MXFP8 is decoded exactly first) and declared the way nvidia/Kimi-K3-NVFP4
    declares it. Shared experts are declared unquantized; the trunk loader decodes them.
    """

    @classmethod
    def configure_requant(cls, config: ModelConfig) -> K31AttentionRequant | None:
        cfg = config.pretrained_config
        if cfg.model_type != "kimi_k31":
            raise ValueError(f"{_REQUANT_SWITCH} requires a K3.1 composite checkpoint")
        text = cfg.text_config
        from .modeling_kimi_linear import KimiK3MoERuntime

        expert_layers = [
            i
            for i in range(text.num_hidden_layers)
            if i >= text.first_k_dense_replace and i % text.moe_layer_freq == 0
        ]
        routed = [KimiK3MoERuntime._resolve_routed_quant_config(config, i) for i in expert_layers]
        if not routed or any(q.quant_algo != QuantAlgo.NVFP4 for q in routed):
            raise ValueError(f"{_REQUANT_SWITCH} requires a native NVFP4 routed-expert checkpoint")
        declarations = dict(config.quant_config_dict or {})
        attention_algos = {q.quant_algo for n, q in declarations.items() if ".self_attn." in n}
        if attention_algos == {QuantAlgo.FP8_BLOCK_SCALES}:
            logger.info(f"{_REQUANT_SWITCH}=1: attention is already FP8 block-scaled; no-op")
            return None
        if not attention_algos <= {QuantAlgo.MXFP8}:
            raise ValueError(
                f"{_REQUANT_SWITCH} expects MXFP8 or unquantized attention, got {attention_algos}"
            )
        prefixes = {
            m.group(1) for n in declarations if (m := re.match(r"(.*?)layers\.\d+\.", n))
        }
        kv_cache_quant_algo = routed[0].kv_cache_quant_algo
        declarations = {n: q for n, q in declarations.items() if q.quant_algo != QuantAlgo.MXFP8}
        attention_keys = set()
        kda_layers = set(text.linear_attn_config["kda_layers"])
        for layer in range(text.num_hidden_layers):
            is_kda = layer + 1 in kda_layers
            for name in _KDA_FP8_DECLARATIONS if is_kda else _MLA_FP8_DECLARATIONS:
                for prefix in prefixes:
                    declarations[f"{prefix}layers.{layer}.self_attn.{name}"] = QuantConfig(
                        quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
                        group_size=None,
                        kv_cache_quant_algo=kv_cache_quant_algo,
                    )
            for projection in _KDA_PROJECTIONS if is_kda else _MLA_PROJECTIONS:
                attention_keys.add(f"model.layers.{layer}.self_attn.{projection}.weight")
        excludes = [
            p for p in config.quant_config.exclude_modules or [] if p not in _REQUANT_EXCLUDES_DROP
        ]
        excludes += [p for p in _REQUANT_EXCLUDES_ADD if p not in excludes]
        quant = config.quant_config.model_copy(update={"exclude_modules": excludes})
        quant.__dict__.pop("quant_mode", None)
        quant.__dict__.pop("layer_quant_mode", None)
        frozen = config._frozen
        config._frozen = False
        config.quant_config = quant
        config.quant_config_dict = declarations
        config._frozen = frozen
        logger.info(
            f"{_REQUANT_SWITCH}=1: requantizing {len(attention_keys)} attention projections to "
            "FP8 128x128 weight-only; routed experts stay NVFP4, shared experts BF16"
        )
        return cls(attention_keys)

    def prepare_weights(self, llm: KimiLinearForCausalLM, weights) -> ConsumableWeightsDict:
        """Replace attention projections with lazy FP8 codes and 128x128 block scales."""
        adapted = dict(weights.items())
        prefix = "language_model." if any(k.startswith("language_model.") for k in adapted) else ""
        device = next(llm.parameters()).device
        if device.type != "cuda" and os.environ.get(_REQUANT_ALLOW_CPU, "0") != "1":
            raise RuntimeError(
                f"{_REQUANT_SWITCH}=1 requantizes on the model device, which is {device}; "
                f"expected CUDA (set {_REQUANT_ALLOW_CPU}=1 only for CPU tests)"
            )
        direct, requant_mx, requant_bf16 = 0, 0, 0
        for name in sorted(self.attention_keys):
            key = prefix + name
            if key not in adapted:
                raise KeyError(f"{_REQUANT_SWITCH} is missing attention projection {key}")
            if key + "_scale_inv" in adapted:
                raise ValueError(f"{_REQUANT_SWITCH} expects MXFP8 or BF16 attention: {key}")
            projection = name.rsplit(".", 2)[-2]
            if key + "_scale" in adapted and _is_kda_bf16_runtime(llm, name, projection):
                # Leave E4M3 + E8M0 in place; the trunk loader decodes MXFP8 to BF16 directly.
                direct += 1
                continue
            mxfp8_scale = adapted.pop(key + "_scale", None)
            if mxfp8_scale is None:
                requant_bf16 += 1
            else:
                requant_mx += 1
            pair = _AttentionPair(adapted[key], device, mxfp8_scale=mxfp8_scale)
            adapted[key] = _AttentionValue(pair)
            adapted[key + "_scale"] = _AttentionValue(pair, scale=True)
        logger.info(
            f"{_REQUANT_SWITCH}=1 on {device}: {requant_mx} MXFP8 projections re-encoded with "
            f"power-of-two 128x128 scales, {requant_bf16} BF16 projections quantized, {direct} "
            "MXFP8 projections left for direct BF16 decode"
        )
        result = ConsumableWeightsDict(adapted)
        checkpoint_dir = getattr(weights, "checkpoint_dir", None)
        if checkpoint_dir is not None:
            result.checkpoint_dir = checkpoint_dir
        result.checkpoint_prefix = getattr(weights, "checkpoint_prefix", "")
        return result


def _is_kda_bf16_runtime(llm: KimiLinearForCausalLM, name: str, projection: str) -> bool:
    """True if ``name`` is a KDA projection that the runtime keeps as a plain BF16 nn.Linear."""
    if projection not in _KDA_BF16_RUNTIME_PROJECTIONS:
        return False
    match = re.match(r"model\.layers\.(\d+)\.self_attn\.", name)
    layers = getattr(getattr(llm, "model", None), "layers", None)
    if match is None or layers is None:
        return False
    layer = layers[int(match.group(1))]
    if not getattr(layer, "is_kda", False):
        return False
    module = getattr(getattr(layer, "linear_attn", None), projection, None)
    if module is None or getattr(module, "has_fp8_block_scales", False):
        return False
    return type(module) is torch.nn.Linear

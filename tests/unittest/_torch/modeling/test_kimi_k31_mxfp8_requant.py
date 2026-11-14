# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""K3.1 MXFP8 attention requantization (TLLM_K31_REQUANT_ATTN_FP8=1)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.kimi_k31_load_quantization import (
    K31LoadQuantization,
    _is_kda_bf16_runtime,
    dequantize_mxfp8_weight,
    e8m0_to_float32,
    mxfp8_requant_loss,
    requantize_mxfp8_weight,
)
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def _mxfp8(rows: int, cols: int, exponents: tuple[int, int], seed: int = 0):
    """Random finite E4M3 codes with E8M0 scales drawn from ``exponents`` (inclusive)."""
    generator = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 256, (rows, cols), dtype=torch.uint8, generator=generator)
    codes[(codes & 0x7F) == 0x7F] = 0  # E4M3 NaN patterns
    scales = torch.randint(
        exponents[0] + 127,
        exponents[1] + 128,
        (rows, (cols + 31) // 32),
        dtype=torch.uint8,
        generator=generator,
    )
    return codes.view(torch.float8_e4m3fn), scales


def _reference(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    factor = torch.pow(2.0, scales.double() - 127).repeat_interleave(32, dim=1)
    return codes.double() * factor[:, : codes.shape[1]]


@pytest.mark.cpu_only
def test_e8m0_scale_is_exact_power_of_two():
    scales = torch.arange(0, 255, dtype=torch.uint8)
    expected = torch.tensor([2.0 ** (int(s) - 127) for s in scales], dtype=torch.float64)
    assert torch.equal(e8m0_to_float32(scales).double(), expected)


@pytest.mark.cpu_only
@pytest.mark.parametrize("cols", [256, 96])
def test_mxfp8_decode_matches_float64_reference(cols):
    codes, scales = _mxfp8(64, cols, (-20, 10))
    weight = dequantize_mxfp8_weight(codes, scales)
    assert weight.dtype == torch.bfloat16
    assert torch.equal(weight.double(), _reference(codes, scales))


@pytest.mark.cpu_only
def test_mxfp8_decode_rejects_nan_scale():
    codes, scales = _mxfp8(4, 64, (0, 0))
    scales[1, 1] = 255
    with pytest.raises(ValueError, match="NaN"):
        dequantize_mxfp8_weight(codes, scales)


@pytest.mark.cpu_only
@pytest.mark.parametrize("rows,cols", [(256, 512), (200, 320)])
def test_mxfp8_power_of_two_requant_is_lossless(rows, cols):
    # Groups of one 128x128 block span up to 3 octaves here; codes of magnitude >= 2^-3 stay
    # normal E4M3 after a shift of up to 3 octaves, so the re-encode must be exact.
    codes, scales = _mxfp8(rows, cols, (-12, -9))
    codes = codes.view(torch.uint8) | 0x20
    codes[(codes & 0x7F) == 0x7F] = 0x20
    codes = codes.view(torch.float8_e4m3fn)
    new_codes, new_scales = requantize_mxfp8_weight(codes, scales)
    assert new_codes.shape == codes.shape and new_codes.dtype == torch.float8_e4m3fn
    assert new_scales.shape == ((rows + 127) // 128, (cols + 127) // 128)
    mantissa, _ = torch.frexp(new_scales)
    assert torch.all(mantissa == 0.5)
    assert mxfp8_requant_loss(codes, scales, new_codes, new_scales) == 0


@pytest.mark.cpu_only
def test_mxfp8_requant_loss_counts_underflow():
    codes, scales = _mxfp8(128, 128, (0, 0))
    scales[0, 0] = 127 + 20  # one group 20 octaves above the rest of the block
    new_codes, new_scales = requantize_mxfp8_weight(codes, scales)
    assert new_scales[0, 0] == 2.0**20
    assert mxfp8_requant_loss(codes, scales, new_codes, new_scales) > 0


@pytest.mark.cpu_only
def test_bf16_runtime_projection_detection():
    attention = nn.Module()
    attention.b_proj = nn.Linear(8, 2, bias=False)
    attention.f_a_proj = nn.Linear(8, 4, bias=False)
    attention.q_proj = nn.Linear(8, 8, bias=False)
    kda = SimpleNamespace(is_kda=True, linear_attn=attention)
    mla = SimpleNamespace(is_kda=False, linear_attn=attention)
    llm = SimpleNamespace(model=SimpleNamespace(layers=[kda, mla]))
    name = "model.layers.{}.self_attn.{}.weight"
    assert _is_kda_bf16_runtime(llm, name.format(0, "b_proj"), "b_proj")
    assert _is_kda_bf16_runtime(llm, name.format(0, "f_a_proj"), "f_a_proj")
    assert not _is_kda_bf16_runtime(llm, name.format(0, "q_proj"), "q_proj")
    assert not _is_kda_bf16_runtime(llm, name.format(1, "b_proj"), "b_proj")


def _native_config(attention_algo):
    text = SimpleNamespace(
        num_hidden_layers=1,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        linear_attn_config={"kda_layers": [1]},
    )
    declarations = {
        "model.layers.0.block_sparse_moe.experts": QuantConfig(
            quant_algo=QuantAlgo.NVFP4, group_size=16
        )
    }
    if attention_algo is not None:
        declarations["model.layers.0.self_attn.q_proj"] = QuantConfig(quant_algo=attention_algo)
    return ModelConfig(
        pretrained_config=SimpleNamespace(model_type="kimi_k31", text_config=text),
        moe_backend="TRTLLM",
        quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8),
        quant_config_dict=declarations,
    )


@pytest.mark.cpu_only
def test_requant_gate_is_noop_on_fp8_attention(monkeypatch):
    monkeypatch.setenv("TLLM_K31_REQUANT_ATTN_FP8", "1")
    monkeypatch.delenv("TLLM_K31_LOAD_QUANTIZE", raising=False)
    config = _native_config(QuantAlgo.FP8_BLOCK_SCALES)
    before = dict(config.quant_config_dict)
    assert K31LoadQuantization.configure(config) is None
    assert config.quant_config_dict == before


@pytest.mark.cpu_only
def test_requant_gate_off_leaves_mxfp8_config(monkeypatch):
    monkeypatch.delenv("TLLM_K31_REQUANT_ATTN_FP8", raising=False)
    monkeypatch.delenv("TLLM_K31_LOAD_QUANTIZE", raising=False)
    config = _native_config(QuantAlgo.MXFP8)
    before = dict(config.quant_config_dict)
    assert K31LoadQuantization.configure(config) is None
    assert config.quant_config_dict == before


@pytest.mark.cpu_only
def test_requant_gate_conflicts_with_load_quantize(monkeypatch):
    monkeypatch.setenv("TLLM_K31_REQUANT_ATTN_FP8", "1")
    monkeypatch.setenv("TLLM_K31_LOAD_QUANTIZE", "1")
    with pytest.raises(ValueError, match="mutually exclusive"):
        K31LoadQuantization.configure(_native_config(QuantAlgo.MXFP8))

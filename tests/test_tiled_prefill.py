"""Studio-only checks: tiled M5 prefill preserves MLX's bits and avoids converting the packed weight."""

from importlib.metadata import version

import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.kernels.qwen.dense.v1 import lane_qmm, tiled_prefill  # noqa: E402


@pytest.fixture(scope="module")
def m5_prefill():
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    if version("mlx") not in tiled_prefill.HEADER_SHAS or not str(info.get("architecture", "")).startswith("applegpu_g17"):
        pytest.skip("requires a qualified MLX version and M5 GPU")
    assert tiled_prefill.prepare(), "the supported Studio must compile and pass the startup equality check"


@pytest.mark.parametrize("nt", [32, 64])
@pytest.mark.parametrize("rows,n,k", [(129, 4096, 256), (191, 5120, 5120), (257, 4096, 5120)])
def test_partial_prompt_tiles_match_mlx_bits(m5_prefill, nt, rows, n, k):
    key = mx.random.key(17)
    x = mx.random.normal((1, rows, k), key=key).astype(mx.bfloat16)
    w = (mx.random.normal((n, k), key=mx.random.split(key)[0]) * 0.02).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=64, bits=4)
    tiled = lane_qmm.tile_weight(q, nt, bits=4)
    assert tiled_prefill.active(x, tiled, 4, 64, nt)
    actual = tiled_prefill.matmul(x, tiled, s, b, nt)
    expect = mx.quantized_matmul(x, q, s, b, transpose=True, group_size=64, bits=4)
    mx.eval(actual, expect)
    assert actual.shape == expect.shape
    assert bool(mx.all(mx.isfinite(actual)).item())
    assert bool(mx.all(actual.view(mx.uint16) == expect.view(mx.uint16)).item())


def test_dispatch_avoids_the_whole_weight_copy_and_preserves_bias(m5_prefill, monkeypatch):
    import mlx.nn as nn

    module = nn.QuantizedLinear(256, 4096, group_size=64, bits=4, bias=True)
    module.scales, module.biases = module.scales.astype(mx.bfloat16), module.biases.astype(mx.bfloat16)
    module.bias = module.bias.astype(mx.bfloat16)
    original = module.weight
    module.weight = lane_qmm.tile_weight(original, 32, bits=4)
    object.__setattr__(module, "_lane_tiled", True)
    object.__setattr__(module, "_lane_nt", 32)
    monkeypatch.setattr(lane_qmm, "enabled", True)
    monkeypatch.setattr(lane_qmm, "max_rows", 128)
    untile, calls = lane_qmm.untile_weight, []

    def counted(*args, **kwargs):
        calls.append(1)
        return untile(*args, **kwargs)

    monkeypatch.setattr(lane_qmm, "untile_weight", counted)
    x = mx.ones((1, 129, 256), dtype=mx.bfloat16)
    expect = mx.quantized_matmul(x, original, module.scales, module.biases,
                                transpose=True, group_size=64, bits=4) + module.bias
    actual = lane_qmm._call(module, x)
    mx.eval(actual, expect)
    assert bool(mx.array_equal(actual, expect).item()) and not calls
    monkeypatch.setattr(tiled_prefill, "_available", False)
    fallback = lane_qmm._call(module, x)
    mx.eval(fallback)
    assert bool(mx.array_equal(fallback, expect).item()) and len(calls) == 1


@pytest.mark.parametrize("rows,n,k,bits,group,nt", [
    (128, 4096, 256, 4, 64, 32),
    (129, 48, 256, 4, 64, 32),
    (129, 4096, 256, 8, 64, 32),
    (129, 4096, 256, 4, 32, 32),
    (129, 64, 4096, 4, 64, 32),              # native MLX uses split-K: its reduction order must stay intact
])
def test_other_dispatches_stay_on_the_existing_path(rows, n, k, bits, group, nt):
    assert not tiled_prefill.supported(rows, n, k, bits, group, nt)

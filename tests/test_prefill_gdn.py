"""Packed prompt recurrence retains both output and resumed-state bits."""

import pytest

mx = pytest.importorskip("mlx.core")

from mlx_lm.models import gated_delta
from tensorfold.kernels.qwen.dense.v1 import prefill_gdn, prompt_attention


def inputs(rows, dtype, batch=1):
    q = (mx.random.normal((batch, rows, 2, 128), key=mx.random.key(1)) / 128).astype(dtype)
    k = (mx.random.normal((batch, rows, 2, 128), key=mx.random.key(2)) / 12).astype(dtype)
    v = mx.random.normal((batch, rows, 6, 128), key=mx.random.key(3)).astype(dtype)
    g = mx.random.uniform(low=.94, high=1., shape=(batch, rows, 6), key=mx.random.key(4))
    beta = mx.sigmoid(mx.random.normal((batch, rows, 6), key=mx.random.key(5))).astype(dtype)
    state = mx.random.normal((batch, 6, 128, 128), key=mx.random.key(6)) * .02
    return q, k, v, g, beta, state


def assert_bits(actual, reference):
    for a, b in zip(actual, reference):
        dtype = mx.uint16 if a.dtype.size == 2 else mx.uint32
        assert bool(mx.array_equal(a.view(dtype), b.view(dtype)).item())


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("rows,batch", [(129, 1), (512, 2), (2048, 1)])
def test_packed_matches_installed_recurrence(dtype, rows, batch):
    args = inputs(rows, dtype, batch)
    stock = prefill_gdn._STOCK or gated_delta.gated_delta_kernel
    assert_bits(prefill_gdn.packed(*args), stock(*args))


def test_prompt_state_resumes_with_identical_bits():
    q, k, v, g, beta, state = inputs(1024, mx.bfloat16)
    whole, final = prefill_gdn.packed(q, k, v, g, beta, state)
    first, kept = prefill_gdn.packed(q[:, :257], k[:, :257], v[:, :257], g[:, :257], beta[:, :257], state)
    tail, resumed = prefill_gdn.packed(q[:, 257:], k[:, 257:], v[:, 257:], g[:, 257:], beta[:, 257:], kept)
    assert_bits((mx.concatenate([first, tail], axis=1), resumed), (whole, final))


@pytest.mark.parametrize("case", ["short", "mask", "vector", "value_dim", "disabled"])
def test_other_shapes_keep_the_installed_kernel(monkeypatch, case):
    q, k, v, g, beta, state = inputs(64 if case == "short" else 128, mx.bfloat16)
    mask = mx.ones((1, q.shape[1]), dtype=mx.bool_) if case == "mask" else None
    if case == "vector":
        g = mx.broadcast_to(g[..., None], (*g.shape, 128))
    if case == "value_dim":
        v, state = v[..., :64], state[:, :, :64]
    calls = []
    monkeypatch.setattr(prefill_gdn, "_available", case != "disabled")
    monkeypatch.setattr(prefill_gdn, "_STOCK", lambda *args: calls.append(args) or ("stock", "state"))
    assert prefill_gdn.dispatch(q, k, v, g, beta, state, mask) == ("stock", "state")
    assert len(calls) == 1 and calls[0][-1] is mask


def test_failed_startup_comparison_disables_the_new_path(monkeypatch):
    monkeypatch.setattr(prefill_gdn, "_available", prefill_gdn._available)
    monkeypatch.setattr(prefill_gdn, "_STOCK", prefill_gdn._STOCK)
    monkeypatch.setattr(prompt_attention, "native_m5", lambda: True)
    monkeypatch.setattr(prefill_gdn, "_check", lambda stock: False)
    monkeypatch.setattr(gated_delta, "gated_delta_kernel", gated_delta.gated_delta_kernel)
    assert not prefill_gdn.prepare()
    assert prefill_gdn.identity() == "packed-prefill-gdn=off"

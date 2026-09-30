"""Packed DeltaNet prompt recurrence, qualified against the installed MLX-LM kernel on M5.

Kernel adapted from mlx-lm gated_delta.py at a9bd8af5c02118882af735cef60705d2efce9fd0.
Copyright (c) 2025 Apple Inc. MIT License; see LICENSES/MLX-LM-MIT.txt.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import mlx.core as mx

_STOCK: Any = None
_kernel: Any = None
_available = False

_SOURCE = r"""
        constexpr int lanes_per_row = 4;
        constexpr int rows_per_simdgroup = 32 / lanes_per_row;
        constexpr int values_per_lane = Dk / lanes_per_row;
        constexpr int partials_per_lane = values_per_lane / 4;

        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);

        auto lane = thread_index_in_simdgroup;
        auto row_in_simdgroup = lane / lanes_per_row;
        auto lane_in_row = lane & (lanes_per_row - 1);
        auto row_group = thread_position_in_grid.y;
        auto dv_idx = row_group * rows_per_simdgroup + row_in_simdgroup;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
        auto k_ = k + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
        y += (b_idx * T * Hv + hv_idx) * Dv;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;

        float state[values_per_lane];
        for (int i = 0; i < values_per_lane; ++i) {
          state[i] = static_cast<float>(i_state[i]);
        }

        // g, beta: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          float gt = static_cast<float>(g_[hv_idx]);

          // Partials mirror the generic kernel: each 4-element chain is one
          // original lane's sequential accumulation.
          float part[partials_per_lane];
          for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
              int e = pb * 4 + i;
              state[e] = state[e] * gt;
              acc += state[e] * static_cast<float>(k_[e]);
            }
            part[pb] = acc;
          }
          // Butterfly levels xor 1,2,4 stay inside this lane (commutative
          // pairwise tree); levels xor 8,16 become the row-group shuffles.
          float kv_mem =
              ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
          kv_mem += simd_shuffle_xor(kv_mem, 1);
          kv_mem += simd_shuffle_xor(kv_mem, 2);

          auto delta =
              (static_cast<float>(v_[dv_idx]) - kv_mem) *
              static_cast<float>(beta_[hv_idx]);

          for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
              int e = pb * 4 + i;
              state[e] = state[e] + static_cast<float>(k_[e]) * delta;
              acc += state[e] * static_cast<float>(q_[e]);
            }
            part[pb] = acc;
          }
          float out =
              ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
          out += simd_shuffle_xor(out, 1);
          out += simd_shuffle_xor(out, 2);
          if (lane_in_row == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }

          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }

        for (int i = 0; i < values_per_lane; ++i) {
          o_state[i] = static_cast<StT>(state[i]);
        }
    """


def _build() -> Any:
    global _kernel
    if _kernel is None:
        digest = hashlib.sha256(_SOURCE.encode()).hexdigest()[:16]
        _kernel = mx.fast.metal_kernel(name=f"tf_prefill_gdn_{digest}",
                  input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
                  output_names=["y", "state_out"], source=_SOURCE)
    return _kernel


def packed(q: Any, k: Any, v: Any, g: Any, beta: Any, state: Any) -> tuple[Any, Any]:
    batch, rows, hk, dk = k.shape
    hv, dv = v.shape[2:]
    result = _build()(inputs=[q, k, v, g, beta, state, rows],
             template=[("InT", q.dtype), ("StT", state.dtype), ("Dk", dk), ("Dv", dv), ("Hk", hk), ("Hv", hv)],
             grid=(32, dv // 8, batch * hv), threadgroup=(32, 2, 1),
             output_shapes=[(batch, rows, hv, dv), state.shape], output_dtypes=[q.dtype, state.dtype])
    return result[0], result[1]


def dispatch(q: Any, k: Any, v: Any, g: Any, beta: Any, state: Any, mask: Any = None) -> tuple[Any, Any]:
    if (_available and mask is None and q.shape[1] >= 128 and k.shape[-1] == 128 and v.shape[-1] % 8 == 0
            and g.ndim == 3 and g.dtype == mx.float32 and state.dtype == mx.float32
            and q.dtype in (mx.bfloat16, mx.float16, mx.float32) and mx.default_device() == mx.gpu):
        return packed(q, k, v, g, beta, state)
    return _STOCK(q, k, v, g, beta, state, mask)


def _check(stock: Any) -> bool:
    for dtype in (mx.bfloat16, mx.float16, mx.float32):
        q = (mx.random.normal((1, 256, 2, 128), key=mx.random.key(1)) / 128).astype(dtype)
        k = (mx.random.normal((1, 256, 2, 128), key=mx.random.key(2)) / 12).astype(dtype)
        v = mx.random.normal((1, 256, 6, 128), key=mx.random.key(3)).astype(dtype)
        g = mx.random.uniform(low=.94, high=1., shape=(1, 256, 6), key=mx.random.key(4))
        beta = mx.sigmoid(mx.random.normal((1, 256, 6), key=mx.random.key(5))).astype(dtype)
        state = mx.random.normal((1, 6, 128, 128), key=mx.random.key(6)) * .02
        actual, reference = packed(q, k, v, g, beta, state), stock(q, k, v, g, beta, state)
        same = [mx.array_equal(a.view(mx.uint16 if a.dtype.size == 2 else mx.uint32),
                              b.view(mx.uint16 if b.dtype.size == 2 else mx.uint32))
                for a, b in zip(actual, reference)]
        mx.eval(*same)
        if not all(bool(value.item()) for value in same):
            return False
    return True


def prepare() -> bool:
    """Install only after a device check and bitwise comparison; other shapes retain MLX-LM."""
    global _available, _STOCK
    from mlx_lm.models import gated_delta
    from tensorfold.kernels.qwen.dense.v1 import prompt_attention

    if _STOCK is None:
        _STOCK = gated_delta.gated_delta_kernel
    _available = False
    reason = "disabled or unsupported GPU/runtime"
    if (os.environ.get("TF_PACKED_PREFILL_GDN", "1") != "0" and prompt_attention.native_m5()
            and mx.default_device() == mx.gpu):
        try:
            _available = _check(_STOCK)
            reason = "equality check failed"
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
    gated_delta.gated_delta_kernel = dispatch
    print(f"[tensorfold] packed prompt DeltaNet: {'on' if _available else 'off (' + reason + ')'}", flush=True)
    return _available


def identity() -> str:
    return f"packed-prefill-gdn={'on' if _available else 'off'}"

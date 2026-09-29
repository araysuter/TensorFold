"""MLX's M5 prompt matmul reading lane-tiled Q4 weights directly, without a whole-weight transpose."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import re
from typing import Any

import mlx.core as mx

# The loader below is an address-only adaptation of MLX's QuantizedBlockLoader.
# Keep other versions on the existing path until their header and dispatch have been qualified.
HEADER_SHAS = {
    "0.32.2": "3dc0dfa3dab060ce1f75dcbb9a4c7ae4546693d74f71cafe7658c1e6743af745",
    "0.32.3": "1a9197c4afaec8dddc869165a4328f712c7b69d3a3a946e9cafe6ddebc456027",
}
_available = False
_kernel: Any = None

_LOADER = r"""
template <int NT, typename T, short BROWS, short BCOLS, short dst_ld,
          short reduction_dim, short tgp_size, short group_size, short bits>
struct TFTiledBlockLoader : QuantizedBlockLoader<T, BROWS, BCOLS, dst_ld,
                                                reduction_dim, tgp_size, group_size, bits> {
  using Base = QuantizedBlockLoader<T, BROWS, BCOLS, dst_ld, reduction_dim,
                                    tgp_size, group_size, bits>;
  static_assert(group_size == 64 && BCOLS == 64 && bits == 4 && reduction_dim == 1);
  static_assert(BROWS == 64 && (NT == 32 || NT == 64));
  TFTiledBlockLoader(const device uint8_t* w, const device T* s, const device T* b,
                    int K, threadgroup T* dst, ushort sg, ushort lane) thread
      : Base(w, s, b, K, dst, sg, lane) {
    // tile_weight stores [N/NT, K/64, NT, 32 packed bytes]. Scales stay [N, K/64].
    this->src = w + (int64_t(this->bi / NT) * (K / 64) * NT + this->bi % NT) * 32 + this->bj;
  }
  void next() thread {
    this->src += NT * 32;
    this->scales++;
    this->biases++;
  }
};
"""

_BODY = r"""
  threadgroup bfloat16_t Ws[64 * (64 + 16 / sizeof(bfloat16_t))];
  tf_tiled_qmm_impl<NT, bfloat16_t, 64, 4, true, 64, 64, 64, 2, 2>(
      W, S, B, X, Y, Ws, dims[0], dims[1], dims[2],
      threadgroup_position_in_grid, thread_index_in_threadgroup,
      simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""


def _helper(source: str) -> str:
    """Extract the installed MLX helper and change only the packed-weight reader."""

    at = source.index("METAL_FUNC void qmm_t_nax_tgp_impl(")
    begin = source.rindex("template <", 0, at)
    end, depth = source.index("{", at), 0
    while True:
        depth += {"{": 1, "}": -1}.get(source[end], 0)
        end += 1
        if depth == 0:
            break
    helper = source[begin:end]
    if helper.count("using loader_w_t = QuantizedBlockLoader<") != 1:
        raise ValueError("unrecognized MLX prompt weight loader")
    return (helper.replace("template <", "template <int NT,", 1)
            .replace("qmm_t_nax_tgp_impl", "tf_tiled_qmm_impl", 1)
            .replace("using loader_w_t = QuantizedBlockLoader<", "using loader_w_t = TFTiledBlockLoader<NT,", 1))


def _build() -> Any:
    global _kernel
    if _kernel is None:
        # Reuse the existing include expander; utils.h is already supplied by metal_kernel.
        from tensorfold.kernels.qwen.flash_next.v1.prefill_mm import _inline

        include = Path(mx.__file__).parent / "include" / "mlx/backend/metal/kernels"
        source = (include / "quantized_nax.h").read_text()
        if hashlib.sha256(source.encode()).hexdigest() != HEADER_SHAS.get(version("mlx")):
            raise ValueError("MLX's M5 matmul header has changed")
        seen: set[str] = set()
        header = "\n".join(_inline(f"mlx/backend/metal/kernels/{path}", seen)
                           for path in ("steel/gemm/gemm_nax.h", "quantized_utils.h", "quantized_nax.h"))
        header += "\n" + _LOADER + "\n" + _helper(source)
        digest = hashlib.sha256((header + _BODY).encode()).hexdigest()[:16]
        _kernel = mx.fast.metal_kernel(name=f"tf_tiled_prefill_{digest}",
                                      input_names=["X", "W", "S", "B", "dims"], output_names=["Y"],
                                      header=header, source=_BODY)
    return _kernel


def supported(rows: int, n: int, k: int, bits: int, group: int, nt: int) -> bool:
    """Only products the supported MLX versions send to their unsplit, 64-row M5 QMM."""

    if rows <= 128 or bits != 4 or group != 64 or nt not in (32, 64) or n % 64 or k % 64:
        return False
    # Match MLX's split-K dispatch; changing that reduction tree would change prompt bits.
    split = min(max(1, 512 // (-(-n // 32) * -(-rows // 32))), k // 64)
    while split > 1 and k % (split * 64):
        split -= 1
    return split <= 1


def matmul(x: Any, weight: Any, scales: Any, biases: Any, nt: int) -> Any:
    """The standard M5 QMM arithmetic, with only its packed-weight addresses changed."""

    from tensorfold.kernels.inputs import ints

    k, n = int(x.shape[-1]), int(weight.shape[0])
    rows = int(x.size // k)
    y = _build()(inputs=[x.reshape(rows, k), weight, scales, biases, ints([k, n, rows])],
                 template=[("NT", int(nt))], grid=(-(-n // 64) * 128, -(-rows // 64), 1),
                 threadgroup=(128, 1, 1), output_shapes=[(rows, n)], output_dtypes=[mx.bfloat16])[0]
    return y.reshape(*x.shape[:-1], n)


def active(x: Any, weight: Any, bits: int, group: int, nt: int) -> bool:
    return (_available and x.dtype == mx.bfloat16 and mx.default_device() == mx.gpu
            and supported(int(x.size // x.shape[-1]), int(weight.shape[0]), int(x.shape[-1]), bits, group, nt))


def prepare() -> bool:
    """Fix the process's path at startup, falling back if the runtime or exact-equality check fails."""

    global _available
    _available = False
    reason = "disabled"
    if os.environ.get("TF_TILED_PREFILL", "1") != "0":
        try:
            info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
            arch = re.match(r"applegpu_g(\d+)", str(info.get("architecture", "")))
            if version("mlx") not in HEADER_SHAS or not arch or int(arch[1]) != 17 or mx.default_device() != mx.gpu:
                reason = "requires MLX 0.32.2 or 0.32.3 on an M5 GPU"
            else:
                _available = _check()
                reason = "equality check failed" if not _available else ""
        except Exception as exc:  # noqa: BLE001 - an unavailable custom kernel leaves the existing path intact
            reason = f"{type(exc).__name__}: {exc}"
    print(f"[tensorfold] tiled prompt matmul: {'on' if _available else 'off (' + reason + ')'}", flush=True)
    return _available


def _check() -> bool:
    """Both tiled layouts against MLX, including a partial row tile; no model weights or global PRNG state."""

    from tensorfold.kernels.qwen.dense.v1.lane_qmm import tile_weight

    key = mx.random.key(20260929)
    x = mx.random.normal((129, 256), key=key).astype(mx.bfloat16)
    w = mx.random.normal((4096, 256), key=mx.random.split(key)[0]).astype(mx.bfloat16)
    q, s, b = mx.quantize(w, group_size=64, bits=4)
    ref = mx.quantized_matmul(x, q, s, b, transpose=True, group_size=64, bits=4)
    same = [mx.all(matmul(x, tile_weight(q, nt, bits=4), s, b, nt).view(mx.uint16) == ref.view(mx.uint16))
            for nt in (32, 64)]
    mx.eval(*same)
    return bool(mx.all(mx.isfinite(ref)).item()) and all(bool(v.item()) for v in same)


def identity() -> str:
    """Do not reuse a prefix from another process's prefill path."""

    return f"tiled-prefill={'on' if _available else 'off'}"

"""Snapshot identities include prompt kernels, their modes, and library versions."""

from importlib import metadata
from types import SimpleNamespace

from tensorfold import cli, families
from tensorfold.families import qwen4_exp
from tensorfold.server import app


def test_snapshot_identity_changes_with_mlx_lm_version(monkeypatch, tmp_path):
    captured = []

    class Captured(Exception):
        pass

    def capture(*args, **kwargs):
        captured.append(kwargs["model_id"])
        raise Captured

    monkeypatch.setattr(app, "ChatApp", capture)
    monkeypatch.setattr(families, "kernel_version", lambda *args: "fixed-kernels")
    family = SimpleNamespace(title="fake", model_type="fake", package=SimpleNamespace(load=lambda *a, **k: (None, None)))
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--no-drafts", "--prompt-cache-gib", "0"])
    for version in ("0.31.3", "0.31.4"):
        monkeypatch.setattr(metadata, "version", lambda name, value=version: value)
        try:
            cli._serve_mlx(args, family, tmp_path, 0, [], 1 << 30)
        except Captured:
            pass
    assert len(captured) == 2 and captured[0] != captured[1]
    assert "mlx_lm=0.31.3" in captured[0] and "mlx_lm=0.31.4" in captured[1]


def test_flash_prompt_attention_mode_changes_the_family_key(monkeypatch):
    family = families.families()["qwen4_exp"]
    model = SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(kernel_select=False))])
    dense = families.kernel_version(family, model)
    model.layers[0].self_attn.kernel_select = True
    sparse = families.kernel_version(family, model)
    assert dense != sparse
    monkeypatch.setenv("TF_FLASH_FUSED", "0")
    assert families.kernel_version(family, model) == sparse
    assert callable(qwen4_exp.kernel_version)


def test_flash_key_still_tracks_kernel_sources(monkeypatch):
    family = families.families()["qwen4_exp"]
    before = families.kernel_version(family, None)
    monkeypatch.setattr(qwen4_exp, "KERNEL_VERSION", "test-revision")
    assert families.kernel_version(family, None) != before


def test_flash_key_carries_the_prefill_key():
    family = families.families()["qwen4_exp"]
    model = SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(kernel_select=True))],
                            prefill_key="flash-prefill=fast;architecture=x;matmul=custom")
    fast = families.kernel_version(family, model)
    model.prefill_key = "flash-prefill=mlx"
    assert families.kernel_version(family, model) != fast and "prompt_attention=1" in fast


def test_dense_prefill_modes_change_snapshot_keys(monkeypatch):
    from tensorfold.families import qwen3_5
    from tensorfold.kernels.qwen.dense.v1 import prefill_gdn, prompt_attention

    monkeypatch.setattr(prompt_attention, "native_m5", lambda: True)
    model = SimpleNamespace(_tensorfold_lanes=True)
    monkeypatch.delenv("TF_NATIVE_PREFILL_ATTENTION", raising=False)
    native = qwen3_5.kernel_version(model)
    monkeypatch.setenv("TF_NATIVE_PREFILL_ATTENTION", "0")
    bounded = qwen3_5.kernel_version(model)
    assert native != bounded
    monkeypatch.setattr(prefill_gdn, "_available", not prefill_gdn._available)
    assert qwen3_5.kernel_version(model) != bounded

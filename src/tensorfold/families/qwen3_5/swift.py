"""Compatibility for the pinned Swift snapshot, preserving the tested BF16 scale conversion."""
def convert_norms(weights):
    import mlx.core as mx
    suffixes = (".input_layernorm.weight", ".post_attention_layernorm.weight",
                ".q_norm.weight", ".k_norm.weight", "model.norm.weight")
    keys = [k for k, v in weights.items() if k.endswith(suffixes) and v.ndim == 1]
    if len(keys) != 161:
        raise ValueError(f"Expected 161 Swift RMSNorm scales, found {len(keys)}")
    result = dict(weights)
    for key in keys:
        value = weights[key]
        result[key] = (value.astype(mx.float32) + 1.0).astype(value.dtype)
    return result

"""MLX affine encoders with per-group reconstruction-error fallback to the standard encoder."""
import mlx.core as mx


def error_groups(value, q, scales, biases, group_size, bits):
    restored = mx.dequantize(q, scales, biases, group_size=group_size, bits=bits)
    error = restored.astype(mx.float32) - value.astype(mx.float32)
    return mx.sum(error.reshape(*value.shape[:-1], -1, group_size)**2, axis=-1)


def quantize(value, *, bits, group_size, mode='affine', method='affine'):
    q, scales, biases = mx.quantize(value, bits=bits, group_size=group_size, mode=mode)
    if method == 'affine' or bits >= 8:
        return q, scales, biases
    if method != 'mse' or mode != 'affine':
        raise ValueError('Unsupported quantization method')
    shape = value.shape
    groups = value.astype(mx.float32).reshape(*shape[:-1], -1, group_size)
    words = group_size * bits // 32
    best_error = error_groups(value, q, scales, biases, group_size, bits)
    mx.eval(q, scales, biases, best_error)

    def choose(candidate_q, candidate_s, candidate_b):
        nonlocal q, scales, biases, best_error
        candidate_error = error_groups(value, candidate_q, candidate_s, candidate_b, group_size, bits)
        better = candidate_error < best_error
        q = mx.where(better[..., None], candidate_q.reshape(*groups.shape[:-1], words),
                     q.reshape(*groups.shape[:-1], words)).reshape(q.shape)
        scales = mx.where(better, candidate_s, scales)
        biases = mx.where(better, candidate_b, biases)
        best_error = mx.minimum(best_error, candidate_error)
        mx.eval(q, scales, biases, best_error)

    low = mx.min(groups, axis=-1, keepdims=True)
    high = mx.max(groups, axis=-1, keepdims=True)
    if bits == 1:
        if group_size != 32:
            raise ValueError('The binary MSE encoder requires groups of 32')
        for _ in range(12):
            lower = groups <= (low + high) * 0.5
            count = mx.sum(lower, axis=-1, keepdims=True)
            new_low = mx.sum(mx.where(lower, groups, 0), axis=-1, keepdims=True) / mx.maximum(count, 1)
            new_high = mx.sum(mx.where(lower, 0, groups), axis=-1, keepdims=True) / mx.maximum(group_size - count, 1)
            low = mx.where(count > 0, new_low, low)
            high = mx.where(count < group_size, new_high, high)
            mx.eval(low, high)
        candidate_s = (high - low)[..., 0].astype(value.dtype)
        candidate_b = low[..., 0].astype(value.dtype)
        threshold = candidate_b.astype(mx.float32) + candidate_s.astype(mx.float32) * 0.5
        codes = (groups > threshold[..., None]).astype(mx.uint32)
        candidate_q = mx.sum(codes << mx.arange(32, dtype=mx.uint32), axis=-1).astype(mx.uint32)
        choose(candidate_q.reshape(q.shape), candidate_s, candidate_b)
    else:
        center = (low + high) * 0.5
        radius = (high - low) * 0.5
        factors = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7] if bits == 2 else [0.99, 0.97, 0.95, 0.92, 0.9]
        for factor in factors:
            clipped = mx.clip(groups, center - radius * factor, center + radius * factor)
            candidate = mx.quantize(clipped.reshape(shape).astype(value.dtype), bits=bits,
                                    group_size=group_size, mode=mode)
            choose(*candidate)
    return q, scales, biases

"""Validate restricted text-chat profiles before queuing or loading an inference backend."""
import math


def finite(value):
    return abs(value) <= 1e12 and (not isinstance(value, float) or math.isfinite(value))


def validate_body(body, name, context):
    if not isinstance(body, dict) or body.get('model', name) != name:
        raise ValueError('Unknown model')
    messages = body.get('messages')
    if (not isinstance(messages, list) or not messages or any(
            not isinstance(m, dict) or m.get('role') not in ('system', 'user', 'assistant')
            or not isinstance(m.get('content'), str) or m.get('tool_calls') for m in messages)):
        raise ValueError('This model supports text chat messages only')
    unsupported = ['tools', 'functions', 'tool_choice', 'function_call', 'adapters', 'draft_model',
                   'chat_template_kwargs', 'role_mapping', 'response_format', 'grammar', 'max_output_tokens']
    if any(body.get(key) is not None for key in unsupported):
        raise ValueError('This text-chat profile does not support tools, drafts, or constrained output')
    maximum = body.get('max_completion_tokens', body.get('max_tokens', 4096))
    if type(maximum) is not int or not 1 <= maximum < context:
        raise ValueError('Invalid output token budget')
    if body.get('n', 1) != 1 or not isinstance(body.get('stream', False), bool):
        raise ValueError('One completion is supported')
    for key in ['temperature', 'top_p', 'top_k', 'min_p', 'repetition_penalty', 'presence_penalty',
                'frequency_penalty', 'xtc_probability', 'xtc_threshold']:
        if key in body and (type(body[key]) not in (int, float) or not finite(body[key])):
            raise ValueError('Sampling parameters must be finite numbers')
    seed = body.get('seed')
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
        raise ValueError('Invalid random seed')
    bias = body.get('logit_bias')
    if bias is not None:
        if not isinstance(bias, dict) or any(type(v) not in (int, float) or not finite(v) for v in bias.values()):
            raise ValueError('Logit biases must be finite numbers')
    options = body.get('stream_options')
    if options is not None and (not isinstance(options, dict) or type(options.get('include_usage', False)) is not bool):
        raise ValueError('Invalid stream options')
    stop = body.get('stop')
    if stop is not None and not (isinstance(stop, str) or isinstance(stop, list) and all(isinstance(s, str) for s in stop)):
        raise ValueError('Invalid stop sequences')
    return maximum


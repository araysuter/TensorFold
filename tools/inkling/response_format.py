"""Inkling response channels for the upstream streaming text state machine."""
from tool_format import parse_tool_call


def configure(tokenizer):
    end = '<|end_message|>'
    start = '<|content_invoke_tool_json|>'
    tokenizer._tool_call_start = start
    tokenizer._tool_call_end = end
    tokenizer._tool_parser = parse_tool_call
    if hasattr(tokenizer, 'encode'):
        tokenizer._tool_call_start_tokens = tuple(tokenizer.encode(start, add_special_tokens=False))
    channels = [('<|content_thinking|>', 'reasoning'), ('<|content_text|>', 'normal'),
                ('<|content_tool_error|>', 'normal'), ('<|content_invoke_tool_text|>', 'normal'),
                ('<|content_xml|>', 'normal'), (start, 'tool')]
    boundaries = [('<|message_model|>', 'header'), (end, 'header'), ('<|end_turn|>', 'normal')]
    tokenizer._response_format = {
        'initial_state': 'normal',
        'transitions': {
            'normal': channels + boundaries,
            'reasoning': channels + boundaries,
            'header': channels + boundaries,
            'tool': [(end, 'header')],
        },
    }

"""Inkling response channels for the upstream streaming text state machine."""


def configure(tokenizer):
    thinking = '<|message_model|><|content_thinking|>'
    content = '<|message_model|><|content_text|>'
    end = '<|end_message|>'
    tokenizer._response_format = {
        'initial_state': 'normal',
        'transitions': {
            'normal': [(thinking, 'reasoning'), (content, 'normal'), (end, 'normal'), ('<|end_turn|>', 'normal')],
            'reasoning': [(end + content, 'normal'), (end + thinking, 'reasoning'), ('<|end_turn|>', 'normal')],
        },
    }

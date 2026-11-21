import unittest
from types import SimpleNamespace
from response_format import configure

try:
    from mlx_lm.generate import TextStateMachine
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, 'Requires the pinned MLX-LM environment')
class ChannelTests(unittest.TestCase):
    def test_named_tool_header_does_not_leak_into_content(self):
        tokenizer = SimpleNamespace(has_thinking=False, has_tool_calling=False, structural_markers=())
        configure(tokenizer)
        tokenizer.has_tool_calling = True
        tokenizer.tool_call_start = tokenizer._tool_call_start
        tokenizer.tool_call_end = tokenizer._tool_call_end
        tokenizer.response_transitions = tokenizer._response_format['transitions']
        text = ('<|message_model|><|content_text|>I will check.'
                '<|end_message|><|message_model|>Shell<|content_invoke_tool_json|>'
                '{"name":"Shell","args":{"command":"pwd"}}<|end_message|>')
        for size in (1, 7, len(text)):
            with self.subTest(size=size):
                reader = TextStateMachine.from_tokenizer(tokenizer).reader(state='normal')
                deltas = [reader.feed(text[i:i + size]) for i in range(0, len(text), size)]
                deltas.append(reader.finish(''))
                self.assertEqual(''.join(d.content for d in deltas), 'I will check.')
                self.assertEqual(tuple(t for d in deltas for t in d.tool_calls),
                                 ('{"name":"Shell","args":{"command":"pwd"}}',))

    def test_reasoning_two_calls_and_final_answer_across_every_boundary(self):
        tokenizer = SimpleNamespace(has_thinking=False, has_tool_calling=False, structural_markers=())
        configure(tokenizer)
        tokenizer.has_tool_calling = True
        tokenizer.tool_call_start = tokenizer._tool_call_start
        tokenizer.tool_call_end = tokenizer._tool_call_end
        tokenizer.response_transitions = tokenizer._response_format['transitions']
        first = '{"name":"Shell","args":{"command":"pwd"}}'
        second = '{"name":"Shell","args":{"command":"ls"}}'
        text = ('<|message_model|><|content_thinking|>Check first.<|end_message|>'
                '<|message_model|>Shell<|content_invoke_tool_json|>' + first + '<|end_message|>'
                '<|message_model|>Shell<|content_invoke_tool_json|>' + second + '<|end_message|>'
                '<|message_model|><|content_text|>Done.<|end_message|>')
        for split in range(len(text) + 1):
            reader = TextStateMachine.from_tokenizer(tokenizer).reader(state='normal')
            delta = reader.feed(text[:split]) + reader.feed(text[split:]) + reader.finish()
            self.assertEqual(delta.content, 'Done.')
            self.assertEqual(delta.reasoning, 'Check first.')
            self.assertEqual(delta.tool_calls, (first, second))

    def test_split_markers_separate_reasoning_from_content(self):
        tokenizer = SimpleNamespace(has_thinking=False, has_tool_calling=False, structural_markers=())
        configure(tokenizer)
        tokenizer.response_transitions = tokenizer._response_format['transitions']
        text = '<|message_model|><|content_thinking|>Work.<|end_message|><|message_model|><|content_text|>Answer.<|end_message|><|end_turn|>'
        for size in (1, 7, len(text)):
            with self.subTest(size=size):
                reader = TextStateMachine.from_tokenizer(tokenizer).reader(state='normal')
                deltas = [reader.feed(text[i:i + size]) for i in range(0, len(text), size)]
                deltas.append(reader.finish(''))
                self.assertEqual(''.join(d.content for d in deltas), 'Answer.')
                self.assertEqual(''.join(d.reasoning for d in deltas), 'Work.')


if __name__ == '__main__':
    unittest.main()

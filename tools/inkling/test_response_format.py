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

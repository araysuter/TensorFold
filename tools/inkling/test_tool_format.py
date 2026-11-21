import json
import unittest
from tool_format import parse_tool_call, prepare_request_tools


TOOLS = [{'type': 'function', 'function': {'name': 'Shell'}},
         {'type': 'function', 'function': {'name': 'lookup_number'}}]


class ToolFormatTests(unittest.TestCase):
    def test_none_removes_active_tools_without_mutating_history(self):
        body = {'tool_choice': 'none', 'tools': TOOLS,
                'messages': [{'role': 'assistant', 'tool_calls': [{'id': 'existing'}]}]}
        result = prepare_request_tools(body)
        self.assertEqual(result['tools'], [])
        self.assertEqual(body['tools'], TOOLS)
        self.assertIs(result['messages'], body['messages'])
        self.assertIs(prepare_request_tools({'tool_choice': 'auto'}).get('tools'), None)
        self.assertIs(prepare_request_tools(body | {'tool_choice': 'auto'})['tools'], TOOLS)

    def test_native_args_and_standard_arguments(self):
        for key in ('args', 'arguments'):
            with self.subTest(key=key):
                result = parse_tool_call(json.dumps({'name': 'Shell', key: {'command': 'pwd'}}), TOOLS)
                self.assertEqual(result, {'name': 'Shell', 'arguments': {'command': 'pwd'}})

    def test_string_arguments_and_zero_argument_call(self):
        self.assertEqual(parse_tool_call('{"name":"Shell","args":"{\\"command\\":\\"pwd\\"}"}', TOOLS),
                         {'name': 'Shell', 'arguments': {'command': 'pwd'}})
        self.assertEqual(parse_tool_call('{"name":"Shell"}', TOOLS), {'name': 'Shell', 'arguments': {}})

    def test_multiple_calls_and_no_model_selected_id(self):
        calls = parse_tool_call('[{"name":"Shell","args":{},"id":"reuse"},'
                                '{"name":"lookup_number","args":{"key":"alpha"}}]', TOOLS)
        self.assertEqual(len(calls), 2)
        self.assertNotIn('id', calls[0])

    def test_rejects_undeclared_tools_and_invalid_arguments(self):
        bad = ['{"name":"unknown","args":{}}', '{"name":"Shell","args":[]}',
               '{"name":"Shell","args":null}', '{"name":"Shell","args":{},"arguments":{"x":1}}',
               '{"args":{}}', '[]', 'null', '{"name":"Shell","args":']
        for text in bad:
            with self.subTest(text=text), self.assertRaises((ValueError, TypeError)):
                parse_tool_call(text, TOOLS)
        with self.assertRaises(ValueError):
            parse_tool_call('{"name":"Shell","args":{}}', [])


if __name__ == '__main__':
    unittest.main()

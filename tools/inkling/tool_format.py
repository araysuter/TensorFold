"""Translate Inkling's native JSON calls into the upstream function-call contract."""
import json


def prepare_request_tools(body):
    if isinstance(body, dict) and body.get('tool_choice') == 'none':
        return body | {'tools': []}
    return body


def parse_tool_call(text, tools=None):
    payload = json.loads(text)
    items = payload if isinstance(payload, list) else [payload]
    if not items or len(items) > 128:
        raise ValueError('Invalid tool-call count')
    allowed = None if tools is None else {
        tool['function']['name'] for tool in tools
        if isinstance(tool, dict) and tool.get('type') == 'function'
        and isinstance(tool.get('function'), dict) and isinstance(tool['function'].get('name'), str)
    }
    calls = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get('name'), str) or not item['name']:
            raise ValueError('Invalid tool-call name')
        if allowed is not None and item['name'] not in allowed:
            raise ValueError('Tool was not declared by the request')
        if 'args' in item and 'arguments' in item and item['args'] != item['arguments']:
            raise ValueError('Conflicting tool-call arguments')
        arguments = item.get('args', item.get('arguments', {}))
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise ValueError('Tool-call arguments must be an object')
        calls.append({'name': item['name'], 'arguments': arguments})
    return calls if isinstance(payload, list) else calls[0]

"""Check streamed/nonstreamed tool calls, tool-result replay and repeated-call IDs."""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

from bench_auth import authorize, read_api_key, redact


TOOLS = [{'type': 'function', 'function': {
    'name': 'lookup_number', 'description': 'Look up the number for a key; call this to obtain its value.',
    'parameters': {'type': 'object', 'properties': {'key': {'type': 'string', 'enum': ['alpha']}},
                   'required': ['key'], 'additionalProperties': False}}}]
PROMPT = 'Call lookup_number with key alpha. Do not guess. After the tool result, reply RESULT=<value>.'


def request(base, key, body, timeout):
    query = urllib.request.Request(base.rstrip('/') + '/v1/chat/completions',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json',
                                               'User-Agent': 'TensorFold-Tool-Qualification/1.0'})
    authorize(query, key)
    started = time.monotonic()
    try:
        with urllib.request.urlopen(query, timeout=timeout) as response:
            if not body.get('stream'):
                data = json.load(response)
                return {'status': response.status, 'elapsed_s': time.monotonic() - started,
                        'message': data['choices'][0]['message'], 'finish_reason': data['choices'][0]['finish_reason']}
            message, calls, finish, done = {'role': 'assistant', 'content': ''}, {}, None, False
            for raw in response:
                line = raw.decode().strip()
                if line == 'data: [DONE]':
                    done = True
                    break
                if not line.startswith('data:'):
                    continue
                chunk = json.loads(line[5:])
                if 'error' in chunk:
                    raise ValueError('Stream error')
                for choice in chunk.get('choices', []):
                    finish = choice.get('finish_reason') or finish
                    delta = choice.get('delta', {})
                    for field in ('content', 'reasoning', 'reasoning_content'):
                        if delta.get(field):
                            message[field] = message.get(field, '') + delta[field]
                    for tool in delta.get('tool_calls', []):
                        tc = calls.setdefault(tool['index'], {'id': '', 'type': 'function',
                                                             'function': {'name': '', 'arguments': ''}})
                        if tool.get('id'):
                            tc['id'] = tool['id']
                        function = tool.get('function', {})
                        for field in ('name', 'arguments'):
                            tc['function'][field] += function.get(field, '')
            if not done:
                raise ValueError('Stream did not finish')
            if calls:
                message['tool_calls'] = [calls[i] for i in sorted(calls)]
            return {'status': response.status, 'elapsed_s': time.monotonic() - started,
                    'message': message, 'finish_reason': finish, 'stream_done': done}
    except urllib.error.HTTPError as exc:
        return {'status': exc.code, 'error': 'http_error'}
    except Exception as exc:
        return {'status': None, 'error': type(exc).__name__}


def run_case(base, key, model, streaming, timeout, maximum):
    row = {'model': model, 'streaming': streaming, 'passed': False, 'replies': []}
    messages, ids = [{'role': 'user', 'content': PROMPT}], []

    def send(choice='auto'):
        body = {'model': model, 'messages': messages, 'tools': TOOLS, 'tool_choice': choice,
                'temperature': 0, 'max_tokens': maximum, 'stream': streaming,
                'reasoning_effort': 'none', 'chat_template_kwargs': {'reasoning_effort': 'none'}}
        reply = request(base, key, body, timeout)
        row['replies'].append(reply)
        if reply['status'] != 200:
            raise ValueError('Request was not admitted')
        return reply

    try:
        for value in (731946, 294617):
            reply = send()
            calls = reply['message'].get('tool_calls', [])
            if len(calls) != 1 or reply['finish_reason'] != 'tool_calls':
                raise ValueError('Expected one structured tool call and tool_calls finish reason')
            call = calls[0]
            if not call.get('id') or call['id'] in ids:
                raise ValueError('Tool-call ID missing or reused within the conversation')
            ids.append(call['id'])
            if call['type'] != 'function' or call['function']['name'] != 'lookup_number':
                raise ValueError('Wrong function')
            if json.loads(call['function']['arguments']) != {'key': 'alpha'}:
                raise ValueError('Wrong function arguments')
            if '<|message_model|>' in (reply['message'].get('content') or ''):
                raise ValueError('Native tool header leaked into content')
            messages.extend([reply['message'], {'role': 'tool', 'tool_call_id': call['id'],
                             'name': 'lookup_number', 'content': json.dumps({'value': value})}])
            answer = send()
            if answer['message'].get('tool_calls'):
                extra = answer['message']['tool_calls']
                if (len(extra) != 1 or answer['finish_reason'] != 'tool_calls'
                        or not extra[0].get('id') or extra[0]['id'] in ids
                        or extra[0].get('type') != 'function'
                        or extra[0].get('function', {}).get('name') != 'lookup_number'
                        or json.loads(extra[0]['function']['arguments']) != {'key': 'alpha'}):
                    raise ValueError('Invalid additional auto tool call')
                ids.append(extra[0]['id'])
                row['extra_auto_calls'] = row.get('extra_auto_calls', 0) + 1
                answer = send('none')
                row['forced_final_reply'] = True
            if (str(value) not in (answer['message'].get('content') or '')
                    or answer['message'].get('tool_calls') or answer['finish_reason'] != 'stop'):
                raise ValueError('Final answer did not use the supplied tool result')
            messages.extend([answer['message'], {'role': 'user', 'content':
                'The database value changed. Call lookup_number with alpha once more. '
                'After its new result, reply only RESULT=<value> using the new value.'}])
        row['passed'] = True
    except Exception as exc:
        row['failure'] = str(exc)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', help='API root without /v1')
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--api-key-env')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=420)
    parser.add_argument('--max-tokens', type=int, default=512)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 1800 or not 64 <= args.max_tokens <= 4096:
        parser.error('Use bounded positive timeout and output budget')
    key = read_api_key(parser, args.api_key_env)
    report = {'status': 'running', 'cases': []}
    try:
        for model in args.models:
            for streaming in (False, True):
                row = run_case(args.base, key, model, streaming, args.timeout, args.max_tokens)
                report['cases'].append(row)
                args.output.write_text(json.dumps(redact(report, key), indent=2) + '\n')
                print(model, 'stream' if streaming else 'nonstream', row['passed'], row.get('failure', ''), flush=True)
        report['status'] = 'passed' if all(r['passed'] for r in report['cases']) else 'failed'
    finally:
        args.output.write_text(json.dumps(redact(report, key), indent=2) + '\n')
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Exercise the shared API, warm chat, tenant isolation and full-window SSD restoration on the Studio."""
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import subprocess
import threading
import time
from urllib.parse import urlsplit


class Client:
    def __init__(self, endpoint, credential):
        self.url, self.credential = urlsplit(endpoint), credential

    def request(self, method, path, body=None, context=None, authorized=True):
        cls = http.client.HTTPSConnection if self.url.scheme == 'https' else http.client.HTTPConnection
        connection = cls(self.url.hostname, self.url.port, timeout=1800)
        headers = {'Content-Type': 'application/json'}
        if authorized:
            headers['Authorization'] = 'Bearer ' + self.credential
        if context:
            headers['X-TensorFold-Context'] = context
        started = time.monotonic()
        connection.request(method, self.url.path.rstrip('/') + path,
                           None if body is None else json.dumps(body).encode(), headers)
        response = connection.getresponse()
        raw = response.read().decode()
        result = {'status': response.status, 'seconds': time.monotonic() - started,
                  'context_status': response.getheader('X-TensorFold-Context-Status')}
        connection.close()
        if body and body.get('stream'):
            result['events'] = [json.loads(s[6:]) for s in raw.splitlines()
                                if s.startswith('data: ') and s != 'data: [DONE]']
            result['done'] = 'data: [DONE]' in raw
        else:
            result['body'] = json.loads(raw)
        return result


def keys(path):
    result = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        label, token = line.strip().split(':', 1)
        result[label] = token
    return result


def tokens(reply):
    return [item['id'] for item in reply['body']['choices'][0]['logprobs']['content']]


def cached(reply):
    return reply['body']['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0)


def message(reply):
    item = reply['body']['choices'][0]['message']
    if item.get('reasoning'):
        raise ValueError('This cache probe expects a short answer without reasoning')
    return {'role': 'assistant', 'content': item.get('content') or ''}


def summarize(reply):
    result = {k: reply[k] for k in ['status', 'seconds', 'context_status']}
    if reply.get('body', {}).get('choices'):
        result['usage'] = reply['body']['usage']
        result['token_sha256'] = hashlib.sha256(json.dumps(tokens(reply)).encode()).hexdigest()
        result['finish_reason'] = reply['body']['choices'][0]['finish_reason']
    return result


def main(args):
    if args.output.exists():
        raise ValueError('Use a new receipt path')
    credentials = keys(args.key_file)
    owner = Client(args.endpoint, credentials['owner'])
    guest = Client(args.endpoint, credentials['friend'])
    row = {'status': 'running', 'context': 131072, 'runtime': 'temporary-mlx', 'dflash2': False,
           'checks': {}, 'max_resident_model_processes': 0}
    stop = threading.Event()

    def save():
        args.output.write_text(json.dumps(row, indent=2) + '\n')

    def monitor():
        while not stop.wait(.2):
            lines = subprocess.check_output(['ps', '-ax', '-o', 'args=']).decode().splitlines()
            count = sum('--supervise' not in s and ('/worker.py --model ' in s
                        or '/tensorfold-gateway-native serve ' in s) for s in lines)
            row['max_resident_model_processes'] = max(row['max_resident_model_processes'], count)

    threading.Thread(target=monitor, daemon=True).start()

    def check(name, value):
        if not value:
            row['status'] = 'failed'
            row['failed_check'] = name
            save()
            raise ValueError('API check failed: ' + name)
        row['checks'][name] = True
        save()

    def chat(messages, model='ripple-65b-preview', maximum=32, **extra):
        result = {'model': model, 'messages': messages, 'max_tokens': maximum, 'temperature': 0, **extra}
        if model == 'ripple-65b-preview':
            result['logprobs'] = True
        return result

    first = [{'role': 'user', 'content': 'What is 17 multiplied by 23? Give only the final number.'}]
    try:
        for client in [owner, guest]:
            result = client.request('GET', '/v1/models')
            check('model_discovery_' + ('owner' if client is owner else 'guest'),
                  result['status'] == 200 and {v['id'] for v in result['body']['data']} >=
                  {'swift-1.5', 'ornith-1.5-122b-mixed-2bit', 'ripple-65b-preview'})
        check('unauthenticated_refused', owner.request('GET', '/v1/models', authorized=False)['status'] == 401)
        check('public_controls_refused', owner.request('POST', '/v1/internal/session/clear', {})['status'] == 404)
        cold = owner.request('POST', '/v1/chat/completions', chat(first), 'ripple-smoke-owner')
        check('ripple_cold_answer', cold['status'] == 200 and '391' in message(cold)['content'] and cached(cold) == 0)
        row['cold'] = summarize(cold)
        history = first + [message(cold), {'role': 'user', 'content': 'Add 9 to that. Give only the final number.'}]
        warm = owner.request('POST', '/v1/chat/completions', chat(history), 'ripple-smoke-owner')
        check('warm_followup', warm['status'] == 200 and cached(warm) > 0 and '400' in message(warm)['content'])
        row['warm'] = summarize(warm)
        fresh = owner.request('POST', '/v1/chat/completions', chat(history), 'ripple-smoke-fresh')
        check('warm_equals_fresh_tokens', fresh['status'] == 200 and tokens(warm) == tokens(fresh) and cached(fresh) == 0)
        other = guest.request('POST', '/v1/chat/completions', chat(history), 'ripple-smoke-owner')
        check('same_context_id_tenant_isolation', other['status'] == 200 and cached(other) == 0)
        row['fresh'], row['guest'] = summarize(fresh), summarize(other)
        continued = history + [message(warm), {'role': 'user', 'content': 'Subtract 1. Give only the final number.'}]
        restored = owner.request('POST', '/v1/chat/completions', chat(continued), 'ripple-smoke-owner')
        check('ssd_restore_positive_cache', restored['status'] == 200 and cached(restored) > 0
              and restored['context_status'] == 'resumed' and '399' in message(restored)['content'])
        reference = owner.request('POST', '/v1/chat/completions', chat(continued), 'ripple-restored-fresh')
        check('ssd_restore_matches_fresh', reference['status'] == 200 and tokens(restored) == tokens(reference))
        row['restored'] = summarize(restored)
        stream_history = continued + [message(restored), {'role': 'user', 'content': 'What was the original product?'}]
        stream = owner.request('POST', '/v1/chat/completions', chat(stream_history, stream=True,
                               stream_options={'include_usage': True}), 'ripple-smoke-owner')
        check('streaming_usage_and_done', stream['status'] == 200 and stream['done'] and any(
              e.get('usage', {}).get('prompt_tokens_details', {}).get('cached_tokens', 0) > 0 for e in stream['events']))
        stream_text = ''.join(c.get('delta', {}).get('content', '') for e in stream['events'] for c in e['choices'])
        swap_history = stream_history + [{'role': 'assistant', 'content': stream_text},
                                        {'role': 'user', 'content': 'Repeat the original product as a number.'}]
        for model in ['swift-1.5', 'ornith-1.5-122b-mixed-2bit']:
            response = owner.request('POST', '/v1/chat/completions', chat(first, model, maximum=16), 'existing-model-smoke')
            row['existing_native_response'] = {'status': response['status'],
                                               'error': response['body'].get('error')}
            check('existing_model_' + model, response['status'] == 200 and response['body']['usage']['completion_tokens'] > 0)
            response = owner.request('POST', '/v1/chat/completions', chat(swap_history), 'ripple-smoke-owner')
            check('swap_back_from_' + model, response['status'] == 200 and cached(response) > 0
                  and response['context_status'] == 'resumed' and '391' in message(response)['content'])
            swap_history += [message(response), {'role': 'user', 'content': 'Repeat that same number.'}]
        if args.capacity:
            from mlx_lm.tokenizer_utils import load
            from probe import long_prompt
            tokenizer = load(args.model)
            target = 130944
            count = target - 6
            for _ in range(5):
                content = tokenizer.decode(long_prompt(tokenizer, count))
                large = [{'role': 'user', 'content': content}]
                prompt_count = len(tokenizer.apply_chat_template(large, tokenize=True, add_generation_prompt=True))
                if prompt_count == target:
                    break
                count += target - prompt_count
            check('capacity_prompt_exact', prompt_count == target)
            before = owner.request('GET', '/stats')
            large_reply = owner.request('POST', '/v1/chat/completions', chat(large, maximum=64), 'ripple-capacity')
            private_reply = args.output.with_suffix('.capacity-reply.private.json')
            private_reply.write_text(json.dumps(large_reply, indent=2) + '\n')
            private_reply.chmod(0o600)
            check('capacity_generation', large_reply['status'] == 200 and cached(large_reply) == 0
                  and large_reply['body']['usage']['prompt_tokens'] == target)
            row['capacity_marker_observed'] = 'JADE-4821-COMET' in message(large_reply)['content']
            row['capacity_cold'] = summarize(large_reply)
            row['capacity_stats'] = owner.request('GET', '/stats').get('body')
            check('capacity_swap_to_swift', owner.request('POST', '/v1/chat/completions',
                  chat(first, 'swift-1.5', maximum=16), 'existing-model-smoke')['status'] == 200)
            followup = large + [message(large_reply), {'role': 'user', 'content': 'Repeat the verification code.'}]
            followup_tokens = len(tokenizer.apply_chat_template(followup, tokenize=True, add_generation_prompt=True))
            maximum = min(32, 131072 - followup_tokens)
            check('capacity_followup_fits', maximum >= 12)
            resumed = owner.request('POST', '/v1/chat/completions', chat(followup, maximum=maximum), 'ripple-capacity')
            check('capacity_ssd_resume', resumed['status'] == 200 and cached(resumed) > 130900
                  and resumed['context_status'] == 'resumed')
            row['capacity_restored_marker_observed'] = 'JADE-4821-COMET' in message(resumed)['content']
            row['full_context_task_accuracy_qualified'] = False
            row['capacity_restored'] = summarize(resumed)
        check('one_resident_model', row['max_resident_model_processes'] == 1)
        row['status'] = 'complete'
        save()
    except BaseException as exc:
        row['status'] = 'failed'
        row['error_type'] = type(exc).__name__
        raise
    finally:
        stop.set()
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8080')
    parser.add_argument('--key-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--capacity', action='store_true')
    main(parser.parse_args())

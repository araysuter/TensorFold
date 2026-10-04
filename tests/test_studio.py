from types import SimpleNamespace
from io import BytesIO
import json
import pytest
from tensorfold.server.studio import Studio, conversation_key
from tensorfold.server.errors import RequestError
from tensorfold.server.http import make_handler
from tensorfold.server.authentication import KeyStore


def job(n, tokens=None, key=None):
    return SimpleNamespace(job_id=str(n), conversation_id=key or str(n), prompt_ids=tokens or [n], error=None)


def test_capacity_and_busy_identity_are_protected():
    s = Studio(2)
    s.submit(job(1)); s.submit(job(2))
    with pytest.raises(RequestError): s.submit(job(3))
    with pytest.raises(RequestError): s.submit(job(4, key='1'))
    s.finish(job(1)); s.submit(job(3))
    assert s.jobs == {'2': 1, '3': 0}


def test_disk_residency_and_restart(tmp_path):
    s = Studio(2, tmp_path, 'model')
    a = job(1, [1, 2, 3]); s.submit(a); s.finish(a)
    file = tmp_path / 'cache'; file.write_bytes(b'abcd')
    s.residency([], [(file, [1, 2])])
    assert s.snapshot()['slots'][0]['state'] == 'ssd'
    assert s.snapshot()['slots'][0]['checkpoint_tokens'] == 2
    s.save()
    restored = Studio(2, tmp_path, 'model')
    restored.residency([], [(file, [1, 2])])
    assert restored.snapshot()['slots'][0]['state'] == 'ssd'
    restored.submit(job(2, [1, 2, 3, 4], key='1'))
    assert restored.jobs['2'] == 0
    assert Studio(2, tmp_path, 'different model').keys == {}
    file.unlink(); s.residency([], [])
    assert s.snapshot()['slots'][0]['state'] == 'uncached'


def test_no_prompt_or_identity_in_metrics():
    s = Studio(1); s.submit(job(1, [98765], 'private-user'))
    payload = json.dumps(s.snapshot())
    assert '98765' not in payload and 'private-user' not in payload


def test_cancellation_frees_slot():
    s=Studio(1); a=job(1); s.submit(a); a.error=RuntimeError('cancelled'); s.finish(a)
    s.submit(job(2)); assert s.errors == 1


def test_identity_fallback_and_override():
    first=[dict(role='system',content='a'),dict(role='user',content='hi')]
    assert conversation_key(first).startswith('auto:')
    assert conversation_key(first,'a')==conversation_key(first+[dict(role='assistant',content='hello')],'a')
    assert conversation_key(first,'a')!=conversation_key(first,'b')
    with pytest.raises(RequestError): conversation_key(first,'')


def request(app, auth=None, path='/v1/models', method='GET'):
    raw=(f'{method} {path} HTTP/1.0\r\n'+(f'Authorization: {auth}\r\n' if auth else '')+'\r\n').encode()
    class Connection:
        def __init__(self): self.output=bytearray()
        def makefile(self,*args): return BytesIO(raw)
        def sendall(self,data): self.output.extend(data)
    c=Connection();make_handler(app)(c,('127.0.0.1',0),None)
    headers,body=bytes(c.output).split(b'\r\n\r\n',1)
    return int(headers.split()[1]),body


def test_auth_all_data_routes_and_post():
    app=SimpleNamespace(auth=KeyStore(['secret']),model_ids=['swift-1.5'])
    assert request(app,path='/health') == (200, b'{"status": "ok"}')
    for route in ['/v1/models','/studio/metrics']:
        assert request(app,path=route)[0]==401
        assert request(app,'Bearer wrong',route)[0]==401
    assert request(app,path='/v1/chat/completions',method='POST')[0]==401
    assert request(app,'Bearer secret')[0]==200
    assert request(app,path='/studio')[0]==200  # shell contains no metrics or key
    assert b'secret' not in request(app,path='/studio')[1]


def test_swift_scales_preserve_dtype_and_fail_closed():
    import mlx.core as mx
    from tensorfold.families.qwen3_5.swift import convert_norms
    weights = {f'model.layers.{i}.q_norm.weight': mx.array([0.25],dtype=mx.bfloat16) for i in range(161)}
    converted = convert_norms(weights)
    assert all(v.dtype==mx.bfloat16 and float(v.item())==1.25 for v in converted.values())
    assert float(next(iter(weights.values())).item())==0.25
    with pytest.raises(ValueError): convert_norms({})


def test_metrics_endpoint_requires_key_and_returns_inventory():
    s=Studio(2)
    app=SimpleNamespace(auth=KeyStore(['secret']),scheduler=SimpleNamespace(studio=s),served_name='swift-1.5',
                        max_batch_size=2,context_window=131072,
                        prompt_memory=SimpleNamespace(memory_snapshot=lambda reset: {'active': 123}))
    status,raw=request(app,'Bearer secret','/studio/metrics')
    data=json.loads(raw)
    assert status==200 and len(data['slots'])==2 and data['memory']['active']==123
    assert 'secret' not in raw.decode()


def test_collector_keeps_measured_rates_without_differencing():
    from importlib.util import spec_from_file_location, module_from_spec
    from pathlib import Path
    spec=spec_from_file_location('collector',Path(__file__).parents[1]/'tools/studio-collector.py')
    module=module_from_spec(spec);spec.loader.exec_module(module)
    data=dict(parallel=2,context=131072,slots=[dict(slot=1,state='prefilling',prompt_tps=296.5,rate_at=123)])
    m=module.tensorfold_metrics(data)
    assert m['slot_1_prompt_tps']==296.5 and 'slot_1_decode_tps' not in m


def auto_job(n, messages):
    from tensorfold.server.studio import conversation_replies
    j = job(n, list(range(100)), conversation_key(messages))
    j.conversation_replies = conversation_replies(messages)
    return j


def test_vscode_instruction_and_user_context_changes_keep_slot():
    s = Studio(8)
    first = auto_job(1, [{'role':'system','content':'workspace v1'}, {'role':'user','content':'task'}])
    s.submit(first); s.finish(first)
    response = {'role':'assistant','content':'Which folder should contain the new feature?'}
    s.record_reply(first.conversation_id, response)
    second = auto_job(2, [{'role':'system','content':'workspace v2'}, {'role':'user','content':'task + updated editor context'},
                          response, {'role':'user','content':'src/ui'}])
    s.submit(second)
    assert s.jobs['2'] == 0 and len(s.prompts) == 1


def test_identical_openings_and_ambiguous_replies_do_not_merge():
    s = Studio(8)
    messages = [{'role':'user','content':'hello'}]
    a,b = auto_job(1,messages),auto_job(2,messages)
    s.submit(a);s.submit(b)
    assert s.jobs['1'] != s.jobs['2']
    for j in (a,b):
        s.finish(j);s.record_reply(j.conversation_id, {'content':'Hello!'})
    c=auto_job(3,messages+[{'role':'assistant','content':'Hello!'}, {'role':'user','content':'More'}])
    s.submit(c)
    assert s.jobs['3'] == 2


def test_latest_reply_disambiguates_branches_without_system_prefix_matching():
    s=Studio(8)
    a=auto_job(1,[{'role':'user','content':'start'}]);s.submit(a);s.finish(a)
    s.record_reply(a.conversation_id,{'content':'ancestor'})
    b=auto_job(2,[{'role':'user','content':'other'}]);s.submit(b);s.finish(b)
    s.record_reply(b.conversation_id,{'content':'ancestor'})
    s.record_reply(b.conversation_id,{'content':'branch B followup'})
    c=auto_job(3,[{'role':'assistant','content':'ancestor'},{'role':'assistant','content':'branch B followup'},
                  {'role':'user','content':'continue'}]);s.submit(c)
    assert s.jobs['3']==1


def test_tool_call_identity_ignores_argument_formatting_and_reasoning():
    from tensorfold.server.studio import reply_fingerprint
    a={'tool_calls':[{'id':'call-123','function':{'name':'search','arguments':'{"b": 2,"a":1}'}}]}
    b={'content':None,'reasoning_content':'private','tool_calls':[{'id':'call-123','function':{'name':'search','arguments':{'a':1,'b':2}}}]}
    assert reply_fingerprint(a)==reply_fingerprint(b)
    b['tool_calls'][0]['id']='different'
    assert reply_fingerprint(a)!=reply_fingerprint(b)


def test_explicit_conversations_never_merge_by_common_reply():
    s=Studio(8)
    a=job(1,key=conversation_key([], 'A'));s.submit(a);s.finish(a)
    s.record_reply(a.conversation_id, {'content':'same reply'})
    b=auto_job(2,[{'role':'assistant','content':'same reply'}]);s.submit(b)
    assert s.jobs['2']==1


def test_shared_checkpoint_is_counted_once_and_not_owned_twice(tmp_path):
    s=Studio(8)
    a,b=job(1,[1,2,3]),job(2,[1,2,4])
    s.submit(a);s.finish(a);s.submit(b);s.finish(b)
    p=tmp_path/'prefix';p.write_bytes(b'x'*20)
    cache=SimpleNamespace(tokens=[1,2],nbytes=10,pinned=False)
    s.residency([cache],[(p,[1,2])])
    data=s.snapshot()
    assert data['cache_stats']['shared_ram_bytes']==10
    assert data['cache_stats']['shared_ssd_bytes']==20
    assert all(row['ram_bytes']==row['ssd_bytes']==row['checkpoint_tokens']==0 for row in data['slots'][:2])
    assert all(row['state']=='shared' and row['shared_checkpoint_tokens']==2 for row in data['slots'][:2])
    unique=SimpleNamespace(tokens=[1,2,3],nbytes=30,pinned=False)
    s.residency([cache,unique],[(p,[1,2])])
    data=s.snapshot()
    assert data['slots'][0]['ram_bytes']==30 and data['slots'][1]['ram_bytes']==0
    assert data['cache_stats']['assigned_ram_bytes']+data['cache_stats']['shared_ram_bytes']==40


def test_legacy_inventory_is_reset_without_touching_disk_caches(tmp_path):
    (tmp_path/'studio.json').write_text(json.dumps({'model_id':'model','slots':[{'state':'ssd'}],'keys':{'old':0},'prompts':{'0':[1]}}))
    cache=tmp_path/'cache.safetensors';cache.write_bytes(b'keep')
    s=Studio(1,tmp_path,'model')
    assert s.snapshot()['slots'][0]['state']=='empty' and s.keys=={}
    assert cache.read_bytes()==b'keep'


@pytest.mark.parametrize('stream', [False, True])
def test_http_records_replies_and_reuses_slot_with_changed_instructions(stream):
    from http_fakes import post
    from test_server_openai_compat import FakeApp
    class TrackingApp(FakeApp):
        accepts_sampling=True
        def __init__(self):
            super().__init__(content='Hello')
            self.scheduler=SimpleNamespace(studio=Studio(8))
            self.counter=0
        def chat(self,messages,*,sampling=None,**kwargs):
            self.counter+=1
            j=job(self.counter,key=sampling['conversation_id'])
            j.conversation_replies=sampling['conversation_replies']
            self.scheduler.studio.submit(j)
            result=super().chat(messages,**kwargs)
            self.scheduler.studio.finish(j)
            return result
    app=TrackingApp()
    initial=[{'role':'system','content':'context one'},{'role':'user','content':'hi'}]
    status,_=post(app,{'messages':initial,'stream':stream})
    assert status==200 and len(app.scheduler.studio.replies)==1
    continuation=[{'role':'system','content':'context two'},{'role':'user','content':'hi'},
                  {'role':'assistant','content':'Hello'},{'role':'user','content':'continue'}]
    status,_=post(app,{'messages':continuation,'stream':stream})
    assert status==200 and len(app.scheduler.studio.prompts)==1


def test_reply_anchors_persist_across_restart(tmp_path):
    s=Studio(8,tmp_path,'m')
    j=auto_job(1,[{'role':'user','content':'question'}]);s.submit(j);s.finish(j)
    response={'role':'assistant','content':'Which file should I edit?'}
    s.record_reply(j.conversation_id,response)
    restored=Studio(8,tmp_path,'m')
    k=auto_job(2,[response,{'role':'user','content':'file.py'}]);restored.submit(k)
    assert restored.jobs['2']==0
    assert restored.snapshot()['slots'][1]['state']=='empty'


def test_streamed_tool_ids_not_reparsed_ids_keep_vscode_followup_in_same_slot():
    from http_fakes import post
    from test_server_openai_compat import FakeApp
    from tensorfold.server.studio import reply_fingerprint

    class TrackingTools(FakeApp):
        accepts_sampling = True
        streams_prose_with_tools = True
        def __init__(self):
            super().__init__()
            self.scheduler = SimpleNamespace(studio=Studio(8))
            self.counter = 0
        def chat(self, messages, *, sampling=None, on_delta=None, **kwargs):
            self.counter += 1
            j = job(self.counter, key=sampling['conversation_id'])
            j.conversation_replies = sampling['conversation_replies']
            self.scheduler.studio.submit(j)
            if on_delta:
                on_delta({'tool_calls': [{'index': 0, 'id': 'wire-call-123', 'type': 'function',
                                         'function': {'name': 'lookup', 'arguments': ''}}]})
                on_delta({'tool_calls': [{'index': 0, 'function': {'arguments': '{"q":'}}]})
                on_delta({'tool_calls': [{'index': 0, 'function': {'arguments': '"test"}'}}]})
            self.scheduler.studio.finish(j)
            return dict(content='<tool_call>{"name":"lookup","arguments":{"q":"test"}}</tool_call>',
                        tool_calls_streamed=True, finish_reason='tool_calls', prompt_tokens=3, completion_tokens=5)

    app = TrackingTools()
    tools = [{'type': 'function', 'function': {'name': 'lookup', 'parameters': {'type': 'object'}}}]
    first = [{'role': 'user', 'content': 'find test'}]
    status, body = post(app, dict(messages=first, tools=tools, stream=True))
    assert status == 200
    assistant = {'role': 'assistant', 'tool_calls': [{'id': 'wire-call-123', 'type': 'function',
                    'function': {'name': 'lookup', 'arguments': '{"q":"test"}'}}]}
    assert reply_fingerprint(assistant) in app.scheduler.studio.replies[0]
    status, _ = post(app, dict(messages=first + [assistant, {'role': 'tool', 'tool_call_id': 'wire-call-123',
                              'content': 'found it'}], tools=tools, stream=True))
    assert status == 200 and len(app.scheduler.studio.prompts) == 1


def test_upstream_auth_keeps_detailed_health_private():
    app = SimpleNamespace(auth=KeyStore(['secret']), served_name='swift-1.5',
                          model_ids=['swift-1.5'], max_batch_size=2,
                          prompt_memory=SimpleNamespace(memory_snapshot=lambda reset: {'active': 123}))
    assert json.loads(request(app, path='/health')[1]) == {'status': 'ok'}
    assert json.loads(request(app, 'Bearer wrong', '/health')[1]) == {'status': 'ok'}
    status, body = request(app, 'Bearer secret', '/health')
    payload = json.loads(body)
    assert status == 200 and payload['model'] == 'swift-1.5' and payload['memory']['active'] == 123

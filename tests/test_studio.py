from types import SimpleNamespace
from io import BytesIO
import json
import pytest
from tensorfold.server.studio import Studio, conversation_key
from tensorfold.server.errors import RequestError
from tensorfold.server.http import make_handler


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
    assert conversation_key(first)==conversation_key(first+[dict(role='assistant',content='hello')])
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
    app=SimpleNamespace(api_key='secret',model_ids=['swift-1.5'])
    for route in ['/health','/v1/models','/studio/metrics']:
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
    app=SimpleNamespace(api_key='secret',scheduler=SimpleNamespace(studio=s),served_name='swift-1.5',
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

"""Concurrent admission reserves future KV growth and scopes cancellation to one request."""
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import http.client
import json
import socket
import time
import unittest
from unittest.mock import patch

import support
from config import ModelConfig, validate
from errors import GatewayError
from memory import GIB


PROFILE = dict(stateless=True, max_concurrent=8, context_window=96000,
               memory_required_gib=18, memory_per_request_gib=3, request_reserve_gib=8)


class ConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = patch("memory.snapshot", return_value=(40 * GIB, 1))
        self.footprint = patch("memory.group_footprint", return_value=15 * GIB)
        self.snapshot.start()
        self.footprint.start()
        self.addCleanup(self.snapshot.stop)
        self.addCleanup(self.footprint.stop)

    def gateway(self, **values):
        return support.gateway(profile_options={"swift": dict(PROFILE, **values)})

    def test_eight_requests_reach_the_child_together_and_ninth_is_rejected(self):
        with self.gateway() as server, ThreadPoolExecutor(max_workers=8) as pool:
            jobs = [pool.submit(server.completion, fixture_delay=1) for _ in range(8)]
            support.eventually(lambda: self.assertEqual(server.server.manager.users, 8))
            support.eventually(lambda: self.assertEqual(sum(r['event'] == 'request' for r in server.rows()), 8))
            status, _, raw = server.completion()
            self.assertEqual(status, 429, raw)
            self.assertTrue(all(job.result()[0] == 200 for job in jobs))
            self.assertEqual(sum(r['event'] == 'start' for r in server.rows()), 1)
            support.eventually(lambda: self.assertEqual(server.server.manager.users, 0))
            self.assertEqual(list(server.state.rglob('*.tfs')), [])
            self.assertFalse(any(r['event'] in ('clear', 'save', 'load') for r in server.rows()))

    def test_eight_connections_survive_a_temporarily_busy_accept_loop(self):
        with self.gateway() as server:
            server.server.shutdown()
            server.thread.join(timeout=2)
            clients = []
            try:
                for _ in range(8):
                    clients.append(socket.create_connection(server.address, timeout=0.3))
            finally:
                for client in clients:
                    client.close()

    def test_future_growth_reservations_admit_two_but_refuse_third(self):
        with self.gateway() as server, ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(server.completion()[0], 200)
            with patch('memory.snapshot', return_value=(14 * GIB, 1)):
                jobs = [pool.submit(server.completion, fixture_delay=0.8) for _ in range(2)]
                support.eventually(lambda: self.assertEqual(server.server.manager.users, 2))
                status, _, raw = server.completion()
                self.assertEqual(status, 503, raw)
                self.assertEqual(json.loads(raw)['error']['code'], 'insufficient_memory')
                self.assertTrue(all(job.result()[0] == 200 for job in jobs))
            self.assertEqual(server.completion()[0], 200)

    def test_already_allocated_request_memory_is_not_reserved_twice(self):
        with self.gateway() as server:
            manager = server.server.manager
            self.assertEqual(server.completion()[0], 200)
            manager.base_footprint = 15 * GIB
            with patch('memory.snapshot', return_value=(11 * GIB, 1)), \
                    patch('memory.group_footprint', return_value=18 * GIB):
                manager._check_slot(manager.model('swift'), 2)
                with self.assertRaises(GatewayError):
                    manager._check_slot(manager.model('swift'), 3)

    def test_model_switch_waits_for_every_user_then_stops_the_previous_child(self):
        with self.gateway() as server, ThreadPoolExecutor(max_workers=3) as pool:
            jobs = [pool.submit(server.completion, fixture_delay=0.5) for _ in range(2)]
            support.eventually(lambda: self.assertEqual(server.server.manager.users, 2))
            other = pool.submit(server.completion, model='ornith')
            time.sleep(0.1)
            self.assertFalse(other.done())
            self.assertTrue(all(job.result()[0] == 200 for job in jobs))
            self.assertEqual(other.result()[0], 200)
            self.assertTrue(all(not r['live_predecessors'] for r in server.rows() if r['event'] == 'start'))

    def test_client_disconnect_does_not_stop_the_shared_child_or_other_user(self):
        with self.gateway() as server, ThreadPoolExecutor(max_workers=1) as pool:
            self.assertEqual(server.completion()[0], 200)
            connection = http.client.HTTPConnection(*server.address, timeout=5)
            body = json.dumps({'model': 'swift', 'messages': [{'role': 'user', 'content': 'public fixture'}],
                               'stream': True, 'fixture_delay': 0.05, 'fixture_chunks': 100})
            connection.request('POST', '/v1/chat/completions', body,
                               {'Authorization': 'Bearer ' + support.OWNER_KEY, 'Content-Type': 'application/json'})
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            response.readline()
            other = pool.submit(server.completion, _key=support.FRIEND_KEY, fixture_delay=0.3)
            support.eventually(lambda: self.assertEqual(server.server.manager.users, 2))
            response.close()
            connection.close()
            self.assertEqual(other.result()[0], 200)
            support.eventually(lambda: self.assertEqual(server.server.manager.users, 0))
            self.assertEqual(server.completion()[0], 200)
            self.assertEqual(sum(r['event'] == 'start' for r in server.rows()), 1)

    def test_context_limit_and_unsafe_profiles_are_rejected(self):
        with self.gateway() as server:
            self.assertEqual(server.completion(max_tokens=96001)[0], 400)
            for values in [dict(stateless=False), dict(memory_per_request_gib=0), dict(max_concurrent=True),
                           dict(context_window=0), dict(request_reserve_gib=float('nan'))]:
                model = dataclasses.replace(server.config.models[0], **values)
                with self.assertRaises(ValueError):
                    validate(dataclasses.replace(server.config, models=(model, server.config.models[1])))

    def test_pressure_or_missing_telemetry_refuses_without_disturbing_users(self):
        with self.gateway() as server, ThreadPoolExecutor(max_workers=1) as pool:
            job = pool.submit(server.completion, fixture_delay=0.6)
            support.eventually(lambda: self.assertEqual(server.server.manager.users, 1))
            support.eventually(lambda: self.assertFalse(server.server.manager.loading))
            for effect in [(40 * GIB, 2), OSError('unavailable')]:
                with patch('memory.snapshot', **({'side_effect': effect} if isinstance(effect, Exception)
                                                 else {'return_value': effect})):
                    self.assertEqual(server.completion()[0], 503)
            self.assertEqual(job.result()[0], 200)

    def test_low_memory_before_switch_preserves_the_resident_snapshot_model(self):
        with self.gateway() as server:
            self.assertEqual(server.completion(model='ornith')[0], 200)
            before = server.rows()
            with patch('memory.snapshot', return_value=(7 * GIB, 1)):
                self.assertEqual(server.completion()[0], 503)
            self.assertEqual(server.rows(), before)
            self.assertEqual(server.server.manager.backend.model.id, 'ornith')

    def test_memory_change_during_measurement_cannot_double_credit_growth(self):
        with self.gateway() as server:
            self.assertEqual(server.completion()[0], 200)
            manager = server.server.manager
            manager.base_footprint = 15 * GIB
            with patch('memory.group_footprint', return_value=18 * GIB):
                for values in [((10 * GIB, 1), (11 * GIB, 1)), ((11 * GIB, 1), (10 * GIB, 1)),
                               ((11 * GIB, 1), (11 * GIB, 2))]:
                    with patch('memory.snapshot', side_effect=values):
                        with self.assertRaises(GatewayError):
                            manager._check_slot(manager.model('swift'), 2)

    def test_released_startup_buffers_lower_the_current_cohort_baseline(self):
        with self.gateway() as server:
            self.assertEqual(server.completion()[0], 200)
            manager = server.server.manager
            manager.base_footprint = 18 * GIB
            with patch('memory.group_footprint', return_value=15 * GIB):
                manager._check_slot(manager.model('swift'), 1)
            with patch('memory.snapshot', return_value=(11 * GIB, 1)), \
                    patch('memory.group_footprint', return_value=18 * GIB):
                manager._check_slot(manager.model('swift'), 2)


if __name__ == '__main__':
    unittest.main()

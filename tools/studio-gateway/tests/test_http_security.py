"""HTTP trust-boundary checks before any backend/model is loaded."""
import http.client
import json
import socket
import unittest

from support import FRIEND_KEY, OWNER_KEY, gateway, private_file


class HTTPSecurityTests(unittest.TestCase):
    def assert_no_backend_started(self, server):
        self.assertEqual([row for row in server.rows() if row["event"] == "start"], [])

    def test_missing_or_wrong_auth_does_not_start_engine(self):
        with gateway() as server:
            for key in (None, "wrong-fixture-key"):
                for path, method in (("/v1/models", "GET"),
                                     ("/v1/chat/completions", "POST")):
                    with self.subTest(path=path, key_is_missing=key is None):
                        status, _, _ = server.request(path, method=method, key=key,
                            body={"model": "swift"} if method == "POST" else None)
                        self.assertEqual(status, 401)
            self.assert_no_backend_started(server)

    def test_models_lists_both_and_auth_does_not_load_any_model(self):
        with gateway() as server:
            status, _, raw = server.request("/v1/models", method="GET")
            self.assertEqual(status, 200)
            model_ids = {item["id"] for item in json.loads(raw)["data"]}
            self.assertTrue({"swift", "ornith"}.issubset(model_ids))
            self.assert_no_backend_started(server)

    def test_removed_friend_key_revokes_next_request_and_owner_survives(self):
        with gateway() as server:
            self.assertEqual(server.request("/v1/models", method="GET", key=FRIEND_KEY)[0], 200)
            server.friend_key_file.unlink()
            self.assertEqual(server.request("/v1/models", method="GET", key=FRIEND_KEY)[0], 401)
            self.assertEqual(server.request("/v1/models", method="GET")[0], 200)
            self.assert_no_backend_started(server)

    def test_key_rotation_reloads_without_restart(self):
        replacement = "tf-test-rotated-owner-ae7021"
        with gateway() as server:
            private_file(server.owner_key_file, replacement + "\n")
            self.assertEqual(server.request("/v1/models", method="GET")[0], 401)
            self.assertEqual(server.request("/v1/models", method="GET", key=replacement)[0], 200)
            self.assert_no_backend_started(server)

    def test_symlink_key_file_replacement_fails_closed(self):
        with gateway() as server:
            actual = server.owner_key_file.with_suffix(".replacement")
            private_file(actual, OWNER_KEY + "\n")
            server.owner_key_file.unlink()
            server.owner_key_file.symlink_to(actual)
            self.assertEqual(server.request("/v1/models", method="GET")[0], 401)
            self.assert_no_backend_started(server)

    def test_labeled_native_key_file_format(self):
        with gateway() as server:
            private_file(server.owner_key_file, "# labeled fixture\nowner: " + OWNER_KEY + "\n")
            status, _, raw = server.request("/v1/models", method="GET")
            self.assertEqual(status, 200, raw)
            self.assert_no_backend_started(server)

    def test_unsafe_key_permissions_fail_closed_without_secret_disclosure(self):
        with gateway() as server:
            server.owner_key_file.chmod(0o644)
            status, _, raw = server.request("/v1/models", method="GET")
            self.assertEqual(status, 401)
            for secret in (OWNER_KEY, FRIEND_KEY, str(server.owner_key_file)):
                self.assertNotIn(secret.encode(), raw)
            self.assert_no_backend_started(server)

    def test_internal_snapshot_routes_are_never_proxied(self):
        with gateway() as server:
            for endpoint in ("save", "load", "clear"):
                with self.subTest(endpoint=endpoint):
                    status, _, _ = server.request("/v1/internal/session/" + endpoint,
                        body={"path": "/tmp/fixture-outside.tfs"})
                    self.assertIn(status, (404, 405))
            self.assert_no_backend_started(server)

    def test_unknown_model_refused_before_start(self):
        with gateway() as server:
            for name in ("unknown", "../../outside", "http://malicious.invalid/model"):
                with self.subTest(model=name):
                    status, _, raw = server.completion(model=name)
                    self.assertIn(status, (400, 404))
                    self.assertNotIn(OWNER_KEY.encode(), raw)
            self.assert_no_backend_started(server)

    def test_private_backend_error_does_not_disclose_generated_credential(self):
        with gateway() as server:
            status, _, raw = server.completion(fixture_backend_error=True)
            self.assertIn(status, (500, 502, 503))
            backend_files = [path for path in server.state.rglob("*")
                             if path.is_file() and "key" in path.name]
            self.assertTrue(backend_files)
            for key_file in backend_files:
                secret = key_file.read_text().strip().split(":", 1)[-1]
                self.assertTrue(secret)
                self.assertNotIn(secret.encode(), raw)
            self.assertNotIn(OWNER_KEY.encode(), raw)
            self.assertNotIn(FRIEND_KEY.encode(), raw)

    def test_duplicate_authorization_is_refused_before_start(self):
        with gateway() as server:
            connection = http.client.HTTPConnection(*server.address, timeout=3)
            try:
                connection.putrequest("GET", "/v1/models")
                connection.putheader("Authorization", "Bearer " + OWNER_KEY)
                connection.putheader("Authorization", "Bearer " + FRIEND_KEY)
                connection.putheader("Connection", "close")
                connection.endheaders()
                response = connection.getresponse()
                self.assertIn(response.status, (400, 401))
                response.read()
            finally:
                connection.close()
            self.assert_no_backend_started(server)

    def test_conflicting_authorization_forms_refused_before_start(self):
        with gateway() as server:
            status, _, _ = server.request("/v1/models", method="GET",
                headers={"x-api-key": FRIEND_KEY})
            self.assertIn(status, (400, 401))
            self.assert_no_backend_started(server)

    def test_disallowed_host_refused_before_start(self):
        with gateway() as server:
            status, _, _ = server.request("/v1/models", method="GET",
                headers={"Host": "untrusted.invalid"})
            self.assertIn(status, (400, 403, 421))
            self.assert_no_backend_started(server)

    def test_body_framing_ambiguity_refused_before_start(self):
        with gateway() as server:
            payload = json.dumps({"model": "swift", "messages": []}).encode()
            for extra in (b"Content-Length: 1\r\nContent-Length: 2\r\n",
                          b"Transfer-Encoding: chunked\r\nContent-Length: 2\r\n"):
                with self.subTest(extra=extra):
                    with socket.create_connection(server.address, timeout=3) as client:
                        client.sendall(b"POST /v1/chat/completions HTTP/1.1\r\n"
                            b"Host: localhost\r\nAuthorization: Bearer " + OWNER_KEY.encode() +
                            b"\r\nConnection: close\r\n" + extra + b"\r\n" + payload)
                        status = client.makefile("rb").readline()
                        self.assertIn(status.split()[1], (b"400", b"411", b"413"))
            self.assert_no_backend_started(server)


if __name__ == "__main__":
    unittest.main()

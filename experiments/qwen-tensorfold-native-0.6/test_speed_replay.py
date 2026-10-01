"""Portable replay schedule, privacy boundary, and curation tests."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
from threading import Thread
import unittest
from unittest.mock import patch

import benchmark_speed
import curate_speed
import speed_protocol
from transport import FIXTURES, request


class FakeSSEHandler(BaseHTTPRequestHandler):
    received = None
    authorization = None

    def do_POST(self):
        FakeSSEHandler.received = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeSSEHandler.authorization = self.headers.get("Authorization")
        events = [
            {"choices": [{"delta": {"content": "fixture answer"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 9, "completion_tokens": 2}},
            "[DONE]",
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for event in events:
            payload = event if event == "[DONE]" else json.dumps(event)
            self.wfile.write(("data: " + payload + "\n\n").encode())

    def log_message(self, *_args):
        pass


class SpeedReplayTests(unittest.TestCase):
    def test_protocol_is_exactly_17_excluded_warmups_then_64_balanced_requests(self):
        meta = speed_protocol.metadata(speed_protocol.VERSION)
        self.assertEqual(speed_protocol.VERSION, "cell-warmup-v2")
        self.assertEqual(meta["fixtures_sha256"],
                         "7a15137eebc72c8a8cd6b3c82e61efbde3868e2b98127cb51fe3095f4754109c")
        warm = speed_protocol.warmup_cells()
        self.assertEqual([(clients, kind) for clients, kind, _ in warm],
                         [(1, "code"), (1, "prose"), (4, "code"), (4, "prose")])
        warm_numbers = [number for _, _, numbers in warm for number in numbers]
        self.assertEqual(len(warm_numbers), 16)
        self.assertEqual(len(set(warm_numbers + [99001])), 17)
        measured = [(rep, clients, kind, speed_protocol.measured_number(rep, i))
                    for rep in range(4)
                    for clients, kind in speed_protocol.measured_cells(rep)
                    for i in range(4)]
        self.assertEqual(len(measured), 64)
        self.assertEqual(speed_protocol.measured_cells(0), list(speed_protocol.CELLS))
        self.assertEqual(speed_protocol.measured_cells(1), list(reversed(speed_protocol.CELLS)))
        self.assertEqual(len({number for _, _, _, number in measured}), 16)
        self.assertEqual(set(FIXTURES), {"code", "prose", "exact"})

    def test_request_body_and_bearer_are_generic_and_transport_only(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeSSEHandler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = request(f"http://127.0.0.1:{server.server_port}/v1", "served-model",
                             "code", 12345, 512, api_key="secret-test-token")
        finally:
            server.shutdown()
            thread.join(2)
            server.server_close()
        self.assertEqual(FakeSSEHandler.authorization, "Bearer secret-test-token")
        body = FakeSSEHandler.received
        self.assertEqual(body["model"], "served-model")
        self.assertEqual(body["seed"], 1000 + 12345)
        self.assertEqual(body["max_tokens"], 512)
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["chat_template_kwargs"],
                         {"enable_thinking": False, "reasoning_effort": "medium"})
        self.assertNotIn("secret-test-token", json.dumps(body))
        self.assertEqual(result["output"], "fixture answer")

    def test_group_stores_only_sanitized_rows_and_computes_eligible_rate(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeSSEHandler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = benchmark_speed.group(
                f"http://127.0.0.1:{server.server_port}/v1", "served-model", None,
                1, "code", [10, 11, 12, 13], 512, "measurement", rep=0,
            )
        finally:
            server.shutdown(); thread.join(2); server.server_close()
        self.assertTrue(result["speed_eligible"])
        self.assertGreater(result["aggregate_output_tokens_per_second"], 0)
        self.assertEqual(len(result["requests"]), 4)
        for row in result["requests"]:
            self.assertNotIn("output", row)
            self.assertNotIn("reasoning", row)
            self.assertTrue(row["visible_output_nonempty"])

    def test_sanitizer_drops_response_text_and_error_details(self):
        row = benchmark_speed.sanitize({
            "kind": "prose", "fixture": 7, "max_tokens": 512,
            "request_sha256": "a", "prompt_sha256": "b", "output_sha256": "c",
            "output": "PRIVATE MODEL ANSWER", "reasoning": "PRIVATE THOUGHT",
            "error": "HTTPError: https://private.invalid/token SECRET BODY",
            "gate": False, "finish": None, "usage": {}, "runtime_stats": {},
        }, 4, "measurement", 0)
        serialized = json.dumps(row)
        self.assertNotIn("PRIVATE MODEL ANSWER", serialized)
        self.assertNotIn("PRIVATE THOUGHT", serialized)
        self.assertNotIn("private.invalid", serialized)
        self.assertEqual(row["error_class"], "HTTPError")
        self.assertTrue(row["visible_output_nonempty"])

    def test_identity_allowlist_rejects_endpoint_or_machine_fields(self):
        good = {
            "schema_version": 1, "image_digest": "sha256:" + "a" * 64,
            "runtime_version": "0.6.0", "runtime_revision": "a" * 40,
            "model_repository": "example/model", "model_revision": "b" * 40,
            "settings": {"context_tokens": 262144, "parallel_slots": 4,
                         "max_output_tokens": 8192, "thinking": "off"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "identity.json"
            path.write_text(json.dumps(good))
            identity, digest = benchmark_speed.load_identity(path)
            self.assertEqual(identity, good)
            self.assertEqual(len(digest), 64)
            path.write_text(json.dumps(good | {"endpoint": "https://private.invalid"}))
            with self.assertRaisesRegex(ValueError, "unknown"):
                benchmark_speed.load_identity(path)
            path.write_text(json.dumps(good | {"node": "node-placeholder"}))
            with self.assertRaisesRegex(ValueError, "unknown"):
                benchmark_speed.load_identity(path)
            path.write_text(json.dumps(good | {"settings": {"parallel_slots": 4}}))
            with self.assertRaisesRegex(ValueError, "context"):
                benchmark_speed.load_identity(path)

    def test_curator_uses_null_when_any_group_fails_and_keeps_warmups_separate(self):
        data = {
            "protocol": {"measured_repetitions_primary": 4},
            "warmup_groups": [{"aggregate_output_tokens_per_second": 999999}],
            "measured_groups": [{"clients": 1, "kind": "code", "rep": 0,
                                 "speed_eligible": False,
                                 "ineligible_reasons": ["usage missing"], "requests": []}],
        }
        result = curate_speed.curate(data)
        self.assertEqual(result["measured_request_count"], 0)
        self.assertTrue(result["warmup_groups_excluded"])
        c1_code = result["cells"][0]
        self.assertFalse(c1_code["eligible"])
        self.assertIsNone(c1_code["group_aggregate_output_tokens_per_second"])
        self.assertIn("usage missing", c1_code["ineligible_reasons"])
        self.assertIsNone(result["cells"][1]["group_aggregate_output_tokens_per_second"])

    def test_curator_rejects_duplicate_repetition_indices(self):
        groups = []
        for rep in (0, 1, 1, 3):
            groups.append({
                "clients": 1, "kind": "code", "rep": rep,
                "speed_eligible": True,
                "aggregate_output_tokens_per_second": 10.0,
                "requests": [{"decode_tokens_per_second_proxy": 2.0,
                              "first_output_event_seconds": 0.1}] * 4,
            })
        result = curate_speed.curate({
            "protocol": {"measured_repetitions_primary": 4},
            "measured_groups": groups,
        })
        cell = result["cells"][0]
        self.assertFalse(cell["eligible"])
        self.assertIsNone(cell["group_aggregate_output_tokens_per_second"])
        self.assertIn("missing, duplicated, or outside", " ".join(cell["ineligible_reasons"]))

    def test_full_runner_saves_sanitized_receipt_with_exact_schedule_counts(self):
        class CompleteHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                prompt = body["messages"][0]["content"]
                output = ('{"sum":465,"sorted":[1,3,7,9],"marker":"TF-CHECK-20260927"}'
                          if "fixture 99001." in prompt else "model text not retained")
                payloads = [
                    {"choices": [{"delta": {"content": output}, "finish_reason": None}]},
                    {"choices": [{"delta": {}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 9, "completion_tokens": 3}},
                    "[DONE]",
                ]
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                for item in payloads:
                    self.wfile.write(("data: " + (item if item == "[DONE]" else json.dumps(item)) + "\n\n").encode())

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), CompleteHandler)
        thread = Thread(target=server.serve_forever, daemon=True); thread.start()
        identity = {
            "schema_version": 1, "image_digest": "sha256:" + "a" * 64,
            "runtime_repository": "https://github.com/ashhart/TensorFold",
            "runtime_version": "0.6.0", "runtime_revision": "a" * 40,
            "model_repository": "example/model", "model_revision": "b" * 40,
            "settings": {"context_tokens": 262144, "parallel_slots": 4,
                         "max_output_tokens": 8192, "thinking": "off"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            identity_path = Path(tmp) / "identity.json"; identity_path.write_text(json.dumps(identity))
            out = Path(tmp) / "receipt.json"
            argv = ["benchmark_speed.py", "--base", f"http://127.0.0.1:{server.server_port}/v1",
                    "--model", "served-model", "--label", "test-arm", "--identity", str(identity_path),
                    "--out", str(out)]
            try:
                with patch("sys.argv", argv), patch.dict("os.environ", {"SPARK_SERVE_API_KEY": "secret"}):
                    with redirect_stdout(io.StringIO()):
                        self.assertEqual(benchmark_speed.main(), 0)
                receipt_text = out.read_text()
                receipt = json.loads(receipt_text)
            finally:
                server.shutdown(); thread.join(2); server.server_close()
        self.assertEqual(receipt["status"], "complete")
        self.assertEqual(sum(len(g["requests"]) for g in receipt["warmup_groups"]), 17)
        self.assertEqual(sum(len(g["requests"]) for g in receipt["measured_groups"]), 64)
        self.assertTrue(all(g["speed_eligible"] for g in receipt["warmup_groups"]))
        self.assertTrue(all(g["speed_eligible"] for g in receipt["measured_groups"]))
        self.assertTrue(all(cell["eligible"] for cell in receipt["summary"]["cells"]))
        self.assertNotIn("127.0.0.1", receipt_text)
        self.assertNotIn("secret", receipt_text)
        self.assertNotIn("model text not retained", receipt_text)
        self.assertNotIn(str(identity_path), receipt_text)

    def test_provenance_pins_runtime_inputs_and_benchmark_sources(self):
        root = Path(__file__).resolve().parent
        provenance = json.loads((root / "benchmark-provenance.json").read_text())
        self.assertEqual(provenance["speed_protocol"]["version"], "cell-warmup-v2")
        for group in provenance["input_sha256"].values():
            for name, expected in group.items():
                actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
                self.assertEqual(actual, expected, name)
        ignored = (root / ".dockerignore").read_text()
        for asset in ("!benchmark_speed.py", "!curate_speed.py", "!speed_protocol.py",
                      "!transport.py", "!quality_fixtures.py", "!coding_cases.py",
                      "!quality_contract.py"):
            self.assertNotIn(asset, ignored)
        for asset in ("!Dockerfile.burst", "!build_burst.py", "!decode-burst/decode-burst.patch"):
            self.assertIn(asset, ignored)


if __name__ == "__main__":
    unittest.main()

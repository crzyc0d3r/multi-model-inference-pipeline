from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from inference.client import Models, SIEClient, SIEError
from inference.replay import ReplayServer
from inference.workflow import cosine, load_case, mixed_request, run_pipeline
from inference.workload import run_workload


class PipelineTests(unittest.TestCase):
    def test_complete_http_workflow_and_sequencing(self):
        with ReplayServer() as server:
            client = SIEClient(server.url)
            result = run_pipeline(client, load_case())
            self.assertEqual(result["answer"], load_case()["answer"])
            self.assertEqual(result["embedding_dimensions"], 384)
            self.assertEqual(result["context_ids"], ["issuer", "payment"])
            self.assertEqual({x["label"] for x in result["entities"]}, {"organization", "date", "amount"})
            self.assertEqual([task for task, _ in server.history], ["encode", "score", "extract", "generate"])
            self.assertEqual(client.stats()["http_attempts"], 4)
            for entity in result["entities"]:
                context = server.history[2][1]["items"][0]["text"]
                self.assertEqual(context[entity["start"]:entity["end"]], entity["text"])

    def test_real_route_and_body_contract(self):
        calls = []
        real_urlopen = urlopen

        def capture(request, **kwargs):
            calls.append((request.full_url, json.loads(request.data), request.get_header("Accept")))
            return real_urlopen(request, **kwargs)

        with ReplayServer() as server:
            client = SIEClient(server.url)
            with patch.object(client._opener, "open", side_effect=capture):
                run_pipeline(client, load_case())
        self.assertTrue(calls[0][0].endswith("/v1/encode/sentence-transformers/all-MiniLM-L6-v2"))
        self.assertTrue(calls[1][0].endswith("/v1/score/cross-encoder/ms-marco-MiniLM-L-6-v2"))
        self.assertTrue(calls[2][0].endswith("/v1/extract/urchade/gliner_multi-v2.1"))
        self.assertTrue(calls[3][0].endswith("/v1/generate/Qwen__Qwen3-0.6B"))
        self.assertEqual(calls[0][1]["params"], {"output_types": ["dense"]})
        self.assertEqual(calls[2][1]["params"]["labels"], ["organization", "date", "amount"])
        self.assertTrue(all(x[2] == "application/json" for x in calls))

    def test_loading_retries_are_reported(self):
        with ReplayServer(loading_responses=2) as server:
            client = SIEClient(server.url, retry_delay=0, loading_retries=2)
            run_pipeline(client, load_case())
            self.assertEqual(client.stats()["http_attempts"], 6)
            self.assertEqual(client.stats()["http_error_responses"], 2)
            self.assertEqual(client.stats()["loading_retries"], 2)

    def test_terminal_error_stops_pipeline_and_counts_failures(self):
        with ReplayServer(fail_task="score") as server:
            client = SIEClient(server.url, retry_delay=0)
            result = run_workload(lambda _: run_pipeline(client, load_case()), 5, 2)
            self.assertEqual(result["failed"], 5)
            self.assertEqual(result["succeeded"], 0)
            self.assertTrue(all(task in {"encode", "score"} for task, _ in server.history))
            self.assertEqual(client.stats()["http_attempts"], 10)
            self.assertEqual(client.stats()["loading_retries"], 0)
            self.assertEqual(client.stats()["http_error_responses"], 5)

    def test_generation_transport_timeout_never_retries(self):
        client = SIEClient("http://127.0.0.1:8080", loading_retries=100)
        with patch.object(client._opener, "open", side_effect=TimeoutError) as transport:
            with self.assertRaisesRegex(SIEError, "not retried"):
                client.generate(Models().generator, "synthetic prompt")
            self.assertEqual(transport.call_count, 1)
            self.assertEqual(client.stats()["transport_errors"], 1)

    def test_per_item_failure_is_not_silently_accepted(self):
        client = SIEClient("http://127.0.0.1:8080")
        response = {"items": [{"id": "context", "entities": [],
                                "error": {"code": "INFERENCE_ERROR", "message": "synthetic"}}]}
        with patch.object(client, "_post", return_value=response):
            with self.assertRaisesRegex(SIEError, "per-item"):
                client.extract(Models().extractor, "test", ["amount"])

    def test_malformed_vectors_and_candidate_ids_rejected(self):
        client = SIEClient("http://127.0.0.1:8080")
        with patch.object(client, "_post", return_value={"items": [{"id": "a", "dense": {"dims": 4, "values": [1]}}]}):
            with self.assertRaisesRegex(SIEError, "dense vector"):
                client.encode(Models().embedder, [{"id": "a", "text": "test"}])
        with patch.object(client, "_post", return_value={"scores": [{"item_id": "unknown", "score": 3.0}]}):
            with self.assertRaisesRegex(SIEError, "IDs"):
                client.score(Models().reranker, "test", [{"id": "a", "text": "test"}])

    def test_workload_is_bounded_and_exact(self):
        lock = threading.Lock()
        active = max_active = 0
        observed = []

        def operation(index):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
                observed.append(index)
            time.sleep(.005)
            with lock:
                active -= 1
            if index % 7 == 0:
                raise ValueError("synthetic failure")
            return {"index": index}

        result = run_workload(operation, 37, 3)
        self.assertEqual(sorted(observed), list(range(37)))
        self.assertEqual(max_active, 3)
        self.assertEqual(result["max_in_flight"], 3)
        self.assertEqual(result["failed"], 6)
        self.assertEqual(result["succeeded"], 31)
        self.assertEqual(sum(result["errors"].values()), 6)
        self.assertLessEqual(result["latency_ms"]["p50"], result["latency_ms"]["p95"])

    def test_mixed_benchmark_means_exact_http_request_count(self):
        with ReplayServer() as server:
            client = SIEClient(server.url)
            report = run_workload(lambda i: mixed_request(client, load_case(), i), 40, 4)
            self.assertEqual(report["failed"], 0)
            self.assertEqual(client.stats()["http_attempts"], 40)
            self.assertEqual({task: sum(1 for t, _ in server.history if t == task)
                              for task in ["encode", "score", "extract", "generate"]},
                             {"encode": 10, "score": 10, "extract": 10, "generate": 10})

    def test_replay_rejects_unknown_model(self):
        with ReplayServer() as server:
            request = Request(server.url + "/v1/generate/Qwen/Qwen3-0.6B", data=b"{}",
                              headers={"Content-Type": "application/json"})
            with self.assertRaises(HTTPError) as error:
                urlopen(request)
            self.assertEqual(error.exception.code, 404)

    def test_configuration_validation(self):
        for url in ["file:///tmp/model", "https://user:secret@example.invalid", "https://example.invalid?a=b",
                    "https://example.invalid#fragment", "https://example.invalid:wrong", "http://example.invalid"]:
            with self.assertRaises(ValueError):
                SIEClient(url)
        for options in [{"timeout": float("nan")}, {"retry_delay": float("inf")}]:
            with self.assertRaises(ValueError):
                SIEClient("http://127.0.0.1:8080", **options)
        with self.assertRaises(ValueError):
            run_workload(lambda _: {}, 0, 1)
        with self.assertRaises(ValueError):
            cosine([1], [1, 2])
        self.assertAlmostEqual(cosine([1e308], [1e308]), 1.0)

    def test_invalid_ids_spans_and_generation_usage_rejected(self):
        client = SIEClient("http://127.0.0.1:8080")
        with patch.object(client, "_post", return_value={"items": [{"id": [], "dense": {}}]}):
            with self.assertRaises(SIEError):
                client.encode(Models().embedder, [{"id": "a", "text": "test"}])
        with patch.object(client, "_post", return_value={"items": [{"id": "context", "entities": [{}]}]}):
            with self.assertRaisesRegex(SIEError, "span"):
                client.extract(Models().extractor, "test", ["amount"])
        valid = {"text": "synthetic answer", "finish_reason": "stop"}
        with patch.object(client, "_post", return_value=dict(valid)):
            self.assertIsNone(client.generate(Models().generator, "test")["usage"])
        invalid = [dict(valid, usage={}), dict(valid, finish_reason="error"),
                   dict(valid, usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 4})]
        for data in invalid:
            with patch.object(client, "_post", return_value=data):
                with self.assertRaises(SIEError):
                    client.generate(Models().generator, "test")

    def test_response_limit_and_redirect_refusal(self):
        from inference.client import NoRedirects
        with self.assertRaisesRegex(SIEError, "2 MiB"):
            SIEClient._json_body(BytesIO(b"x" * 2_097_153))
        self.assertIsNone(NoRedirects().redirect_request(None, None, 302, "redirect", {}, "https://example.invalid"))

    def test_cli_offline_smoke(self):
        root = Path(__file__).parent.parent
        result = subprocess.run([sys.executable, "-B", "main.py", "demo", "--requests", "2",
                                 "--concurrency", "2"], cwd=root, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["http"]["http_successes"], 8)
        self.assertEqual(report["failed"], 0)
        self.assertIn("Synthetic", report["measurement"])


if __name__ == "__main__":
    unittest.main()

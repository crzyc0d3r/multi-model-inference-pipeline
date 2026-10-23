"""A localhost HTTP contract fixture, not an inference engine or GPU simulator."""

from __future__ import annotations

import json
import threading
import time
from contextlib import AbstractContextManager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

from .client import Models
from .workflow import load_case


class ReplayServer(AbstractContextManager):
    def __init__(self, *, delay: float = .002, loading_responses: int = 0,
                 fail_task: str | None = None):
        self.case = load_case()
        self.delay = delay
        self.loading_responses = loading_responses
        self.fail_task = fail_task
        self.history: list[tuple[str, dict]] = []
        self.lock = threading.Lock()
        owner = self
        models = Models()
        paths = {f"/v1/encode/{models.embedder}": "encode",
                 f"/v1/score/{models.reranker}": "score",
                 f"/v1/extract/{models.extractor}": "extract",
                 f"/v1/generate/{models.generator.replace('/', '__')}": "generate"}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:
                pass

            def respond(self, status: int, value: dict) -> None:
                body = json.dumps(value).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                self.respond(200 if self.path == "/readyz" else 404, {"fixture": True})

            def do_POST(self) -> None:
                task = paths.get(unquote(self.path))
                if task is None:
                    self.respond(404, {"error": {"code": "MODEL_NOT_FOUND"}})
                    return
                try:
                    size = int(self.headers.get("Content-Length", 0))
                    if not 0 < size <= 100_000:
                        raise ValueError("invalid request size")
                    if self.headers.get("Content-Type") != "application/json":
                        raise ValueError("JSON required")
                    payload = json.loads(self.rfile.read(size))
                    with owner.lock:
                        owner.history.append((task, payload))
                        loading = owner.loading_responses > 0
                        if loading:
                            owner.loading_responses -= 1
                    if loading:
                        self.respond(503, {"error": {"code": "MODEL_LOADING"}})
                        return
                    if owner.fail_task == task:
                        self.respond(503, {"error": {"code": "INFERENCE_ERROR"}})
                        return
                    time.sleep(owner.delay)
                    result = owner.dispatch(task, payload)
                    self.respond(200, result)
                except (ValueError, KeyError, TypeError, AttributeError):
                    self.respond(400, {"error": {"code": "INVALID_INPUT"}})

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            request_queue_size = 128

        self.server = Server(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def dispatch(self, task: str, body: dict) -> dict:
        case = self.case
        if task == "encode":
            if body["params"]["output_types"] != ["dense"]:
                raise ValueError("fixture supports dense only")
            text_ids = {x["text"]: x["id"] for x in case["passages"]}
            text_ids[case["query"]] = "query"
            items = []
            for item in body["items"]:
                head = case["vector_heads"][text_ids[item["text"]]]
                items.append({"id": item["id"], "dense": {"dims": 384, "dtype": "float32",
                                                          "values": head + [0.0] * 381}})
            if not items:
                raise ValueError("items must not be empty")
            return {"model": Models().embedder, "items": items}
        if task == "score":
            if body["query"]["text"] != case["query"]:
                raise ValueError("fixture query differs")
            valid = {x["id"]: x["text"] for x in case["passages"]}
            scores = []
            for item in body["items"]:
                if valid[item["id"]] != item["text"]:
                    raise ValueError("fixture passage differs")
                scores.append({"item_id": item["id"], "score": case["scores"][item["id"]]})
            scores.sort(key=lambda x: x["score"], reverse=True)
            return {"model": Models().reranker, "scores": [dict(x, rank=i) for i, x in enumerate(scores)]}
        if task == "extract":
            labels = body["params"]["labels"]
            items = []
            for item in body["items"]:
                entities = []
                for span in case["spans"]:
                    start = item["text"].find(span["text"])
                    if start >= 0 and span["label"] in labels:
                        entities.append(dict(span, start=start, end=start + len(span["text"]),
                                             score=.99, bbox=None))
                items.append({"id": item["id"], "entities": entities})
            return {"model": Models().extractor, "items": items}
        if task == "generate":
            if not 1 <= body["max_new_tokens"] <= 1024:
                raise ValueError("token budget out of range")
            if not all(x["text"] in body["prompt"] for x in case["passages"][:2]):
                raise ValueError("fixture answer needs both source passages")
            return {"text": case["answer"], "finish_reason": "stop",
                    "usage": {"prompt_tokens": 100, "completion_tokens": 19, "total_tokens": 119}}
        raise ValueError("unknown task")

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

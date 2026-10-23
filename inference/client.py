"""Minimal JSON client for the four native SIE HTTP primitives."""

from __future__ import annotations

import json
import math
import threading
import time
import ipaddress
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


@dataclass(frozen=True)
class Models:
    embedder: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    extractor: str = "urchade/gliner_multi-v2.1"
    generator: str = "Qwen/Qwen3-0.6B"


class SIEError(RuntimeError):
    """An actionable failure without echoing request contents or credentials."""


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def finite_number(value) -> bool:
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


class SIEClient:
    def __init__(self, base_url: str, *, api_key: str | None = None,
                 timeout: float = 120, loading_retries: int = 60,
                 retry_delay: float = 1):
        parsed = urlparse(base_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment or any(x.isspace() for x in base_url)):
            raise ValueError("base_url must be an HTTP(S) URL without credentials")
        try:
            parsed.port
        except ValueError:
            raise ValueError("base_url has an invalid port") from None
        try:
            loopback = parsed.hostname == "localhost" or ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = False
        if parsed.scheme == "http" and not loopback:
            raise ValueError("use HTTPS for a server outside the loopback interface")
        if (not math.isfinite(timeout) or timeout <= 0 or loading_retries < 0
                or not math.isfinite(retry_delay) or retry_delay < 0):
            raise ValueError("timeout must be positive; retries and delay must be nonnegative")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.loading_retries = loading_retries
        self.retry_delay = retry_delay
        self._opener = build_opener(ProxyHandler({}), NoRedirects())
        self._lock = threading.Lock()
        self._stats = {"http_attempts": 0, "http_successes": 0,
                       "http_error_responses": 0, "transport_errors": 0,
                       "loading_retries": 0}

    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def _count(self, key: str) -> None:
        with self._lock:
            self._stats[key] += 1

    @staticmethod
    def _json_body(response) -> dict:
        raw = response.read(2_097_153)
        if len(raw) > 2_097_152:
            raise SIEError("response exceeds the 2 MiB client limit")
        return json.loads(raw)

    def ready(self) -> bool:
        try:
            with self._opener.open(self.base_url + "/readyz", timeout=self.timeout) as response:
                return response.status == 200
        except (URLError, TimeoutError):
            return False

    def _post(self, task: str, model: str, payload: dict) -> dict:
        # Native encode/score/extract retain the slash. Generation uses __.
        route_model = model.replace("/", "__") if task == "generate" else model
        path = f"/v1/{task}/{quote(route_model, safe='/:-_.')}"
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        for attempt in range(self.loading_retries + 1):
            self._count("http_attempts")
            request = Request(self.base_url + path, data=body, headers=headers, method="POST")
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    self._count("http_successes")
                    data = self._json_body(response)
                if not isinstance(data, dict):
                    raise SIEError(f"{task}: response must be an object")
                return data
            except HTTPError as exc:
                self._count("http_error_responses")
                try:
                    error = self._json_body(exc)
                    code = error.get("error", {}).get("code", "HTTP_ERROR")
                except (ValueError, AttributeError):
                    code = "HTTP_ERROR"
                # Retry only an explicit pre-inference loading response. In
                # particular, a generation read timeout must never be replayed.
                if exc.code == 503 and code == "MODEL_LOADING" and attempt < self.loading_retries:
                    self._count("loading_retries")
                    time.sleep(self.retry_delay)
                    continue
                allowed = {"MODEL_LOADING", "MODEL_LOAD_FAILED", "MODEL_NOT_FOUND",
                           "INVALID_INPUT", "INFERENCE_ERROR", "RESOURCE_EXHAUSTED", "HTTP_ERROR"}
                safe_code = code if isinstance(code, str) and code in allowed else "HTTP_ERROR"
                raise SIEError(f"{task}: HTTP {exc.code} {safe_code}") from None
            except (URLError, TimeoutError, OSError):
                self._count("transport_errors")
                raise SIEError(f"{task}: connection failed or timed out; request was not retried") from None
            except (ValueError, UnicodeError):
                raise SIEError(f"{task}: response is not valid JSON") from None
        raise AssertionError("unreachable")

    @staticmethod
    def _items(data: dict, expected_ids: list[str], task: str) -> list[dict]:
        items = data.get("items")
        if not isinstance(items, list) or not all(isinstance(x, dict) for x in items):
            raise SIEError(f"{task}: missing result items")
        actual = [x.get("id") for x in items]
        if (not all(isinstance(x, str) and x for x in actual)
                or len(actual) != len(expected_ids) or len(set(actual)) != len(actual)
                or set(actual) != set(expected_ids)):
            raise SIEError(f"{task}: result IDs do not match submitted items")
        by_id = {x["id"]: x for x in items}
        ordered = [by_id[item_id] for item_id in expected_ids]
        if any(x.get("error") is not None for x in ordered):
            raise SIEError(f"{task}: server reported a per-item inference error")
        return ordered

    def encode(self, model: str, items: list[dict]) -> list[list[float]]:
        data = self._post("encode", model, {"items": items, "params": {"output_types": ["dense"]}})
        results = self._items(data, [x["id"] for x in items], "encode")
        vectors = []
        for item in results:
            dense = item.get("dense", {})
            values = dense.get("values") if isinstance(dense, dict) else None
            if (not isinstance(values, list) or not values or dense.get("dims") != len(values)
                    or not all(finite_number(v) for v in values)):
                raise SIEError("encode: invalid dense vector envelope")
            vectors.append(values)
        if len({len(x) for x in vectors}) > 1:
            raise SIEError("encode: vector dimensions differ")
        return vectors

    def score(self, model: str, query: str, items: list[dict]) -> list[dict]:
        data = self._post("score", model, {"query": {"text": query}, "items": items})
        scores = data.get("scores")
        if not isinstance(scores, list) or not all(isinstance(x, dict) for x in scores):
            raise SIEError("score: missing scores")
        ids = [x.get("item_id") for x in scores]
        if (not all(isinstance(x, str) and x for x in ids) or len(set(ids)) != len(ids)
                or len(ids) != len(items) or set(ids) != {x["id"] for x in items}):
            raise SIEError("score: candidate IDs do not match")
        for entry in scores:
            value = entry.get("score")
            if not finite_number(value):
                raise SIEError("score: invalid relevance score")
        return sorted(scores, key=lambda x: x["score"], reverse=True)

    def extract(self, model: str, text: str, labels: list[str]) -> list[dict]:
        data = self._post("extract", model, {"items": [{"id": "context", "text": text}],
                                            "params": {"labels": labels}})
        [item] = self._items(data, ["context"], "extract")
        entities = item.get("entities")
        if not isinstance(entities, list) or not all(isinstance(x, dict) for x in entities):
            raise SIEError("extract: missing entities")
        for entity in entities:
            score = entity.get("score")
            start, end = entity.get("start"), entity.get("end")
            if (not isinstance(entity.get("text"), str) or entity.get("label") not in labels
                    or not finite_number(score) or not 0 <= score <= 1
                    or not isinstance(start, int) or isinstance(start, bool)
                    or not isinstance(end, int) or isinstance(end, bool)
                    or not 0 <= start <= end <= len(text) or text[start:end] != entity["text"]):
                raise SIEError("extract: invalid entity span")
        return entities

    def generate(self, model: str, prompt: str, max_new_tokens: int = 64) -> dict:
        data = self._post("generate", model, {"prompt": prompt, "max_new_tokens": max_new_tokens,
                                              "temperature": 0.0})
        if not isinstance(data.get("text"), str) or not data["text"].strip():
            raise SIEError("generate: missing answer text")
        reason = data.get("finish_reason")
        if not isinstance(reason, str) or reason not in {"stop", "length"}:
            raise SIEError("generate: generation did not finish successfully")
        usage = data.get("usage")
        if usage is not None:
            keys = ("prompt_tokens", "completion_tokens", "total_tokens")
            if (not isinstance(usage, dict) or not all(isinstance(usage.get(k), int)
                    and not isinstance(usage[k], bool) and usage[k] >= 0 for k in keys)
                    or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]):
                raise SIEError("generate: invalid token usage")
        data["usage"] = usage
        return data

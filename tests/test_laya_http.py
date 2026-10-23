import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from bench import laya_http as lh
from bench.jev_advisory import QUESTIONS


def _answers(review=0.9):
    out = {}
    for name, q in QUESTIONS.items():
        opts = list(q["criteria"])
        rest = (1 - review) / (len(opts) - 1)
        out[name] = {"type": "choice", "choice": opts[0],
                     "probabilities": {o: (review if o == opts[0] else rest) for o in opts}}
    return out


def _ok(route="english", usage=None):
    return {"model": "laya-rl-agent", "answers": _answers(), "routing": {"model": route, "reason": "x"},
            "usage": usage or {"input_tokens": 300, "state_tokens": 250, "state_tokens_dropped": 0,
                               "truncated": False, "truncated_questions": []}}


@pytest.fixture
def server():
    """Fake laya-serve; `plan` is a list of (status, body or callable) consumed per POST."""
    state = {"plan": [], "posts": [], "health": {"revisions": {"english": "abc"}}}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, status, body):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("X-Inference-Time-Ms", "12.5")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200, state["health"])

        def do_POST(self):
            state["posts"].append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            status, body = state["plan"].pop(0) if state["plan"] else (200, _ok())
            if callable(body):
                body = body()
            self._send(status, body)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield state
    httpd.shutdown()


def _client(server, timeout=5.0):
    return lh.HttpLaya(server["url"], timeout=timeout, pace=0.0)


def test_success_pins_english_and_parses(server):
    res = lh.assess(_client(server), "synthetic")
    assert res["available"] and res["elevated"] is True
    assert server["posts"][0]["model"] == "english" and server["posts"][0]["questions"] == QUESTIONS
    assert res["routing"]["model"] == "english" and res["server_inference_ms"] == "12.5"


def test_http_500_twice_is_unavailable_after_one_retry(server):
    server["plan"] = [(500, {"detail": "inference failed"})] * 2
    client = _client(server)
    res = lh.assess(client, "synthetic")
    assert res["available"] is False and res["elevated"] is None and "HTTPError" in res["error"]
    assert client.calls == 2 and client.retried == 1 and not lh.cacheable(res)


def test_retry_recovers_once(server):
    server["plan"] = [(500, {"detail": "x"}), (200, _ok())]
    assert lh.assess(_client(server), "synthetic")["available"]


def test_malformed_json_is_unavailable(server):
    server["plan"] = [(200, b"not json")] * 2
    res = lh.assess(_client(server), "synthetic")
    assert res["available"] is False and res["answers"] is None


def test_timeout_is_unavailable(server):
    def slow():
        time.sleep(1.0)
        return _ok()
    server["plan"] = [(200, slow)] * 2
    res = lh.assess(_client(server, timeout=0.2), "synthetic")
    assert res["available"] is False and not lh.cacheable(res)


def test_tunnel_down_is_unavailable():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    res = lh.assess(lh.HttpLaya(f"http://127.0.0.1:{port}", timeout=1, pace=0), "synthetic")
    assert res["available"] is False and "URLError" in res["error"]


def test_non_english_route_is_flagged_not_retried(server):
    server["plan"] = [(200, _ok(route="multilingual"))]
    client = _client(server)
    res = lh.assess(client, "synthetic")
    assert res["available"] is False and res["error"].startswith("RoutingError")
    assert client.calls == 1 and lh.cacheable(res)


def test_truncation_is_flagged(server):
    server["plan"] = [(200, _ok(usage={"truncated": True, "state_tokens_dropped": 5}))]
    res = lh.assess(_client(server), "synthetic")
    assert res["available"] is False and res["error"].startswith("ValueError: truncated")
    assert lh.cacheable(res)


def test_resume_skips_cached_answers_but_retries_transport_failures(server, tmp_path):
    prov = {"revision": "abc", "laya_version": "t"}
    due = {1: "a", 2: "b"}
    server["plan"] = [(200, _ok()), (500, {}), (500, {})]
    client = _client(server)
    first = lh.run_assessments(client, prov, due, tmp_path, None, "t", lambda: None)
    assert first[1]["available"] and not first[2]["available"]
    again = lh.run_assessments(client, prov, due, tmp_path, None, "t", lambda: None)
    assert again[2]["available"] and client.calls == 4  # only the failed one was re-sent

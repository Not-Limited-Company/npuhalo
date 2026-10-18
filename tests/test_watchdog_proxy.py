#!/usr/bin/env python3
"""
test_watchdog_proxy.py — Unit and Integration tests for NPU Watchdog & Smart Proxy.
"""

import sys
import os
import json
import asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer, TestClient

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "verifier", "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from watchdog_analyzer import WatchdogAnalyzer
from npuhalo_proxy import NPUHaloProxy, create_app

class TestWatchdogAnalyzer:
    def setup_method(self):
        self.analyzer = WatchdogAnalyzer(enable_npu_eval=False)

    def test_detects_destructive_root_deletion(self):
        malicious_samples = [
            "<parameter=command>rm -rf /</parameter>",
            "<parameter=command>rm -rf /*</parameter>",
            "```bash\nrm -rf ~\n```",
            "<parameter=command>rm -r .git</parameter>",
            "<parameter=command>mkfs.ext4 /dev/sda1</parameter>",
            "<parameter=command>dd if=/dev/zero of=/dev/nvme0n1</parameter>",
            "<parameter=command>git clean -fdx</parameter>",
        ]
        for sample in malicious_samples:
            verdict = self.analyzer.check_regex_rules(sample)
            assert verdict is not None, f"Failed to detect destructive command: {sample}"
            assert verdict.is_anomaly is True
            assert verdict.category == "DESTRUCTIVE_COMMAND"
            assert verdict.risk_level in ("HIGH", "CRITICAL")

    def test_detects_secret_leakage(self):
        secret_samples = [
            "Here is the token: sk-proj-1234567890abcdef1234567890abcdef",
            "Exporting token: ghp_123456789012345678901234567890123456",
            "Command: <parameter=command>cat .env</parameter>",
            "Command: <parameter=command>cat ~/.ssh/id_rsa</parameter>",
        ]
        for sample in secret_samples:
            verdict = self.analyzer.check_regex_rules(sample)
            assert verdict is not None, f"Failed to detect secret leak: {sample}"
            assert verdict.is_anomaly is True
            assert verdict.category == "SECRET_LEAK"

    def test_allows_benign_commands(self):
        safe_samples = [
            "<parameter=command>ls -la</parameter>",
            "<parameter=command>pytest tests/test_api.py</parameter>",
            "<parameter=command>git status</parameter>",
            "<parameter=command>python3 scripts/npu_benchmark.py</parameter>",
            "```python\ndef solve(): return 42\n```",
        ]
        for sample in safe_samples:
            verdict = self.analyzer.check_regex_rules(sample)
            assert verdict is None, f"False positive on safe sample: {sample}"

    def test_detects_infinite_loops(self):
        session_id = "test_session_1"
        cmd = "<parameter=command>cargo test --workspace</parameter>"
        
        # 1st time: safe
        assert self.analyzer.check_loop_oscillation(session_id, cmd) is None
        # 2nd time: safe
        assert self.analyzer.check_loop_oscillation(session_id, cmd) is None
        # 3rd time: triggers loop anomaly
        v = self.analyzer.check_loop_oscillation(session_id, cmd)
        assert v is not None
        assert v.is_anomaly is True
        assert v.category == "LOOP_DETECTED"
        assert v.risk_level == "HIGH"

def _sse_event(model: str, content: str) -> bytes:
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


async def _gpu_streaming_upstream(request):
    """Fake GPU lane that streams a benign head, a destructive command, then a tail marker."""
    body = await request.json()
    model = body.get("model", "m")
    resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    try:
        await resp.write(_sse_event(model, "Here is a command: "))
        await asyncio.sleep(0.05)
        await resp.write(_sse_event(model, "<parameter=command>rm -rf /</parameter>"))
        await asyncio.sleep(0.05)
        await resp.write(_sse_event(model, "TAIL-MUST-NOT-ARRIVE"))
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
    except Exception:
        pass  # the guard cut the client stream, so later writes are expected to fail
    return resp


def _sse_contents(sse_text: str):
    """Collect assistant content deltas from an SSE transcript."""
    contents = []
    for line in sse_text.splitlines():
        line = line.strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        payload = json.loads(line[6:])
        delta = (payload.get("choices") or [{}])[0].get("delta") or {}
        if delta.get("content"):
            contents.append(delta["content"])
    return contents


def test_streaming_guard_blocks_mid_stream():
    """guard_mode=block must cut a live SSE stream when a destructive command appears."""
    async def _run():
        gpu_app = web.Application()
        gpu_app.router.add_post("/v1/chat/completions", _gpu_streaming_upstream)
        gpu_server = TestServer(gpu_app)
        await gpu_server.start_server()

        proxy = NPUHaloProxy(
            gpu_url=str(gpu_server.make_url("")).rstrip("/"),
            guard_mode="block",
        )
        await proxy.start()
        server = TestServer(create_app(proxy))
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post("/v1/chat/completions", json={
                "model": "test-model",
                "stream": True,
                "messages": [{"role": "user", "content": "deploy it"}],
            })
            assert resp.status == 200
            assert "text/event-stream" in resp.headers.get("Content-Type", "")

            text = await resp.text()
            contents = _sse_contents(text)

            # The benign head reached the client, then the stream was cut.
            assert contents[0] == "Here is a command: "
            assert contents[-1].startswith("[NPUHALO WATCHDOG INTERCEPTION]")
            # The destructive tool call was never emitted as assistant content.
            assert not any("<parameter=command>" in c for c in contents)
            assert "TAIL-MUST-NOT-ARRIVE" not in text
            assert proxy._interceptions_count == 1
            assert proxy.watchdog.get_stats()["total_anomalies"] >= 1
        finally:
            await client.close()
            await server.close()
            await proxy.stop()
            await gpu_server.close()

    asyncio.run(_run())


def test_streaming_audit_mode_never_cuts_stream():
    """guard_mode=audit must stay a pure pass-through (zero added delay, no rewriting)."""
    async def _run():
        gpu_app = web.Application()
        gpu_app.router.add_post("/v1/chat/completions", _gpu_streaming_upstream)
        gpu_server = TestServer(gpu_app)
        await gpu_server.start_server()

        proxy = NPUHaloProxy(
            gpu_url=str(gpu_server.make_url("")).rstrip("/"),
            guard_mode="audit",
        )
        await proxy.start()
        server = TestServer(create_app(proxy))
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post("/v1/chat/completions", json={
                "model": "test-model",
                "stream": True,
                "messages": [{"role": "user", "content": "deploy it"}],
            })
            text = await resp.text()
            assert "TAIL-MUST-NOT-ARRIVE" in text
            assert "NPUHALO WATCHDOG INTERCEPTION" not in text
            assert proxy._interceptions_count == 0
        finally:
            await client.close()
            await server.close()
            await proxy.stop()
            await gpu_server.close()

    asyncio.run(_run())


def test_fast_lane_stream_wraps_json_as_sse():
    """Router fast-lane must return SSE to a stream=true client, not a bare JSON body."""
    async def _run():
        async def npu_handler(request):
            return web.json_response({
                "id": "chatcmpl-npu",
                "object": "chat.completion",
                "created": 0,
                "model": "qwen3.5:0.8b",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello! 42"},
                    "finish_reason": "stop",
                }],
            })

        npu_app = web.Application()
        npu_app.router.add_post("/v1/chat/completions", npu_handler)
        npu_server = TestServer(npu_app)
        await npu_server.start_server()

        proxy = NPUHaloProxy(
            gpu_url="http://127.0.0.1:1",
            npu_url=str(npu_server.make_url("")).rstrip("/"),
            enable_router=True,
            guard_mode="block",
        )
        await proxy.start()
        server = TestServer(create_app(proxy))
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post("/v1/chat/completions", json={
                "model": "test-model",
                "stream": True,
                "messages": [{"role": "user", "content": "Hello there"}],
            })
            assert resp.status == 200
            assert "text/event-stream" in resp.headers.get("Content-Type", "")
            assert resp.headers.get("X-NPUHalo-Route") == "npu-direct"

            text = await resp.text()
            assert "Hello! 42" in text
            assert text.strip().endswith("data: [DONE]")
        finally:
            await client.close()
            await server.close()
            await proxy.stop()
            await npu_server.close()

    asyncio.run(_run())


def test_stream_guard_window_ignores_safe_text():
    proxy = NPUHaloProxy(guard_mode="block")
    assert proxy._stream_guard_verdict("ls -la\npytest tests/test_api.py") is None
    verdict = proxy._stream_guard_verdict("```bash\nrm -rf ~\n```")
    assert verdict is not None
    assert verdict.risk_level in ("HIGH", "CRITICAL")


def test_fast_lane_falls_back_to_gpu_when_npu_unreachable():
    """A deterministic NPU fast-lane hit must never 500 when the NPU is down."""
    async def _run():
        async def gpu_handler(request):
            await request.json()
            return web.json_response({
                "id": "chatcmpl-gpu",
                "object": "chat.completion",
                "model": "gpu-model",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "GPU answered"},
                    "finish_reason": "stop",
                }],
            })

        gpu_app = web.Application()
        gpu_app.router.add_post("/v1/chat/completions", gpu_handler)
        gpu_server = TestServer(gpu_app)
        await gpu_server.start_server()

        proxy = NPUHaloProxy(
            gpu_url=str(gpu_server.make_url("")).rstrip("/"),
            npu_url="http://127.0.0.1:1",  # nothing listening
            enable_router=True,
            guard_mode="block",
        )
        await proxy.start()
        server = TestServer(create_app(proxy))
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post("/v1/chat/completions", json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "What is 25 * 4?"}],  # NPU fast-lane pattern
            })
            assert resp.status == 200
            data = await resp.json()
            assert data["choices"][0]["message"]["content"] == "GPU answered"
            assert resp.headers.get("X-NPUHalo-Route") == "gpu"
        finally:
            await client.close()
            await server.close()
            await proxy.stop()
            await gpu_server.close()

    asyncio.run(_run())


def test_fast_lane_falls_back_when_npu_returns_error_body():
    """flm answers 200 {"error": ...} on a bad NPU stack; that must fall back too."""
    async def _run():
        async def npu_handler(request):
            return web.json_response({"error": "qds_device::wait() unexpected command state"})

        async def gpu_handler(request):
            await request.json()
            return web.json_response({
                "id": "chatcmpl-gpu",
                "object": "chat.completion",
                "model": "gpu-model",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "GPU answered"},
                    "finish_reason": "stop",
                }],
            })

        npu_app = web.Application()
        npu_app.router.add_post("/v1/chat/completions", npu_handler)
        npu_server = TestServer(npu_app)
        await npu_server.start_server()

        gpu_app = web.Application()
        gpu_app.router.add_post("/v1/chat/completions", gpu_handler)
        gpu_server = TestServer(gpu_app)
        await gpu_server.start_server()

        proxy = NPUHaloProxy(
            gpu_url=str(gpu_server.make_url("")).rstrip("/"),
            npu_url=str(npu_server.make_url("")).rstrip("/"),
            enable_router=True,
            guard_mode="block",
        )
        await proxy.start()
        server = TestServer(create_app(proxy))
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post("/v1/chat/completions", json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "What is 25 * 4?"}],
            })
            assert resp.status == 200
            data = await resp.json()
            assert data["choices"][0]["message"]["content"] == "GPU answered"
        finally:
            await client.close()
            await server.close()
            await proxy.stop()
            await gpu_server.close()
            await npu_server.close()

    asyncio.run(_run())


def _completion(model: str, content: str) -> dict:
    return {
        "id": "chatcmpl-test", "object": "chat.completion", "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
    }


async def _serve(payload_for, hits):
    async def handler(request):
        body = await request.json()
        hits.append(body)
        return web.json_response(payload_for(body))
    return handler


def _is_audit_request(hit: dict) -> bool:
    """The watchdog's semantic tier posts its own prompt to the NPU.

    With the semantic audit enabled, GPU-served requests still touch the NPU for
    safety auditing; those calls must not be mistaken for generation traffic.
    """
    content = (hit.get("messages") or [{}])[0].get("content")
    return isinstance(content, str) and "AI Safety Watchdog" in content


def _generations(hits):
    return [h for h in hits if not _is_audit_request(h)]


def test_explicit_npu_model_is_served_by_npu():
    """`model: qwen3.5` must go to the NPU backend, with the upstream tag rewritten."""
    async def _run():
        npu_hits, gpu_hits = [], []
        npu_app = web.Application()
        npu_app.router.add_post("/v1/chat/completions",
                                await _serve(lambda b: _completion("qwen3.5:0.8b", "from the NPU"), npu_hits))
        gpu_app = web.Application()
        gpu_app.router.add_post("/v1/chat/completions",
                                await _serve(lambda b: _completion("gpu", "from the GPU"), gpu_hits))
        npu_server, gpu_server = TestServer(npu_app), TestServer(gpu_app)
        await npu_server.start_server()
        await gpu_server.start_server()

        proxy = NPUHaloProxy(
            gpu_url=str(gpu_server.make_url("")).rstrip("/"),
            npu_url=str(npu_server.make_url("")).rstrip("/"),
            npu_model_names=["qwen3.5"],
            guard_mode="block",
        )
        await proxy.start()
        server = TestServer(create_app(proxy))
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post("/v1/chat/completions", json={
                "model": "qwen3.5",
                "messages": [{"role": "user", "content": "describe this"}],
            })
            assert resp.status == 200
            data = await resp.json()
            assert data["choices"][0]["message"]["content"] == "from the NPU"
            assert resp.headers.get("X-NPUHalo-Route") == "npu-direct"
            gens = _generations(npu_hits)
            assert len(gens) == 1                      # one NPU generation (plus any audit calls)
            assert gens[0]["model"] == "qwen3.5:0.8b"  # rewritten to the upstream tag
            assert _generations(gpu_hits) == []        # the GPU never generated this answer
        finally:
            await client.close()
            await server.close()
            await proxy.stop()
            await gpu_server.close()
            await npu_server.close()

    asyncio.run(_run())


def test_explicit_npu_model_never_falls_back_to_gpu():
    """A request that named the NPU model must fail loudly, never be answered by DeepSeek."""
    async def _run():
        gpu_hits = []
        gpu_app = web.Application()
        gpu_app.router.add_post("/v1/chat/completions",
                                await _serve(lambda b: _completion("gpu", "from the GPU"), gpu_hits))
        gpu_server = TestServer(gpu_app)
        await gpu_server.start_server()

        proxy = NPUHaloProxy(
            gpu_url=str(gpu_server.make_url("")).rstrip("/"),
            npu_url="http://127.0.0.1:1",  # nothing listening
            npu_model_names=["qwen3.5"],
            guard_mode="block",
        )
        await proxy.start()
        server = TestServer(create_app(proxy))
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post("/v1/chat/completions", json={
                "model": "qwen3.5",
                "messages": [{"role": "user", "content": "describe this"}],
            })
            assert resp.status == 502, f"expected a loud failure, got {resp.status}"
            body = await resp.json()
            assert body["backend"] == "npu"
            assert gpu_hits == []  # the GPU model was never substituted
        finally:
            await client.close()
            await server.close()
            await proxy.stop()
            await gpu_server.close()

    asyncio.run(_run())


def test_gpu_model_answers_from_gpu_not_npu():
    """Without router gating, the GPU model never generates on the NPU.

    (NPU *audit* traffic is expected and allowed - that is the watchdog's semantic tier.)
    """
    async def _run():
        npu_hits, gpu_hits = [], []
        npu_app = web.Application()
        npu_app.router.add_post("/v1/chat/completions",
                                await _serve(lambda b: _completion("npu", "from the NPU"), npu_hits))
        gpu_app = web.Application()
        gpu_app.router.add_post("/v1/chat/completions",
                                await _serve(lambda b: _completion("gpu", "from the GPU"), gpu_hits))
        npu_server, gpu_server = TestServer(npu_app), TestServer(gpu_app)
        await npu_server.start_server()
        await gpu_server.start_server()

        proxy = NPUHaloProxy(
            gpu_url=str(gpu_server.make_url("")).rstrip("/"),
            npu_url=str(npu_server.make_url("")).rstrip("/"),
            npu_model_names=["qwen3.5"],
            guard_mode="block",
        )
        await proxy.start()
        server = TestServer(create_app(proxy))
        client = TestClient(server)
        await client.start_server()
        try:
            # "Summarize the plot of Hamlet" is exactly the kind of prompt the router
            # used to hijack; with gating off it must reach the GPU.
            resp = await client.post("/v1/chat/completions", json={
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "Summarize the plot of Hamlet"}],
            })
            assert resp.status == 200
            data = await resp.json()
            assert data["choices"][0]["message"]["content"] == "from the GPU"
            assert resp.headers.get("X-NPUHalo-Route") == "gpu"
            assert _generations(npu_hits) == []        # no NPU generation for this request
            assert _is_audit_request(npu_hits[0])      # only the out-of-loop audit hit the NPU
        finally:
            await client.close()
            await server.close()
            await proxy.stop()
            await gpu_server.close()
            await npu_server.close()

    asyncio.run(_run())


def test_multimodal_content_is_forwarded_intact():
    """Image blocks must reach the NPU untouched, byte for byte."""
    async def _run():
        npu_hits = []
        npu_app = web.Application()
        npu_app.router.add_post("/v1/chat/completions",
                                await _serve(lambda b: _completion("qwen3.5:0.8b", "blue circle on red square"), npu_hits))
        npu_server = TestServer(npu_app)
        await npu_server.start_server()

        text_block = {"type": "text", "text": "What is in this image?"}
        image_block = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}

        proxy = NPUHaloProxy(
            gpu_url="http://127.0.0.1:1",
            npu_url=str(npu_server.make_url("")).rstrip("/"),
            npu_model_names=["qwen3.5"],
            guard_mode="block",
        )
        await proxy.start()
        server = TestServer(create_app(proxy))
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post("/v1/chat/completions", json={
                "model": "qwen3.5",
                "messages": [{"role": "user", "content": [text_block, image_block]}],
            })
            assert resp.status == 200
            data = await resp.json()
            assert data["choices"][0]["message"]["content"] == "blue circle on red square"
            gens = _generations(npu_hits)
            assert len(gens) == 1
            assert gens[0]["messages"][0]["content"] == [text_block, image_block]
        finally:
            await client.close()
            await server.close()
            await proxy.stop()
            await npu_server.close()

    asyncio.run(_run())


def test_proxy_endpoints_sync():
    async def _run():
        proxy = NPUHaloProxy(
            gpu_url="http://127.0.0.1:8012",
            npu_url="http://127.0.0.1:8001",
            guard_mode="block",
        )
        await proxy.start()
        app = create_app(proxy)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()

        try:
            # 1. Test /v1/models
            resp = await client.get("/v1/models")
            assert resp.status == 200
            data = await resp.json()
            model_ids = [m["id"] for m in data.get("data", [])]
            assert "Ornith-1.5-35B-A3B-ROCmFP4.gguf" in model_ids
            assert "qwen3.5:0.8b-watchdog" in model_ids


            # 2. Test /v1/watchdog/status
            resp = await client.get("/v1/watchdog/status")
            assert resp.status == 200
            stats = await resp.json()
            assert "total_audited" in stats
            assert stats["guard_mode"] == "block"

            # 3. Test /v1/watchdog/clear
            resp = await client.post("/v1/watchdog/clear")
            assert resp.status == 200
            res = await resp.json()
            assert res["status"] == "cleared"
        finally:
            await client.close()
            await server.close()
            await proxy.stop()

    asyncio.run(_run())


def test_oversized_request_body_is_reported_as_too_large():
    """An image request that exceeds the body limit must say so, not "Invalid JSON".

    aiohttp caps request bodies at 1 MiB by default; images arrive base64-encoded
    (~1.37x the file size), so the old behaviour rejected every real image with a
    misleading 400.
    """
    async def _run():
        proxy = NPUHaloProxy(max_body_mb=1)  # 1 MiB, so a small body trips it
        await proxy.start()
        server = TestServer(create_app(proxy))
        client = TestClient(server)
        await client.start_server()
        try:
            payload = {"model": "qwen3.5",
                       "messages": [{"role": "user", "content": "x" * (2 * 1024 * 1024)}]}
            resp = await client.post("/v1/chat/completions", json=payload)
            assert resp.status == 413, f"expected 413, got {resp.status}"
            body = await resp.json()
            assert "too large" in body["error"]
            assert "Invalid JSON" not in body["error"]
        finally:
            await client.close()
            await server.close()
            await proxy.stop()

    asyncio.run(_run())


def test_malformed_json_still_reports_400():
    async def _run():
        proxy = NPUHaloProxy()
        await proxy.start()
        server = TestServer(create_app(proxy))
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post("/v1/chat/completions", data=b"{not json",
                                     headers={"Content-Type": "application/json"})
            assert resp.status == 400
            assert (await resp.json())["error"] == "Invalid JSON body"
        finally:
            await client.close()
            await server.close()
            await proxy.stop()

    asyncio.run(_run())


def test_default_body_limit_fits_real_images():
    """Regression guard: the default must never drop back to aiohttp's 1 MiB."""
    proxy = NPUHaloProxy()
    assert proxy.max_body_mb >= 16, f"default body limit too small: {proxy.max_body_mb} MiB"
    app = create_app(proxy)
    assert app._client_max_size == proxy.max_body_mb * 1024 * 1024
    # a 390-byte test image must fit with enormous headroom
    assert app._client_max_size > 10 * 1024 * 1024



class TestImageRouting:
    """Image-bearing requests go to the local vision model and are not retried elsewhere."""

    def _proxy(self, **overrides):
        return NPUHaloProxy(**overrides)

    def _select(self, proxy, messages, model="deepseek-v4-flash", session="session-1"):
        body = {"model": model, "messages": messages}
        prompt = messages[-1].get("content")
        return asyncio.run(proxy._select_backend(body, prompt, session))

    @staticmethod
    def _user(text):
        return [{"role": "user", "content": text}]

    def test_an_image_is_served_locally_without_a_text_model_fallback(self):
        """The NPU is the only vision backend, so an image must not fall back to the GPU.

        It did, and the retry failed on the text model with *its* error ("image input is not
        supported - you may need to provide the mmproj"), which reads like a misconfigured
        text model rather than a vision backend that could not serve the image.
        """
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}]
        selection = self._select(self._proxy(), messages, session="image-chat")
        assert selection["backend"] == "npu"
        assert selection["allow_fallback"] is False

    def test_the_image_lane_follows_the_configured_target(self):
        """--private-image-target gpu sends image requests to the GPU model instead."""
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}]
        selection = self._select(self._proxy(image_target="gpu"), messages, session="image-chat")
        assert selection["backend"] == "gpu"

    def test_a_text_request_is_unaffected(self):
        selection = self._select(self._proxy(), self._user("write a fibonacci function"))
        assert selection["backend"] == "gpu"

    def test_an_explicit_npu_model_still_wins(self):
        selection = self._select(self._proxy(), self._user("anything at all"), model="qwen3.5")
        assert selection["backend"] == "npu"


def test_session_key_prefers_the_client_header():
    from npuhalo_proxy import session_key
    assert session_key("abc-123", [{"role": "user", "content": "hi"}]) == "abc-123"


def test_session_key_is_stable_across_turns_without_a_header():
    """An agent sends no session header, so the opening ask identifies the conversation."""
    from npuhalo_proxy import session_key
    opening = [{"role": "user", "content": "please review the billing service"}]
    later = opening + [
        {"role": "assistant", "content": "sure"},
        {"role": "user", "content": "and now the retry logic"},
    ]
    other = [{"role": "user", "content": "a completely unrelated conversation"}]
    assert session_key(None, opening) == session_key(None, later)
    assert session_key(None, later) != session_key(None, other)
    assert session_key(None, later).startswith("auto-")


def test_npu_responses_get_a_unique_id():
    """FastFlowLM reuses one response id, and LiteLLM keys its spend log on that id.

    The first NPU-served request was logged and every one after it failed the unique
    constraint on `LiteLLM_SpendLogs.request_id` and was dropped -- silently, while the
    response looked perfectly fine to the client. Every image request routes to the NPU, so
    images were the visible symptom; NPU text vanished the same way.
    """
    from npuhalo_proxy import NPUHaloProxy, new_response_id

    first, second = new_response_id("npu"), new_response_id("gpu")
    assert first != second
    # The lane tag is what makes the spend-log row self-describing: LiteLLM keys that row on this
    # id, and records the *requested* model in its other columns.
    assert first.startswith("chatcmpl-npu-")
    assert second.startswith("chatcmpl-gpu-")
    assert new_response_id().startswith("chatcmpl-local-")

    seen: set = set()
    chunk = b'data: {"id": "fastflowlm-chat-completion", "choices": []}\n\n'
    out = NPUHaloProxy._rewrite_chunk_ids(chunk, first, seen)
    assert b"fastflowlm-chat-completion" not in out
    assert first.encode() in out

    # the unspaced form, and the replacement is reused for later chunks in the same stream
    spaced = b'data: {"id":"fastflowlm-chat-completion"}\n\n'
    assert second.encode() in NPUHaloProxy._rewrite_chunk_ids(spaced, second, seen)
    # a chunk carrying no id is passed through untouched
    assert NPUHaloProxy._rewrite_chunk_ids(b"data: [DONE]\n\n", first, seen) == b"data: [DONE]\n\n"

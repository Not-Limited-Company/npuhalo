#!/usr/bin/env python3
"""
npuhalo_proxy.py — NPU+iGPU guardrail reverse proxy for AMD Strix Halo.
Exposes an OpenAI-compatible /v1/chat/completions server on port 8000.

Backend selection is explicit: a request is served by the NPU when its `model`
names one of the configured NPU models (e.g. "qwen3.5"), and by the primary GPU
model otherwise. There is no heuristic downgrade of GPU traffic.

Features (applied uniformly to both backends):
  1. Zero-Latency Streaming: tokens are passed through as they arrive.
  2. Security Interception: destructive commands, secret leaks and infinite loops
     are detected. In "block" mode the guard also cuts a live SSE stream mid-flight
     (incremental deterministic scan, no added latency for safe tokens).
  3. Asynchronous NPU Watchdog: the XDNA 2 NPU audits tool calls & commands
     out-of-loop at ~2-4W.
  4. In-Flight Compression: prompts over the threshold are distilled on the NPU
     before the upstream prefill (text-only; multimodal content is left intact).
  5. Optional Pre-Route Gating: --enable-router additionally lets a small NPU model
     classify and absorb "trivial" queries that did not explicitly ask for the NPU.
     Off by default: it can silently downgrade answer quality.
  6. Image Routing: image-bearing requests are served by the vision model on the NPU,
     which is the only vision-capable backend here. Text-only requests are unaffected.
"""

import os
import re
import sys
import json
import time
import uuid
import asyncio
import hashlib
import logging
import argparse
from typing import Any, Dict, List, Optional

import aiohttp
from aiohttp import web

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "verifier", "src"))
from watchdog_analyzer import WatchdogAnalyzer, AnomalyVerdict
from npu_router import HybridNPURouter
from compressor_sidecar import DualCompressorSidecar
from energy import EnergyMeter

# Backend lanes. They key the telemetry counters and the lane tags in response ids.
LOCAL_GPU = "gpu"
LOCAL_NPU = "npu"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("npuhalo_proxy")

DEFAULT_PROXY_PORT = 8000
DEFAULT_GPU_URL = "http://127.0.0.1:8012"
DEFAULT_NPU_URL = "http://127.0.0.1:8001"
DEFAULT_GPU_MODEL = "Ornith-1.5-35B-A3B-ROCmFP4.gguf"
DEFAULT_NPU_MODEL = "qwen3.5:0.8b"
DEFAULT_NPU_MODEL_NAMES = ["qwen3.5"]
# aiohttp caps request bodies at 1 MiB by default, which is far below what an
# image request needs: images arrive base64-encoded (~1.37x the file size).
DEFAULT_MAX_BODY_MB = 64

SSE_HEADERS = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
# Characters of streamed assistant text kept for the incremental (Tier-1) stream guard.
STREAM_GUARD_WINDOW = 8192
# Matches a response `id` in either raw JSON or an SSE line, with or without a space.
SSE_ID_PATTERN = re.compile(rb'"id"\s*:\s*"([^"]{1,120})"')

# FastFlowLM answers *every* request with the same response id ("fastflowlm-chat-completion").
# LiteLLM uses that id as the primary key of its spend log, so the first NPU-served request was
# logged and every one after it failed the unique constraint and was dropped -- silently, while
# the responses themselves looked perfectly fine to the client. In this stack that means every
# image request (they all route to the NPU) vanished from the logs.
#
# So npuhalo stamps its own id on every response it returns, streamed or not. That is the fix
# for log completeness, and a unique id is more useful to a client than a constant one anyway.
#
# The lane tag is deliberate. LiteLLM's spend row records what the client *asked for*, and the row
# cannot be rebuilt inside the logging path. The id is the one field of that row this proxy
# controls, so it is the one that can say where the request actually ran -- `chatcmpl-npu-…` for the
# NPU, `chatcmpl-gpu-…` for the local GPU. A wildcard image request then reads honestly at a glance.
LANE_TAGS = {"gpu": "gpu", "npu": "npu"}


def new_response_id(lane: str = "") -> str:
    return f"chatcmpl-{LANE_TAGS.get(lane, 'local')}-{uuid.uuid4().hex[:20]}"


# Content-part types that carry pixels. Two client APIs spell the image part differently and both
# reach this proxy, so the set is what "this request has an image" means here.
IMAGE_PART_TYPES = frozenset({"image_url", "input_image", "image"})

# Where the per-request energy record is appended. The compose already mounts this directory as
# NPUHALO_ARTIFACTS_DIR, so the record is readable on the host without another mount.
DEFAULT_ENERGY_LOG = "/tmp/npuhalo-compressor-artifacts/energy.jsonl"


def request_has_image(messages: Any) -> bool:
    """True when any message in the request carries an image part."""
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in IMAGE_PART_TYPES:
                return True
    return False


def session_key(header_value: Optional[str], messages: Any) -> str:
    """A stable per-conversation key for one request.

    Prefers the client's ``X-Session-ID``; otherwise it fingerprints the *first* user message.
    An agent resends the whole conversation every turn, so the opening ask stays constant while
    the request would otherwise look like a brand-new session each time. It keys the correlation
    id when a request does not come through nginx, which is what keeps a conversation's routing
    records joinable across turns instead of each looking like a fresh, unrelated request.
    """
    if isinstance(header_value, str) and header_value.strip():
        return header_value.strip()
    for message in messages or []:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            text = " ".join(
                str(part.get("text") or "") for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        else:
            text = str(content or "")
        if text.strip():
            digest = hashlib.sha1(text.strip().encode("utf-8", "replace")).hexdigest()
            return f"auto-{digest[:16]}"
    return f"anon-{uuid.uuid4().hex[:8]}"

# Transport failures that justify retrying a request on the primary GPU backend.
FALLBACK_ERRORS = (aiohttp.ClientError, asyncio.TimeoutError)


class UpstreamUnusable(RuntimeError):
    """Upstream answered but produced no usable completion.

    Retryable on the GPU only for heuristic (router-gated) requests. An explicitly
    requested NPU model must never be silently answered by the primary GPU model.
    """


class NPUHaloProxy:
    def __init__(
        self,
        gpu_url: str = DEFAULT_GPU_URL,
        npu_url: str = DEFAULT_NPU_URL,
        gpu_model: str = DEFAULT_GPU_MODEL,
        npu_model: str = DEFAULT_NPU_MODEL,
        npu_model_names: Optional[List[str]] = None,
        guard_mode: str = "audit",  # "audit" (alert/telemetry only) | "block" (reject destructive steps)
        enable_router: bool = False,
        enable_compressor: bool = False,
        max_body_mb: int = DEFAULT_MAX_BODY_MB,
        image_target: str = LOCAL_NPU,
        energy_log: Optional[str] = None,
    ):
        self.gpu_url = gpu_url.rstrip("/")
        self.npu_url = npu_url.rstrip("/")
        self.gpu_model = gpu_model
        self.npu_model = npu_model
        # Client-facing model names served by the NPU. Defaults to DEFAULT_NPU_MODEL_NAMES
        # only when the caller passes None; an explicit empty list means "never route by name".
        self.npu_model_names = {
            str(n).strip().lower()
            for n in (DEFAULT_NPU_MODEL_NAMES if npu_model_names is None else npu_model_names)
            if str(n).strip()
        }
        self.guard_mode = guard_mode
        self.enable_router = enable_router
        self.enable_compressor = enable_compressor
        self.max_body_mb = max_body_mb
        # Which local backend serves an image-bearing request. The vision model is the only one
        # here that can read pixels, so this switches between the NPU and the GPU, not on/off.
        self.image_target = image_target
        # Per-request package energy. The meter samples the socket power sensor while a request is
        # in flight and integrates it; a missing sensor degrades the record, never the request.
        self.energy = EnergyMeter()
        self.energy_log = energy_log or os.environ.get("NPUHALO_ENERGY_LOG") or DEFAULT_ENERGY_LOG
        self._energy_lock = asyncio.Lock()


        if "minicpm" in self.npu_model.lower():
            logger.warning(
                "MiniCPM5-2B (42 layers) may trip ERT_CMD_STATE_TIMEOUT in FastFlowLM during decode. "
                "Recommended: qwen3.5:0.8b or qwen3:1.7b. See ROCm/FastFlowLM#712."
            )
        if self.enable_router:
            logger.warning(
                "Router gating is ENABLED: prompts that did not explicitly name an NPU model may be "
                "served by the small NPU model, which can be a silent quality downgrade."
            )

        self.watchdog = WatchdogAnalyzer(
            npu_url=f"{self.npu_url}/v1/chat/completions",
            npu_model=self.npu_model,
            enable_npu_eval=True,
        )
        self.router = HybridNPURouter(url=self.npu_url, model=self.npu_model) if self.enable_router else None
        self.compressor = DualCompressorSidecar(npu_url=f"{self.npu_url}/v1/chat/completions", npu_model=self.npu_model) if self.enable_compressor else None
        self.session: Optional[aiohttp.ClientSession] = None
        self._interceptions_count = 0
        self._backend_counts = {"gpu": 0, "npu": 0}

    async def start(self):
        connector = aiohttp.TCPConnector(limit=100, force_close=False)
        self.session = aiohttp.ClientSession(connector=connector)
        if self.router:
            await self.router.start()
        await self.energy.start()
        logger.info(
            f"NPUHalo Proxy initialized | GPU={self.gpu_url} ({self.gpu_model}) | "
            f"NPU={self.npu_url} ({self.npu_model}) for models {sorted(self.npu_model_names) or ['<none>']} | "
            f"GuardMode={self.guard_mode} | ImageTarget={self.image_target} | "
            f"Energy={self.energy.sensor_path or 'sensor unavailable'}"
        )

    async def stop(self):
        await self.energy.stop()
        if self.session:
            await self.session.close()
        if self.router:
            await self.router.close()

    # ---- backend selection ---------------------------------------------------
    def _is_npu_model(self, name: str) -> bool:
        return str(name).strip().lower() in self.npu_model_names

    def _gpu_selection(self, body: dict, reason: str = "primary GPU model") -> Dict[str, Any]:
        return {
            "backend": LOCAL_GPU,
            "target_url": f"{self.gpu_url}/v1/chat/completions",
            "upstream_model": self.gpu_model,
            "payload": body,
            "reason": reason,
            "allow_fallback": False,  # nothing left to fall back to
        }

    def _npu_selection(self, body: dict, reason: str, allow_fallback: bool) -> Dict[str, Any]:
        return {
            "backend": LOCAL_NPU,
            "target_url": f"{self.npu_url}/v1/chat/completions",
            "upstream_model": self.npu_model,
            "payload": {**body, "model": self.npu_model},
            "reason": reason,
            "allow_fallback": allow_fallback,
        }

    async def _heuristic_selection(self, body: dict, user_prompt: Any) -> Dict[str, Any]:
        """Pre-routing behaviour: optional router gating, else the primary GPU."""
        if self.enable_router and self.router:
            prompt_text = user_prompt if isinstance(user_prompt, str) else ""
            dec = await self.router.classify(prompt_text)
            if dec.get("route") == "npu":
                # Heuristic power optimisation only, so a failure may be retried on the GPU.
                return self._npu_selection(body, f"router gating: {dec.get('reason')}", True)
        return self._gpu_selection(body)

    async def _select_backend(
        self, body: dict, user_prompt: Any, session_id: str = ""
    ) -> Dict[str, Any]:
        """Explicit model -> backend mapping, then image routing, then the optional router.

        The ordering is deliberate. An explicit NPU model name is served by the NPU or fails
        loudly. An image-bearing request goes to the configured image lane, because the vision
        model is the only backend here that can read pixels.
        """
        requested = str(body.get("model") or "")
        if self._is_npu_model(requested):
            # The client asked for this model by name: it must be served by the NPU or
            # fail loudly. Never silently substitute the GPU model.
            return self._npu_selection(body, f"explicit NPU model '{requested}'", False)

        if request_has_image(body.get("messages") or []):
            # Retrying an image on a text-only model would fail again and report the *text*
            # model's error ("image input is not supported - you may need to provide the
            # mmproj"), which reads like a misconfiguration rather than "this image could not be
            # served". So the image lane is chosen here and the request is not retried elsewhere.
            if self.image_target == LOCAL_NPU:
                return self._npu_selection(
                    body, "image request served by the local vision model", False)
            return self._gpu_selection(body, "image request served by the local GPU model")

        return await self._heuristic_selection(body, user_prompt)

    # ---- endpoints -----------------------------------------------------------
    async def handle_models(self, request: web.Request) -> web.Response:
        """OpenAI-compatible /v1/models endpoint."""
        models = [
            {"id": self.gpu_model, "object": "model", "owned_by": "amd-rdna3.5-igpu"},
        ]
        models += [
            {"id": name, "object": "model", "owned_by": f"amd-xdna2-npu ({self.npu_model})"}
            for name in sorted(self.npu_model_names)
        ]
        models += [
            {"id": f"{self.npu_model}-watchdog", "object": "model", "owned_by": "amd-xdna2-npu"},
            {"id": "npuhalo-hybrid", "object": "model", "owned_by": "npuhalo-coprocessor"},
        ]
        return web.json_response({"object": "list", "data": models})

    async def handle_watchdog_status(self, request: web.Request) -> web.Response:
        """Watchdog telemetry and health status endpoint."""
        stats = self.watchdog.get_stats()
        stats["interceptions_blocked"] = self._interceptions_count
        stats["guard_mode"] = self.guard_mode
        stats["npu_target"] = f"{self.npu_url} ({self.npu_model})"
        stats["gpu_target"] = f"{self.gpu_url} ({self.gpu_model})"
        stats["npu_model_names"] = sorted(self.npu_model_names)
        stats["router_enabled"] = self.enable_router
        stats["compressor_enabled"] = self.enable_compressor
        stats["requests_by_backend"] = dict(self._backend_counts)
        return web.json_response(stats)

    async def handle_watchdog_clear(self, request: web.Request) -> web.Response:
        """Reset telemetry and state history."""
        self.watchdog._history.clear()
        self.watchdog._verdict_history.clear()
        self._interceptions_count = 0
        self._backend_counts = {"gpu": 0, "npu": 0}
        return web.json_response({"status": "cleared"})

    async def handle_chat_completions(self, request: web.Request) -> web.StreamResponse:
        """Proxy /v1/chat/completions with zero streaming delay and async NPU audit."""
        try:
            body = await request.json()
        except web.HTTPRequestEntityTooLarge:
            # aiohttp rejects bodies over client_max_size while .json() reads them.
            # Without this clause the generic handler below would report a size
            # problem as malformed JSON, which is actively misleading.
            limit = self.max_body_mb
            logger.error(
                f"Request body exceeds the {limit} MiB limit (images are base64-encoded, ~1.37x their size)"
            )
            return web.json_response(
                {"error": f"request body too large: limit is {limit} MiB. Images are sent "
                          f"base64-encoded and inflate ~1.37x; raise --max-body-mb for larger ones."},
                status=413,
            )
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)

        stream = body.get("stream", False)
        messages = body.get("messages", [])
        # Stable per-conversation key: the client's header when it sends one, otherwise a
        # fingerprint of the opening user message, so the correlation id stays stable across a
        # conversation's turns and its routing records remain joinable.
        session_id = session_key(request.headers.get("X-Session-ID"), messages)
        user_prompt = messages[-1].get("content", "") if messages else ""

        # In-Flight Context Compression: If enabled and prompt > 16k chars, distill on NPU first
        if self.enable_compressor and self.compressor and user_prompt and len(user_prompt) >= 16000:
            logger.info(f"Compressor: Distilling input prompt ({len(user_prompt)} chars) on NPU before GPU prefill")
            comp_res = self.compressor.compress_input_context(user_prompt)
            if comp_res.action == "compressed":
                messages[-1]["content"] = comp_res.inserted_text
                body["messages"] = messages
                logger.info(f"Compressor: Prompt compressed from {comp_res.raw_tokens} to {comp_res.compressed_tokens} tokens")

        selection = await self._select_backend(body, user_prompt, session_id)
        backend = selection["backend"]
        payload = selection["payload"]
        headers = {"Content-Type": "application/json", **selection.get("auth_headers", {})}
        selection["requested_model"] = body.get("model")

        # Ask the upstream for token usage even when the client did not. Both lanes report usage on
        # a streamed response only when asked, and without the token count the energy record cannot
        # be an energy-per-token record. A client that did not ask does not see it: the streaming
        # path drops the usage-only chunk before forwarding it.
        wants_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        selection["wants_usage"] = wants_usage
        if stream:
            options = dict(payload.get("stream_options") or {})
            options["include_usage"] = True
            payload["stream_options"] = options

        # The routing record, and the authoritative answer to "where did this actually run".
        #
        # `requested_model` is what the client asked for; `upstream_model` is what served it.
        # Stating both on one line is the whole point: an image asked for as deepseek-v4-flash
        # is served by qwen3.5:2b on the NPU, and any log that records only the requested name
        # misreports where the tokens were produced. LiteLLM's own spend-log row cannot show
        # this -- it keeps the requested name and re-stamps it into the response -- so this
        # line is the record of truth, and `correlation` ties it to that row (LiteLLM forwards
        # its call id in the request body when it is available; otherwise the session key is
        # the join, and its absence is stated rather than guessed).
        # The correlation key. nginx mints one per request ($request_id) and sends it two
        # ways: as spend-log metadata, which LiteLLM writes into the row, and as a plain
        # header, which LiteLLM forwards to npuhalo (it forwards `x-*` when
        # forward_client_headers_to_llm_api is on). npuhalo prints it on its routing record,
        # so the row and the routing record join on an exact id rather than on a timestamp.
        correlation = (request.headers.get("X-NPUHalo-Correlation")
                       or body.get("litellm_call_id") or body.get("litellm_trace_id")
                       or request.headers.get("X-Litellm-Call-Id")
                       or request.headers.get("X-Litellm-Trace-Id") or "")
        logger.info(
            "route backend=%s upstream=%s upstream_model=%s requested_model=%s correlation=%s reason=%s",
            backend,
            selection["target_url"].split("://")[-1].split("/")[0],
            selection.get("upstream_model"),
            body.get("model"),
            correlation or f"session:{session_id}",
            selection["reason"],
        )

        self._backend_counts[backend] = self._backend_counts.get(backend, 0) + 1

        runner = self._handle_streaming_completion if stream else self._handle_non_streaming_completion
        args = (request, selection, headers, session_id, user_prompt)

        # Meter the request across both lanes and both paths. The meter's own finally runs when
        # this block exits, so the measurement is complete before the record is written below --
        # and for a streamed response this block is not exited until every chunk has been written.
        served = selection
        served_backend = backend
        response: Optional[web.StreamResponse] = None
        async with self.energy.measure() as measurement:
            try:
                response = await runner(*args)
            except (FALLBACK_ERRORS + (UpstreamUnusable,)) as e:
                logger.error(f"Backend '{backend}' unusable: {type(e).__name__}: {e}")
                if backend == LOCAL_NPU and selection.get("allow_fallback"):
                    # Router gating is a power optimisation, so a failed NPU attempt may be
                    # retried on the primary GPU model.
                    logger.warning(f"Falling back to {self.gpu_model}")
                    self._backend_counts["gpu"] = self._backend_counts.get("gpu", 0) + 1
                    local = self._gpu_selection(payload)
                    local["extra_headers"] = dict(selection.get("extra_headers", {}))
                    local["wants_usage"] = wants_usage
                    local["requested_model"] = selection["requested_model"]
                    served, served_backend = local, LOCAL_GPU
                    try:
                        response = await runner(
                            request, local, {"Content-Type": "application/json"},
                            session_id, user_prompt,
                        )
                    except FALLBACK_ERRORS as e2:
                        logger.error(f"GPU fallback also failed: {type(e2).__name__}: {e2}")
                        response = web.json_response(
                            {"error": f"upstream unavailable: {e2}"}, status=502)
                else:
                    # Explicitly requested NPU model (or a GPU failure): fail loudly rather than
                    # silently answering with a different model.
                    response = web.json_response(
                        {"error": f"{backend} backend unavailable: {e}", "backend": backend},
                        status=502,
                    )

        await self._record_energy(
            measurement, served, served_backend, correlation, session_id, stream, user_prompt
        )
        return response

    # ---- request handling ----------------------------------------------------
    @staticmethod
    def _rewrite_chunk_ids(chunk: bytes, response_id: str, seen: set) -> bytes:
        """Replace the upstream's response id inside a streamed chunk.

        FastFlowLM reuses one id for every response and LiteLLM keys its spend log on that id,
        so all but the first NPU-served request were dropped by the unique constraint. The id
        is rewritten as the stream passes through, which is what makes those requests loggable.
        Substitution is done on bytes rather than by re-parsing JSON, to keep it off the
        latency path.
        """
        if not chunk:
            return chunk
        for match in SSE_ID_PATTERN.finditer(chunk):
            seen.add(match.group(1))
        replacement = response_id.encode()
        for upstream in seen:
            if upstream == replacement:
                continue
            chunk = chunk.replace(b'"id": "' + upstream + b'"', b'"id": "' + replacement + b'"')
            chunk = chunk.replace(b'"id":"' + upstream + b'"', b'"id":"' + replacement + b'"')
        return chunk

    @staticmethod
    def _route_headers(selection: Dict[str, Any]) -> Dict[str, str]:
        """Upstream routing, echoed to the caller.

        `X-NPUHalo-Upstream-Model` is what a LiteLLM callback needs to record where a request
        actually ran: the spend-log row records the *requested* model, and the response body's
        `model` is re-stamped to the requested name, so the header is the only place the serving
        model is visible to the gateway.
        """
        backend = selection["backend"]
        if backend == LOCAL_NPU:
            route = "npu-direct"
        else:
            route = "gpu"
        headers = {"X-NPUHalo-Route": route, **selection.get("extra_headers", {})}
        upstream_model = selection.get("upstream_model")
        if upstream_model:
            headers["X-NPUHalo-Upstream-Model"] = str(upstream_model)
        return headers

    # ---- energy record -------------------------------------------------------
    @staticmethod
    def _is_usage_only_chunk(chunk: bytes) -> bool:
        """True for the final SSE chunk that carries usage and no choices.

        Cheap substring test on purpose: this runs on the streaming path, where the code
        deliberately avoids re-parsing JSON.
        """
        if b'"usage"' not in chunk:
            return False
        return b'"choices":[]' in chunk or b'"choices": []' in chunk

    @staticmethod
    def _extract_usage(text: str) -> Optional[dict]:
        """The token usage object from an SSE body or a JSON completion, if the upstream sent one.

        Both lanes report usage on a stream only when asked, which is why the request asks; this is
        the read side of that. FastFlowLM's block also carries `prefill_duration_ttft` and
        `decoding_duration`, so the prefill/decode split comes free on the NPU lane.
        """
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line.startswith("data:") or '"usage"' not in line:
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                usage = json.loads(payload).get("usage")
            except ValueError:
                continue
            if isinstance(usage, dict):
                return usage
        if '"usage"' in text:
            try:
                usage = json.loads(text).get("usage")
            except ValueError:
                return None
            if isinstance(usage, dict):
                return usage
        return None

    def classify_tier(self, text: str) -> str:
        """A coarse bucket for this request, recorded beside the energy.

        LiteLLM's complexity router decides the real tier, but it does not persist it: the
        `router_metadata` and `routing_decision` slots it stamps into the spend log are JSON null
        on every row, and the tier lives only in the router's in-memory decision (see
        docs/ENERGY.md). So this bucket is the shim's own -- the same kind of keyword test the
        router's heuristic classifier uses -- and every record says `tier_source=shim-heuristic`
        so nobody mistakes it for the router's answer.
        """
        if not isinstance(text, str) or not text.strip():
            return "empty"
        return "simple" if self._is_trivial_query(text) else "complex"

    async def _record_energy(
        self,
        measurement,
        selection: Dict[str, Any],
        backend: str,
        correlation: str,
        session_id: str,
        stream: bool,
        user_prompt: Any,
    ) -> None:
        """Append the per-request energy record, and put its numbers on the routing line.

        Two destinations on purpose: the log line is what a human reads, and the JSONL is what the
        report tool and the spend-log reconciler read. Neither is on the response path.
        """
        usage = selection.get("usage") if isinstance(selection, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        prefill_s = usage.get("prefill_duration_ttft")
        decode_s = usage.get("decoding_duration")
        record = {
            "ts": time.time(),
            "correlation": correlation or f"session:{session_id}",
            "lane": backend,
            "requested_model": selection.get("requested_model"),
            "upstream_model": selection.get("upstream_model"),
            "reason": selection.get("reason"),
            "stream": bool(stream),
            "tier": self.classify_tier(user_prompt),
            "tier_source": "shim-heuristic",
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "prefill_ms": round(prefill_s * 1000, 2) if isinstance(prefill_s, (int, float)) else None,
            "decode_ms": round(decode_s * 1000, 2) if isinstance(decode_s, (int, float)) else None,
            **measurement.as_dict(),
        }
        logger.info(
            "energy lane=%s tier=%s joules=%s marginal_joules=%s idle_w=%s duration_ms=%s "
            "tokens=%s+%s samples=%s concurrent=%s",
            record["lane"], record["tier"], record["energy_j"], record["energy_marginal_j"],
            record["energy_idle_w"], round(measurement.duration_s * 1000, 1),
            record["prompt_tokens"], record["completion_tokens"], record["energy_samples"],
            record["energy_concurrent"],
        )
        try:
            async with self._energy_lock:
                with open(self.energy_log, "a") as handle:
                    handle.write(json.dumps(record, default=str) + "\n")
        except OSError as e:
            # A log that cannot be written must not fail a request that has already been answered.
            logger.warning(f"energy: could not append to {self.energy_log}: {e}")

    async def _handle_streaming_completion(
        self,
        request: web.Request,
        selection: Dict[str, Any],
        headers: dict,
        session_id: str,
        user_prompt: Any,
    ) -> web.StreamResponse:
        """Stream upstream chunks to the client, guarding incrementally.

        The upstream connection is established *before* any response is prepared, so a
        transport failure can still be retried on the other backend by the caller.
        """
        target_url = selection["target_url"]
        payload = selection["payload"]
        route_headers = self._route_headers(selection)
        response_id = new_response_id(selection["backend"])
        seen_ids: set = set()

        async with self.session.post(target_url, json=payload, headers=headers) as upstream_resp:
            if upstream_resp.status != 200:
                err_txt = await upstream_resp.text()
                logger.error(f"Upstream {selection['backend']} returned {upstream_resp.status}: {err_txt[:200]}")
                if selection.get("allow_fallback"):
                    raise UpstreamUnusable(f"upstream status {upstream_resp.status}: {err_txt[:200]}")
                return web.json_response(
                    {"error": err_txt, "backend": selection["backend"]},
                    status=502,
                    headers=route_headers,
                )

            is_sse = "text/event-stream" in upstream_resp.headers.get("Content-Type", "")
            wrapped = None
            if not is_sse:
                # Upstream answered with a plain JSON completion despite stream=true.
                data = await upstream_resp.json()
                if not isinstance(data, dict) or not data.get("choices"):
                    logger.error(f"Upstream {selection['backend']} returned no choices: {str(data)[:200]}")
                    if selection.get("allow_fallback"):
                        raise UpstreamUnusable(f"upstream returned no completion: {str(data)[:200]}")
                    return web.json_response(
                        {"error": "upstream returned no completion", "backend": selection["backend"],
                         "detail": data},
                        status=502,
                        headers=route_headers,
                    )
                data["id"] = response_id
                selection["usage"] = data.get("usage")
                wrapped = self._json_to_sse(data, fallback_model=payload.get("model", "")).encode("utf-8")

            response = web.StreamResponse(status=200, headers={**SSE_HEADERS, **route_headers})
            await response.prepare(request)

            accumulated_chunks = []
            streamed_text = ""
            pending_sse = ""
            blocked_verdict: Optional[AnomalyVerdict] = None
            t0 = time.perf_counter()

            try:
                if wrapped is not None:
                    await response.write(wrapped)
                    accumulated_chunks.append(wrapped)
                else:
                    async for chunk in upstream_resp.content.iter_any():
                        if not chunk:
                            continue
                        chunk = self._rewrite_chunk_ids(chunk, response_id, seen_ids)
                        accumulated_chunks.append(chunk)

                        # Incremental deterministic guard: only armed in "block" mode, scans
                        # complete SSE events as they arrive, so safe tokens keep zero delay.
                        if self.guard_mode == "block":
                            pending_sse += chunk.decode("utf-8", errors="replace")
                            while "\n\n" in pending_sse:
                                event, pending_sse = pending_sse.split("\n\n", 1)
                                streamed_text += self._sse_delta_text(event)
                            blocked_verdict = self._stream_guard_verdict(streamed_text)
                            if blocked_verdict is not None:
                                break

                        # The usage-only chunk is what we asked the upstream for on the client's
                        # behalf; a client that did not ask for it must not see it.
                        if not selection.get("wants_usage") and self._is_usage_only_chunk(chunk):
                            continue

                        await response.write(chunk)

                if blocked_verdict is not None:
                    self.watchdog.record_verdict(blocked_verdict)
                    self._interceptions_count += 1
                    logger.warning(
                        f"INTERCEPTED dangerous agent action mid-stream: {blocked_verdict.reason} "
                        f"(client stream cut after {len(accumulated_chunks)} chunks)"
                    )
                    model = payload.get("model", "") or self.gpu_model
                    await response.write(self._interception_sse(model, blocked_verdict).encode("utf-8"))
                    await response.write_eof()
                    return response

                # Reconstruct full response text in background
                full_raw = b"".join(accumulated_chunks).decode("utf-8", errors="replace")
                generated_text = self._extract_text_from_sse(full_raw)
                stream_ms = (time.perf_counter() - t0) * 1000
                if selection.get("usage") is None:
                    selection["usage"] = self._extract_usage(full_raw)

                # Schedule Out-of-Loop NPU Audit Task (0ms delay to client stream)
                prompt_text = user_prompt if isinstance(user_prompt, str) else ""
                asyncio.create_task(
                    self._async_audit_background(session_id, prompt_text, generated_text, stream_ms)
                )
            except ConnectionResetError as e:
                # The client hung up (or a proxy in front of us did) mid-stream: there is
                # nobody left to write an error to, so just abandon the response quietly.
                logger.info(f"Client disconnected mid-stream ({type(e).__name__}); abandoning response")
                return response
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                try:
                    await response.write(f"data: {json.dumps({'error': str(e)})}\n\n".encode("utf-8"))
                except Exception:
                    pass

        try:
            await response.write_eof()
        except Exception:
            pass
        return response

    async def _handle_non_streaming_completion(
        self,
        request: web.Request,
        selection: Dict[str, Any],
        headers: dict,
        session_id: str,
        user_prompt: Any,
    ) -> web.Response:
        """Handle non-streaming completion with guardrail interception."""
        async with self.session.post(selection["target_url"], json=selection["payload"], headers=headers) as upstream_resp:
            data = await upstream_resp.json()
            route_headers = self._route_headers(selection)
            # A unique, lane-tagged id per response: the upstream's may be a constant, and
            # LiteLLM keys its spend log on it (see new_response_id).
            if isinstance(data, dict):
                data["id"] = new_response_id(selection["backend"])
            if upstream_resp.status != 200:
                return web.json_response(data, status=upstream_resp.status, headers=route_headers)

            choices = data.get("choices") if isinstance(data, dict) else None
            if not choices:
                # FastFlowLM answers HTTP 200 with {"error": ...} when the NPU stack is
                # unhealthy; surface that instead of a fake empty completion.
                logger.error(f"Upstream {selection['backend']} returned no choices: {str(data)[:200]}")
                if selection.get("allow_fallback"):
                    raise UpstreamUnusable(f"upstream returned no completion: {str(data)[:200]}")
                return web.json_response(
                    {"error": "upstream returned no completion", "backend": selection["backend"],
                     "detail": data},
                    status=502,
                    headers=route_headers,
                )

            generated_text = (choices[0].get("message") or {}).get("content", "") or ""
            selection["usage"] = data.get("usage") if isinstance(data, dict) else None

            prompt_text = user_prompt if isinstance(user_prompt, str) else ""
            verdict = await self.watchdog.audit_step(session_id, prompt_text, generated_text, session=self.session)

            resp_headers = {
                **route_headers,
                "X-NPU-Watchdog-Risk": verdict.risk_level,
                "X-NPU-Watchdog-Category": verdict.category,
            }

            if verdict.is_anomaly and self.guard_mode == "block" and verdict.risk_level in ("HIGH", "CRITICAL"):
                self._interceptions_count += 1
                logger.warning(f"INTERCEPTED dangerous agent action: {verdict.reason}")
                choices[0]["message"]["content"] = self._interception_message(verdict)
                resp_headers["X-NPU-Watchdog-Action"] = "BLOCKED"

            return web.json_response(data, headers=resp_headers)

    def _is_trivial_query(self, text: str) -> bool:
        """Fast heuristic for routing simple queries to NPU."""
        text_lower = text.lower().strip()
        if len(text.split()) <= 8 and not any(kw in text_lower for kw in ["code", "def ", "class ", "bash", "run ", "test"]):
            if any(text_lower.startswith(q) for q in ["what is", "who is", "who wrote", "calculate", "hello", "hi", "help"]):
                return True
        return False

    # ---- SSE helpers ---------------------------------------------------------
    def _extract_text_from_sse(self, sse_data: str) -> str:
        """Extract combined text content from SSE chunks."""
        text_pieces = []
        for line in sse_data.splitlines():
            line = line.strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    payload = json.loads(line[6:])
                    delta = payload.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        text_pieces.append(content)
                except Exception:
                    continue
        return "".join(text_pieces)

    def _sse_delta_text(self, event_text: str) -> str:
        """Extract assistant content deltas from one complete SSE event block."""
        pieces = []
        for line in event_text.splitlines():
            line = line.strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                payload = json.loads(line[6:])
                delta = payload.get("choices", [{}])[0].get("delta", {}) or {}
                content = delta.get("content", "")
                if content:
                    pieces.append(content)
            except Exception:
                continue
        return "".join(pieces)

    def _stream_guard_verdict(self, streamed_text: str) -> Optional[AnomalyVerdict]:
        """Incremental Tier-1 guard over a bounded tail window of the streamed text."""
        window = streamed_text[-STREAM_GUARD_WINDOW:]
        verdict = self.watchdog.check_regex_rules(window)
        if verdict and verdict.risk_level in ("HIGH", "CRITICAL"):
            return verdict
        return None

    @staticmethod
    def _interception_message(verdict: AnomalyVerdict) -> str:
        return (
            "[NPUHALO WATCHDOG INTERCEPTION]\n"
            f"Action blocked by NPU safety guardrail: {verdict.reason}.\n"
            f"Please reformulate a safe command."
        )

    def _json_to_sse(self, data: dict, fallback_model: str = "") -> str:
        """Render a non-streaming OpenAI chat completion as an SSE event stream."""
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or message.get("reasoning_content") or ""
        model = data.get("model") or fallback_model or self.npu_model
        cid = data.get("id") or f"chatcmpl-npuhalo-{uuid.uuid4().hex[:12]}"
        created = data.get("created") or int(time.time())
        events = [
            {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
             "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}]},
            {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
             "choices": [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason") or "stop"}]},
        ]
        return "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"

    def _interception_sse(self, model: str, verdict: AnomalyVerdict) -> str:
        """SSE replacement payload emitted when a live stream is cut by the guard."""
        payload = {
            "model": model or self.gpu_model,
            "choices": [{"message": {"content": self._interception_message(verdict)}, "finish_reason": "stop"}],
        }
        return self._json_to_sse(payload, fallback_model=self.gpu_model)

    async def _async_audit_background(self, session_id: str, prompt: str, text: str, stream_ms: float):
        """Asynchronously executes on NPU without delaying the client."""
        t0 = time.perf_counter()
        verdict = await self.watchdog.audit_step(session_id, prompt, text, session=self.session)
        audit_ms = (time.perf_counter() - t0) * 1000

        if verdict.is_anomaly:
            logger.warning(
                f"[WATCHDOG ALERT] Risk={verdict.risk_level} | Cat={verdict.category} | "
                f"Reason: {verdict.reason} | Audit Latency: {audit_ms:.1f}ms"
            )
        else:
            logger.debug(f"[WATCHDOG OK] Step verified safe in {audit_ms:.1f}ms")

def create_app(proxy: NPUHaloProxy) -> web.Application:
    app = web.Application(client_max_size=proxy.max_body_mb * 1024 * 1024)
    app.router.add_get("/v1/models", proxy.handle_models)
    app.router.add_post("/v1/chat/completions", proxy.handle_chat_completions)
    app.router.add_get("/v1/watchdog/status", proxy.handle_watchdog_status)
    app.router.add_post("/v1/watchdog/clear", proxy.handle_watchdog_clear)
    return app

def main():
    parser = argparse.ArgumentParser(description="NPUHalo Smart Guardrail Reverse Proxy")
    parser.add_argument("--port", type=int, default=DEFAULT_PROXY_PORT, help="Port to listen on (default: 8000)")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--gpu-url", default=DEFAULT_GPU_URL, help="Primary GPU server URL")
    parser.add_argument("--npu-url", default=DEFAULT_NPU_URL, help="NPU FastFlowLM server URL")
    parser.add_argument("--npu-model", default=DEFAULT_NPU_MODEL, help="NPU model tag sent upstream (default: qwen3.5:0.8b)")
    parser.add_argument(
        "--npu-model-name",
        action="append",
        default=None,
        metavar="NAME",
        help="Client-facing model name served by the NPU (repeatable). "
             f"Default: {', '.join(DEFAULT_NPU_MODEL_NAMES)}. Pass an empty string to disable name-based routing.",
    )
    parser.add_argument("--gpu-model", default=DEFAULT_GPU_MODEL, help="Primary GPU model tag")
    parser.add_argument("--guard-mode", choices=["audit", "block"], default="audit", help="Guard mode: audit or block")
    parser.add_argument("--enable-router", action="store_true", help="Additionally route unrouted 'trivial' queries to the NPU (can downgrade quality)")
    parser.add_argument("--enable-compressor", action="store_true", help="Enable in-flight NPU prompt compression for long inputs (>16k chars)")
    parser.add_argument(
        "--energy-log", default=None,
        help="Append the per-request energy record here "
             f"(default: {DEFAULT_ENERGY_LOG}, or NPUHALO_ENERGY_LOG)",
    )
    parser.add_argument(
        "--max-body-mb", type=int, default=DEFAULT_MAX_BODY_MB, metavar="MB",
        help=f"Maximum request body size in MiB (default: {DEFAULT_MAX_BODY_MB}). Images are base64-encoded and inflate ~1.37x",
    )
    parser.add_argument(
        "--private-image-target", choices=["npu", "gpu"], default="npu",
        help="Local backend for image-bearing requests (default: npu, the vision-capable "
             "local model).",
    )
    args = parser.parse_args()

    npu_model_names: Optional[List[str]]
    if args.npu_model_name is None:
        npu_model_names = None  # library default
    else:
        npu_model_names = [n for n in args.npu_model_name if n.strip()]

    proxy = NPUHaloProxy(
        gpu_url=args.gpu_url,
        npu_url=args.npu_url,
        gpu_model=args.gpu_model,
        npu_model=args.npu_model,
        npu_model_names=npu_model_names,
        guard_mode=args.guard_mode,
        enable_router=args.enable_router,
        enable_compressor=args.enable_compressor,
        max_body_mb=args.max_body_mb,
        image_target=args.private_image_target,
        energy_log=args.energy_log,
    )

    app = create_app(proxy)
    
    async def on_startup(app):
        await proxy.start()

    async def on_cleanup(app):
        await proxy.stop()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    print("\n========================================================")
    print("  NPUHALO GUARDRAIL PROXY FOR STRIX HALO")
    print(f"  Listening on : http://{args.host}:{args.port}/v1")
    print(f"  GPU backend  : {args.gpu_url} ({args.gpu_model})")
    print(f"  NPU backend  : {args.npu_url} ({args.npu_model})")
    print(f"  NPU models   : {', '.join(sorted(proxy.npu_model_names)) or '<none>'}")
    print(f"  Guard Mode   : {args.guard_mode.upper()}")
    print(f"  Router Gating: {'ENABLED' if args.enable_router else 'DISABLED'}")
    print(f"  Compressor   : {'ENABLED' if args.enable_compressor else 'DISABLED'}")
    print(f"  Max body     : {args.max_body_mb} MiB")
    print(f"  Image lane   : {args.private_image_target.upper()}")
    print(f"  Energy       : {proxy.energy.sensor_path or 'sensor unavailable'}"
          f" (sampled every {proxy.energy.interval_s}s)")
    print(f"  Energy log   : {proxy.energy_log}")
    print("========================================================\n")

    web.run_app(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()

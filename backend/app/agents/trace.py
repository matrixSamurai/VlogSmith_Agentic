"""Live tracing of the ADK workflow: graph structure, hand-offs, routes, state writes and LLM calls.

`TracePlugin` is a standard ADK `BasePlugin` registered on the Runner. It sees every event the workflow
engine emits (`on_event_callback`) and every model call (`before/after_model_callback`). From the event
stream it reconstructs *hand-offs*: when a graph node finishes, ADK passes its output as `node_input` to
the successor nodes selected by the edges (and by the emitted route, for routing nodes). Each record is
pushed to the job's event stream as a `trace` event, which the UI renders as a live graph.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable

from google.adk.plugins.base_plugin import BasePlugin
from google.adk.workflow import JoinNode, Workflow

log = logging.getLogger(__name__)

EmitFn = Callable[[dict], Awaitable[None]]
PREVIEW_CHARS = 1600


def _preview(value: Any, limit: int = PREVIEW_CHARS) -> str:
    if value is None:
        return ""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if hasattr(value, "parts"):  # google.genai Content
        value = "\n".join(p.text for p in (value.parts or []) if getattr(p, "text", None))
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str, indent=1)
    return text if len(text) <= limit else text[:limit] + f"… (+{len(text) - limit} chars)"


def _shape(value: Any) -> str:
    """Short type description of a payload, e.g. list[4]·dict{index,feedback}."""
    if isinstance(value, list):
        inner = _shape(value[0]) if value else "empty"
        return f"list[{len(value)}] of {inner}"
    if isinstance(value, dict):
        keys = list(value)[:6]
        return "dict{" + ", ".join(keys) + ("…" if len(value) > 6 else "") + "}"
    if value is None:
        return "None"
    return type(value).__name__


def _segments(path: str) -> list[tuple[str, str]]:
    out = []
    for seg in (path or "").split("/"):
        name, _, run = seg.partition("@")
        out.append((name, run))
    return out


class TracePlugin(BasePlugin):
    def __init__(self, workflow: Workflow, emit: EmitFn, node_agents: dict[str, list[dict]]):
        super().__init__(name="vlogsmith_trace")
        self._wf = workflow
        self._emit = emit
        self._node_agents = node_agents
        graph = workflow.graph
        self._edges = [(e.from_node.name, e.to_node.name, e.route) for e in graph.edges] if graph else []
        self._routing = {f for f, _, r in self._edges if r is not None}
        self._pending: dict[str, Any] = {}  # routing node -> output awaiting its route
        self._started = False

    # ------------------------------------------------------------------ structure
    def graph_record(self) -> dict:
        from google.adk.workflow._parallel_worker import _ParallelWorker

        nodes = []
        for n in self._wf.graph.nodes if self._wf.graph else []:
            kind = "start" if n.name == "__START__" else "join" if isinstance(n, JoinNode) else \
                "parallel" if isinstance(n, _ParallelWorker) else "function"
            nodes.append({"id": n.name, "kind": kind, "agents": self._node_agents.get(n.name, []),
                          "routes": sorted({str(r) for f, _, r in self._edges if f == n.name and r is not None})})
        edges = [{"from": f, "to": t, "route": None if r is None else str(r)} for f, t, r in self._edges]
        return {"type": "graph", "workflow": self._wf.name, "nodes": nodes, "edges": edges}

    async def _send(self, record: dict) -> None:
        try:
            await self._emit(record)
        except Exception as exc:  # tracing must never break the pipeline
            log.debug("trace emit failed: %s", exc)

    def _successors(self, node: str, route: Any) -> list[str]:
        if route is None:
            return [t for f, t, r in self._edges if f == node and r is None]
        routes = route if isinstance(route, list) else [route]
        return [t for f, t, r in self._edges if f == node and r is not None and (r in routes or str(r) in map(str, routes))]

    async def _handoff(self, node: str, payload: Any, route: Any = None) -> None:
        targets = self._successors(node, route)
        if not targets:
            return
        await self._send({"type": "handoff", "from": node, "to": targets, "route": None if route is None else str(route),
                          "shape": _shape(payload), "payload": _preview(payload)})

    # ------------------------------------------------------------------ ADK hooks
    async def on_user_message_callback(self, *, invocation_context, user_message):
        if not self._started:
            self._started = True
            await self._send(self.graph_record())
            await self._handoff("__START__", user_message)
        return None

    async def before_model_callback(self, *, callback_context, llm_request):
        segs = _segments(getattr(callback_context, "node_path", "") or "")
        node = segs[1][0] if len(segs) > 1 else callback_context.agent_name
        instance = segs[2][1] if len(segs) > 3 and segs[2][0] == node else None
        contents = llm_request.contents or []
        last = contents[-1] if contents else None
        text = "\n".join(p.text for p in (last.parts if last else []) or [] if getattr(p, "text", None))
        images = sum(1 for c in contents for p in (c.parts or []) if getattr(p, "inline_data", None))
        schema = getattr(getattr(llm_request, "config", None), "response_schema", None)
        await self._send({
            "type": "llm_start", "node": node, "instance": instance, "agent": callback_context.agent_name,
            "model": llm_request.model, "schema": getattr(schema, "__name__", None), "images": images,
            "prompt_chars": len(text), "prompt": _preview(text, 1200),
        })
        return None

    async def after_model_callback(self, *, callback_context, llm_response):
        if getattr(llm_response, "partial", False):
            return None
        segs = _segments(getattr(callback_context, "node_path", "") or "")
        node = segs[1][0] if len(segs) > 1 else callback_context.agent_name
        instance = segs[2][1] if len(segs) > 3 and segs[2][0] == node else None
        usage = getattr(llm_response, "usage_metadata", None)
        text = ""
        if llm_response.content and llm_response.content.parts:
            text = "".join(p.text or "" for p in llm_response.content.parts)
        await self._send({
            "type": "llm_end", "node": node, "instance": instance, "agent": callback_context.agent_name,
            "tokens_in": getattr(usage, "prompt_token_count", None), "tokens_out":
                (getattr(usage, "candidates_token_count", 0) or 0) + (getattr(usage, "thoughts_token_count", 0) or 0),
            "response": _preview(text, 1200),
        })
        return None

    async def on_model_error_callback(self, *, callback_context, llm_request, error):
        await self._send({"type": "error", "node": callback_context.agent_name, "message": str(error)[:400]})
        return None

    async def on_event_callback(self, *, invocation_context, event):
        info = event.node_info
        if info is None or not info.path:
            return None
        segs = _segments(info.path)
        if len(segs) < 2:
            return None
        node, run = segs[1]
        state_keys = sorted((event.actions.state_delta or {}).keys()) if event.actions else []
        route = event.actions.route if event.actions else None
        is_output = bool(info.output_for) and info.path in info.output_for

        if is_output and len(segs) == 2:  # a top-level graph node finished
            await self._send({"type": "node_done", "node": node, "run": run, "shape": _shape(event.output),
                              "output": _preview(event.output), "state": state_keys})
            if node in self._routing:
                self._pending[node] = event.output
            else:
                await self._handoff(node, event.output)
        elif is_output and len(segs) == 3 and segs[2][0] == node:  # one parallel worker item finished
            await self._send({"type": "worker_done", "node": node, "instance": segs[2][1], "shape": _shape(event.output),
                              "output": _preview(event.output, 900)})
        elif state_keys and len(segs) == 2:
            await self._send({"type": "state", "node": node, "keys": state_keys})
        if route is not None and len(segs) == 2:
            await self._send({"type": "route", "node": node, "route": str(route)})
            await self._handoff(node, self._pending.pop(node, None), route)
        return None


def summarize(r: dict) -> str:
    """One-line human summary of a trace record (used in logs and the CLI)."""
    t = r.get("type")
    if t == "graph":
        return f"ADK Workflow '{r['workflow']}' compiled: {len(r['nodes'])} nodes, {len(r['edges'])} edges"
    if t == "handoff":
        route = f" [route={r['route']}]" if r.get("route") else ""
        return f"{r['from']} → {', '.join(r['to'])}{route}: node_input = {r['shape']}"
    if t == "llm_start":
        inst = f" #{r['instance']}" if r.get("instance") else ""
        return (f"{r['node']}{inst} → {r['agent']} calls {r['model']}"
                + (f" (output_schema={r['schema']})" if r.get("schema") else "")
                + (f", {r['images']} image(s)" if r.get("images") else ""))
    if t == "llm_end":
        inst = f" #{r['instance']}" if r.get("instance") else ""
        return f"{r['agent']}{inst} answered: {r.get('tokens_in') or 0:,} tokens in / {r.get('tokens_out') or 0:,} out"
    if t == "worker_done":
        return f"{r['node']} worker #{r['instance']} finished → {r['shape']}"
    if t == "node_done":
        st = f"; state += {', '.join(r['state'])}" if r.get("state") else ""
        return f"{r['node']} finished → output {r['shape']}{st}"
    if t == "route":
        return f"{r['node']} set ctx.route = '{r['route']}'"
    if t == "state":
        return f"{r['node']} wrote state: {', '.join(r['keys'])}"
    if t == "error":
        return f"{r.get('node')}: model error {r.get('message')}"
    return t or "trace"

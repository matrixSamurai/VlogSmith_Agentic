import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { ArrowRight, Bot, Braces, ChevronDown, ChevronRight, GitBranch, GitMerge, Layers, Play, Split, Zap } from "lucide-react";
import type { JobEvent } from "../api";

/* ------------------------------------------------------------------ trace record types (from backend/app/agents/trace.py) */
interface GNode { id: string; kind: "start" | "function" | "parallel" | "join"; agents: { agent: string; model: string; schema: string | null }[]; routes: string[] }
interface GEdge { from: string; to: string; route: string | null }
type Rec =
  | { type: "graph"; workflow: string; nodes: GNode[]; edges: GEdge[] }
  | { type: "handoff"; from: string; to: string[]; route: string | null; shape: string; payload: string }
  | { type: "llm_start"; node: string; instance: string | null; agent: string; model: string; schema: string | null; images: number; prompt_chars: number; prompt: string }
  | { type: "llm_end"; node: string; instance: string | null; agent: string; tokens_in: number | null; tokens_out: number | null; response: string }
  | { type: "worker_done"; node: string; instance: string; shape: string; output: string }
  | { type: "node_done"; node: string; run: string; shape: string; output: string; state: string[] }
  | { type: "route"; node: string; route: string }
  | { type: "state"; node: string; keys: string[] }
  | { type: "error"; node: string; message: string };

type Status = "idle" | "active" | "done" | "error";
interface NodeState {
  status: Status;
  runs: number;
  workers: Record<string, Status>;
  tokensIn: number;
  tokensOut: number;
  calls: number;
  lastInput?: { from: string; shape: string; payload: string; route: string | null };
  lastOutput?: { shape: string; output: string };
  state: string[];
  llm: { agent: string; instance: string | null; model: string; schema: string | null; prompt: string; response?: string; tokens?: string }[];
}

const NODE_W = 178, NODE_H = 70, ROW_GAP = 50, COL_GAP = 14, PAD = 20;

const LABELS: Record<string, string> = {
  __START__: "START", ingest: "Ingest media", analyze_media: "Analyze media", plan_story: "Plan story",
  write_sections: "Write sections", review_script: "Editor review", rewrite_sections: "Rewrite sections",
  merge_revisions: "Merge revisions", narrate: "Narrate (TTS)", compose_music: "Compose music",
  write_article: "Write article", production_join: "Join", render_video: "Render video", publish: "Publish",
};

/* ------------------------------------------------------------------ layout: layered DAG, back-edges detected by DFS */
function layout(nodes: GNode[], edges: GEdge[]) {
  const out = new Map<string, string[]>();
  nodes.forEach((n) => out.set(n.id, []));
  edges.forEach((e) => out.get(e.from)?.push(e.to));
  const back = new Set<string>();
  const state = new Map<string, number>();
  const dfs = (id: string) => {
    state.set(id, 1);
    for (const t of out.get(id) || []) {
      if (state.get(t) === 1) back.add(`${id}>${t}`);
      else if (!state.get(t)) dfs(t);
    }
    state.set(id, 2);
  };
  dfs("__START__");
  nodes.forEach((n) => !state.get(n.id) && dfs(n.id));
  const level = new Map<string, number>([["__START__", 0]]);
  for (let i = 0; i < nodes.length; i++) {
    for (const e of edges) {
      if (back.has(`${e.from}>${e.to}`)) continue;
      const l = (level.get(e.from) ?? 0) + 1;
      if (l > (level.get(e.to) ?? -1)) level.set(e.to, l);
    }
  }
  const rows: string[][] = [];
  nodes.forEach((n) => { const l = level.get(n.id) ?? 0; (rows[l] ||= []).push(n.id); });
  const maxCols = Math.max(...rows.map((r) => r?.length || 0));
  const width = PAD * 2 + maxCols * NODE_W + (maxCols - 1) * COL_GAP;
  const pos = new Map<string, { x: number; y: number }>();
  rows.forEach((row, li) => {
    const rw = row.length * NODE_W + (row.length - 1) * COL_GAP;
    row.forEach((id, i) => pos.set(id, { x: (width - rw) / 2 + i * (NODE_W + COL_GAP), y: PAD + li * (NODE_H + ROW_GAP) }));
  });
  return { pos, back, width, height: PAD * 2 + rows.length * (NODE_H + ROW_GAP) - ROW_GAP };
}

function edgePath(a: { x: number; y: number }, b: { x: number; y: number }, isBack: boolean, width: number) {
  const sx = a.x + NODE_W / 2, sy = a.y + (isBack ? NODE_H / 2 : NODE_H);
  const tx = b.x + NODE_W / 2, ty = b.y + (isBack ? NODE_H / 2 : 0);
  if (isBack) {
    const side = a.x + NODE_W / 2 > width / 2 ? 1 : -1;
    const ox = side > 0 ? Math.max(a.x, b.x) + NODE_W + 30 : Math.min(a.x, b.x) - 30;
    const ax = side > 0 ? a.x + NODE_W : a.x, bx = side > 0 ? b.x + NODE_W : b.x;
    return { d: `M ${ax} ${sy} C ${ox} ${sy}, ${ox} ${ty}, ${bx} ${ty}`, mx: ox - side * 4, my: (sy + ty) / 2 };
  }
  const my = (sy + ty) / 2;
  return { d: `M ${sx} ${sy} C ${sx} ${my}, ${tx} ${my}, ${tx} ${ty}`, mx: (sx + tx) / 2, my };
}

/* ------------------------------------------------------------------ reducer: trace records -> node / edge state */
function reduce(recs: Rec[]) {
  const nodes: Record<string, NodeState> = {};
  const used = new Map<string, number>(); // edge key -> times traversed
  const get = (id: string) => (nodes[id] ||= { status: "idle", runs: 0, workers: {}, tokensIn: 0, tokensOut: 0, calls: 0, state: [], llm: [] });
  for (const r of recs) {
    if (r.type === "handoff") {
      for (const t of r.to) {
        used.set(`${r.from}>${t}`, (used.get(`${r.from}>${t}`) || 0) + 1);
        const n = get(t);
        n.status = "active";
        n.lastInput = { from: r.from, shape: r.shape, payload: r.payload, route: r.route };
      }
      get(r.from).status = get(r.from).status === "error" ? "error" : "done";
    } else if (r.type === "llm_start") {
      const n = get(r.node);
      n.status = "active";
      if (r.instance) n.workers[r.instance] = "active";
      n.llm.push({ agent: r.agent, instance: r.instance, model: r.model, schema: r.schema, prompt: r.prompt });
    } else if (r.type === "llm_end") {
      const n = get(r.node);
      n.calls += 1;
      n.tokensIn += r.tokens_in || 0;
      n.tokensOut += r.tokens_out || 0;
      const call = [...n.llm].reverse().find((c) => c.instance === r.instance && !c.response);
      if (call) { call.response = r.response; call.tokens = `${(r.tokens_in || 0).toLocaleString()} in / ${(r.tokens_out || 0).toLocaleString()} out`; }
    } else if (r.type === "worker_done") {
      get(r.node).workers[r.instance] = "done";
    } else if (r.type === "node_done") {
      const n = get(r.node);
      n.status = "done";
      n.runs += 1;
      n.lastOutput = { shape: r.shape, output: r.output };
      n.state = Array.from(new Set([...n.state, ...r.state]));
    } else if (r.type === "state") {
      const n = get(r.node);
      n.state = Array.from(new Set([...n.state, ...r.keys]));
    } else if (r.type === "error") {
      get(r.node).status = "error";
    }
  }
  return { nodes, used };
}

/* ------------------------------------------------------------------ component */
export default function AgentFlow({ events, live }: { events: JobEvent[]; live: boolean }) {
  const recs = useMemo(() => events.filter((e) => e.stage === "trace" && e.data).map((e) => e.data as unknown as Rec), [events]);
  const graph = recs.find((r) => r.type === "graph") as Extract<Rec, { type: "graph" }> | undefined;
  const { nodes, used } = useMemo(() => reduce(recs), [recs]);
  const [selected, setSelected] = useState<string | null>(null);
  const [flash, setFlash] = useState<Set<string>>(new Set());
  const lastHandoffs = useRef(0);

  // Animate the edges a payload just travelled along.
  useEffect(() => {
    const hs = recs.filter((r) => r.type === "handoff") as Extract<Rec, { type: "handoff" }>[];
    if (hs.length <= lastHandoffs.current) return;
    const fresh = hs.slice(lastHandoffs.current);
    lastHandoffs.current = hs.length;
    const keys = new Set(fresh.flatMap((h) => h.to.map((t) => `${h.from}>${t}`)));
    setFlash((prev) => new Set([...prev, ...keys]));
    const timer = setTimeout(() => setFlash((prev) => new Set([...prev].filter((k) => !keys.has(k)))), 2600);
    return () => clearTimeout(timer);
  }, [recs]);

  if (!graph) return null;
  const { pos, back, width, height } = layout(graph.nodes, graph.edges);
  const sel = selected ? nodes[selected] : null;
  const selNode = graph.nodes.find((n) => n.id === selected);

  return (
    <div className="card overflow-hidden p-0">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/5 px-5 py-3">
        <div className="flex items-center gap-2 text-sm font-semibold">
          <GitBranch size={15} className="text-brand-400" /> ADK Workflow <span className="font-mono text-xs text-zinc-500">{graph.workflow}</span>
          {live && <span className="ml-1 inline-flex items-center gap-1 rounded-full bg-sky-500/15 px-2 py-0.5 text-[10px] font-medium text-sky-300"><span className="h-1.5 w-1.5 animate-pulse rounded-full bg-sky-400" /> LIVE</span>}
        </div>
        <div className="flex gap-3 text-[11px] text-zinc-500">
          <span className="flex items-center gap-1"><Layers size={12} /> parallel worker</span>
          <span className="flex items-center gap-1"><Split size={12} /> router</span>
          <span className="flex items-center gap-1"><GitMerge size={12} /> join</span>
          <span className="flex items-center gap-1"><Bot size={12} /> LlmAgent</span>
        </div>
      </div>
      <div className="grid lg:grid-cols-[minmax(0,1fr)_380px]">
        <div className="overflow-x-auto border-b border-white/5 lg:border-b-0 lg:border-r">
          <div className="relative mx-auto" style={{ width, height }}>
            <svg className="absolute inset-0" width={width} height={height}>
              <defs>
                <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" />
                </marker>
              </defs>
              {graph.edges.map((e) => {
                const a = pos.get(e.from), b = pos.get(e.to);
                if (!a || !b) return null;
                const key = `${e.from}>${e.to}`;
                const isBack = back.has(key);
                const { d, mx, my } = edgePath(a, b, isBack, width);
                const hot = flash.has(key), done = (used.get(key) || 0) > 0;
                const color = hot ? "#ff7a5f" : done ? "rgba(255,122,95,0.55)" : "rgba(255,255,255,0.14)";
                return (
                  <g key={key} style={{ color }}>
                    <path d={d} fill="none" stroke={color} strokeWidth={hot ? 2.6 : 1.6} strokeDasharray={done || hot ? (hot ? "6 6" : undefined) : "4 5"}
                      markerEnd="url(#arrow)" className={hot ? "flow-dash" : ""} />
                    {e.route && (
                      <g transform={`translate(${mx}, ${my})`}>
                        <rect x={-30} y={-9} width={60} height={18} rx={9} fill={done ? "#3b1f19" : "#18181b"} stroke={color} />
                        <text textAnchor="middle" y={4} fontSize={10} fill={done ? "#ffb4a3" : "#a1a1aa"} fontFamily="ui-monospace,monospace">{e.route}</text>
                      </g>
                    )}
                    {(used.get(key) || 0) > 1 && (
                      <text x={mx + 8} y={my - 6} fontSize={10} fill="#ffb4a3">×{used.get(key)}</text>
                    )}
                  </g>
                );
              })}
            </svg>
            {graph.nodes.map((n) => {
              const p = pos.get(n.id)!;
              const st = nodes[n.id];
              const status: Status = n.id === "__START__" ? (recs.length > 1 ? "done" : "active") : st?.status || "idle";
              const workers = Object.entries(st?.workers || {});
              const Icon = n.kind === "parallel" ? Layers : n.kind === "join" ? GitMerge : n.routes.length ? Split : n.id === "__START__" ? Play : Braces;
              return (
                <button key={n.id} onClick={() => setSelected(n.id === selected ? null : n.id)}
                  className={`absolute rounded-xl border px-3 py-2 text-left transition ${
                    selected === n.id ? "ring-2 ring-brand-500/60" : ""} ${
                    status === "active" ? "border-sky-400/60 bg-sky-500/10 shadow-lg shadow-sky-500/10"
                      : status === "done" ? "border-brand-500/40 bg-brand-500/[0.07]"
                      : status === "error" ? "border-red-500/50 bg-red-500/10" : "border-white/10 bg-zinc-900/70"}`}
                  style={{ left: p.x, top: p.y, width: NODE_W, height: NODE_H }}>
                  <div className="flex items-center gap-1.5">
                    <Icon size={13} className={status === "active" ? "text-sky-300" : status === "done" ? "text-brand-400" : "text-zinc-500"} />
                    <span className="truncate text-[13px] font-medium text-zinc-100">{LABELS[n.id] || n.id}</span>
                    {status === "active" && <span className="ml-auto h-2 w-2 shrink-0 animate-pulse rounded-full bg-sky-400" />}
                    {(st?.runs || 0) > 1 && <span className="ml-auto rounded bg-white/10 px-1 text-[10px] text-zinc-300">run {st!.runs}</span>}
                  </div>
                  <div className="mt-0.5 truncate font-mono text-[10px] text-zinc-500">
                    {n.agents.length ? `${n.agents[0].agent} · ${n.agents[0].model}` : n.kind === "join" ? "waits for all inputs" : n.kind === "start" ? "user message" : "FunctionNode"}
                  </div>
                  <div className="mt-1 flex items-center gap-1">
                    {workers.slice(0, 10).map(([id, s]) => (
                      <span key={id} title={`worker #${id}`} className={`h-1.5 w-3 rounded-full ${s === "done" ? "bg-brand-400" : "animate-pulse bg-sky-400"}`} />
                    ))}
                    {workers.length > 10 && <span className="text-[9px] text-zinc-500">+{workers.length - 10}</span>}
                    {st && st.calls > 0 && <span className="ml-auto text-[10px] tabular-nums text-zinc-400">{(st.tokensIn + st.tokensOut).toLocaleString()} tok</span>}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
        <div className="max-h-[760px] overflow-y-auto">
          {sel && selNode ? <NodeDetail id={selected!} node={selNode} st={sel} /> : <HandoffFeed recs={recs} live={live} onPick={setSelected} />}
        </div>
      </div>
    </div>
  );
}

function Block({ title, children, defaultOpen = false }: { title: ReactNode; children: ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="rounded-lg border border-white/5 bg-black/20">
      <button className="flex w-full items-center gap-1.5 px-3 py-2 text-left text-xs text-zinc-300" onClick={() => setOpen(!open)}>
        {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />} {title}
      </button>
      {open && <div className="border-t border-white/5 px-3 py-2">{children}</div>}
    </div>
  );
}

function Pre({ text }: { text: string }) {
  return <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed text-zinc-400">{text || "—"}</pre>;
}

function NodeDetail({ id, node, st }: { id: string; node: GNode; st: NodeState }) {
  return (
    <div className="space-y-3 p-4 text-sm">
      <div>
        <div className="font-semibold">{LABELS[id] || id}</div>
        <div className="font-mono text-[11px] text-zinc-500">
          {node.kind === "parallel" ? "_ParallelWorker → one run per list item" : node.kind === "join" ? "JoinNode → waits for every predecessor" : node.routes.length ? `FunctionNode with routes: ${node.routes.join(", ")}` : "FunctionNode"}
        </div>
      </div>
      {node.agents.map((a) => (
        <div key={a.agent} className="rounded-lg border border-white/5 bg-white/[0.02] px-3 py-2 text-xs">
          <div className="flex items-center gap-1.5 text-zinc-200"><Bot size={13} className="text-brand-400" /> {a.agent}</div>
          <div className="mt-0.5 font-mono text-[11px] text-zinc-500">model={a.model}{a.schema ? ` · output_schema=${a.schema}` : ""}</div>
        </div>
      ))}
      {st.lastInput && (
        <Block defaultOpen title={<span>node_input from <b className="text-zinc-100">{LABELS[st.lastInput.from] || st.lastInput.from}</b>{st.lastInput.route ? ` (route ${st.lastInput.route})` : ""} · <span className="font-mono text-zinc-500">{st.lastInput.shape}</span></span>}>
          <Pre text={st.lastInput.payload} />
        </Block>
      )}
      {st.lastOutput && (
        <Block title={<span>output · <span className="font-mono text-zinc-500">{st.lastOutput.shape}</span></span>}>
          <Pre text={st.lastOutput.output} />
        </Block>
      )}
      {st.state.length > 0 && (
        <div className="text-xs text-zinc-400">session state written: {st.state.map((k) => <code key={k} className="mr-1 rounded bg-white/5 px-1 text-zinc-300">{k}</code>)}</div>
      )}
      {st.llm.length > 0 && <div className="pt-1 text-xs font-medium text-zinc-300">Model calls ({st.llm.length}) · {(st.tokensIn).toLocaleString()} in / {(st.tokensOut).toLocaleString()} out</div>}
      {st.llm.map((c, i) => (
        <Block key={i} title={<span>{c.agent}{c.instance ? ` #${c.instance}` : ""} → <span className="font-mono">{c.model}</span>{c.tokens ? <span className="text-zinc-500"> · {c.tokens}</span> : <span className="text-sky-300"> · running…</span>}</span>}>
          <div className="mb-1 text-[10px] uppercase tracking-wide text-zinc-500">Prompt (user turn)</div>
          <Pre text={c.prompt} />
          {c.response && <><div className="mb-1 mt-2 text-[10px] uppercase tracking-wide text-zinc-500">Structured response{c.schema ? ` (${c.schema})` : ""}</div><Pre text={c.response} /></>}
        </Block>
      ))}
    </div>
  );
}

function HandoffFeed({ recs, live, onPick }: { recs: Rec[]; live: boolean; onPick: (id: string) => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const [showLlm, setShowLlm] = useState(true);
  const items = recs.filter((r) => r.type === "handoff" || r.type === "route" || (showLlm && r.type === "llm_start") || r.type === "error");
  useEffect(() => { if (live) ref.current?.scrollTo({ top: ref.current.scrollHeight, behavior: "smooth" }); }, [items.length, live]);
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between px-4 pt-3 text-xs">
        <span className="font-medium text-zinc-300">Control flow · {items.length} events</span>
        <label className="flex items-center gap-1.5 text-zinc-500"><input type="checkbox" checked={showLlm} onChange={(e) => setShowLlm(e.target.checked)} /> model calls</label>
      </div>
      <p className="px-4 pb-2 pt-1 text-[11px] text-zinc-500">Click a node to inspect its node_input, output, state writes and prompts.</p>
      <div ref={ref} className="flex-1 space-y-1.5 overflow-y-auto px-4 pb-4">
        {items.map((r, i) => {
          if (r.type === "handoff") return (
            <Block key={i} title={
              <span className="flex flex-wrap items-center gap-1">
                <b className="cursor-pointer text-zinc-100" onClick={(e) => { e.stopPropagation(); onPick(r.from); }}>{LABELS[r.from] || r.from}</b>
                <ArrowRight size={12} className="text-brand-400" />
                {r.to.map((t) => <b key={t} className="cursor-pointer text-zinc-100" onClick={(e) => { e.stopPropagation(); onPick(t); }}>{LABELS[t] || t}</b>)}
                {r.route && <span className="rounded bg-brand-500/20 px-1.5 font-mono text-[10px] text-brand-400">{r.route}</span>}
                <span className="font-mono text-[10px] text-zinc-500">{r.shape}</span>
              </span>}>
              <Pre text={r.payload} />
            </Block>
          );
          if (r.type === "route") return (
            <div key={i} className="flex items-center gap-1.5 px-1 text-[11px] text-amber-300"><Split size={12} /> {LABELS[r.node] || r.node} set <code>ctx.route = "{r.route}"</code></div>
          );
          if (r.type === "llm_start") return (
            <div key={i} className="flex items-center gap-1.5 px-1 text-[11px] text-zinc-400"><Zap size={12} className="text-sky-400" />
              <span className="truncate">{r.agent}{r.instance ? ` #${r.instance}` : ""} → {r.model}{r.schema ? ` · ${r.schema}` : ""}{r.images ? ` · ${r.images} img` : ""}</span></div>
          );
          if (r.type === "error") return <div key={i} className="px-1 text-[11px] text-red-300">{r.node}: {r.message}</div>;
          return null;
        })}
        {live && <div className="px-1 pt-1 text-[11px] text-zinc-600">waiting for the next hand-off…</div>}
      </div>
    </div>
  );
}

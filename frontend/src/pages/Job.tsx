import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  ArrowLeft, Check, Clapperboard, Copy, MonitorPlay, Download, FileText, Film, Loader2, Mic, Package, PenLine, RefreshCcw, ScanEye,
  ShieldCheck, Upload, X, XCircle, Map as MapIcon,
} from "lucide-react";
import { api, subscribe, usd, type Job, type JobCost, type JobEvent } from "../api";
import { ProgressBar, StatusBadge } from "../components/ui";

const STAGES = [
  { id: "ingest", label: "Ingest", desc: "Normalize photos, sample video keyframes", icon: Upload },
  { id: "analyze", label: "Vision analysis", desc: "Parallel Gemini agents study every shot", icon: ScanEye },
  { id: "plan", label: "Story plan", desc: "Showrunner builds the narrative arc", icon: MapIcon },
  { id: "write", label: "Section writers", desc: "Captions, narration and article per section", icon: PenLine },
  { id: "review", label: "Editor review", desc: "Critique and revision loop", icon: ShieldCheck },
  { id: "production", label: "Production", desc: "Voiceover · music · article in parallel", icon: Mic },
  { id: "render", label: "Video edit", desc: "Ken Burns, captions, transitions, mix", icon: Film },
  { id: "publish", label: "Publish", desc: "Article page, package and assets", icon: Package },
] as const;

type StageState = "pending" | "active" | "done" | "failed";
type Tab = "video" | "article" | "youtube";

export default function JobPage() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const [job, setJob] = useState<Job | null>(null);
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [tab, setTab] = useState<Tab>("video");
  const [error, setError] = useState<string | null>(null);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setEvents([]);
    api.job(id).then(setJob).catch((e) => setError(e.message));
    const seen = new Set<number>();
    return subscribe(id, (ev) => {
      if (seen.has(ev.seq)) return;
      seen.add(ev.seq);
      setEvents((prev) => [...prev, ev]);
    }, setJob);
  }, [id]);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: "smooth" });
  }, [events.length]);

  const stageStates = useMemo(() => {
    const order = STAGES.map((s) => s.id as string);
    const states: Record<string, StageState> = Object.fromEntries(order.map((s) => [s, "pending"]));
    let furthest = -1;
    for (const ev of events) furthest = Math.max(furthest, order.indexOf(ev.stage));
    order.forEach((s, i) => { if (i < furthest) states[s] = "done"; });
    if (furthest >= 0) states[order[furthest]] = "active";
    if (job?.status === "succeeded") order.forEach((s) => (states[s] = "done"));
    if (job?.status === "failed" && furthest >= 0) states[order[furthest]] = "failed";
    return states;
  }, [events, job?.status]);

  if (error) return <p className="py-20 text-center text-red-300">{error}</p>;
  if (!job) return <div className="grid place-items-center py-32"><Loader2 className="animate-spin text-zinc-500" /></div>;

  const running = job.status === "queued" || job.status === "running";
  const done = job.status === "succeeded";
  const elapsed = job.started_at ? Math.round((job.finished_at || Date.now() / 1000) - job.started_at) : 0;

  async function retry() {
    const fresh = await api.retry(id);
    navigate(`/jobs/${fresh.id}`);
  }

  return (
    <div className="space-y-6">
      <Link to="/library" className="inline-flex items-center gap-1.5 text-sm text-zinc-400 hover:text-white">
        <ArrowLeft size={15} /> Library
      </Link>

      <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div className="min-w-0 space-y-2">
          <div className="flex items-center gap-3">
            <StatusBadge status={job.status} />
            <span className="text-xs text-zinc-500">
              {job.media.filter((m) => m.kind !== "audio").length} media · {job.settings.aspect_ratio} · {elapsed > 0 && `${Math.floor(elapsed / 60)}m ${elapsed % 60}s`}
            </span>
          </div>
          <h1 className="truncate text-2xl font-semibold tracking-tight sm:text-3xl">{job.title || "Creating your vlog…"}</h1>
          {job.summary && <p className="text-zinc-400">{job.summary}</p>}
        </div>
        <div className="flex shrink-0 gap-2">
          {running && <button className="btn-ghost" onClick={() => api.cancel(id)}><X size={15} /> Cancel</button>}
          {(job.status === "failed" || job.status === "cancelled") && <button className="btn-primary" onClick={retry}><RefreshCcw size={15} /> Retry</button>}
          {done && <button className="btn-ghost" onClick={retry}><RefreshCcw size={15} /> Regenerate</button>}
        </div>
      </div>

      {!done && <ProgressBar value={job.progress} active={running} />}

      {job.status === "failed" && (
        <div className="flex items-start gap-3 rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-200">
          <XCircle size={18} className="mt-0.5 shrink-0" /> {job.error || "Something went wrong."}
        </div>
      )}

      <div className={`grid gap-6 ${done ? "lg:grid-cols-[minmax(0,1fr)_340px]" : "lg:grid-cols-[360px_minmax(0,1fr)]"}`}>
        {done ? (
          <Results job={job} tab={tab} setTab={setTab} />
        ) : (
          <Pipeline states={stageStates} />
        )}
        <div className="space-y-6">
        {job.cost && !running && <CostCard cost={job.cost} final={done} />}
        <div className="card flex max-h-[640px] flex-col p-0">
          <div className="border-b border-white/5 px-5 py-3 text-sm font-semibold">Agent activity</div>
          <div ref={logRef} className="flex-1 space-y-2 overflow-y-auto p-4 text-sm">
            {events.map((ev) => (
              <div key={ev.seq} className="flex gap-3">
                <span className="w-14 shrink-0 pt-0.5 text-[11px] tabular-nums text-zinc-600">
                  {new Date(ev.ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                </span>
                <span className={
                  ev.level === "error" ? "text-red-300" : ev.level === "warning" ? "text-amber-300"
                    : ev.level === "success" ? "text-emerald-300" : "text-zinc-300"}>
                  <span className="mr-1.5 rounded bg-white/5 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-zinc-500">{ev.stage}</span>
                  {ev.message}
                </span>
              </div>
            ))}
            {running && <div className="flex items-center gap-2 pl-[68px] text-zinc-500"><Loader2 size={13} className="animate-spin" /> working…</div>}
          </div>
        </div>
        </div>
      </div>
    </div>
  );
}

function Pipeline({ states }: { states: Record<string, StageState> }) {
  return (
    <div className="card p-5">
      <div className="mb-4 text-sm font-semibold">ADK workflow</div>
      <ol className="relative space-y-1">
        {STAGES.map((s, i) => {
          const st = states[s.id];
          const Icon = s.icon;
          return (
            <li key={s.id} className="relative flex gap-3 pb-3">
              {i < STAGES.length - 1 && (
                <span className={`absolute left-[17px] top-9 h-[calc(100%-24px)] w-px ${st === "done" ? "bg-brand-500/60" : "bg-white/10"}`} />
              )}
              <span className={`relative z-10 grid h-9 w-9 shrink-0 place-items-center rounded-xl ring-1 transition ${
                st === "done" ? "bg-brand-500 text-white ring-brand-500"
                  : st === "active" ? "bg-brand-500/15 text-brand-400 ring-brand-500/50"
                  : st === "failed" ? "bg-red-500/15 text-red-300 ring-red-500/40"
                  : "bg-white/5 text-zinc-500 ring-white/10"}`}>
                {st === "done" ? <Check size={16} /> : st === "active" ? <Loader2 size={16} className="animate-spin" /> : <Icon size={16} />}
              </span>
              <span className="pt-0.5">
                <span className={`block text-sm font-medium ${st === "pending" ? "text-zinc-500" : "text-zinc-100"}`}>{s.label}</span>
                <span className="block text-xs text-zinc-500">{s.desc}</span>
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function Results({ job, tab, setTab }: { job: Job; tab: Tab; setTab: (t: Tab) => void }) {
  const o = job.outputs;
  const vertical = job.settings.aspect_ratio === "9:16";
  const tabs: Tab[] = ["video", ...(o.youtube ? ["youtube" as Tab] : []), ...(o.article_html ? ["article" as Tab] : [])];
  const icon = { video: <Clapperboard size={14} />, youtube: <MonitorPlay size={14} />, article: <FileText size={14} /> };
  return (
    <div className="space-y-4">
      {tabs.length > 1 && (
        <div className="flex gap-1 rounded-xl border border-white/10 bg-zinc-900/80 p-1 sm:w-fit">
          {tabs.map((t) => (
            <button key={t} onClick={() => setTab(t)}
              className={`flex flex-1 items-center justify-center gap-2 rounded-lg px-4 py-1.5 text-sm font-medium capitalize ${tab === t ? "bg-brand-500 text-white" : "text-zinc-400 hover:text-white"}`}>
              {icon[t]} {t === "youtube" ? "YouTube" : t}
            </button>
          ))}
        </div>
      )}
      {tab === "article" && o.article_html ? (
        <iframe title="Article preview" sandbox="" src={api.fileUrl(job.id, o.article_html)}
          className="h-[75vh] w-full rounded-2xl border border-white/10 bg-white" />
      ) : tab === "youtube" && o.youtube ? (
        <YouTubePanel job={job} />
      ) : (
        <div className="card overflow-hidden bg-black p-0">
          <video
            key={job.id}
            src={api.fileUrl(job.id, o.video || "vlog.mp4")}
            poster={o.thumbnail ? api.fileUrl(job.id, o.thumbnail) : undefined}
            controls playsInline
            className={`mx-auto block bg-black ${vertical ? "max-h-[75vh]" : "w-full"}`}
          />
        </div>
      )}
      <div className="flex flex-wrap gap-2">
        <a className="btn-primary" href={api.fileUrl(job.id, o.video || "vlog.mp4", true)}><Download size={15} /> Video (MP4)</a>
        {o.thumbnail && <a className="btn-ghost" href={api.fileUrl(job.id, o.thumbnail, true)}><Download size={15} /> Thumbnail</a>}
        {o.youtube && <a className="btn-ghost" href={api.fileUrl(job.id, o.youtube, true)}><MonitorPlay size={15} /> YouTube text</a>}
        {o.article_html && <a className="btn-ghost" href={api.fileUrl(job.id, o.article_html, true)}><FileText size={15} /> Article HTML</a>}
        {o.article_md && <a className="btn-ghost" href={api.fileUrl(job.id, o.article_md, true)}><FileText size={15} /> Markdown</a>}
        {o.package && <a className="btn-ghost" href={api.fileUrl(job.id, o.package, true)}><Package size={15} /> Full package (.zip)</a>}
      </div>
      {job.duration && <p className="text-xs text-zinc-500">{Math.round(job.duration)}s · {job.settings.resolution} · {job.settings.fps} fps</p>}
    </div>
  );
}

function YouTubePanel({ job }: { job: Job }) {
  const [text, setText] = useState<string>("");
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    fetch(api.fileUrl(job.id, job.outputs.youtube!)).then((r) => r.text()).then(setText).catch(() => setText(""));
  }, [job.id, job.outputs.youtube]);
  const copy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  return (
    <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_320px]">
      <div className="card relative p-5">
        <button className="btn-ghost absolute right-4 top-4 px-3 py-1.5 text-xs" onClick={copy}>
          {copied ? <Check size={13} /> : <Copy size={13} />} {copied ? "Copied" : "Copy all"}
        </button>
        <pre className="max-h-[60vh] overflow-auto whitespace-pre-wrap pr-24 font-sans text-sm leading-relaxed text-zinc-200">{text || "Loading…"}</pre>
      </div>
      {job.outputs.thumbnail && (
        <div className="space-y-2">
          <span className="label">Thumbnail</span>
          <img src={api.fileUrl(job.id, job.outputs.thumbnail)} alt="Thumbnail" className="w-full rounded-xl ring-1 ring-white/10" />
        </div>
      )}
    </div>
  );
}

function CostCard({ cost, final }: { cost: JobCost; final: boolean }) {
  const [open, setOpen] = useState(false);
  const tokens = cost.lines.reduce((a, l) => a + l.input_tokens + l.output_tokens, 0);
  return (
    <div className="card p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="label">{final ? "Generation cost" : "Spent before stopping"}</div>
          <div className="mt-1 text-3xl font-semibold tabular-nums">{usd(cost.total_usd)}</div>
          <div className="mt-1 text-xs text-zinc-500">
            AI {usd(cost.ai_usd)}{cost.compute_usd ? ` · compute ${usd(cost.compute_usd)}` : ""} · {tokens.toLocaleString()} tokens
          </div>
        </div>
        <button className="btn-ghost px-3 py-1.5 text-xs" onClick={() => setOpen(!open)}>{open ? "Hide" : "Breakdown"}</button>
      </div>
      {open && (
        <div className="mt-4 space-y-3">
          <table className="w-full text-xs">
            <thead className="text-left text-zinc-500">
              <tr><th className="pb-2 font-medium">Step</th><th className="pb-2 text-right font-medium">Calls</th>
                <th className="pb-2 text-right font-medium">Tokens in / out</th><th className="pb-2 text-right font-medium">Cost</th></tr>
            </thead>
            <tbody className="divide-y divide-white/5 tabular-nums text-zinc-300">
              {cost.lines.map((l) => (
                <tr key={l.label + l.model} title={`${l.model}${l.thinking_tokens ? ` · ${l.thinking_tokens.toLocaleString()} thinking tokens (billed as output)` : ""}`}>
                  <td className="py-1.5">{l.label}</td>
                  <td className="py-1.5 text-right">{l.calls}</td>
                  <td className="py-1.5 text-right">{l.input_tokens.toLocaleString()} / {l.output_tokens.toLocaleString()}</td>
                  <td className="py-1.5 text-right">{usd(l.cost_usd)}</td>
                </tr>
              ))}
              {cost.compute_usd > 0 && (
                <tr><td className="py-1.5">Cloud Run compute</td><td className="py-1.5 text-right">—</td>
                  <td className="py-1.5 text-right">{Math.round(cost.compute_seconds)}s</td><td className="py-1.5 text-right">{usd(cost.compute_usd)}</td></tr>
              )}
            </tbody>
          </table>
          {cost.notes.map((n) => <p key={n} className="text-[11px] leading-relaxed text-zinc-500">{n}</p>)}
          <p className="text-[11px] leading-relaxed text-zinc-500">Estimated from the token counts Gemini reports and published list prices; your Google bill is authoritative.</p>
        </div>
      )}
    </div>
  );
}

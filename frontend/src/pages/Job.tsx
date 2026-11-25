import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  ArrowLeft, Check, Clapperboard, Download, FileText, Film, Loader2, Mic, Package, PenLine, RefreshCcw, ScanEye,
  ShieldCheck, Upload, X, XCircle, Map as MapIcon,
} from "lucide-react";
import { api, subscribe, type Job, type JobEvent } from "../api";
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

export default function JobPage() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const [job, setJob] = useState<Job | null>(null);
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [tab, setTab] = useState<"video" | "article">("video");
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

function Results({ job, tab, setTab }: { job: Job; tab: "video" | "article"; setTab: (t: "video" | "article") => void }) {
  const o = job.outputs;
  const vertical = job.settings.aspect_ratio === "9:16";
  return (
    <div className="space-y-4">
      {o.article_html && (
        <div className="flex gap-1 rounded-xl border border-white/10 bg-zinc-900/80 p-1 sm:w-fit">
          {(["video", "article"] as const).map((t) => (
            <button key={t} onClick={() => setTab(t)}
              className={`flex flex-1 items-center justify-center gap-2 rounded-lg px-4 py-1.5 text-sm font-medium capitalize ${tab === t ? "bg-brand-500 text-white" : "text-zinc-400 hover:text-white"}`}>
              {t === "video" ? <Clapperboard size={14} /> : <FileText size={14} />} {t}
            </button>
          ))}
        </div>
      )}
      {tab === "video" || !o.article_html ? (
        <div className="card overflow-hidden bg-black p-0">
          <video
            key={job.id}
            src={api.fileUrl(job.id, o.video || "vlog.mp4")}
            poster={o.thumbnail ? api.fileUrl(job.id, o.thumbnail) : undefined}
            controls playsInline
            className={`mx-auto block bg-black ${vertical ? "max-h-[75vh]" : "w-full"}`}
          />
        </div>
      ) : (
        <iframe title="Article preview" sandbox="" src={api.fileUrl(job.id, o.article_html)}
          className="h-[75vh] w-full rounded-2xl border border-white/10 bg-white" />
      )}
      <div className="flex flex-wrap gap-2">
        <a className="btn-primary" href={api.fileUrl(job.id, o.video || "vlog.mp4", true)}><Download size={15} /> Video (MP4)</a>
        {o.article_html && <a className="btn-ghost" href={api.fileUrl(job.id, o.article_html, true)}><FileText size={15} /> Article HTML</a>}
        {o.article_md && <a className="btn-ghost" href={api.fileUrl(job.id, o.article_md, true)}><FileText size={15} /> Markdown</a>}
        {o.package && <a className="btn-ghost" href={api.fileUrl(job.id, o.package, true)}><Package size={15} /> Full package (.zip)</a>}
      </div>
      {job.duration && <p className="text-xs text-zinc-500">{Math.round(job.duration)}s · {job.settings.resolution} · {job.settings.fps} fps</p>}
    </div>
  );
}

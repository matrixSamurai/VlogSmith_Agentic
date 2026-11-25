import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Clapperboard, Loader2, Plus, Trash2 } from "lucide-react";
import { api, type Job } from "../api";
import { ProgressBar, StatusBadge, timeAgo } from "../components/ui";

export default function LibraryPage() {
  const [jobs, setJobs] = useState<Job[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const load = () => api.jobs().then((j) => alive && setJobs(j)).catch((e) => alive && setError(e.message));
    load();
    const t = setInterval(load, 5000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  async function remove(id: string) {
    if (!confirm("Delete this blog from the library?")) return;
    try {
      await api.remove(id);
      setJobs((j) => j?.filter((x) => x.id !== id) ?? null);
    } catch (e) {
      alert((e as Error).message);
    }
  }

  if (error) return <p className="py-20 text-center text-red-300">{error}</p>;
  if (!jobs) return <div className="grid place-items-center py-32"><Loader2 className="animate-spin text-zinc-500" /></div>;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold tracking-tight">Library</h1>
        <Link to="/" className="btn-primary"><Plus size={16} /> New vlog</Link>
      </div>
      {jobs.length === 0 ? (
        <div className="card grid place-items-center gap-3 py-24 text-zinc-400">
          <Clapperboard size={32} className="text-brand-400" />
          No blogs yet. Create your first one.
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {jobs.map((job) => (
            <div key={job.id} className="card group overflow-hidden p-0 transition hover:border-white/20">
              <Link to={`/jobs/${job.id}`} className="block">
                <div className="relative aspect-video bg-zinc-900">
                  {job.status === "succeeded" && job.outputs.thumbnail ? (
                    <img src={api.fileUrl(job.id, job.outputs.thumbnail)} alt="" className="h-full w-full object-cover" loading="lazy" />
                  ) : (
                    <div className="grid h-full place-items-center text-zinc-600">
                      {job.status === "running" || job.status === "queued" ? <Loader2 className="animate-spin" /> : <Clapperboard />}
                    </div>
                  )}
                  <div className="absolute left-3 top-3"><StatusBadge status={job.status} /></div>
                </div>
                <div className="space-y-2 p-4">
                  <div className="truncate font-medium">{job.title || job.settings.context.slice(0, 60)}</div>
                  <div className="text-xs text-zinc-500">
                    {timeAgo(job.created_at)} · {job.settings.aspect_ratio}
                    {job.duration ? ` · ${Math.round(job.duration)}s` : ""}
                  </div>
                  {(job.status === "running" || job.status === "queued") && <ProgressBar value={job.progress} />}
                </div>
              </Link>
              {!(job.status === "running" || job.status === "queued") && (
                <div className="flex justify-end border-t border-white/5 px-3 py-2">
                  <button onClick={() => remove(job.id)} className="rounded-lg p-1.5 text-zinc-500 hover:bg-red-500/10 hover:text-red-300" title="Delete">
                    <Trash2 size={15} />
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

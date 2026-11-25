import type { ReactNode } from "react";
import type { JobStatus } from "../api";

export function Section({ title, icon, children }: { title: string; icon?: ReactNode; children: ReactNode }) {
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 text-sm font-semibold text-zinc-200">
        {icon}
        {title}
      </div>
      {children}
    </div>
  );
}

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="block space-y-1.5">
      <span className="label">{label}</span>
      {children}
      {hint && <span className="block text-xs text-zinc-500">{hint}</span>}
    </label>
  );
}

export function Segmented<T extends string | number>({
  value, options, onChange, labels,
}: { value: T; options: readonly T[]; onChange: (v: T) => void; labels?: Partial<Record<string, string>> }) {
  return (
    <div className="flex flex-wrap gap-1 rounded-xl border border-white/10 bg-zinc-900/80 p-1">
      {options.map((o) => (
        <button
          type="button"
          key={String(o)}
          onClick={() => onChange(o)}
          className={`flex-1 whitespace-nowrap rounded-lg px-2.5 py-1.5 text-xs font-medium capitalize transition ${
            value === o ? "bg-brand-500 text-white shadow" : "text-zinc-400 hover:bg-white/5 hover:text-zinc-200"
          }`}
        >
          {labels?.[String(o)] ?? String(o).replace("_", " ")}
        </button>
      ))}
    </div>
  );
}

export function Toggle({ checked, onChange, label, hint }: { checked: boolean; onChange: (v: boolean) => void; label: string; hint?: string }) {
  return (
    <button type="button" onClick={() => onChange(!checked)} className="flex w-full items-start justify-between gap-4 text-left">
      <span>
        <span className="block text-sm text-zinc-200">{label}</span>
        {hint && <span className="block text-xs text-zinc-500">{hint}</span>}
      </span>
      <span className={`relative mt-0.5 h-5 w-9 shrink-0 rounded-full transition ${checked ? "bg-brand-500" : "bg-zinc-700"}`}>
        <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition ${checked ? "left-4.5" : "left-0.5"}`} />
      </span>
    </button>
  );
}

export function Slider({
  label, value, min, max, step, onChange, format,
}: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; format?: (v: number) => string }) {
  return (
    <label className="block space-y-1.5">
      <span className="flex items-center justify-between">
        <span className="label">{label}</span>
        <span className="text-xs tabular-nums text-zinc-300">{format ? format(value) : value}</span>
      </span>
      <input type="range" className="w-full" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}

const STATUS_STYLE: Record<JobStatus, string> = {
  queued: "bg-zinc-700/60 text-zinc-200",
  running: "bg-sky-500/15 text-sky-300 ring-1 ring-sky-500/30",
  succeeded: "bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/30",
  failed: "bg-red-500/15 text-red-300 ring-1 ring-red-500/30",
  cancelled: "bg-zinc-700/60 text-zinc-300",
};

export function StatusBadge({ status }: { status: JobStatus }) {
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium capitalize ${STATUS_STYLE[status]}`}>
      {status === "running" && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-sky-400" />}
      {status}
    </span>
  );
}

export function ProgressBar({ value, active }: { value: number; active?: boolean }) {
  return (
    <div className="h-2 overflow-hidden rounded-full bg-white/5">
      <div
        className={`h-full rounded-full bg-gradient-to-r from-brand-500 to-amber-400 transition-all duration-700 ${active ? "animate-pulse" : ""}`}
        style={{ width: `${Math.max(2, Math.round(value * 100))}%` }}
      />
    </div>
  );
}

export function timeAgo(ts: number): string {
  const s = Math.max(1, Math.round(Date.now() / 1000 - ts));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return new Date(ts * 1000).toLocaleDateString();
}

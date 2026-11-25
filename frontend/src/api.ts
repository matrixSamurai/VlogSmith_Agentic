export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";

export interface BlogSettings {
  title?: string | null;
  context: string;
  tone: string;
  language: string;
  audience?: string | null;
  target_duration: number;
  aspect_ratio: "16:9" | "9:16" | "1:1";
  resolution: "720p" | "1080p";
  fps: 24 | 30;
  caption_style: "bold" | "minimal" | "cinematic" | "pop" | "none";
  subtitles: boolean;
  color_look: string;
  letterbox: boolean;
  framing: "smart_crop" | "fit_blur";
  beat_sync: boolean;
  sfx: boolean;
  channel_name?: string | null;
  caption_position: "bottom" | "center" | "top";
  transition: string;
  transition_duration: number;
  ken_burns: boolean;
  include_title_card: boolean;
  include_outro: boolean;
  outro_text?: string | null;
  max_clip_seconds: number;
  clip_audio: "keep" | "mute";
  clip_audio_volume: number;
  music_source: "generated" | "ai" | "upload" | "none";
  music_mood: string;
  music_volume: number;
  music_upload_id?: string | null;
  voiceover: boolean;
  voice: string;
  generate_article: boolean;
  article_length: "short" | "medium" | "long";
  review_rounds: number;
  quality: "fast" | "balanced" | "high";
}

export interface ServerConfig {
  app_name: string;
  llm_configured: boolean;
  vertex_enabled: boolean;
  defaults: BlogSettings;
  options: { tones: string[]; music_moods: string[]; voices: string[]; transitions: string[]; looks: string[] };
  limits: { max_files: number; max_mb: Record<"image" | "video" | "audio", number> };
  accept: Record<"image" | "video" | "audio", string[]>;
}

export interface MediaRef { id: string; filename: string; kind: "image" | "video" | "audio" }

export interface Job {
  id: string;
  status: JobStatus;
  stage: string;
  progress: number;
  settings: BlogSettings;
  media: MediaRef[];
  outputs: { video?: string; thumbnail?: string; article_html?: string; article_md?: string; package?: string; youtube?: string };
  title?: string | null;
  summary?: string | null;
  duration?: number | null;
  error?: string | null;
  created_at: number;
  started_at?: number | null;
  finished_at?: number | null;
}

export interface JobEvent {
  seq: number;
  ts: number;
  stage: string;
  level: "info" | "warning" | "error" | "success";
  message: string;
  progress?: number | null;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch { /* not JSON */ }
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

export const api = {
  config: () => request<ServerConfig>("/api/config"),
  jobs: () => request<Job[]>("/api/jobs"),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  createJob: (upload_ids: string[], settings: BlogSettings) =>
    request<Job>("/api/jobs", { method: "POST", body: JSON.stringify({ upload_ids, settings }) }),
  cancel: (id: string) => request(`/api/jobs/${id}/cancel`, { method: "POST" }),
  retry: (id: string) => request<Job>(`/api/jobs/${id}/retry`, { method: "POST" }),
  remove: (id: string) => request(`/api/jobs/${id}`, { method: "DELETE" }),
  fileUrl: (id: string, name: string, download = false) =>
    `/api/jobs/${id}/files/${name}${download ? "?download=true" : ""}`,
};

/** Two-step upload: get a ticket (signed URL on GCS), PUT the bytes, then confirm. */
export async function uploadFile(file: File, onProgress: (fraction: number) => void, signal?: AbortSignal): Promise<string> {
  const contentType = file.type || guessType(file.name);
  const ticket = await request<{ upload_id: string; upload_url: string; headers: Record<string, string> }>(
    "/api/uploads",
    { method: "POST", body: JSON.stringify({ filename: file.name, content_type: contentType, size: file.size }) },
  );
  await new Promise<void>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", ticket.upload_url);
    Object.entries(ticket.headers).forEach(([k, v]) => xhr.setRequestHeader(k, v));
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => (xhr.status < 300 ? resolve() : reject(new Error(`Upload failed (${xhr.status})`)));
    xhr.onerror = () => reject(new Error("Network error while uploading"));
    xhr.onabort = () => reject(new DOMException("Upload cancelled", "AbortError"));
    signal?.addEventListener("abort", () => xhr.abort());
    xhr.send(file);
  });
  await request(`/api/uploads/${ticket.upload_id}/complete`, { method: "POST" });
  onProgress(1);
  return ticket.upload_id;
}

const TYPES: Record<string, string> = {
  jpg: "image/jpeg", jpeg: "image/jpeg", png: "image/png", webp: "image/webp", gif: "image/gif", bmp: "image/bmp",
  mp4: "video/mp4", mov: "video/quicktime", m4v: "video/mp4", webm: "video/webm", mkv: "video/x-matroska", avi: "video/x-msvideo",
  mp3: "audio/mpeg", wav: "audio/wav", m4a: "audio/mp4", aac: "audio/aac", ogg: "audio/ogg", flac: "audio/flac",
};

export function guessType(name: string): string {
  return TYPES[name.split(".").pop()?.toLowerCase() || ""] || "application/octet-stream";
}

export function kindOf(file: File): "image" | "video" | "audio" | null {
  const t = file.type || guessType(file.name);
  if (t.startsWith("image/")) return "image";
  if (t.startsWith("video/")) return "video";
  if (t.startsWith("audio/")) return "audio";
  return null;
}

export function subscribe(jobId: string, onEvent: (e: JobEvent) => void, onJob: (j: Job) => void): () => void {
  const es = new EventSource(`/api/jobs/${jobId}/events`);
  es.addEventListener("progress", (e) => onEvent(JSON.parse((e as MessageEvent).data)));
  es.addEventListener("job", (e) => {
    const job: Job = JSON.parse((e as MessageEvent).data);
    onJob(job);
    if (["succeeded", "failed", "cancelled"].includes(job.status)) es.close();
  });
  return () => es.close();
}

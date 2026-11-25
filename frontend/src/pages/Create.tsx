import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  AlertTriangle, Bot, Film, ImagePlus, Loader2, Music2, Sparkles, Trash2, Type, UploadCloud, Video, Wand2,
} from "lucide-react";
import { api, kindOf, uploadFile, type BlogSettings, type ServerConfig } from "../api";
import { Field, Section, Segmented, Slider, Toggle } from "../components/ui";

interface Item {
  key: string;
  file: File;
  kind: "image" | "video" | "audio";
  preview: string;
  progress: number;
  error?: string;
}

const PRESETS: { name: string; hint: string; patch: Partial<BlogSettings> }[] = [
  { name: "YouTube vlog", hint: "16:9 · 90s · voice + subtitles", patch: { aspect_ratio: "16:9", target_duration: 90, caption_style: "bold", transition: "smart", voiceover: true, subtitles: true, color_look: "auto", letterbox: false } },
  { name: "Reel / Short", hint: "9:16 · 30s · pop captions", patch: { aspect_ratio: "9:16", target_duration: 30, caption_style: "pop", transition: "smart", music_mood: "upbeat", voiceover: true, subtitles: true, color_look: "vibrant", letterbox: false } },
  { name: "Cinematic story", hint: "16:9 · 2 min · letterbox", patch: { aspect_ratio: "16:9", target_duration: 120, caption_style: "cinematic", transition: "smart", tone: "cinematic", music_mood: "epic", color_look: "cinematic", letterbox: true } },
  { name: "Clean square", hint: "1:1 · 45s · minimal", patch: { aspect_ratio: "1:1", target_duration: 45, caption_style: "minimal", transition: "smart", color_look: "natural", letterbox: false } },
];

const DRAFT_KEY = "vlogsmith.settings";

export default function CreatePage() {
  const navigate = useNavigate();
  const [config, setConfig] = useState<ServerConfig | null>(null);
  const [settings, setSettings] = useState<BlogSettings | null>(null);
  const [items, setItems] = useState<Item[]>([]);
  const [music, setMusic] = useState<Item | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const musicInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.config().then((cfg) => {
      setConfig(cfg);
      let saved: Partial<BlogSettings> = {};
      try { saved = JSON.parse(localStorage.getItem(DRAFT_KEY) || "{}"); } catch { /* ignore */ }
      setSettings({ ...cfg.defaults, context: "", title: "", ...saved, music_upload_id: null });
    }).catch((e) => setError(`Cannot reach the server: ${e.message}`));
  }, []);

  useEffect(() => {
    if (!settings) return;
    const { context: _c, title: _t, music_upload_id: _m, ...rest } = settings;
    localStorage.setItem(DRAFT_KEY, JSON.stringify(rest));
  }, [settings]);

  useEffect(() => () => items.forEach((i) => URL.revokeObjectURL(i.preview)), []); // eslint-disable-line react-hooks/exhaustive-deps

  const set = <K extends keyof BlogSettings>(key: K, value: BlogSettings[K]) => setSettings((s) => (s ? { ...s, [key]: value } : s));

  const addFiles = useCallback((files: FileList | File[]) => {
    if (!config) return;
    const next: Item[] = [];
    for (const file of Array.from(files)) {
      const kind = kindOf(file);
      if (!kind) continue;
      if (kind === "audio") {
        setMusic({ key: crypto.randomUUID(), file, kind, preview: "", progress: 0 });
        setSettings((s) => (s ? { ...s, music_source: "upload" } : s));
        continue;
      }
      const tooBig = file.size > config.limits.max_mb[kind] * 1024 * 1024;
      next.push({ key: crypto.randomUUID(), file, kind, preview: URL.createObjectURL(file), progress: 0,
        error: tooBig ? `Larger than ${config.limits.max_mb[kind]} MB` : undefined });
    }
    setItems((prev) => [...prev, ...next].slice(0, config.limits.max_files));
  }, [config]);

  const remove = (key: string) => setItems((prev) => {
    const it = prev.find((i) => i.key === key);
    if (it) URL.revokeObjectURL(it.preview);
    return prev.filter((i) => i.key !== key);
  });

  const valid = items.filter((i) => !i.error);
  const counts = useMemo(() => ({
    photos: valid.filter((i) => i.kind === "image").length,
    videos: valid.filter((i) => i.kind === "video").length,
    mb: valid.reduce((a, i) => a + i.file.size, 0) / 1024 / 1024,
  }), [valid]);

  const canSubmit = !!settings && valid.length > 0 && settings.context.trim().length >= 3 && !busy
    && (settings.music_source !== "upload" || !!music);

  async function submit() {
    if (!settings || !canSubmit) return;
    setBusy(true);
    setError(null);
    try {
      const queue = [...valid];
      const ids = new Map<string, string>();
      const worker = async () => {
        for (let it = queue.shift(); it; it = queue.shift()) {
          const current = it;
          const id = await uploadFile(current.file, (p) =>
            setItems((prev) => prev.map((x) => (x.key === current.key ? { ...x, progress: p } : x))));
          ids.set(current.key, id);
        }
      };
      await Promise.all([worker(), worker(), worker()]);
      let musicId: string | null = null;
      if (settings.music_source === "upload" && music) {
        musicId = await uploadFile(music.file, (p) => setMusic((m) => (m ? { ...m, progress: p } : m)));
      }
      const ordered = valid.map((i) => ids.get(i.key)!);
      const job = await api.createJob(ordered, { ...settings, music_upload_id: musicId });
      navigate(`/jobs/${job.id}`);
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  }

  if (!settings || !config) {
    return (
      <div className="grid place-items-center py-32 text-zinc-400">
        {error ? <p className="text-red-300">{error}</p> : <Loader2 className="animate-spin" />}
      </div>
    );
  }

  const uploadProgress = valid.length ? valid.reduce((a, i) => a + i.progress, 0) / valid.length : 0;

  return (
    <div className="space-y-8">
      <div className="max-w-3xl space-y-3">
        <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
          Turn your photos and clips into a <span className="text-brand-400">finished video blog</span>.
        </h1>
        <p className="text-zinc-400">
          Drop your media, describe the story, press create. A team of Gemini agents analyses every shot, plans the story,
          writes captions, narration and an article, then edits the final video with music and transitions.
        </p>
      </div>

      {!config.llm_configured && (
        <div className="flex items-start gap-3 rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 text-sm text-amber-200">
          <AlertTriangle size={18} className="mt-0.5 shrink-0" />
          The server has no Gemini credentials. Set GOOGLE_API_KEY or enable Vertex AI before creating a blog.
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_380px]">
        <div className="space-y-6">
          {/* Media */}
          <div
            className={`card p-5 transition ${dragging ? "border-brand-500/60 bg-brand-500/5" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => { e.preventDefault(); setDragging(false); addFiles(e.dataTransfer.files); }}
          >
            <div className="mb-4 flex items-center justify-between">
              <Section title="Media" icon={<ImagePlus size={16} className="text-brand-400" />}>{null}</Section>
              <span className="text-xs text-zinc-500">
                {counts.photos} photos · {counts.videos} videos · {counts.mb.toFixed(1)} MB
              </span>
            </div>
            <input ref={fileInput} type="file" multiple hidden
              accept={[...config.accept.image, ...config.accept.video, ...config.accept.audio].join(",")}
              onChange={(e) => { if (e.target.files) addFiles(e.target.files); e.target.value = ""; }} />
            {items.length === 0 ? (
              <button type="button" onClick={() => fileInput.current?.click()}
                className="flex w-full flex-col items-center gap-3 rounded-xl border-2 border-dashed border-white/10 py-16 text-zinc-400 transition hover:border-brand-500/50 hover:text-zinc-200">
                <UploadCloud size={36} className="text-brand-400" />
                <span className="text-sm font-medium">Drag photos & videos here, or click to browse</span>
                <span className="text-xs text-zinc-500">JPG, PNG, WEBP, MP4, MOV, WEBM · drop an MP3/WAV to use as music</span>
              </button>
            ) : (
              <div className="grid grid-cols-3 gap-3 sm:grid-cols-4 xl:grid-cols-5">
                {items.map((it, idx) => (
                  <div key={it.key} className="group relative aspect-square overflow-hidden rounded-xl bg-zinc-900 ring-1 ring-white/10">
                    {it.kind === "image"
                      ? <img src={it.preview} alt="" className="h-full w-full object-cover" />
                      : <video src={it.preview} muted className="h-full w-full object-cover" />}
                    <div className="absolute inset-x-0 top-0 flex items-center justify-between p-1.5">
                      <span className="rounded-md bg-black/60 px-1.5 py-0.5 text-[10px] font-semibold">
                        {it.kind === "video" ? <Video size={11} className="inline" /> : idx + 1}
                      </span>
                      {!busy && (
                        <button type="button" onClick={() => remove(it.key)}
                          className="rounded-md bg-black/60 p-1 opacity-0 transition group-hover:opacity-100 hover:bg-red-500">
                          <Trash2 size={12} />
                        </button>
                      )}
                    </div>
                    {it.error && <div className="absolute inset-0 grid place-items-center bg-red-950/80 p-2 text-center text-xs text-red-200">{it.error}</div>}
                    {busy && !it.error && (
                      <div className="absolute inset-x-1.5 bottom-1.5 h-1 overflow-hidden rounded-full bg-black/60">
                        <div className="h-full bg-brand-500 transition-all" style={{ width: `${it.progress * 100}%` }} />
                      </div>
                    )}
                  </div>
                ))}
                {!busy && (
                  <button type="button" onClick={() => fileInput.current?.click()}
                    className="grid aspect-square place-items-center rounded-xl border-2 border-dashed border-white/10 text-zinc-500 hover:border-brand-500/50 hover:text-zinc-200">
                    <ImagePlus size={22} />
                  </button>
                )}
              </div>
            )}
          </div>

          {/* Story */}
          <div className="card space-y-4 p-5">
            <Section title="Story brief" icon={<Type size={16} className="text-brand-400" />}>
              <Field label="What is this blog about?" hint="Where, who, what happened, and how it should feel. The agents use this everywhere.">
                <textarea className="field min-h-32 resize-y" maxLength={4000} value={settings.context}
                  placeholder="e.g. A 3-day rail trip through the Swiss Alps with my partner: sunrise at Lake Geneva, the Glacier Express, fondue in Zermatt. Warm, dreamy, a little funny. End with tips for first-timers."
                  onChange={(e) => set("context", e.target.value)} />
              </Field>
              <div className="grid gap-4 sm:grid-cols-3">
                <Field label="Title (optional)">
                  <input className="field" maxLength={140} value={settings.title || ""} placeholder="Let the AI decide"
                    onChange={(e) => set("title", e.target.value)} />
                </Field>
                <Field label="Language">
                  <input className="field" maxLength={40} value={settings.language} onChange={(e) => set("language", e.target.value)} />
                </Field>
                <Field label="Audience (optional)">
                  <input className="field" maxLength={200} value={settings.audience || ""} placeholder="e.g. budget travellers"
                    onChange={(e) => set("audience", e.target.value)} />
                </Field>
              </div>
              <Field label="Tone">
                <Segmented value={settings.tone} options={config.options.tones} onChange={(v) => set("tone", v)} />
              </Field>
            </Section>
          </div>
        </div>

        {/* Settings */}
        <aside className="space-y-6">
          <div className="card space-y-3 p-5">
            <Section title="Quick presets" icon={<Wand2 size={16} className="text-brand-400" />}>
              <div className="grid grid-cols-2 gap-2">
                {PRESETS.map((p) => (
                  <button key={p.name} type="button" onClick={() => setSettings((s) => (s ? { ...s, ...p.patch } : s))}
                    className="rounded-xl border border-white/10 bg-white/[0.02] p-3 text-left transition hover:border-brand-500/50 hover:bg-brand-500/5">
                    <span className="block text-sm font-medium">{p.name}</span>
                    <span className="block text-xs text-zinc-500">{p.hint}</span>
                  </button>
                ))}
              </div>
            </Section>
          </div>

          <div className="card space-y-5 p-5">
            <Section title="Video" icon={<Film size={16} className="text-brand-400" />}>
              <Field label="Aspect ratio">
                <Segmented value={settings.aspect_ratio} options={["16:9", "9:16", "1:1"] as const} onChange={(v) => set("aspect_ratio", v)}
                  labels={{ "16:9": "16:9 Wide", "9:16": "9:16 Vertical", "1:1": "1:1 Square" }} />
              </Field>
              <Slider label="Target length" value={settings.target_duration} min={20} max={300} step={5}
                onChange={(v) => set("target_duration", v)} format={(v) => (v >= 60 ? `${Math.floor(v / 60)}m ${v % 60}s` : `${v}s`)} />
              <div className="grid grid-cols-2 gap-3">
                <Field label="Resolution">
                  <Segmented value={settings.resolution} options={["720p", "1080p"] as const} onChange={(v) => set("resolution", v)} />
                </Field>
                <Field label="Frame rate">
                  <Segmented value={settings.fps} options={[24, 30] as const} onChange={(v) => set("fps", v)} labels={{ 24: "24 fps", 30: "30 fps" }} />
                </Field>
              </div>
              <Field label="Motion graphics style" hint="Animated titles, captions, location tags, lower thirds and end screen.">
                <Segmented value={settings.caption_style} options={["bold", "pop", "minimal", "cinematic", "none"] as const}
                  onChange={(v) => set("caption_style", v)}
                  labels={{ bold: "Creator", pop: "Pop", minimal: "Clean", cinematic: "Cinematic", none: "No text" }} />
              </Field>
              {settings.caption_style !== "none" && (
                <Field label="Caption position" hint={settings.voiceover && settings.subtitles ? "Captions move to the top while subtitles are on." : undefined}>
                  <Segmented value={settings.caption_position} options={["top", "center", "bottom"] as const} onChange={(v) => set("caption_position", v)} />
                </Field>
              )}
              <Field label="Colour grade">
                <select className="field capitalize" value={settings.color_look} onChange={(e) => set("color_look", e.target.value)}>
                  {(config.options.looks ?? ["auto"]).map((l) => <option key={l} value={l}>{l === "auto" ? "Auto (AI picks)" : l === "bw" ? "Black & white" : l}</option>)}
                </select>
              </Field>
              <Field label="Framing">
                <Segmented value={settings.framing} options={["smart_crop", "fit_blur"] as const} onChange={(v) => set("framing", v)}
                  labels={{ smart_crop: "Smart crop (AI focus)", fit_blur: "Fit + blur" }} />
              </Field>
              <Field label="Transitions">
                <select className="field capitalize" value={settings.transition} onChange={(e) => set("transition", e.target.value)}>
                  {config.options.transitions.map((t) => <option key={t} value={t}>{t === "smart" ? "Smart (editor's choice)" : t.replace("_", " ")}</option>)}
                </select>
              </Field>
              <Slider label="Max seconds per video clip" value={settings.max_clip_seconds} min={2} max={30} step={1}
                onChange={(v) => set("max_clip_seconds", v)} format={(v) => `${v}s`} />
              <Toggle label="Ken Burns motion on photos" checked={settings.ken_burns} onChange={(v) => set("ken_burns", v)} />
              <Toggle label="Cut on the beat" hint="Shot lengths snap to the soundtrack tempo" checked={settings.beat_sync} onChange={(v) => set("beat_sync", v)} />
              <Toggle label="Cinematic letterbox" checked={settings.letterbox} onChange={(v) => set("letterbox", v)} />
              <Toggle label="Cold open with animated title" hint="Starts on your best shot" checked={settings.include_title_card} onChange={(v) => set("include_title_card", v)} />
              <Toggle label="End screen with subscribe animation" checked={settings.include_outro} onChange={(v) => set("include_outro", v)} />
              {settings.include_outro && (
                <input className="field" maxLength={60} placeholder="Channel name (e.g. @ujjwaltravels)" value={settings.channel_name || ""}
                  onChange={(e) => set("channel_name", e.target.value)} />
              )}
              {settings.include_outro && (
                <input className="field" maxLength={120} placeholder="Outro text (e.g. Follow for part 2!)" value={settings.outro_text || ""}
                  onChange={(e) => set("outro_text", e.target.value)} />
              )}
            </Section>
          </div>

          <div className="card space-y-5 p-5">
            <Section title="Sound" icon={<Music2 size={16} className="text-brand-400" />}>
              <Field label="Music" hint={settings.music_source === "ai" && !config.vertex_enabled ? "AI music (Lyria) needs Vertex AI; the built-in composer is used as a fallback." : undefined}>
                <Segmented value={settings.music_source} options={["generated", "ai", "upload", "none"] as const}
                  onChange={(v) => { set("music_source", v); if (v === "upload" && !music) musicInput.current?.click(); }}
                  labels={{ generated: "Composer", ai: "AI (Lyria)", upload: "My track", none: "None" }} />
              </Field>
              <input ref={musicInput} type="file" hidden accept={config.accept.audio.join(",")}
                onChange={(e) => { if (e.target.files?.[0]) addFiles([e.target.files[0]]); e.target.value = ""; }} />
              {settings.music_source === "upload" && (
                <button type="button" onClick={() => musicInput.current?.click()} className="btn-ghost w-full justify-start truncate">
                  <Music2 size={14} /> {music ? music.file.name : "Choose an audio file…"}
                </button>
              )}
              {settings.music_source !== "none" && (
                <>
                  {settings.music_source !== "upload" && (
                    <Field label="Mood">
                      <select className="field capitalize" value={settings.music_mood} onChange={(e) => set("music_mood", e.target.value)}>
                        {config.options.music_moods.map((m) => <option key={m} value={m}>{m === "auto" ? "Auto (match the story)" : m}</option>)}
                      </select>
                    </Field>
                  )}
                  <Slider label="Music volume" value={settings.music_volume} min={0} max={1} step={0.02}
                    onChange={(v) => set("music_volume", v)} format={(v) => `${Math.round(v * 100)}%`} />
                </>
              )}
              <Toggle label="AI voiceover narration" hint="Gemini TTS reads a script written for each section" checked={settings.voiceover} onChange={(v) => set("voiceover", v)} />
              {settings.voiceover && (
                <Toggle label="Animated subtitles" hint="Word-by-word highlight synced to the voice" checked={settings.subtitles} onChange={(v) => set("subtitles", v)} />
              )}
              <Toggle label="Transition sound effects" hint="Whooshes on chapter changes" checked={settings.sfx} onChange={(v) => set("sfx", v)} />
              {settings.voiceover && (
                <Field label="Voice">
                  <select className="field" value={settings.voice} onChange={(e) => set("voice", e.target.value)}>
                    {config.options.voices.map((v) => <option key={v}>{v}</option>)}
                  </select>
                </Field>
              )}
              <Toggle label="Keep original clip audio" checked={settings.clip_audio === "keep"} onChange={(v) => set("clip_audio", v ? "keep" : "mute")} />
            </Section>
          </div>

          <div className="card space-y-5 p-5">
            <Section title="Agents" icon={<Bot size={16} className="text-brand-400" />}>
              <Field label="Quality" hint="Higher quality uses Gemini Pro for planning and writing and renders slower, sharper video.">
                <Segmented value={settings.quality} options={["fast", "balanced", "high"] as const} onChange={(v) => set("quality", v)} />
              </Field>
              <Slider label="Editor review rounds" value={settings.review_rounds} min={0} max={3} step={1} onChange={(v) => set("review_rounds", v)} />
              <Toggle label="Write a blog article too" checked={settings.generate_article} onChange={(v) => set("generate_article", v)} />
              {settings.generate_article && (
                <Field label="Article length">
                  <Segmented value={settings.article_length} options={["short", "medium", "long"] as const} onChange={(v) => set("article_length", v)} />
                </Field>
              )}
            </Section>
          </div>
        </aside>
      </div>

      {/* Sticky action bar */}
      <div className="sticky bottom-4 z-10">
        <div className="card flex flex-col gap-3 border-white/15 bg-zinc-900/90 p-4 shadow-2xl sm:flex-row sm:items-center">
          <div className="min-w-0 flex-1 text-sm">
            {error ? <span className="text-red-300">{error}</span>
              : busy ? <span className="text-zinc-300">Uploading media… {Math.round(uploadProgress * 100)}%</span>
              : valid.length === 0 ? <span className="text-zinc-500">Add at least one photo or video to start.</span>
              : settings.context.trim().length < 3 ? <span className="text-zinc-500">Describe your story in the brief.</span>
              : <span className="text-zinc-300">Ready: {counts.photos} photos, {counts.videos} videos → ~{settings.target_duration}s {settings.aspect_ratio} vlog{settings.generate_article ? " + article" : ""}</span>}
          </div>
          <button type="button" className="btn-primary px-6 py-3 text-base" disabled={!canSubmit} onClick={submit}>
            {busy ? <Loader2 size={18} className="animate-spin" /> : <Sparkles size={18} />}
            {busy ? "Starting…" : "Create my vlog"}
          </button>
        </div>
      </div>
    </div>
  );
}

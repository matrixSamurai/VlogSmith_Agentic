import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";
import { Clapperboard, LayoutGrid, Sparkles } from "lucide-react";
import "./index.css";
import CreatePage from "./pages/Create";
import JobPage from "./pages/Job";
import LibraryPage from "./pages/Library";

function Shell() {
  const link = ({ isActive }: { isActive: boolean }) =>
    `inline-flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm font-medium transition ${
      isActive ? "bg-white/10 text-white" : "text-zinc-400 hover:text-white"
    }`;
  return (
    <div className="min-h-screen bg-[radial-gradient(ellipse_at_top,_rgba(255,90,60,0.12),_transparent_55%)]">
      <header className="sticky top-0 z-20 border-b border-white/5 bg-zinc-950/70 backdrop-blur-xl">
        <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4 sm:px-6">
          <NavLink to="/" className="flex items-center gap-2 font-semibold tracking-tight">
            <span className="grid h-8 w-8 place-items-center rounded-lg bg-brand-500 shadow-lg shadow-brand-500/30">
              <Clapperboard size={17} />
            </span>
            Vlogsmith
          </NavLink>
          <nav className="flex gap-1">
            <NavLink to="/" end className={link}><Sparkles size={15} /> Create</NavLink>
            <NavLink to="/library" className={link}><LayoutGrid size={15} /> Library</NavLink>
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
        <Routes>
          <Route path="/" element={<CreatePage />} />
          <Route path="/library" element={<LibraryPage />} />
          <Route path="/jobs/:id" element={<JobPage />} />
        </Routes>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <Shell />
    </BrowserRouter>
  </StrictMode>,
);

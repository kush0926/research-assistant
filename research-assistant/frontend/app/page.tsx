"use client";
import { useCallback, useEffect, useState } from "react";
import { BookOpen, CaretDown, SignOut, UploadSimple, ArrowClockwise } from "@phosphor-icons/react";
import { api, ApiError, token } from "@/lib/api";
import Auth from "@/components/Auth";
import Library from "@/components/Library";
import Ask from "@/components/Ask";
import Compare from "@/components/Compare";
import Gaps from "@/components/Gaps";
import Discover from "@/components/Discover";
import Review from "@/components/Review";

const TABS = ["Library", "Ask", "Compare", "Gaps", "Discover", "Review"] as const;
type Tab = (typeof TABS)[number];
const PAGE: Record<Tab, { title: string; description: string }> = {
  Library: { title: "Your research library", description: "Bring your papers together. Follow every finding back to its source." },
  Ask: { title: "Ask your papers", description: "Answers grounded in your selected research." },
  Compare: { title: "Compare papers", description: "See where studies agree, differ, and leave questions open." },
  Gaps: { title: "Candidate research gaps", description: "Explore what is missing in your corpus, with the evidence in view." },
  Discover: { title: "Discover connections", description: "Follow the relationships between papers, methods, and datasets." },
  Review: { title: "Literature review", description: "Turn selected research into a structured draft with traceable references." },
};

function AppLoading() {
  return <div className="app-loading" role="status" aria-label="Loading your research workspace">
    <div className="app-loading-brand"><BookOpen size={30} /><span>Research Assistant</span></div>
    <div className="app-loading-body"><div className="skeleton loading-title" /><div className="skeleton loading-subtitle" />
      <div className="loading-workspace"><div className="skeleton" /><div className="skeleton" /><div className="skeleton" /></div>
    </div>
    <span className="sr-only">Loading your research workspace…</span>
  </div>;
}

export default function Home() {
  const [user, setUser] = useState<any>(null); const [checked, setChecked] = useState(false);
  const [tab, setTab] = useState<Tab>("Library");
  const [papers, setPapers] = useState<any[]>([]); const [selected, setSelected] = useState<number[]>([]); const [gapId, setGapId] = useState<number | null>(null);
  const [loadError, setLoadError] = useState("");
  const [uploadPending, setUploadPending] = useState(false);

  const reload = useCallback(async () => {
    try {
      const result = await api("/papers"); setPapers(result.papers); setLoadError("");
      const ready = new Set(result.papers.filter((paper: any) => paper.status === "ready").map((paper: any) => paper.id));
      setSelected((previous) => previous.filter((id) => ready.has(id)));
    }
    catch (error) { setLoadError(error instanceof Error ? error.message : "Unable to load your papers."); }
  }, []);
  const loadUser = useCallback(async () => {
    try { const current = await api("/auth/me"); setUser(current); setLoadError(""); await reload(); }
    catch (error) {
      if (error instanceof ApiError && error.status === 401) { token.clear(); setUser(null); }
      else setLoadError("The research service could not be reached. Check your connection and try again.");
    } finally { setChecked(true); }
  }, [reload]);
  useEffect(() => { if (!token.get()) { setChecked(true); return; } void loadUser(); }, [loadUser]);
  useEffect(() => {
    if (!user || !papers.some((paper) => paper.status === "processing")) return;
    const timer = setInterval(() => void reload(), 3000);
    return () => clearInterval(timer);
  }, [papers, user, reload]);
  useEffect(() => {
    if (!uploadPending || tab !== "Library") return;
    const input = document.getElementById("paper-upload-input") as HTMLInputElement | null;
    input?.click();
    setUploadPending(false);
  }, [uploadPending, tab]);
  const openUpload = () => { setTab("Library"); setUploadPending(true); };
  const signOut = () => { token.clear(); setUser(null); setPapers([]); setSelected([]); setTab("Library"); setLoadError(""); };

  if (!checked) return <AppLoading />;
  if (!user) return <>
    {loadError && <div className="connection-banner" role="alert"><span>{loadError}</span><button className="btn small ghost" onClick={() => void loadUser()}><ArrowClockwise size={16} />Retry</button></div>}
    <Auth onDone={() => { setChecked(false); void loadUser(); }} />
  </>;
  const initials = (user.name || "Researcher").trim().split(/\s+/).slice(0, 2).map((part: string) => part[0]).join("").toUpperCase();
  return (
    <div className="app-shell">
      <a href="#main-content" className="skip-link">Skip to content</a>
      <header className="app-header">
        <div className="topbar-inner">
          <button className="brand" onClick={() => setTab("Library")} aria-label="Research Assistant home"><BookOpen size={32} weight="regular" aria-hidden="true" /><span>Research Assistant</span></button>
          <nav className="nav" aria-label="Sections">{TABS.map((item) => <button key={item} aria-current={tab === item ? "page" : undefined} onClick={() => setTab(item)}>{item}</button>)}</nav>
          <details className="account-menu"><summary aria-label={`Account: ${user.name}`}><span className="avatar">{initials}</span><CaretDown size={14} aria-hidden="true" /></summary>
            <div className="account-popover"><strong>{user.name}</strong><span>{user.email}</span><button onClick={signOut}><SignOut size={17} />Sign out</button></div>
          </details>
        </div>
      </header>
      <main id="main-content" className={`main ${tab === "Ask" ? "main-workspace" : "main-standard"}`}>
        <div className="page-heading"><div><h1>{PAGE[tab].title}</h1><p>{PAGE[tab].description}</p></div>
          <button className="btn upload-action" onClick={openUpload}><UploadSimple size={19} aria-hidden="true" /><span>Upload PDFs</span></button>
        </div>
        {loadError && <div className="notice error" role="alert"><span>{loadError}</span><button className="btn ghost small" onClick={() => void reload()}><ArrowClockwise size={16} />Retry</button></div>}
        {tab === "Library" && <Library papers={papers} reload={reload} selected={selected} setSelected={setSelected} />}
        {tab === "Ask" && <Ask selected={selected} papers={papers} setSelected={setSelected} onLibrary={() => setTab("Library")} />}
        {tab === "Compare" && <Compare selected={selected} />}
        {tab === "Gaps" && <Gaps selected={selected} onGap={(id) => { setGapId(id); setTab("Discover"); }} />}
        {tab === "Discover" && <Discover papers={papers} gapId={gapId} />}
        {tab === "Review" && <Review selected={selected} />}
      </main>
    </div>
  );
}

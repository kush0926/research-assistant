"use client";
import { useCallback, useEffect, useState } from "react";
import { api, token } from "@/lib/api";
import Auth from "@/components/Auth";
import Library from "@/components/Library";
import Ask from "@/components/Ask";
import Compare from "@/components/Compare";
import Gaps from "@/components/Gaps";
import Discover from "@/components/Discover";
import Review from "@/components/Review";

const TABS = ["Library", "Ask", "Compare", "Gaps", "Discover", "Review"] as const;

export default function Home() {
  const [user, setUser] = useState<any>(null); const [checked, setChecked] = useState(false);
  const [tab, setTab] = useState<(typeof TABS)[number]>("Library");
  const [papers, setPapers] = useState<any[]>([]); const [selected, setSelected] = useState<number[]>([]); const [gapId, setGapId] = useState<number | null>(null);

  const reload = useCallback(() => api("/papers").then((r) => setPapers(r.papers)).catch(() => {}), []);
  useEffect(() => { if (!token.get()) { setChecked(true); return; }
    api("/auth/me").then((u) => { setUser(u); reload(); }).catch(() => token.clear()).finally(() => setChecked(true)); }, []);

  if (!checked) return null;
  if (!user) return <Auth onDone={() => { setChecked(false); api("/auth/me").then((u) => { setUser(u); reload(); setChecked(true); }); }} />;
  return (
    <div className="shell">
      <aside className="side">
        <div className="brand">Research Assistant<small>Evidence-grounded literature analysis</small></div>
        <nav className="nav" aria-label="Sections">{TABS.map((t) => <button key={t} aria-current={tab === t ? "page" : undefined} onClick={() => setTab(t)}>{t}</button>)}</nav>
        <div className="foot">{user.name}<br /><button className="btn ghost small" style={{ marginTop: ".4rem" }} onClick={() => { token.clear(); setUser(null); }}>Sign out</button></div>
      </aside>
      <main className="main">
        {tab === "Library" && <Library papers={papers} reload={reload} selected={selected} setSelected={setSelected} />}
        {tab === "Ask" && <Ask selected={selected} />}
        {tab === "Compare" && <Compare selected={selected} />}
        {tab === "Gaps" && <Gaps selected={selected} onGap={(id) => { setGapId(id); setTab("Discover"); }} />}
        {tab === "Discover" && <Discover papers={papers} gapId={gapId} />}
        {tab === "Review" && <Review selected={selected} />}
      </main>
    </div>
  );
}

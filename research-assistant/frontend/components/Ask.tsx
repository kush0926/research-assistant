"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import Passage from "./Passage";

export default function Ask({ selected }: { selected: number[] }) {
  const [q, setQ] = useState(""); const [auto, setAuto] = useState(false);
  const [res, setRes] = useState<any>(null); const [route, setRoute] = useState<any>(null);
  const [busy, setBusy] = useState(false); const [err, setErr] = useState(""); const [chunk, setChunk] = useState<number | null>(null);

  async function go(e: React.FormEvent) {
    e.preventDefault(); setErr(""); setBusy(true); setRes(null); setRoute(null);
    try {
      const ids = selected.length ? selected : undefined;
      if (auto) { const r = await api("/agent", { body: { request: q, paper_ids: ids } }); setRoute(r); if (r.route === "query") setRes(r.result); }
      else setRes(await api("/query", { body: { question: q, paper_ids: ids } }));
    } catch (x: any) { setErr(x.message); } finally { setBusy(false); }
  }
  const cite = (m: number) => res.citations.find((c: any) => c.marker === m);
  function renderClaim(cl: any, i: number) {
    const parts = cl.text.split(/(\[\d+\])/g);
    return <span key={i} className={cl.verified ? "" : "unverified"} title={cl.verified ? "" : "Not verified against the cited passage"}>
      {parts.map((p: string, j: number) => { const m = /^\[(\d+)\]$/.exec(p);
        if (!m) return p; const c = cite(+m[1]);
        return c ? <button key={j} className="cite" onClick={() => setChunk(c.chunk_id)} aria-label={`Open source ${m[1]}`}>{m[1]}</button> : null; })}{" "}
    </span>;
  }
  return (<>
    <h1>Ask your papers</h1>
    <form className="card" onSubmit={go}>
      <textarea value={q} onChange={(e) => setQ(e.target.value)} placeholder="Which papers used deep learning, and how did accuracy compare?" required minLength={3} />
      <div className="row" style={{ marginTop: ".6rem" }}>
        <button className="btn" disabled={busy}>{busy ? "Thinking…" : "Ask"}</button>
        <label className="muted"><input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} /> Route automatically (search, compare, gaps, review)</label>
        <span className="muted">Scope: {selected.length ? `${selected.length} selected paper(s)` : "all processed papers"}</span>
      </div>
      {err && <p className="err">{err}</p>}
    </form>
    {route && <div className="card"><strong>Route: {route.route}</strong><ul className="muted">{route.steps.map((s: string, i: number) => <li key={i}>{s}</li>)}</ul>
      {route.route !== "query" && <details><summary>Raw result</summary><pre style={{ whiteSpace: "pre-wrap", fontSize: ".8rem" }}>{JSON.stringify(route.result, null, 2).slice(0, 6000)}</pre></details>}</div>}
    {res && <div className="card">
      <p className="prose">{res.claims.length ? res.claims.map(renderClaim) : res.answer}</p>
      <p className="muted">{res.mode === "refusal" ? "No supporting passages found, so nothing was generated." :
        `${res.citations.length} source(s) · ${res.unverified ? `${res.unverified} claim(s) underlined red could not be verified` : "all claims verified against their sources"} · ${res.mode} mode`}</p>
      {res.citations.map((c: any) => <div key={c.marker} className="quote"><strong>[{c.marker}] {c.paper_title}</strong> <span className="muted">({c.section}, p.{c.page})</span><br />{c.text.slice(0, 260)}…</div>)}
    </div>}
    {chunk && <Passage chunkId={chunk} onClose={() => setChunk(null)} />}
  </>);
}

"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import Passage from "./Passage";

const NAMES = ["novelty", "sparsity", "importance", "feasibility", "temporal", "evidence", "opportunity", "contradiction"];
const TYPE_LABEL: any = { limitation: "Limitation", future_work: "Future work", contradiction: "Conflicting result", sparsity: "Sparsity" };

export default function Gaps({ selected, onGap }: { selected: number[]; onGap?: (id: number) => void }) {
  const [topic, setTopic] = useState(""); const [res, setRes] = useState<any>(null);
  const [w, setW] = useState<Record<string, number>>(Object.fromEntries(NAMES.map((n) => [n, 0.125])));
  const [busy, setBusy] = useState(false); const [err, setErr] = useState(""); const [chunk, setChunk] = useState<number | null>(null);
  async function go() {
    setErr(""); setBusy(true);
    try { setRes(await api("/research-gaps", { body: { topic: topic || null, paper_ids: selected.length ? selected : null, weights: w } })); }
    catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  }
  return (<>
    <h1>Research gaps</h1>
    <div className="card">
      <div className="row"><input className="grow" type="text" placeholder="Optional topic focus, e.g. Parkinson's voice detection" value={topic} onChange={(e) => setTopic(e.target.value)} />
        <button className="btn" onClick={go} disabled={busy}>{busy ? "Analysing…" : "Find candidate gaps"}</button></div>
      <p className="muted" style={{ marginTop: ".5rem" }}>Scope: {selected.length ? `${selected.length} selected paper(s)` : "all processed papers"}. Needs at least 3.</p>
      <details><summary>Scoring weights (hand-set; normalised to sum to 1)</summary>
        <div className="bars">{NAMES.map((n) => <label key={n} className="bar"><span>{n}</span>
          <input type="range" min={0} max={1} step={0.025} value={w[n]} onChange={(e) => setW({ ...w, [n]: +e.target.value })} /><span>{w[n].toFixed(2)}</span></label>)}</div></details>
      {err && <p className="err">{err}</p>}
    </div>
    {res?.error && <div className="empty">{res.error}</div>}
    {res && !res.error && <>
      <p className="muted">{res.disclaimer} Corpus: {res.corpus_size} papers.</p>
      {res.gaps.map((g: any, i: number) => (
        <div className="card gap" key={g.id}>
          <div className="score">{g.score.toFixed(2)}<small>#{i + 1} · {g.confidence}</small></div>
          <div>
            <div className={`tier ${g.tier}`}>{g.tier === "evidence_backed" ? "Evidence-backed gap" : "Under-explored combination (lower confidence)"}{g.trend?.kind === "explored_early_then_abandoned" ? " · explored early, then dropped" : ""}</div>
            <h3>{g.description}</h3>
            {g.rationale && <p className="prose">{g.rationale}</p>}
            <div className="bars">{NAMES.map((n) => <div className="bar" key={n}><span>{n}</span><i><b style={{ width: `${Math.round(g.components[n] * 100)}%` }} /></i><span>{g.components[n].toFixed(2)}</span></div>)}</div>
            {g.evidence.map((e: any, j: number) => e.type === "sparsity"
              ? <p key={j} className="muted">{e.text}</p>
              : <div className="quote" key={j} style={{ cursor: e.chunk_id ? "pointer" : "default" }} onClick={() => e.chunk_id && setChunk(e.chunk_id)}>
                  <span className="mark">{e.text}</span><div className="muted">{TYPE_LABEL[e.type]} — {e.paper_title}</div></div>)}
            {onGap && g.id && <button className="btn ghost small" onClick={() => onGap(g.id)}>Papers to read for this gap</button>}
          </div>
        </div>))}
      {res.matrix && <div className="card scroll"><h3>Problem × dataset coverage</h3>
        <table className="heat"><thead><tr><th></th>{res.matrix.problem_dataset.cols.map((c: string) => <th key={c}>{c}</th>)}</tr></thead>
          <tbody>{res.matrix.problem_dataset.rows.map((r: string, i: number) => <tr key={r}><th>{r}</th>
            {res.matrix.problem_dataset.cells[i].map((v: number, j: number) => <td key={j} style={{ background: v ? `rgba(14,107,102,${Math.min(0.15 + v * 0.18, 0.85)})` : "transparent", color: v > 3 ? "#fff" : "inherit" }}>{v}</td>)}</tr>)}</tbody></table></div>}
      {res.trends?.years?.length > 0 && <div className="card scroll"><h3>Method usage by year</h3>
        <table><thead><tr><th>Method</th>{res.trends.years.map((y: number) => <th key={y}>{y}</th>)}</tr></thead>
          <tbody>{Object.entries(res.trends.series).map(([m, v]: any) => <tr key={m}><th>{m}</th>{v.map((n: number, i: number) => <td key={i}>{n || ""}</td>)}</tr>)}</tbody></table></div>}
    </>}
    {chunk && <Passage chunkId={chunk} onClose={() => setChunk(null)} />}
  </>);
}

"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import Passage from "./Passage";

const LABEL: any = { problem: "Problem", dataset: "Dataset", methodology: "Methodology", algorithm: "Algorithm / model", metric: "Metrics", results: "Best results", limitations: "Limitations", future_work: "Future work" };

export default function Compare({ selected }: { selected: number[] }) {
  const [res, setRes] = useState<any>(null); const [err, setErr] = useState(""); const [busy, setBusy] = useState(false); const [chunk, setChunk] = useState<number | null>(null);
  async function go() { setErr(""); setBusy(true); try { setRes(await api("/compare", { body: { paper_ids: selected } })); } catch (e: any) { setErr(e.message); } finally { setBusy(false); } }
  const ok = selected.length >= 2 && selected.length <= 5;
  return (<>
    <h1>Compare papers</h1>
    <div className="card row">
      <button className="btn" onClick={go} disabled={!ok || busy}>{busy ? "Comparing…" : `Compare ${selected.length} selected`}</button>
      <span className="muted">{ok ? "Click any cell to see the source sentence." : "Select 2–5 processed papers in the Library."}</span>
    </div>
    {err && <p className="err">{err}</p>}
    {res && <>
      <div className="card scroll"><table>
        <thead><tr><th></th>{res.table.rows.map((r: any) => <th key={r.paper_id}>{r.title}<div className="muted">{r.year}</div></th>)}</tr></thead>
        <tbody>{res.table.dimensions.map((d: string) => <tr key={d}><th>{LABEL[d]}</th>
          {res.table.rows.map((r: any) => <td key={r.paper_id} className={r[d].chunk_id ? "click" : ""} onClick={() => r[d].chunk_id && setChunk(r[d].chunk_id)}>{r[d].text}</td>)}</tr>)}</tbody></table></div>
      <div className="card"><h3>Synthesis</h3><p className="prose">{res.narrative}</p></div>
      {res.contradictions.length > 0 && <div className="card"><h3>Possible contradictions (needs human review)</h3>
        {res.contradictions.map((c: any, i: number) => <div key={i}><strong>{c.metric} on {c.dataset} ({c.method || "same method"}): differs by {c.delta} points</strong>
          {c.papers.map((p: any) => <div key={p.paper_id} className="quote" style={{ cursor: "pointer" }} onClick={() => setChunk(p.chunk_id)}>{p.sentence}</div>)}</div>)}</div>}
    </>}
    {chunk && <Passage chunkId={chunk} onClose={() => setChunk(null)} />}
  </>);
}

"use client";
import { useState } from "react";
import { ArrowsLeftRight, ArrowRight, CheckCircle, FileText, WarningCircle } from "@phosphor-icons/react";
import { api } from "@/lib/api";
import Markdown from "./Markdown";
import Passage from "./Passage";
import "./Analysis.css";

const LABEL: Record<string, string> = { problem: "Research problem", dataset: "Dataset", methodology: "Methodology", algorithm: "Algorithm / model", metric: "Metrics", results: "Reported results", limitations: "Limitations", future_work: "Future work" };

export default function Compare({ selected }: { selected: number[] }) {
  const [res, setRes] = useState<any>(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [chunk, setChunk] = useState<number | null>(null);
  async function go() {
    setErr(""); setBusy(true);
    try { setRes(await api("/compare", { body: { paper_ids: selected } })); }
    catch (e: any) { setErr(e.message); }
    finally { setBusy(false); }
  }
  const ok = selected.length >= 2 && selected.length <= 5;
  return <div className="analysis-screen compare-screen">
    <section className="analysis-controls" aria-label="Comparison settings">
      <div className="analysis-control-copy"><span className="analysis-eyebrow">Side by side</span><p>{ok ? `${selected.length} selected papers, compared across eight research dimensions.` : "Select 2–5 ready papers from your library to compare."}</p></div>
      <button className="btn" onClick={go} disabled={!ok || busy}>{busy ? "Comparing papers…" : "Compare papers"}{!busy && <ArrowRight size={17} />}</button>
    </section>
    {err && <div className="err analysis-error" role="alert"><WarningCircle size={18} />{err}</div>}
    {busy && <div className="analysis-loading" role="status"><span className="analysis-loading-line" /><span className="analysis-loading-line" /><span className="analysis-loading-line" /><p>Building the table and checking the synthesis against extracted facts…</p></div>}
    {!res && !busy && <div className="analysis-empty"><span className="analysis-empty-icon"><ArrowsLeftRight size={30} weight="light" /></span><h2>Find the connections between papers</h2><p>Compare their methods, datasets, reported results, and open questions. Each available source opens the original passage.</p><span className="analysis-empty-note">2–5 papers · 8 dimensions · Source passages</span></div>}
    {res && !busy && <>
      <section className="analysis-table-panel" aria-labelledby="comparison-table-heading">
        <div className="analysis-section-header"><div><h2 id="comparison-table-heading">Paper comparison</h2><p className="muted">Open a numbered source to inspect the extracted evidence. Missing fields appear as —.</p></div><span className="analysis-count">{res.table.rows.length} papers</span></div>
        <div className="analysis-table-scroll" tabIndex={0} role="region" aria-label="Scrollable comparison table">
          <table className="comparison-table"><caption className="sr-only">Selected research papers compared by research problem, dataset, methodology, model, metric, results, limitations, and future work.</caption>
            <thead><tr><th scope="col">Research dimension</th>{res.table.rows.map((r: any, i: number) => <th scope="col" key={r.paper_id}><span className="analysis-paper-index">[{i + 1}] <FileText size={15} /></span><span className="comparison-paper-title">{r.title}</span><span className="comparison-paper-year">{r.year || "Year not available"}</span></th>)}</tr></thead>
            <tbody>{res.table.dimensions.map((d: string) => <tr key={d}><th scope="row">{LABEL[d] || d}</th>{res.table.rows.map((r: any) => {
              const cell = r[d] || { text: "—" };
              const sources: number[] = cell.chunk_ids || (cell.chunk_id ? [cell.chunk_id] : []);
              return <td key={r.paper_id}><p className={cell.text === "—" ? "comparison-missing" : ""}>{cell.text}</p>{sources.length > 0 && <div className="comparison-cell-sources"><span>Source{sources.length > 1 ? "s" : ""}</span>{sources.map((id, i) => <button key={id} className="analysis-source-number" aria-label={`Open source ${i + 1} for ${LABEL[d] || d} in ${r.title}`} onClick={() => setChunk(id)}>{i + 1}</button>)}</div>}</td>;
            })}</tr>)}</tbody>
          </table>
        </div>
      </section>
      <section className="analysis-reading-panel" aria-labelledby="synthesis-heading"><div className="analysis-section-header"><h2 id="synthesis-heading">Comparative synthesis</h2>{res.verification?.status === "passed" && <span className="analysis-check"><CheckCircle weight="fill" size={16} />Extracted facts checked</span>}</div>
        {res.verification?.status === "fallback" && <p className="analysis-notice">Automatic checks could not verify every generated claim. This summary uses the extracted facts.</p>}
        {res.mode === "template" && res.verification?.status !== "fallback" && <p className="analysis-notice">A factual summary of the extracted fields. Open source passages to check the original context.</p>}
        <Markdown>{res.narrative || "No synthesis was returned."}</Markdown>
      </section>
      {res.contradictions?.length > 0 && <section className="analysis-reading-panel" aria-labelledby="conflicts-heading"><div className="analysis-section-header"><h2 id="conflicts-heading">Possible conflicts in reported results</h2><WarningCircle size={19} /></div><p className="muted">Matching extracted dataset, metric, and method; study settings may differ.</p>
        {res.contradictions.map((c: any, i: number) => <article className="comparison-conflict" key={i}><h3>{c.metric} on {c.dataset}</h3><p className="muted">{c.method || "Same extracted method"} · {c.delta} points difference</p>{c.papers.map((p: any) => p.chunk_id ? <button className="analysis-evidence-source" key={p.paper_id} onClick={() => setChunk(p.chunk_id)}><span>{p.sentence}</span><span className="analysis-evidence-footer">Open supporting passage <ArrowRight size={15} /></span></button> : <blockquote key={p.paper_id}>{p.sentence}</blockquote>)}</article>)}
      </section>}
    </>}
    {chunk !== null && <Passage chunkId={chunk} onClose={() => setChunk(null)} />}
  </div>;
}

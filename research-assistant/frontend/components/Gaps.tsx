"use client";
import { useState } from "react";
import { ArrowRight, ChartBar, MagnifyingGlass, SlidersHorizontal, Target, WarningCircle } from "@phosphor-icons/react";
import { api } from "@/lib/api";
import Markdown from "./Markdown";
import Passage from "./Passage";
import "./Analysis.css";

const NAMES = ["novelty", "sparsity", "importance", "feasibility", "temporal", "evidence", "opportunity", "contradiction"];
const TYPE_LABEL: Record<string, string> = { limitation: "Limitation", future_work: "Future work", contradiction: "Conflicting result", sparsity: "Corpus coverage", related_context: "Related context" };

export default function Gaps({ selected, onGap }: { selected: number[]; onGap?: (id: number) => void }) {
  const [topic, setTopic] = useState("");
  const [res, setRes] = useState<any>(null);
  const [w, setW] = useState<Record<string, number>>(Object.fromEntries(NAMES.map((n) => [n, 0.125])));
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [chunk, setChunk] = useState<number | null>(null);
  const [coverageKind, setCoverageKind] = useState("problem_dataset");
  async function go(e: React.FormEvent) {
    e.preventDefault(); setErr(""); setBusy(true);
    try { setRes(await api("/research-gaps", { body: { topic: topic || null, paper_ids: selected.length ? selected : null, weights: w } })); }
    catch (e: any) { setErr(e.message); }
    finally { setBusy(false); }
  }
  const enough = selected.length === 0 || selected.length >= 3;
  const coverage = res?.matrix?.[coverageKind];
  return <div className="analysis-screen gaps-screen">
    <form className="gap-query-panel" onSubmit={go}>
      <label htmlFor="gap-topic" className="analysis-eyebrow">Focus your exploration <span className="gap-optional">Optional</span></label>
      <div className="gap-topic-row"><div className="gap-topic-input"><MagnifyingGlass size={19} /><input id="gap-topic" type="text" placeholder="e.g. Parkinson’s voice detection" value={topic} onChange={(e) => setTopic(e.target.value)} /></div><button className="btn" disabled={busy || !enough}>{busy ? "Analysing papers…" : "Find candidate gaps"}{!busy && <ArrowRight size={17} />}</button></div>
      <p className="muted gap-scope">Scope: {selected.length ? `${selected.length} selected paper${selected.length === 1 ? "" : "s"}` : "all ready papers"} · At least 3 papers required.</p>
      <details className="gap-weight-settings"><summary><SlidersHorizontal size={17} />Adjust scoring weights</summary><p className="muted">Weights are normalised to a total of 1 when the analysis runs.</p><div className="gap-weight-grid">{NAMES.map((n) => <label className="gap-weight" key={n}><span>{n}</span><input type="range" min={0} max={1} step={0.025} value={w[n]} onChange={(e) => setW({ ...w, [n]: +e.target.value })} /><output>{w[n].toFixed(2)}</output></label>)}</div></details>
    </form>
    {err && <div className="err analysis-error" role="alert"><WarningCircle size={18} />{err}</div>}
    {busy && <div className="analysis-loading" role="status"><span className="analysis-loading-line" /><span className="analysis-loading-line" /><span className="analysis-loading-line" /><p>Looking for sparse combinations, repeated limitations, and supporting evidence…</p></div>}
    {!res && !busy && <div className="analysis-empty"><span className="analysis-empty-icon"><Target size={31} weight="light" /></span><h2>Explore the questions your corpus leaves open</h2><p>Find patterns in methods, datasets, limitations, and future work. Ranked candidates stay connected to their supporting evidence.</p><span className="analysis-empty-note">Corpus-level candidates · Evidence passages · Adjustable scoring</span></div>}
    {res?.error && !busy && <div className="analysis-empty"><WarningCircle size={27} /><h2>More research context is needed</h2><p>{res.error}</p></div>}
    {res && !res.error && !busy && <>
      <div className="gap-results-intro"><div><h2>{res.gaps.length} candidate{res.gaps.length === 1 ? "" : "s"} to explore</h2><p className="muted">Ranked within a corpus of {res.corpus_size} papers.</p></div><span className="analysis-count">{res.corpus_size} papers</span></div>
      <p className="analysis-notice gap-disclaimer"><WarningCircle size={18} /><span>{res.disclaimer || "Candidates are scored within this corpus. Check the sources and wider literature before treating them as research gaps."} Global novelty is unconfirmed.</span></p>
      {res.gaps.length === 0 && <div className="analysis-empty"><h2>No candidates found for this scope</h2><p>Try a broader topic or include more ready papers.</p></div>}
      <div className="gap-candidates">{res.gaps.map((g: any, i: number) => <article className="gap-candidate" key={g.id || i}>
        <div className="gap-candidate-rank"><span className="gap-rank-label">Candidate {String(i + 1).padStart(2, "0")}</span><strong>{Number(g.score).toFixed(2)}</strong><span>Weighted score</span><span className="gap-confidence">{g.confidence} confidence</span></div>
        <div className="gap-candidate-main"><div className={`gap-tier ${g.tier}`}><span />{g.tier === "evidence_backed" ? "Evidence-backed candidate" : "Under-explored combination"}</div><h3>{g.description}</h3>
          {g.trend?.kind === "explored_early_then_abandoned" && <p className="gap-trend-note">Earlier studies appear in this corpus, with no recent use in its extracted records.</p>}
          {g.rationale && <Markdown className="gap-rationale">{g.rationale}</Markdown>}
          <details className="gap-score-details"><summary><ChartBar size={16} />Score breakdown</summary><div className="gap-component-grid">{NAMES.map((n) => <div className="gap-component" key={n}><span>{n}</span><meter min={0} max={1} value={g.components?.[n] || 0} aria-label={`${n} component`} /><span>{Number(g.components?.[n] || 0).toFixed(2)}</span></div>)}</div></details>
          {g.evidence?.length > 0 && <div className="gap-evidence-list"><h4>Evidence & context</h4>{g.evidence.map((e: any, j: number) => e.type === "sparsity" ? <div key={j} className="gap-coverage-note"><ChartBar size={17} /><p>{e.text}</p></div> : <div className={`gap-evidence ${e.type === "related_context" ? "related-context" : ""}`} key={j}>{e.chunk_id ? <button className="analysis-evidence-source" onClick={() => setChunk(e.chunk_id)}><span className="gap-evidence-kind">{TYPE_LABEL[e.type] || e.type}</span><span className="gap-evidence-text">{e.text}</span><span className="analysis-evidence-footer"><span>{e.paper_title}{e.page ? ` · Page ${e.page}` : ""}</span><ArrowRight size={16} /></span></button> : <blockquote><span className="gap-evidence-kind">{TYPE_LABEL[e.type] || e.type}</span><p>{e.text}</p><span className="muted">{e.paper_title}</span></blockquote>}{e.type === "related_context" && <p className="gap-context-note">Background context; does not establish this specific gap.</p>}</div>)}</div>}
          {onGap && g.id && <button className="btn ghost small gap-reading-button" onClick={() => onGap(g.id)}>Papers to read for this candidate <ArrowRight size={16} /></button>}
        </div>
      </article>)}</div>
      {res.matrix && <section className="analysis-table-panel gap-coverage-panel" aria-labelledby="coverage-heading"><div className="analysis-section-header"><div><h2 id="coverage-heading">Corpus coverage</h2><p className="muted">Number of papers linking each research problem to a dataset or method.</p></div><div className="analysis-segmented" aria-label="Coverage dimension"><button aria-pressed={coverageKind === "problem_dataset"} onClick={() => setCoverageKind("problem_dataset")}>Datasets</button><button aria-pressed={coverageKind === "problem_method"} onClick={() => setCoverageKind("problem_method")}>Methods</button></div></div>
        {coverage?.rows?.length > 0 && coverage?.cols?.length > 0 ? <div className="analysis-table-scroll" tabIndex={0} role="region" aria-label="Scrollable corpus coverage table"><table className="analysis-data-table gap-heatmap"><caption className="sr-only">Paper counts by research problem and {coverageKind === "problem_dataset" ? "dataset" : "method"}.</caption><thead><tr><th scope="col">Research problem</th>{coverage.cols.map((c: string) => <th scope="col" key={c}>{c}</th>)}</tr></thead><tbody>{coverage.rows.map((r: string, i: number) => <tr key={r}><th scope="row">{r}</th>{coverage.cells[i].map((v: number, j: number) => <td key={j}><span className={v ? "gap-heat-value populated" : "gap-heat-value"} style={v ? { backgroundColor: `rgba(14,107,102,${Math.min(0.1 + v * 0.09, 0.75)})`, color: v >= 5 ? "#fff" : "var(--ink)" } : undefined} title={`${v} paper${v === 1 ? "" : "s"}`}>{v}</span></td>)}</tr>)}</tbody></table></div> : <div className="analysis-inline-empty">No {coverageKind === "problem_dataset" ? "dataset" : "method"} coverage was extracted for this scope.</div>}
      </section>}
      {res.trends?.years?.length > 0 && <section className="analysis-table-panel" aria-labelledby="method-trends-heading"><div className="analysis-section-header"><div><h2 id="method-trends-heading">Method usage over time</h2><p className="muted">Extracted paper counts by publication year within this corpus.</p></div></div><div className="analysis-table-scroll" tabIndex={0} role="region" aria-label="Scrollable method usage table"><table className="analysis-data-table"><thead><tr><th scope="col">Method</th>{res.trends.years.map((y: number) => <th scope="col" key={y}>{y}</th>)}</tr></thead><tbody>{Object.entries(res.trends.series).map(([m, values]: any) => <tr key={m}><th scope="row">{m}</th>{values.map((n: number, i: number) => <td key={i} className={n ? "" : "comparison-missing"}>{n}</td>)}</tr>)}</tbody></table></div></section>}
    </>}
    {chunk !== null && <Passage chunkId={chunk} onClose={() => setChunk(null)} />}
  </div>;
}

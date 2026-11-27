"use client";
import { useState } from "react";
import { ArrowRight, ChartBar, MagnifyingGlass, SlidersHorizontal, Target, WarningCircle } from "@phosphor-icons/react";
import { api } from "@/lib/api";
import Markdown from "./Markdown";
import Passage from "./Passage";
import SparsityMatrix from "./SparsityMatrix";
import "./Analysis.css";

const NAMES = ["novelty", "sparsity", "importance", "feasibility", "temporal", "evidence", "opportunity", "contradiction"];
const COMPONENT_HELP: Record<string, string> = {
  novelty: "Pair coverage and semantic distance from combinations already explored in this corpus.",
  sparsity: "Missing coverage compared with expected coverage, using method support locally and across the corpus.",
  importance: "How strongly this candidate is supported by relevant papers and their citations.",
  feasibility: "How much the available datasets and methods support exploring this candidate.",
  temporal: "Recency relative to the corpus. Missing publication dates contribute a neutral value.",
  evidence: "Support from independent papers with relevant limitations, future work, or conflicting results.",
  opportunity: "Use of this method outside the candidate’s problem and dataset context.",
  contradiction: "Strength of conflicting results and the number of independent papers involved.",
};
const TYPE_LABEL: Record<string, string> = { limitation: "Limitation", future_work: "Future work", contradiction: "Conflicting result", sparsity: "Corpus coverage", related_context: "Related context" };

export default function Gaps({ selected, onGap }: { selected: number[]; onGap?: (id: number) => void }) {
  const [topic, setTopic] = useState("");
  const [res, setRes] = useState<any>(null);
  const [w, setW] = useState<Record<string, number>>(Object.fromEntries(NAMES.map((n) => [n, 0.125])));
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [chunk, setChunk] = useState<number | null>(null);
  const weightTotal = NAMES.reduce((total, name) => total + w[name], 0);
  const weightPercent = (name: string) => weightTotal > 0 ? (w[name] / weightTotal * 100).toFixed(1) : "0.0";
  async function go(e: React.FormEvent) {
    e.preventDefault();
    if (weightTotal <= 0) { setErr("Choose at least one scoring weight above zero."); return; }
    setErr(""); setBusy(true);
    try { setRes(await api("/research-gaps", { body: { topic: topic || null, paper_ids: selected.length ? selected : null, weights: w } })); }
    catch (e: any) { setErr(e.message); }
    finally { setBusy(false); }
  }
  const enough = selected.length === 0 || selected.length >= 3;
  return <div className="analysis-screen gaps-screen">
    <form className="gap-query-panel" onSubmit={go}>
      <label htmlFor="gap-topic" className="analysis-eyebrow">Focus your exploration <span className="gap-optional">Optional</span></label>
      <div className="gap-topic-row"><div className="gap-topic-input"><MagnifyingGlass size={19} /><input id="gap-topic" type="text" placeholder="e.g. Parkinson’s voice detection" value={topic} onChange={(e) => setTopic(e.target.value)} /></div><button className="btn" disabled={busy || !enough || weightTotal <= 0}>{busy ? "Analysing papers…" : "Find candidate gaps"}{!busy && <ArrowRight size={17} />}</button></div>
      <p className="muted gap-scope">Scope: {selected.length ? `${selected.length} selected paper${selected.length === 1 ? "" : "s"}` : "all ready papers"} · At least 3 papers required.</p>
      <details className="gap-weight-settings"><summary><SlidersHorizontal size={17} />Adjust scoring weights</summary><p className="muted">Percentages show each component’s share of the weighted score. Run the analysis again to apply changes.</p><div className="gap-weight-grid">{NAMES.map((n) => <label className="gap-weight" key={n} title={COMPONENT_HELP[n]}><span>{n}</span><input type="range" min={0} max={1} step={0.025} value={w[n]} aria-valuetext={`${weightPercent(n)}% of scoring weight`} onChange={(e) => setW({ ...w, [n]: +e.target.value })} /><output>{weightPercent(n)}%</output></label>)}</div>{weightTotal <= 0 && <p className="err" role="alert">Choose at least one weight above zero to calculate scores.</p>}</details>
    </form>
    {err && <div className="err analysis-error" role="alert"><WarningCircle size={18} />{err}</div>}
    {busy && <div className="analysis-loading" role="status"><span className="analysis-loading-line" /><span className="analysis-loading-line" /><span className="analysis-loading-line" /><p>Looking for sparse combinations, repeated limitations, and supporting evidence…</p></div>}
    {!res && !busy && <div className="analysis-empty"><span className="analysis-empty-icon"><Target size={31} weight="light" /></span><h2>Explore the questions your corpus leaves open</h2><p>Find patterns in methods, datasets, limitations, and future work. Ranked candidates stay connected to their supporting evidence.</p><span className="analysis-empty-note">Corpus-level candidates · Evidence passages · Adjustable scoring</span></div>}
    {res?.error && !busy && <div className="analysis-empty"><WarningCircle size={27} /><h2>More research context is needed</h2><p>{res.error}</p></div>}
    {res && !res.error && !busy && <>
      <div className="gap-results-intro"><div><h2>{res.gaps.length} candidate{res.gaps.length === 1 ? "" : "s"} to explore</h2><p className="muted">Ranked within a corpus of {res.corpus_size} papers.</p></div><span className="analysis-count">{res.corpus_size} papers</span></div>
      <p className="analysis-notice gap-disclaimer"><WarningCircle size={18} /><span>{res.disclaimer || "Candidates are scored within this corpus. Check the sources and wider literature before treating them as research gaps."} Global novelty is unconfirmed.</span></p>
      {res.matrix && <SparsityMatrix key={res.run_id} matrix={res.matrix} corpusSize={res.corpus_size} />}
      {res.gaps.length === 0 && <div className="analysis-empty"><h2>No candidates found for this scope</h2><p>Try a broader topic or include more ready papers.</p></div>}
      <div className="gap-candidates">{res.gaps.map((g: any, i: number) => {
        const appliedWeights = g.weights || res.weights || {};
        const breakdown = g.score_breakdown;
        return <article className="gap-candidate" key={g.id || i}>
        <div className="gap-candidate-rank"><span className="gap-rank-label">Candidate {String(i + 1).padStart(2, "0")}</span><strong>{Number(g.score).toFixed(4)}</strong><span>Weighted score · 0–1</span><span className="gap-confidence">{g.confidence} confidence</span></div>
        <div className="gap-candidate-main"><div className={`gap-tier ${g.tier}`}><span />{g.tier === "evidence_backed" ? "Evidence-backed candidate" : "Under-explored combination"}</div><h3>{g.description}</h3>
          {g.trend?.kind === "explored_early_then_abandoned" && <p className="gap-trend-note">Earlier studies appear in this corpus, with no recent use in its extracted records.</p>}
          {g.rationale && <Markdown className="gap-rationale">{g.rationale}</Markdown>}
          <details className="gap-score-details"><summary><ChartBar size={16} />Score breakdown</summary>
            <p className="gap-score-note">Components range from 0 to 1. Each shows its weight from this analysis and its contribution (+) to the weighted sum. Candidates without direct evidence receive a lower final score.</p>
            {breakdown && <p className="gap-score-formula">Weighted sum <strong>{Number(breakdown.weighted_sum).toFixed(4)}</strong> × evidence factor <strong>{Number(breakdown.penalty).toFixed(3)}</strong> = final score <strong>{Number(g.score).toFixed(4)}</strong></p>}
            {g.support && <p className="gap-score-note">{g.support.paper_count} supporting papers · {g.support.evidence_paper_count} with direct evidence · {g.support.dated_paper_count} with publication dates</p>}
            <p className="gap-score-note">Confidence reflects independent evidence sources: one is Low, two Medium, and three or more High. Structural-only candidates remain Low. It is not a calibrated probability that the gap is valid.</p>
            <div className="gap-component-grid">{NAMES.map((n) => {
              const value = Number(g.components?.[n] || 0);
              const weight = Number(appliedWeights[n] || 0);
              const contribution = Number(breakdown?.contributions?.[n] ?? value * weight);
              return <div className="gap-component" key={n}>
                <span>{n}<small>{(weight * 100).toFixed(1)}% weight</small></span>
                <meter min={0} max={1} value={value} aria-label={`${n}: ${COMPONENT_HELP[n]}`} />
                <span>{value.toFixed(3)}<small title="Contribution to the weighted sum">+{contribution.toFixed(4)}</small></span>
                <p className="gap-component-help">{COMPONENT_HELP[n]}</p>
              </div>;
            })}</div>
          </details>
          {g.evidence?.length > 0 && <div className="gap-evidence-list"><h4>Evidence & context</h4>{g.evidence.map((e: any, j: number) => e.type === "sparsity" ? <div key={j} className="gap-coverage-note"><ChartBar size={17} /><p>{e.text}</p></div> : <div className={`gap-evidence ${e.type === "related_context" ? "related-context" : ""}`} key={j}>{e.chunk_id ? <button className="analysis-evidence-source" onClick={() => setChunk(e.chunk_id)}><span className="gap-evidence-kind">{TYPE_LABEL[e.type] || e.type}</span><span className="gap-evidence-text">{e.text}</span><span className="analysis-evidence-footer"><span>{e.paper_title}{e.page ? ` · Page ${e.page}` : ""}</span><ArrowRight size={16} /></span></button> : <blockquote><span className="gap-evidence-kind">{TYPE_LABEL[e.type] || e.type}</span><p>{e.text}</p><span className="muted">{e.paper_title}</span></blockquote>}{e.type === "related_context" && <p className="gap-context-note">Background context; does not establish this specific gap.</p>}</div>)}</div>}
          {onGap && g.id && <button className="btn ghost small gap-reading-button" onClick={() => onGap(g.id)}>Papers to read for this candidate <ArrowRight size={16} /></button>}
        </div>
      </article>; })}</div>
      {res.trends?.years?.length > 0 && <section className="analysis-table-panel" aria-labelledby="method-trends-heading"><div className="analysis-section-header"><div><h2 id="method-trends-heading">Method usage over time</h2><p className="muted">Extracted paper counts by publication year within this corpus.</p></div></div><div className="analysis-table-scroll" tabIndex={0} role="region" aria-label="Scrollable method usage table"><table className="analysis-data-table"><thead><tr><th scope="col">Method</th>{res.trends.years.map((y: number) => <th scope="col" key={y}>{y}</th>)}</tr></thead><tbody>{Object.entries(res.trends.series).map(([m, values]: any) => <tr key={m}><th scope="row">{m}</th>{values.map((n: number, i: number) => <td key={i} className={n ? "" : "comparison-missing"}>{n}</td>)}</tr>)}</tbody></table></div></section>}
    </>}
    {chunk !== null && <Passage chunkId={chunk} onClose={() => setChunk(null)} />}
  </div>;
}

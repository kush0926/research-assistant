"use client";

import { useEffect, useRef, useState } from "react";
import { ArrowLeft, ArrowRight, ArrowUpRight, Books, CaretDown, CaretRight, CheckCircle, FilePdf, MagnifyingGlass, Quotes, SlidersHorizontal } from "@phosphor-icons/react";
import { api } from "@/lib/api";
import Markdown from "./Markdown";
import Passage from "./Passage";
import "./Ask.css";

type AskProps = { selected: number[]; papers: any[]; setSelected: (ids: number[]) => void; onLibrary: () => void };
type OpenSource = { chunk_id: number; marker?: number; quotes?: { text: string }[] };

const ROUTE_NAMES: Record<string, string> = { query: "Answer", compare: "Paper comparison", gap: "Candidate research gaps", review: "Literature review draft", recommend: "Recommended reading", search: "Paper search" };

function safeUrl(value: unknown): string | null {
  if (typeof value !== "string") return null;
  try { const url = new URL(value); return url.protocol === "https:" || url.protocol === "http:" ? url.href : null; } catch { return null; }
}

/** React escapes all text. Only bold text and valid numeric citation markers become elements. */
function InlineAnswer({ text, citations, onSource }: { text: string; citations: any[]; onSource: (source: any) => void }) {
  return <>{text.split(/(\*\*[^*]+\*\*|\[\d+\])/g).map((part, i) => {
    const marker = /^\[(\d+)\]$/.exec(part);
    if (marker) {
      const citation = citations.find((c) => String(c.marker) === marker[1]);
      return citation ? <button key={i} className="answer-citation" onClick={() => onSource(citation)} aria-label={`Open source ${marker[1]}: ${citation.paper_title}`}>[{marker[1]}]</button> : <span key={i}>{part}</span>;
    }
    return part.startsWith("**") && part.endsWith("**") ? <strong key={i}>{part.slice(2, -2)}</strong> : <span key={i}>{part}</span>;
  })}</>;
}

function RoutedResult({ route, onSource }: { route: any; onSource: (source: OpenSource) => void }) {
  const result = route.result;
  if (!result || result.error) return <p className="ask-result-message">{result?.error || "No results were returned. Try a more specific request."}</p>;
  if (route.route === "compare") return <>
    {result.verification?.status === "fallback" && <p className="ask-check-note">Showing extracted facts because every generated claim could not be verified.</p>}
    <Markdown>{result.narrative || ""}</Markdown>
    <div className="ask-routed-papers">{(result.table?.rows || []).map((paper: any) => <article key={paper.paper_id}>
      <h3>{paper.title}</h3>
      <p className="ask-routed-meta">{paper.year || "Year unavailable"}</p>
      {["dataset", "algorithm", "results"].map((key) => paper[key]?.text && paper[key].text !== "—" && <div className="ask-routed-fact" key={key}><span>{key === "algorithm" ? "Methods" : key === "dataset" ? "Datasets" : "Results"}</span><p>{paper[key].text}</p>
        {(paper[key].chunk_ids || (paper[key].chunk_id ? [paper[key].chunk_id] : [])).map((id: number, i: number) => <button className="ask-text-action" key={id} onClick={() => onSource({ chunk_id: id })}>Source {i + 1}<ArrowUpRight size={12} /></button>)}</div>)}
    </article>)}</div>
  </>;
  if (route.route === "review") return <>
    {result.verification?.status === "fallback" && <p className="ask-check-note">Showing a draft assembled from extracted facts.</p>}
    <Markdown>{result.markdown || "No draft was returned."}</Markdown>
  </>;
  if (route.route === "gap") return <>
    <p className="ask-check-note">{result.disclaimer || "These are candidates within the selected papers. Broader novelty is not established."}</p>
    {(result.gaps || []).length ? (result.gaps || []).map((gap: any, i: number) => <article className="ask-gap-result" key={gap.id || i}>
      <p className="ask-routed-meta">Candidate {i + 1} · {gap.tier === "evidence_backed" ? "Source evidence available" : "Under-explored combination"}{typeof gap.score === "number" ? ` · Score ${gap.score.toFixed(2)}` : ""}</p>
      <h3>{gap.description}</h3>{gap.rationale && <Markdown>{gap.rationale}</Markdown>}
      {(gap.evidence || []).filter((e: any) => e.chunk_id).map((e: any, j: number) => <button className="ask-evidence-link" key={j} onClick={() => onSource({ chunk_id: e.chunk_id, quotes: e.text ? [{ text: e.text }] : [] })}><Quotes size={17} /><span>{e.paper_title || "Source paper"}</span><CaretRight size={16} /></button>)}
    </article>) : <p className="ask-result-message">No candidate gaps met the criteria. Try selecting more related papers.</p>}
  </>;
  if (route.route === "search" || route.route === "recommend") {
    const items = Array.isArray(result) ? result : result.results || result.recommendations || result.papers || [];
    return items.length ? <div className="ask-routed-papers">{items.map((paper: any, i: number) => {
      const href = safeUrl(paper.url || (paper.doi ? `https://doi.org/${paper.doi}` : ""));
      return <article key={paper.paper_id || paper.external_id || i}><h3>{paper.title || "Untitled paper"}</h3><p className="ask-routed-meta">{paper.year || "Year unavailable"}{paper.source ? ` · ${paper.source}` : ""}</p>
        <p>{paper.reason || paper.abstract || "No summary available."}</p>{href && <a className="ask-text-action" href={href} target="_blank" rel="noopener noreferrer">View paper<ArrowUpRight size={14} /></a>}</article>;
    })}</div> : <p className="ask-result-message">No matching papers were found. Try different search terms.</p>;
  }
  return <p className="ask-result-message">The request completed. Open the relevant workspace to continue.</p>;
}

export default function Ask({ selected, papers, setSelected, onLibrary }: AskProps) {
  const [question, setQuestion] = useState("");
  const [automatic, setAutomatic] = useState(false);
  const [filter, setFilter] = useState("");
  const [papersExpanded, setPapersExpanded] = useState(false);
  const [answer, setAnswer] = useState<any>(null);
  const [route, setRoute] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [activeSource, setActiveSource] = useState<OpenSource | null>(null);
  const [submittedScope, setSubmittedScope] = useState("");
  const request = useRef<AbortController | null>(null);
  const sequence = useRef(0);
  const composer = useRef<HTMLTextAreaElement>(null);
  const ready = papers.filter((paper) => paper.status === "ready");
  const selectedReady = selected.filter((id) => ready.some((paper) => paper.id === id));
  const visible = papers.filter((paper) => `${paper.title || ""} ${paper.year || ""} ${paper.filename || ""}`.toLowerCase().includes(filter.toLowerCase()));
  const rawCitations: any[] = Array.isArray(answer?.citations) ? answer.citations : [];
  const markerCounts = new Map<number, number>();
  rawCitations.forEach((citation) => markerCounts.set(citation?.marker, (markerCounts.get(citation?.marker) || 0) + 1));
  // Ambiguous or malformed source records remain plain citation text rather than linking elsewhere.
  const citations = rawCitations.filter((citation) => Number.isSafeInteger(citation?.marker) && citation.marker > 0
    && Number.isSafeInteger(citation.chunk_id) && citation.chunk_id > 0 && markerCounts.get(citation.marker) === 1);
  const sourceIndex = activeSource?.marker === undefined ? -1 : citations.findIndex((c) => c.marker === activeSource.marker);
  const scope = selectedReady.length ? `${selectedReady.length} selected paper${selectedReady.length === 1 ? "" : "s"}` : `All ${ready.length} ready paper${ready.length === 1 ? "" : "s"}`;

  useEffect(() => () => { request.current?.abort(); sequence.current += 1; }, []);

  function openCitation(citation: any) {
    if (!citations.includes(citation)) return;
    const quotes = (answer?.claims || []).flatMap((claim: any) => claim.supporting_quotes || []).filter((quote: any) => quote.marker === citation.marker);
    setActiveSource({ chunk_id: citation.chunk_id, marker: citation.marker, quotes });
  }

  function stopWaiting() {
    request.current?.abort(); sequence.current += 1; setBusy(false);
  }

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (busy || question.trim().length < 3 || (!ready.length && !automatic)) return;
    request.current?.abort();
    const controller = new AbortController(); request.current = controller;
    const current = ++sequence.current;
    const ids = selectedReady.length ? [...selectedReady] : undefined;
    setBusy(true); setError(""); setAnswer(null); setRoute(null); setActiveSource(null); setSubmittedScope(scope);
    try {
      const result = await api(automatic ? "/agent" : "/query", { signal: controller.signal, body: automatic ? { request: question.trim(), paper_ids: ids } : { question: question.trim(), paper_ids: ids } });
      if (controller.signal.aborted || current !== sequence.current) return;
      if (automatic) { setRoute(result); if (result.route === "query") setAnswer(result.result); }
      else setAnswer(result);
    } catch (err: any) {
      if (!controller.signal.aborted && current === sequence.current) setError(err.message || "The request failed. Please try again.");
    } finally {
      if (current === sequence.current) setBusy(false);
    }
  }

  function chooseQuestion(value: string) { setQuestion(value); composer.current?.focus(); }

  return <div className={`ask-workspace ${activeSource ? "ask-workspace--source" : ""}`}>
    <aside className={`ask-paper-panel ${papersExpanded ? "ask-paper-panel--expanded" : ""}`} aria-label="Papers used for questions">
      <div className="ask-paper-heading"><h2>Selected papers</h2><span>{selectedReady.length} selected</span><button className="icon-btn ask-paper-toggle" aria-label={papersExpanded ? "Collapse paper selector" : "Expand paper selector"} aria-expanded={papersExpanded} onClick={() => setPapersExpanded((value) => !value)}><CaretDown size={19} /></button></div>
      <div className="ask-paper-controls">
        <label className="ask-filter"><MagnifyingGlass size={18} /><input type="search" value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="Filter papers" aria-label="Filter papers" /></label>
        <div className="ask-selection-actions"><button disabled={!ready.length || selectedReady.length === ready.length} onClick={() => setSelected(ready.map((paper) => paper.id))}>Select all</button><button disabled={!selected.length} onClick={() => setSelected([])}>Clear selection</button></div>
      </div>
      <div className="ask-paper-list">
        {visible.map((paper) => <label key={paper.id} className={`ask-paper-item ${selected.includes(paper.id) ? "ask-paper-item--selected" : ""} ${paper.status !== "ready" ? "ask-paper-item--disabled" : ""}`}>
          <input type="checkbox" checked={selected.includes(paper.id)} disabled={paper.status !== "ready"} onChange={(event) => setSelected(event.target.checked ? [...selected, paper.id] : selected.filter((id) => id !== paper.id))} aria-label={`Use ${paper.title || "untitled paper"}`} />
          <span className="ask-pdf-icon"><FilePdf size={25} weight="light" /></span>
          <span className="ask-paper-information"><span className="ask-paper-meta"><span>{paper.year || "No date"}</span><span className={`ask-paper-status ask-paper-status--${paper.status}`}><i aria-hidden="true" />{paper.status === "ready" ? "Ready" : paper.status === "processing" ? "Processing" : paper.status === "metadata_only" ? "Metadata only" : paper.status === "failed" ? "Failed" : paper.status}</span></span><span className="ask-paper-title">{paper.title || paper.filename || "Untitled paper"}</span></span>
        </label>)}
        {!visible.length && <div className="ask-no-papers"><Books size={30} weight="light" /><p>{papers.length ? "No papers match your filter." : "Add your first paper to start asking questions."}</p>{!papers.length && <button className="ask-text-action" onClick={onLibrary}>Go to library<ArrowRight size={15} /></button>}</div>}
      </div>
      <div className="ask-paper-footer"><p>{selectedReady.length ? "Only selected papers will be searched." : "No selection: questions search all ready papers."}</p><button onClick={onLibrary}><ArrowLeft size={18} />Back to library</button></div>
    </aside>

    <section className="ask-main-panel" aria-label="Ask and answer">
      <form className="ask-composer" onSubmit={submit}>
        <label className="sr-only" htmlFor="research-question">Your research question</label>
        <textarea id="research-question" ref={composer} value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); event.currentTarget.form?.requestSubmit(); } }} placeholder="What would you like to know about your papers?" required minLength={3} maxLength={1000} rows={4} />
        <div className="ask-composer-footer"><span className="ask-scope">Scope: <strong>{scope}</strong></span><label className="ask-automatic" title="Choose search, comparison, gaps, review, or a source-based answer from your request"><input type="checkbox" checked={automatic} disabled={busy} onChange={(event) => setAutomatic(event.target.checked)} />Route automatically</label><button className="btn ask-submit" type="submit" disabled={busy || question.trim().length < 3 || (!ready.length && !automatic)}>{busy ? "Working…" : "Ask"}<ArrowRight size={19} /></button></div>
      </form>
      {!ready.length && <p className="ask-check-note">{papers.some((paper) => paper.status === "processing") ? "Your papers are processing. Questions become available when a paper is ready." : "Upload and process a paper in your library to ask a question."}</p>}
      {error && <div className="ask-error" role="alert"><p>{error}</p><span>Your question is still above. You can try again.</span></div>}
      {busy && <div className="ask-answer-loading" role="status" aria-live="polite"><div className="ask-answer-heading"><h2>{automatic ? "Working on your request" : "Reading your papers"}</h2><button className="ask-text-action" onClick={stopWaiting}>Stop waiting</button></div><p>Finding relevant passages and checking source support.</p><div className="ask-skeleton-line" /><div className="ask-skeleton-line" /><div className="ask-skeleton-line" /></div>}
      {!busy && !answer && !route && !error && <div className="ask-start"><span className="ask-start-icon"><Quotes size={30} weight="light" /></span><h2>A question. An answer. The evidence.</h2><p>Ask across your papers, then open a citation to read the exact supporting passage.</p><div className="ask-suggestions"><button onClick={() => chooseQuestion("What methods and datasets do these papers use?")}>Explore methods and datasets<ArrowUpRight size={15} /></button><button onClick={() => chooseQuestion("What limitations do the authors report?")}>Find reported limitations<ArrowUpRight size={15} /></button></div><span className="ask-keyboard-hint">Ctrl / ⌘ + Enter to ask</span></div>}

      {route && <div className="ask-route-note"><SlidersHorizontal size={15} /><span>Routed to <strong>{ROUTE_NAMES[route.route] || route.route}</strong></span>{route.steps?.length > 0 && <details><summary>View steps</summary><ol>{route.steps.map((step: string, i: number) => <li key={i}>{step}</li>)}</ol></details>}</div>}
      {answer && <article className="ask-answer" aria-live="polite">
        <div className="ask-answer-heading"><h2>Answer</h2>{answer.mode !== "refusal" && !answer.unverified && <span className="ask-checked"><CheckCircle size={18} weight="fill" />Source support checked</span>}</div>
        <div className="ask-answer-copy">{answer.claims?.length ? answer.claims.map((claim: any, i: number) => <p key={i} className={!claim.verified ? "ask-unverified" : ""}><InlineAnswer text={String(claim.text || "").replace(/^\s*[-*]\s+/, "")} citations={citations} onSource={openCitation} /></p>) : <p>{answer.answer}</p>}</div>
        <p className="ask-answer-footnote"><CheckCircle size={14} />{answer.mode === "refusal" ? "No supporting passages were found for this question. Try a more specific question or select other papers." : answer.unverified ? "Some displayed claims could not be verified. Check their sources before use." : "Checked against cited passages. Open a source to inspect the evidence."}</p>
        {!!answer.omitted_unverified_claims && <p className="ask-check-note">{answer.omitted_unverified_claims} draft claim{answer.omitted_unverified_claims === 1 ? " was" : "s were"} withheld because source support could not be verified.</p>}
        {answer.mode === "extractive" && <p className="ask-check-note">Showing excerpts from the papers.</p>}
        {!!citations.length && <section className="ask-sources" aria-label="Answer sources"><div className="ask-source-heading"><h3>Sources</h3><span>{citations.length} passage{citations.length === 1 ? "" : "s"} · {submittedScope}</span></div>
          {citations.map((citation) => <button key={citation.marker} className={`ask-source-item ${activeSource?.marker === citation.marker ? "ask-source-item--active" : ""}`} onClick={() => openCitation(citation)} aria-pressed={activeSource?.marker === citation.marker}><span className="ask-source-number">[{citation.marker}]</span><span className="ask-source-details"><strong>{citation.paper_title}</strong><span>{papers.find((paper) => paper.id === citation.paper_id)?.year || "No date"}<i>·</i>{citation.section || "Source text"}<i>·</i>{citation.page ? `Page ${citation.page}` : "Page unavailable"}</span></span><CaretRight size={19} /></button>)}
        </section>}
      </article>}
      {route && route.route !== "query" && <article className="ask-routed-result"><div className="ask-answer-heading"><h2>{ROUTE_NAMES[route.route] || "Result"}</h2></div><RoutedResult route={route} onSource={setActiveSource} /></article>}
    </section>

    {activeSource && <Passage inline chunkId={activeSource.chunk_id} marker={activeSource.marker} supportingQuotes={activeSource.quotes || []} onClose={() => setActiveSource(null)} sourceIndex={Math.max(0, sourceIndex)} sourceCount={sourceIndex >= 0 ? citations.length : 1} onPrevious={sourceIndex > 0 ? () => openCitation(citations[sourceIndex - 1]) : undefined} onNext={sourceIndex >= 0 && sourceIndex < citations.length - 1 ? () => openCitation(citations[sourceIndex + 1]) : undefined} />}
  </div>;
}

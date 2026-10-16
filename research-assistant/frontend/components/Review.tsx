"use client";
import { useState } from "react";
import { ArrowRight, BookOpenText, CheckCircle, DownloadSimple, WarningCircle } from "@phosphor-icons/react";
import { api } from "@/lib/api";
import Markdown from "./Markdown";
import "./Analysis.css";

export default function Review({ selected }: { selected: number[] }) {
  const [result, setResult] = useState<any>(null);
  const [draftCount, setDraftCount] = useState(0);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  async function go() {
    setErr(""); setBusy(true);
    try { const r = await api("/literature-review", { body: { paper_ids: selected } }); setResult(r); setDraftCount(selected.length); }
    catch (e: any) { setErr(e.message); }
    finally { setBusy(false); }
  }
  function download() {
    if (!result?.markdown) return;
    const url = URL.createObjectURL(new Blob([result.markdown], { type: "text/markdown;charset=utf-8" }));
    const a = document.createElement("a"); a.href = url; a.download = "literature-review.md";
    document.body.appendChild(a); a.click(); a.remove(); window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return <div className="analysis-screen review-screen">
    <section className="analysis-controls" aria-label="Review settings"><div className="analysis-control-copy"><span className="analysis-eyebrow">From papers to a first draft</span><p>{selected.length ? `${selected.length} selected paper${selected.length === 1 ? "" : "s"}. Paper tags map each cited claim to the references.` : "Select ready papers from your library to draft a literature review."}</p></div><button className="btn" onClick={go} disabled={!selected.length || busy}>{busy ? "Drafting review…" : "Generate draft"}{!busy && <ArrowRight size={17} />}</button></section>
    {err && <div className="err analysis-error" role="alert"><WarningCircle size={18} />{err}</div>}
    {busy && <div className="analysis-loading" role="status"><span className="analysis-loading-line" /><span className="analysis-loading-line" /><span className="analysis-loading-line" /><p>Organising the selected research and checking the draft against extracted facts…</p></div>}
    {!result && !busy && <div className="analysis-empty"><span className="analysis-empty-icon"><BookOpenText size={31} weight="light" /></span><h2>A starting point for your literature review</h2><p>Bring together research themes, methods, findings, and limitations in a readable draft with paper references.</p><span className="analysis-empty-note">Cited paper tags · Structured sections · Markdown export</span></div>}
    {result && !busy && <article className="analysis-reading-panel review-document"><div className="analysis-section-header"><div><span className="analysis-eyebrow">Working draft · {draftCount} paper{draftCount === 1 ? "" : "s"}</span><h2>Your literature review</h2></div><button className="btn ghost small" onClick={download}><DownloadSimple size={17} />Download .md</button></div>
      {result.verification?.status === "passed" ? <p className="analysis-check"><CheckCircle size={17} weight="fill" />Generated claims checked against extracted paper facts.</p> : <p className="analysis-notice">{result.verification?.status === "fallback" ? "Automatic checks could not verify every generated claim. Showing a structured draft built from the extracted facts." : "This structured draft uses the extracted paper facts."}</p>}
      <p className="review-reuse-note">Review the cited papers before reusing this draft. Paper tags such as [P1] refer to its reference list.</p>
      <div className="review-document-body"><Markdown>{result.markdown || "No draft was returned."}</Markdown></div>
      <footer className="review-document-footer"><span>{result.mode === "llm" ? "LLM-assisted draft" : "Structured factual draft"}</span><span>Export preserves the original Markdown and references.</span></footer>
    </article>}
  </div>;
}

"use client";

import { useEffect, useRef, useState } from "react";
import { CaretLeft, CaretRight, CheckCircle, FileText, X } from "@phosphor-icons/react";
import { api } from "@/lib/api";
import "./Passage.css";

type SupportingQuote = string | { text: string };

export type PassageProps = {
  chunkId: number;
  onClose: () => void;
  inline?: boolean;
  marker?: number;
  supportingQuotes?: SupportingQuote[];
  sourceIndex?: number;
  sourceCount?: number;
  onPrevious?: () => void;
  onNext?: () => void;
};

/** Highlight only literal supporting quotations, keeping the original passage intact. */
function SourceText({ text, quotes }: { text: string; quotes: SupportingQuote[] }) {
  const spans: [number, number][] = [];
  for (const quote of quotes) {
    const literal = typeof quote === "string" ? quote : quote.text;
    if (!literal?.trim()) continue;
    let start = 0;
    while ((start = text.indexOf(literal, start)) !== -1) {
      spans.push([start, start + literal.length]);
      start += literal.length;
    }
  }
  spans.sort((a, b) => a[0] - b[0]);
  const merged: [number, number][] = [];
  for (const span of spans) {
    const last = merged[merged.length - 1];
    if (last && span[0] <= last[1]) last[1] = Math.max(last[1], span[1]);
    else merged.push([...span]);
  }
  let offset = 0;
  const parts = merged.map(([start, end], i) => {
    const before = text.slice(offset, start);
    offset = end;
    return <span key={i}>{before}<mark>{text.slice(start, end)}</mark></span>;
  });
  return <>{parts}{text.slice(offset)}</>;
}

/** Exact source inspector shared by citations, comparisons and gap evidence. */
export default function Passage({ chunkId, onClose, inline = false, marker, supportingQuotes = [],
  sourceIndex = 0, sourceCount = 1, onPrevious, onNext }: PassageProps) {
  const [chunk, setChunk] = useState<any>(null);
  const [err, setErr] = useState("");
  const [retry, setRetry] = useState(0);
  const [overlay, setOverlay] = useState(!inline);
  const heading = useRef<HTMLHeadingElement>(null);
  const panel = useRef<HTMLElement>(null);
  const close = useRef(onClose);
  close.current = onClose;

  useEffect(() => {
    const media = window.matchMedia("(max-width: 1279px)");
    const update = () => setOverlay(!inline || media.matches);
    update(); media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, [inline]);

  useEffect(() => {
    if (!overlay) return;
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = previous; };
  }, [overlay]);

  useEffect(() => {
    const controller = new AbortController();
    setChunk(null); setErr("");
    api(`/chunks/${chunkId}`, { signal: controller.signal })
      .then((value) => { if (!controller.signal.aborted) setChunk(value); })
      .catch((error) => { if (!controller.signal.aborted) setErr(error.message || "Could not load this passage."); });
    return () => controller.abort();
  }, [chunkId, retry]);

  useEffect(() => {
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    heading.current?.focus({ preventScroll: true });
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") { event.preventDefault(); close.current(); }
      const overlay = !inline || window.matchMedia("(max-width: 1279px)").matches;
      if (event.key !== "Tab" || !overlay || !panel.current) return;
      const controls = Array.from(panel.current.querySelectorAll<HTMLElement>("button:not(:disabled), a[href], [tabindex='0']"));
      const first = controls[0]; const last = controls[controls.length - 1];
      if (!first) return;
      if (event.shiftKey && (document.activeElement === first || document.activeElement === heading.current)) {
        event.preventDefault(); last.focus();
      } else if (!event.shiftKey && (document.activeElement === last || document.activeElement === heading.current)) {
        event.preventDefault(); first.focus();
      }
    }
    document.addEventListener("keydown", onKey);
    return () => { document.removeEventListener("keydown", onKey); if (opener?.isConnected) opener.focus({ preventScroll: true }); };
  }, [inline]);

  return <>
    <button className={`passage-backdrop ${inline ? "passage-backdrop--inline" : ""}`} onClick={onClose} aria-label="Close source passage" tabIndex={-1} />
    <aside ref={panel} className={`source-panel ${inline ? "source-panel--inline" : "source-panel--drawer"}`} role="dialog" aria-modal={overlay || undefined} aria-labelledby="source-panel-title">
      <div className="source-panel-header">
        <h2 id="source-panel-title" ref={heading} tabIndex={-1}>Source passage</h2>
        <button className="icon-btn" onClick={onClose} aria-label="Close source passage"><X size={20} /></button>
      </div>
      {marker !== undefined && <span className="source-marker">[{marker}]</span>}
      {!chunk && !err && <div className="source-loading" role="status" aria-label="Loading source passage"><div /><div /><div /><span>Loading original passage…</span></div>}
      {err && <div className="source-error" role="alert"><FileText size={28} /><p>{err}</p><button className="btn ghost small" onClick={() => setRetry((v) => v + 1)}>Try again</button></div>}
      {chunk && <>
        <h3 className="source-paper-title">{chunk.paper_title || "Untitled paper"}</h3>
        <p className="source-meta">{chunk.section || "Source text"}<span aria-hidden="true">·</span>{chunk.page ? `Page ${chunk.page}` : "Page unavailable"}</p>
        {chunk.flagged_suspicious && <p className="source-warning">Instruction-like text in this passage was neutralised during processing.</p>}
        <div className="source-reading" tabIndex={0} aria-label="Original passage text">
          <p><SourceText text={chunk.text || ""} quotes={supportingQuotes} /></p>
          <div className="source-reading-note"><CheckCircle size={15} /> Original passage{supportingQuotes.length ? "; matching supporting sentences are highlighted." : "; no supporting sentence highlight supplied."}</div>
        </div>
      </>}
      <div className="source-pagination">
        <button className="icon-btn" onClick={onPrevious} disabled={!onPrevious} aria-label="Previous source"><CaretLeft size={20} /></button>
        <span>{sourceCount > 1 ? `${sourceIndex + 1} of ${sourceCount} sources` : "Original source passage"}</span>
        <button className="icon-btn" onClick={onNext} disabled={!onNext} aria-label="Next source"><CaretRight size={20} /></button>
      </div>
    </aside>
  </>;
}

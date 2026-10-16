"use client";

import { useEffect, useRef, useState } from "react";
import { ArrowDown, ArrowUpRight, CloudArrowUp, FilePdf, MagnifyingGlass, Trash, X } from "@phosphor-icons/react";
import { api } from "@/lib/api";
import "./Library.css";

type Paper = { id: number; title: string; authors?: string[]; year?: number | null; status: string; error?: string };
type LibraryProps = { papers: Paper[]; reload: () => void | Promise<void>; selected: number[]; setSelected: (ids: number[]) => void };
const statusLabels: Record<string, string> = { ready: "Ready", processing: "Processing", failed: "Failed", metadata_only: "Abstract only" };

export default function Library({ papers, reload, selected, setSelected }: LibraryProps) {
  const [msg, setMsg] = useState("");
  const [err, setErr] = useState("");
  const [q, setQ] = useState("");
  const [filter, setFilter] = useState("");
  const [source, setSource] = useState("semanticscholar");
  const [results, setResults] = useState<any[]>([]);
  const [searched, setSearched] = useState(false);
  const [searching, setSearching] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [importing, setImporting] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<number | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const selectAllRef = useRef<HTMLInputElement>(null);
  const dragDepth = useRef(0);
  const mounted = useRef(true);
  const uploadLock = useRef(false);
  const latestSelected = useRef(selected);
  latestSelected.current = selected;
  const readyPapers = papers.filter((paper) => paper.status === "ready");
  const selectedReady = readyPapers.filter((paper) => selected.includes(paper.id));
  const allReadySelected = readyPapers.length > 0 && selectedReady.length === readyPapers.length;
  const processingCount = papers.filter((paper) => paper.status === "processing").length;
  const filteredPapers = papers.filter((paper) => `${paper.title} ${(paper.authors || []).join(" ")} ${paper.year || ""}`.toLowerCase().includes(filter.toLowerCase()));

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);
  useEffect(() => {
    if (selectAllRef.current) selectAllRef.current.indeterminate = selectedReady.length > 0 && !allReadySelected;
  }, [selectedReady.length, allReadySelected]);

  async function upload(files: FileList | File[] | null) {
    if (!files?.length || uploadLock.current) return;
    uploadLock.current = true;
    setErr(""); setMsg(""); setUploading(true);
    const errors: string[] = [];
    let uploaded = 0;
    for (const file of Array.from(files)) {
      if (!/\.pdf$/i.test(file.name)) { errors.push(`${file.name}: choose a PDF file.`); continue; }
      const form = new FormData(); form.append("file", file);
      try {
        await api("/papers/upload", { form }); uploaded += 1;
        if (mounted.current) setMsg(`${uploaded} ${uploaded === 1 ? "paper uploaded" : "papers uploaded"}. Processing will continue in the background.`);
        await reload();
      } catch (error: any) { errors.push(`${file.name}: ${error.message}`); }
    }
    if (fileRef.current) fileRef.current.value = "";
    uploadLock.current = false;
    if (mounted.current) { setErr(errors.join(" ")); setUploading(false); }
    void reload();
  }
  async function search(event: React.FormEvent) {
    event.preventDefault(); if (q.trim().length < 2) return;
    setErr(""); setSearching(true); setSearched(false);
    try {
      const response = await api(`/papers/search?q=${encodeURIComponent(q.trim())}&source=${source}`);
      if (mounted.current) { setResults(response.results || []); setSearched(true); }
    } catch (error: any) { if (mounted.current) setErr(error.message); } finally { if (mounted.current) setSearching(false); }
  }
  async function importPaper(paper: any, key: string) {
    setErr(""); setImporting(key);
    try {
      await api("/papers/import", { body: paper });
      if (mounted.current) setMsg(paper.pdf_url ? "Paper imported. Processing the open-access PDF." : "Abstract imported. A full-text PDF is not available from this source.");
      await reload();
    } catch (error: any) { if (mounted.current) setErr(error.message); } finally { if (mounted.current) setImporting(null); }
  }
  async function deletePaper(paper: Paper) {
    if (!window.confirm(`Delete “${paper.title}” and its extracted data?`)) return;
    setErr(""); setDeleting(paper.id);
    try {
      await api(`/papers/${paper.id}`, { method: "DELETE" });
      if (mounted.current) setSelected(latestSelected.current.filter((id) => id !== paper.id));
      await reload();
      if (mounted.current) setMsg("Paper deleted.");
    } catch (error: any) { if (mounted.current) setErr(error.message); } finally { if (mounted.current) setDeleting(null); }
  }
  const toggle = (id: number) => setSelected(selected.includes(id) ? selected.filter((item) => item !== id) : [...selected, id]);

  return (
    <div className="library-view">
      <input id="paper-upload-input" className="library-file-input" ref={fileRef} type="file" accept="application/pdf,.pdf" multiple tabIndex={-1} disabled={uploading} onChange={(event) => { void upload(event.target.files); }} aria-label="Upload PDF papers" />
      <section className="library-acquisition" aria-label="Add research papers">
        <div className={`library-dropzone${dragging ? " is-dragging" : ""}${uploading ? " is-uploading" : ""}`}
          onDragEnter={(event) => { event.preventDefault(); dragDepth.current += 1; setDragging(true); }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={(event) => { event.preventDefault(); dragDepth.current -= 1; if (dragDepth.current <= 0) { dragDepth.current = 0; setDragging(false); } }}
          onDrop={(event) => { event.preventDefault(); dragDepth.current = 0; setDragging(false); void upload(event.dataTransfer.files); }} aria-busy={uploading}>
          <CloudArrowUp size={30} weight="regular" aria-hidden="true" />
          <div><h2>{uploading ? "Uploading your papers…" : "Bring your research together"}</h2>
            <p>Drop PDFs here or <button className="library-text-button" type="button" disabled={uploading} onClick={() => fileRef.current?.click()}>browse files</button>.</p>
            <span className="library-upload-note">Your uploads stay private to your account.</span></div>
        </div>
        <form className="library-discovery" onSubmit={search}>
          <h2>Find a paper</h2><p className="muted">Search research sources and import an available PDF or abstract.</p>
          <div className="library-search-controls">
            <label className="library-search-field"><MagnifyingGlass size={19} aria-hidden="true" /><input type="text" aria-label="Search research papers" placeholder="Title, topic, or author" value={q} onChange={(event) => setQ(event.target.value)} /></label>
            <select value={source} onChange={(event) => setSource(event.target.value)} aria-label="Paper search source"><option value="semanticscholar">Semantic Scholar</option><option value="arxiv">arXiv</option><option value="openalex">OpenAlex</option></select>
            <button className="btn" disabled={searching || q.trim().length < 2}>{searching ? "Searching…" : "Search"}</button>
          </div>
        </form>
      </section>
      <div className="library-notices" aria-live="polite">{msg && <p className="library-success" role="status">{msg}</p>}{err && <p className="err" role="alert">{err}</p>}</div>
      {searched && <section className="library-search-results" aria-labelledby="search-results-heading">
        <div className="library-section-heading"><h2 id="search-results-heading">Search results <span>{results.length}</span></h2><button type="button" className="library-text-button" onClick={() => { setSearched(false); setResults([]); }}><X size={15} aria-hidden="true" /> Close results</button></div>
        {results.length === 0 ? <p className="library-no-results">No papers found. Try a broader topic or another source.</p> : results.map((paper, index) => {
          const key = `${paper.id || paper.title}-${index}`;
          return <article key={key} className="library-search-result"><FilePdf size={24} weight="regular" aria-hidden="true" />
            <div className="library-result-copy"><h3>{paper.title}</h3><p className="muted">{(paper.authors || []).slice(0, 3).join(", ") || "Authors unavailable"} · {paper.year || "Year unavailable"}{paper.citation_count != null ? ` · ${paper.citation_count} citations` : ""}</p>
              {paper.abstract && <p className="library-result-abstract">{paper.abstract.slice(0, 260)}{paper.abstract.length > 260 ? "…" : ""}</p>}</div>
            <button className="btn ghost small" disabled={importing !== null} onClick={() => { void importPaper(paper, key); }}>{importing === key ? "Importing…" : paper.pdf_url ? "Import PDF" : "Import abstract"}<ArrowDown size={15} aria-hidden="true" /></button></article>;
        })}
      </section>}
      <section className="library-collection" aria-labelledby="library-papers-heading">
        <div className="library-section-heading library-collection-heading"><div><h2 id="library-papers-heading">Your papers <span>{papers.length}</span></h2><p className="muted">{selectedReady.length} selected for analysis{processingCount ? ` · ${processingCount} processing` : ""}</p></div>
          <label className="library-search-field library-filter"><MagnifyingGlass size={18} aria-hidden="true" /><input type="text" placeholder="Filter your library" aria-label="Filter your library" value={filter} onChange={(event) => setFilter(event.target.value)} /></label></div>
        {papers.length === 0 ? <div className="library-empty"><span className="library-empty-icon"><FilePdf size={30} aria-hidden="true" /></span><h3>Your research starts here</h3><p>Upload a PDF or find a paper above. Once it is ready, select it to ask questions, compare findings, or explore research gaps.</p><button type="button" className="btn ghost" onClick={() => fileRef.current?.click()} disabled={uploading}>Choose PDFs <ArrowUpRight size={17} aria-hidden="true" /></button></div> : <>
          <div className="library-table-scroll"><table className="library-table"><thead><tr>
            <th scope="col" className="library-checkbox-cell"><input ref={selectAllRef} type="checkbox" checked={allReadySelected} disabled={!readyPapers.length} onChange={() => setSelected(allReadySelected ? [] : readyPapers.map((paper) => paper.id))} aria-label="Select all ready papers" /></th>
            <th scope="col">Paper</th><th scope="col" className="library-year-cell">Year</th><th scope="col">Status</th><th scope="col" className="library-action-cell"><span className="library-sr-only">Actions</span></th>
          </tr></thead><tbody>{filteredPapers.map((paper) => <tr key={paper.id} className={selected.includes(paper.id) ? "is-selected" : undefined}>
            <td className="library-checkbox-cell"><input type="checkbox" checked={selected.includes(paper.id)} onChange={() => toggle(paper.id)} disabled={paper.status !== "ready"} aria-label={`Select ${paper.title}`} /></td>
            <td><div className="library-paper-heading"><span className="library-paper-icon"><FilePdf size={23} aria-hidden="true" /></span><div><span className="library-paper-title">{paper.title}</span><p className="library-paper-authors">{(paper.authors || []).slice(0, 3).join(", ") || "Authors unavailable"}{(paper.authors || []).length > 3 ? " et al." : ""}</p>{paper.error && <p className="err library-paper-error">{paper.error}</p>}</div></div></td>
            <td className="library-year-cell">{paper.year || "N/A"}</td><td><span className={`library-status library-status-${paper.status}`}><span aria-hidden="true" />{statusLabels[paper.status] || paper.status.replace(/_/g, " ")}</span></td>
            <td className="library-action-cell"><button type="button" className="icon-btn library-delete" disabled={deleting !== null} onClick={() => { void deletePaper(paper); }} aria-label={`Delete ${paper.title}`} title={deleting === paper.id ? "Deleting paper" : "Delete paper"}><Trash size={18} aria-hidden="true" /></button></td></tr>)}
            {!filteredPapers.length && <tr><td colSpan={5} className="library-no-results">No papers match “{filter}”.</td></tr>}
          </tbody></table></div>
          <footer className="library-collection-footer"><span>{filteredPapers.length === papers.length ? `${papers.length} ${papers.length === 1 ? "paper" : "papers"}` : `${filteredPapers.length} of ${papers.length} papers`} · {readyPapers.length} ready</span><div className="library-selection-actions"><button type="button" className="library-text-button" disabled={!readyPapers.length || allReadySelected} onClick={() => setSelected(readyPapers.map((paper) => paper.id))}>Select all ready</button><button type="button" className="library-text-button" disabled={!selected.length} onClick={() => setSelected([])}>Clear selection</button></div></footer>
        </>}
      </section>
    </div>
  );
}

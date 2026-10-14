"use client";
import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";

export default function Library({ papers, reload, selected, setSelected }: any) {
  const [msg, setMsg] = useState(""); const [err, setErr] = useState("");
  const [q, setQ] = useState(""); const [source, setSource] = useState("semanticscholar");
  const [results, setResults] = useState<any[]>([]); const [busy, setBusy] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!papers.some((p: any) => p.status === "processing")) return;
    const t = setInterval(reload, 3000); return () => clearInterval(t);
  }, [papers]);

  async function upload(files: FileList | null) {
    if (!files) return; setErr(""); setMsg("");
    for (const f of Array.from(files)) {
      const fd = new FormData(); fd.append("file", f);
      try { await api("/papers/upload", { form: fd }); setMsg(`Uploaded ${f.name}; processing…`); }
      catch (e: any) { setErr(`${f.name}: ${e.message}`); }
    }
    if (fileRef.current) fileRef.current.value = ""; reload();
  }
  async function search(e: React.FormEvent) {
    e.preventDefault(); setErr(""); setBusy(true);
    try { setResults((await api(`/papers/search?q=${encodeURIComponent(q)}&source=${source}`)).results); }
    catch (x: any) { setErr(x.message); } finally { setBusy(false); }
  }
  async function imp(r: any) {
    try { await api("/papers/import", { body: r }); setMsg(r.pdf_url ? "Imported; processing open-access PDF…" : "Imported abstract only (no open-access PDF)."); reload(); }
    catch (x: any) { setErr(x.message); }
  }
  async function del(id: number) {
    if (!confirm("Delete this paper and its extracted data?")) return;
    await api(`/papers/${id}`, { method: "DELETE" }); setSelected(selected.filter((s: number) => s !== id)); reload();
  }
  const toggle = (id: number) => setSelected(selected.includes(id) ? selected.filter((s: number) => s !== id) : [...selected, id]);

  return (<>
    <h1>Library</h1>
    <div className="card">
      <h3>Add papers</h3>
      <div className="row">
        <input ref={fileRef} type="file" accept="application/pdf" multiple onChange={(e) => upload(e.target.files)} aria-label="Upload PDF papers" />
        <span className="muted">PDF only, parsed in an isolated worker. Unpublished drafts stay private to your account.</span>
      </div>
      <form className="row" style={{ marginTop: ".9rem" }} onSubmit={search}>
        <input className="grow" type="text" placeholder="Search papers, e.g. deep learning Parkinson's voice" value={q} onChange={(e) => setQ(e.target.value)} />
        <select style={{ width: 170 }} value={source} onChange={(e) => setSource(e.target.value)} aria-label="Source">
          <option value="semanticscholar">Semantic Scholar</option><option value="arxiv">arXiv</option><option value="openalex">OpenAlex</option>
        </select>
        <button className="btn" disabled={busy || q.length < 2}>{busy ? "Searching…" : "Search"}</button>
      </form>
      {msg && <p className="okmsg">{msg}</p>}{err && <p className="err">{err}</p>}
    </div>

    {results.length > 0 && <div className="card"><h3>Search results</h3>
      {results.map((r, i) => (
        <div key={i} style={{ padding: ".6rem 0", borderTop: i ? "1px solid var(--rule)" : 0 }}>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <strong>{r.title}</strong>
            <button className="btn small" onClick={() => imp(r)}>{r.pdf_url ? "Import PDF" : "Import abstract"}</button>
          </div>
          <div className="muted">{(r.authors || []).slice(0, 3).join(", ")} · {r.year || "n.d."}{r.citation_count != null ? ` · ${r.citation_count} citations` : ""}</div>
          <p className="muted" style={{ margin: ".25rem 0 0" }}>{(r.abstract || "").slice(0, 240)}{(r.abstract || "").length > 240 ? "…" : ""}</p>
        </div>))}
    </div>}

    <div className="card">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h3>Your papers ({papers.length})</h3>
        <span className="muted">{selected.length} selected for Ask / Compare / Gaps / Review</span>
      </div>
      {papers.length === 0 ? <div className="empty">No papers yet. Upload PDFs above to begin.</div> :
        <div className="scroll"><table>
          <thead><tr><th></th><th>Title</th><th>Year</th><th>Status</th><th></th></tr></thead>
          <tbody>{papers.map((p: any) => (
            <tr key={p.id}>
              <td><input type="checkbox" checked={selected.includes(p.id)} onChange={() => toggle(p.id)} disabled={p.status !== "ready"} aria-label={`Select ${p.title}`} /></td>
              <td>{p.title}<div className="muted">{(p.authors || []).slice(0, 3).join(", ")}</div>{p.error && <div className="err">{p.error}</div>}</td>
              <td>{p.year || "—"}</td>
              <td><span className={`badge ${p.status}`}>{p.status.replace("_", " ")}</span></td>
              <td><button className="btn ghost small" onClick={() => del(p.id)}>Delete</button></td>
            </tr>))}</tbody></table></div>}
      {papers.length > 0 && <p style={{ marginTop: ".75rem" }}><button className="btn ghost small" onClick={() => setSelected(papers.filter((p: any) => p.status === "ready").map((p: any) => p.id))}>Select all ready</button>{" "}
        <button className="btn ghost small" onClick={() => setSelected([])}>Clear</button></p>}
    </div>
  </>);
}

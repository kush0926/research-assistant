"use client";
import { useState } from "react";
import { api } from "@/lib/api";

export default function Review({ selected }: { selected: number[] }) {
  const [md, setMd] = useState(""); const [mode, setMode] = useState(""); const [busy, setBusy] = useState(false); const [err, setErr] = useState("");
  async function go() { setErr(""); setBusy(true); try { const r = await api("/literature-review", { body: { paper_ids: selected } }); setMd(r.markdown); setMode(r.mode); } catch (e: any) { setErr(e.message); } finally { setBusy(false); } }
  function download() { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([md], { type: "text/markdown" })); a.download = "literature-review.md"; a.click(); }
  return (<>
    <h1>Literature review draft</h1>
    <div className="card row"><button className="btn" onClick={go} disabled={!selected.length || busy}>{busy ? "Drafting…" : `Draft from ${selected.length} selected`}</button>
      {md && <button className="btn ghost" onClick={download}>Download .md</button>}<span className="muted">Every statement carries a [P#] tag that maps to the reference list.</span></div>
    {err && <p className="err">{err}</p>}
    {md && <div className="card"><p className="muted">Generated in {mode} mode — check it against the sources before reusing.</p><pre className="prose" style={{ whiteSpace: "pre-wrap" }}>{md}</pre></div>}
  </>);
}

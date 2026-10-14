"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";

/** Side drawer showing the exact source passage behind any citation / table cell / gap evidence. */
export default function Passage({ chunkId, onClose }: { chunkId: number; onClose: () => void }) {
  const [c, setC] = useState<any>(null);
  const [err, setErr] = useState("");
  useEffect(() => { api(`/chunks/${chunkId}`).then(setC).catch((e) => setErr(e.message)); }, [chunkId]);
  return (
    <aside className="drawer" role="dialog" aria-label="Source passage">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h3>Source passage</h3>
        <button className="btn ghost small" onClick={onClose}>Close</button>
      </div>
      {err && <p className="err">{err}</p>}
      {c && (<>
        <p className="muted">{c.paper_title} — {c.section}, page {c.page}</p>
        {c.flagged_suspicious && <p className="err">This passage contained instruction-like text that was neutralised during ingestion.</p>}
        <p className="prose"><span className="mark">{c.text}</span></p>
      </>)}
    </aside>
  );
}

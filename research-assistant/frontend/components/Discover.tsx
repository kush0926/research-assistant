"use client";
import dynamic from "next/dynamic";
import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
const ForceGraph = dynamic(() => import("react-force-graph-2d"), { ssr: false });

const COLORS: any = { Paper: "#0e6b66", Method: "#c0762b", Dataset: "#5b5fc7", Topic: "#8a8f98", Author: "#b9c0c7" };

export default function Discover({ papers, gapId }: { papers: any[]; gapId: number | null }) {
  const [pid, setPid] = useState<string>(""); const [graph, setGraph] = useState<any>({ nodes: [], links: [] });
  const [recs, setRecs] = useState<any[]>([]); const [err, setErr] = useState("");
  const box = useRef<HTMLDivElement>(null); const [w, setW] = useState(700);
  useEffect(() => { if (box.current) setW(box.current.clientWidth); }, []);
  useEffect(() => {
    api(`/knowledge-graph${pid ? `?paper_id=${pid}` : ""}`).then((g) => setGraph({ nodes: g.nodes, links: g.edges })).catch((e) => setErr(e.message));
    const qs = gapId ? `gap_id=${gapId}` : pid ? `paper_id=${pid}` : "";
    api(`/recommendations?${qs}`).then((r) => setRecs(r.recommendations)).catch(() => {});
  }, [pid, gapId]);
  return (<>
    <h1>Discover</h1>
    <div className="card row">
      <select style={{ maxWidth: 420 }} value={pid} onChange={(e) => setPid(e.target.value)} aria-label="Centre graph on paper">
        <option value="">Whole library</option>{papers.filter((p) => p.status === "ready").map((p) => <option key={p.id} value={p.id}>{p.title?.slice(0, 70)}</option>)}</select>
      <span className="legend">{Object.entries(COLORS).map(([k, c]: any) => <span key={k}><i style={{ background: c }} />{k}</span>)}</span>
    </div>
    {err && <p className="err">{err}</p>}
    <div className="graphbox" ref={box}>
      <ForceGraph graphData={graph} width={w} height={460} nodeLabel="label" nodeColor={(n: any) => COLORS[n.type] || "#999"}
        nodeVal={(n: any) => (n.type === "Paper" ? 4 : 2)} linkColor={() => "#c4ccd0"} linkLabel="type" linkDirectionalArrowLength={3} />
    </div>
    <div className="card" style={{ marginTop: "1rem" }}>
      <h3>{gapId ? "Papers most relevant to the selected gap" : pid ? "Related papers" : "Recommended reading"}</h3>
      {recs.length === 0 ? <div className="empty">Nothing to recommend yet.</div> : recs.map((r) => <div key={r.paper_id} style={{ padding: ".4rem 0", borderTop: "1px solid var(--rule)" }}><strong>{r.title}</strong> <span className="muted">({r.year || "n.d."}) — {r.reason}</span></div>)}
    </div>
  </>);
}

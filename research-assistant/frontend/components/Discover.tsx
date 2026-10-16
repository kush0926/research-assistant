"use client";
import dynamic from "next/dynamic";
import { useEffect, useRef, useState } from "react";
import { ArrowRight, ArrowsClockwise, ArrowsOutSimple, Graph, WarningCircle } from "@phosphor-icons/react";
import { api } from "@/lib/api";
import "./Analysis.css";

const ForceGraph = dynamic(() => import("react-force-graph-2d"), { ssr: false });
const COLORS: Record<string, string> = { Paper: "#0e6b66", Method: "#b5844b", Dataset: "#698caa", Topic: "#85998d", Author: "#b7ada0" };
const tooltipText = (value: unknown) => String(value ?? "").replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[character] || character));

export default function Discover({ papers, gapId }: { papers: any[]; gapId: number | null }) {
  const [pid, setPid] = useState("");
  const [graph, setGraph] = useState<any>({ nodes: [], links: [] });
  const [recs, setRecs] = useState<any[]>([]);
  const [graphError, setGraphError] = useState("");
  const [recError, setRecError] = useState("");
  const [graphBusy, setGraphBusy] = useState(true);
  const [recBusy, setRecBusy] = useState(true);
  const [refresh, setRefresh] = useState(0);
  const [focused, setFocused] = useState<any>(null);
  const box = useRef<HTMLDivElement>(null);
  const graphRef = useRef<any>(null);
  const fitted = useRef(false);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const element = box.current;
    if (!element) return;
    const update = () => setWidth(Math.floor(element.clientWidth));
    update();
    const observer = new ResizeObserver(update);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    let current = true;
    setGraphBusy(true); setRecBusy(true); setGraphError(""); setRecError(""); setFocused(null); fitted.current = false;
    api(`/knowledge-graph${pid ? `?paper_id=${pid}` : ""}`).then((g) => {
      if (current) setGraph({ nodes: g.nodes || [], links: g.edges || [] });
    }).catch((e) => { if (current) { setGraphError(e.message); setGraph({ nodes: [], links: [] }); } }).finally(() => { if (current) setGraphBusy(false); });
    const qs = gapId ? `gap_id=${gapId}` : pid ? `paper_id=${pid}` : "";
    api(`/recommendations${qs ? `?${qs}` : ""}`).then((r) => { if (current) setRecs(r.recommendations || []); }).catch((e) => { if (current) { setRecError(e.message); setRecs([]); } }).finally(() => { if (current) setRecBusy(false); });
    return () => { current = false; };
  }, [pid, gapId, refresh]);
  function fitGraph() { graphRef.current?.zoomToFit(400, 42); }
  const ready = papers.filter((p) => p.status === "ready");
  const title = gapId ? "Reading for your selected gap" : pid ? "Related papers" : "Recommended reading";
  return <div className="analysis-screen discover-screen">
    <section className="discover-controls" aria-label="Discovery settings"><div className="discover-scope"><label className="analysis-eyebrow" htmlFor="graph-scope">Explore connections around</label><select id="graph-scope" value={pid} onChange={(e) => setPid(e.target.value)}><option value="">Whole library</option>{ready.map((p) => <option key={p.id} value={p.id}>{p.title || "Untitled paper"}</option>)}</select></div><button className="btn ghost small" onClick={() => setRefresh((r) => r + 1)} disabled={graphBusy || recBusy}><ArrowsClockwise size={17} />Refresh</button></section>
    <section className="discover-graph-panel" aria-labelledby="graph-heading"><div className="analysis-section-header"><div><h2 id="graph-heading">Your research network</h2><p className="muted">{graphBusy ? "Loading connections…" : `${graph.nodes.length} entities · ${graph.links.length} connections`} · Drag to explore, scroll to zoom.</p></div><button className="btn ghost small" onClick={fitGraph} disabled={graphBusy || !graph.nodes.length}><ArrowsOutSimple size={17} />Fit graph</button></div>
      <div className="discover-legend" aria-label="Graph entity types">{Object.entries(COLORS).map(([type, color]) => <span key={type}><i style={{ backgroundColor: color }} />{type}</span>)}</div>
      <div className="discover-graph-canvas" ref={box} aria-busy={graphBusy}>
        {graphBusy ? <div className="discover-graph-empty" role="status"><Graph size={34} weight="light" /><p>Connecting papers, methods, and datasets…</p></div> : graphError ? <div className="discover-graph-empty"><WarningCircle size={30} /><h3>Could not load the graph</h3><p className="err" role="alert">{graphError}</p><button className="btn ghost small" onClick={() => setRefresh((r) => r + 1)}>Try again</button></div> : graph.nodes.length === 0 ? <div className="discover-graph-empty"><Graph size={34} weight="light" /><h3>Your connections will appear here</h3><p>Add research papers to your library. Ready papers connect to their extracted methods, datasets, and topics.</p></div> : width > 0 && <ForceGraph ref={graphRef} graphData={graph} width={width} height={460} backgroundColor="transparent" nodeLabel={(n: any) => tooltipText(`${n.label} (${n.type})`)} nodeColor={(n: any) => COLORS[n.type] || COLORS.Topic} nodeVal={(n: any) => n.type === "Paper" ? 4 : 2} nodeRelSize={5} linkColor={() => "#d6e0df"} linkLabel={(link: any) => tooltipText(link.type)} linkDirectionalArrowLength={3} linkDirectionalArrowRelPos={1} cooldownTicks={70} onNodeClick={(node: any) => setFocused(node)} onBackgroundClick={() => setFocused(null)} onEngineStop={() => { if (!fitted.current) { fitGraph(); fitted.current = true; } }} />}
      </div>
      {focused && <div className="discover-node-detail" role="status"><span className="analysis-eyebrow">{focused.type}</span><strong>{focused.label}</strong>{focused.year && <span className="muted">{focused.year}</span>}{focused.paper_id && Number(pid) !== focused.paper_id && <button className="btn ghost small" onClick={() => setPid(String(focused.paper_id))}>Explore this paper <ArrowRight size={15} /></button>}</div>}
      {graph.nodes.length > 0 && !graphBusy && <details className="discover-text-view"><summary>View network as a list</summary><ul>{graph.nodes.map((node: any) => <li key={node.id}><span>{node.type}</span><strong>{node.label}</strong>{node.paper_id && <button className="discover-text-action" onClick={() => setPid(String(node.paper_id))}>Explore paper</button>}</li>)}</ul></details>}
    </section>
    <section className="discover-reading-panel" aria-labelledby="reading-heading"><div className="analysis-section-header"><div><h2 id="reading-heading">{title}</h2><p className="muted">{gapId ? "Ranked by relevance to the selected candidate gap, content similarity, and citation links." : "From your library, using content similarity and citation connections."}</p></div>{!recBusy && <span className="analysis-count">{recs.length} papers</span>}</div>
      {recBusy ? <div className="analysis-loading" role="status"><span className="analysis-loading-line" /><span className="analysis-loading-line" /><p>Finding relevant papers…</p></div> : recError ? <div className="analysis-error err" role="alert"><WarningCircle size={18} /><span>Could not load recommendations: {recError}</span><button className="btn ghost small" onClick={() => setRefresh((r) => r + 1)}>Retry</button></div> : recs.length === 0 ? <div className="analysis-inline-empty">No related reading is available for this scope yet. Add more ready papers or explore the whole library.</div> : <ol className="discover-reading-list">{recs.map((r, i) => <li key={r.paper_id}><span className="discover-reading-rank">{String(i + 1).padStart(2, "0")}</span><div><span className="discover-reading-year">{r.year || "Year not available"}</span><h3>{r.title}</h3><p>{r.reason}</p></div><button className="icon-btn discover-reading-action" onClick={() => setPid(String(r.paper_id))} aria-label={`Explore connections for ${r.title}`} title="Explore this paper"><ArrowRight size={20} /></button></li>)}</ol>}
    </section>
  </div>;
}

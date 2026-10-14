"""Lightweight knowledge graph (Part 16) built from Postgres with NetworkX; derived, never a second source of truth."""
import networkx as nx
from sqlalchemy.orm import Session

from ..db import (Author, Citation, Dataset, Method, Paper, PaperAuthor, PaperDataset, PaperMethod, PaperTopic, Topic)


def build_graph(db: Session, user_id: int) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    papers = {p.id: p for p in db.query(Paper).filter(Paper.upload_user_id == user_id, Paper.status == "ready")}
    for p in papers.values():
        G.add_node(f"paper:{p.id}", label=(p.title or "Untitled")[:70], type="Paper", year=p.year, paper_id=p.id)
    ids = list(papers)
    ds = {d.id: d.name for d in db.query(Dataset)}
    ms = {m.id: m.name for m in db.query(Method)}
    tp = {t.id: t.label for t in db.query(Topic)}
    au = {a.id: a.name for a in db.query(Author)}
    methods_of, datasets_of = {}, {}
    for r in db.query(PaperDataset).filter(PaperDataset.paper_id.in_(ids)):
        G.add_node(f"dataset:{ds[r.dataset_id]}", label=ds[r.dataset_id], type="Dataset")
        G.add_edge(f"paper:{r.paper_id}", f"dataset:{ds[r.dataset_id]}", type="USES")
        datasets_of.setdefault(r.paper_id, []).append(ds[r.dataset_id])
    for r in db.query(PaperMethod).filter(PaperMethod.paper_id.in_(ids)):
        n = ms[r.method_id]
        G.add_node(f"method:{n}", label=n, type="Method")
        G.add_edge(f"paper:{r.paper_id}", f"method:{n}", type="PROPOSES" if r.role == "proposed" else "APPLIES")
        methods_of.setdefault(r.paper_id, []).append(n)
    for pid, mlist in methods_of.items():  # Method -EVALUATED_ON-> Dataset
        for m in mlist:
            for d in datasets_of.get(pid, []):
                if not G.has_edge(f"method:{m}", f"dataset:{d}"):
                    G.add_edge(f"method:{m}", f"dataset:{d}", type="EVALUATED_ON")
    for r in db.query(PaperTopic).filter(PaperTopic.paper_id.in_(ids)):
        G.add_node(f"topic:{tp[r.topic_id]}", label=tp[r.topic_id], type="Topic")
        G.add_edge(f"paper:{r.paper_id}", f"topic:{tp[r.topic_id]}", type="BELONGS_TO")
    for r in db.query(PaperAuthor).filter(PaperAuthor.paper_id.in_(ids)):
        G.add_node(f"author:{au[r.author_id]}", label=au[r.author_id], type="Author")
        G.add_edge(f"paper:{r.paper_id}", f"author:{au[r.author_id]}", type="AUTHORED_BY")
    for r in db.query(Citation).filter(Citation.citing_paper_id.in_(ids)):
        if r.cited_paper_id in papers:
            G.add_edge(f"paper:{r.citing_paper_id}", f"paper:{r.cited_paper_id}", type="CITES")
    return G


def graph_view(db: Session, user_id: int, paper_id: int | None = None, topic: str | None = None, hops: int = 2) -> dict:
    G = build_graph(db, user_id)
    if paper_id is not None and f"paper:{paper_id}" in G:
        keep = nx.ego_graph(G.to_undirected(as_view=True), f"paper:{paper_id}", radius=hops).nodes
        G = G.subgraph(keep)
    elif topic:
        seeds = [n for n, d in G.nodes(data=True) if d["type"] == "Topic" and topic.lower() in d["label"].lower()]
        keep = set(seeds)
        for s in seeds:
            keep |= set(nx.ego_graph(G.to_undirected(as_view=True), s, radius=hops).nodes)
        G = G.subgraph(keep)
    nodes = [{"id": n, **d} for n, d in G.nodes(data=True)]
    edges = [{"source": u, "target": v, "type": d["type"]} for u, v, d in G.edges(data=True)]
    return {"nodes": nodes, "edges": edges}

"""External paper search (arXiv, Semantic Scholar, OpenAlex) + safe open-access PDF download (Part 20)."""
import ipaddress
import socket
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import httpx

from ..config import settings

UA = {"User-Agent": "ResearchAssistant/1.0 (academic project)"}


def search_arxiv(q: str, limit: int = 10) -> list[dict]:
    r = httpx.get("http://export.arxiv.org/api/query", headers=UA, timeout=20,
                  params={"search_query": f"all:{q}", "start": 0, "max_results": limit, "sortBy": "relevance"})
    r.raise_for_status()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in ET.fromstring(r.text).findall("a:entry", ns):
        aid = e.findtext("a:id", namespaces=ns).rsplit("/abs/", 1)[-1]
        out.append({"source": "arxiv", "external_id": aid, "title": " ".join(e.findtext("a:title", namespaces=ns).split()),
                    "abstract": " ".join((e.findtext("a:summary", namespaces=ns) or "").split()),
                    "year": int(e.findtext("a:published", namespaces=ns)[:4]),
                    "authors": [a.findtext("a:name", namespaces=ns) for a in e.findall("a:author", ns)],
                    "pdf_url": f"https://arxiv.org/pdf/{aid}", "doi": None, "citation_count": None,
                    "url": f"https://arxiv.org/abs/{aid}"})
    return out


def search_semantic_scholar(q: str, limit: int = 10) -> list[dict]:
    headers = dict(UA)
    if settings.SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = settings.SEMANTIC_SCHOLAR_API_KEY
    r = httpx.get("https://api.semanticscholar.org/graph/v1/paper/search", headers=headers, timeout=20,
                  params={"query": q, "limit": limit,
                          "fields": "title,abstract,year,authors,externalIds,openAccessPdf,citationCount,venue,url"})
    r.raise_for_status()
    return [{"source": "semanticscholar", "external_id": p["paperId"], "title": p.get("title"),
             "abstract": p.get("abstract") or "", "year": p.get("year"),
             "authors": [a["name"] for a in p.get("authors", [])],
             "pdf_url": (p.get("openAccessPdf") or {}).get("url"), "doi": (p.get("externalIds") or {}).get("DOI"),
             "citation_count": p.get("citationCount"), "venue": p.get("venue"), "url": p.get("url")}
            for p in r.json().get("data", [])]


def search_openalex(q: str, limit: int = 10) -> list[dict]:
    r = httpx.get("https://api.openalex.org/works", headers=UA, timeout=20, params={"search": q, "per-page": limit})
    r.raise_for_status()
    out = []
    for w in r.json().get("results", []):
        inv = w.get("abstract_inverted_index") or {}
        words = sorted(((pos, word) for word, poss in inv.items() for pos in poss))
        out.append({"source": "openalex", "external_id": w["id"].rsplit("/", 1)[-1], "title": w.get("title"),
                    "abstract": " ".join(wd for _, wd in words), "year": w.get("publication_year"),
                    "authors": [a["author"]["display_name"] for a in w.get("authorships", [])],
                    "pdf_url": (w.get("open_access") or {}).get("oa_url"), "doi": (w.get("doi") or "").replace("https://doi.org/", "") or None,
                    "citation_count": w.get("cited_by_count"), "url": w.get("id")})
    return out


def search(q: str, source: str = "semanticscholar", limit: int = 10) -> list[dict]:
    fn = {"arxiv": search_arxiv, "semanticscholar": search_semantic_scholar, "openalex": search_openalex}.get(source)
    if not fn:
        raise ValueError("source must be arxiv | semanticscholar | openalex")
    return fn(q, limit)


def _is_safe_url(url: str) -> bool:
    """SSRF guard: https only, no IP literals, hostname must resolve to public addresses."""
    u = urlparse(url)
    if u.scheme != "https" or not u.hostname:
        return False
    try:
        ipaddress.ip_address(u.hostname)
        return False
    except ValueError:
        pass
    try:
        return all(ipaddress.ip_address(i[4][0]).is_global for i in socket.getaddrinfo(u.hostname, 443))
    except Exception:
        return False


def download_pdf(url: str) -> bytes:
    if not _is_safe_url(url):
        raise ValueError("URL rejected (must be a public https URL)")
    with httpx.stream("GET", url, headers=UA, timeout=60, follow_redirects=True) as r:
        r.raise_for_status()
        data = bytearray()
        for chunk in r.iter_bytes():
            data += chunk
            if len(data) > settings.MAX_UPLOAD_MB * 1024 * 1024:
                raise ValueError("PDF too large")
    return bytes(data)

"""External paper search (arXiv, Semantic Scholar, OpenAlex) + safe open-access PDF download (Part 20)."""
import ipaddress
import re
import socket
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import httpx

from ..config import settings

UA = {"User-Agent": "ResearchAssistant/1.0 (academic project)"}
MAX_PDF_REDIRECTS = 5


def search_arxiv(q: str, limit: int = 10) -> list[dict]:
    terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9._+-]*", q)
    if not terms:
        return []
    search_query = " AND ".join(f"all:{term}" for term in terms)
    r = httpx.get("https://export.arxiv.org/api/query", headers=UA, timeout=20,
                  params={"search_query": search_query, "start": 0, "max_results": limit, "sortBy": "relevance"})
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
    try:
        u = urlparse(url)
        port = u.port
    except ValueError:
        return False
    if u.scheme != "https" or not u.hostname or u.username or u.password or port not in (None, 443):
        return False
    try:
        ipaddress.ip_address(u.hostname)
        return False
    except ValueError:
        pass
    try:
        addresses = socket.getaddrinfo(u.hostname, 443)
        return bool(addresses) and all(ipaddress.ip_address(i[4][0]).is_global for i in addresses)
    except Exception:
        return False


def download_pdf(url: str) -> bytes:
    current = url
    with httpx.Client(timeout=60, follow_redirects=False) as client:
        for redirect_count in range(MAX_PDF_REDIRECTS + 1):
            if not _is_safe_url(current):
                raise ValueError("URL rejected (must be a public https URL)")
            with client.stream("GET", current, headers=UA) as r:
                if r.is_redirect:
                    location = r.headers.get("location")
                    if not location or redirect_count == MAX_PDF_REDIRECTS:
                        raise ValueError("Too many or invalid PDF redirects")
                    current = str(r.url.join(location))
                    continue
                r.raise_for_status()
                data = bytearray()
                for chunk in r.iter_bytes():
                    data.extend(chunk)
                    if len(data) > settings.MAX_UPLOAD_MB * 1024 * 1024:
                        raise ValueError("PDF too large")
                return bytes(data)
    raise ValueError("Too many PDF redirects")

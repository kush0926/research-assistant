"""PDF validation -> sandboxed parse -> clean -> section detection -> chunking -> metadata (Part 11 steps 1-6)."""
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
import re
from collections import Counter

import httpx

from ..config import settings
from ..security import sanitize_document_text
from .pdf_worker import classify_heading

log = logging.getLogger("pdf")


class PDFError(Exception):
    pass


# ---------- 1. validation ----------
def validate_pdf_bytes(data: bytes, content_type: str | None = None):
    if len(data) > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise PDFError(f"File too large (max {settings.MAX_UPLOAD_MB} MB)")
    if content_type and content_type not in ("application/pdf", "application/octet-stream", "application/x-pdf"):
        raise PDFError("Only PDF files are accepted")
    if not data.startswith(b"%PDF-"):
        raise PDFError("File is not a valid PDF (bad header)")
    if b"%%EOF" not in data[-2048:]:
        raise PDFError("Malformed PDF (missing EOF marker)")


# ---------- 2. sandboxed parse ----------
def parse_sandboxed(path: str) -> dict:
    """Parse in an isolated child process: memory/CPU rlimits (POSIX) + wall-clock timeout. Never executes PDF JS."""
    backend_dir = str(Path(__file__).resolve().parent.parent.parent)
    cmd = [sys.executable, "-m", "app.services.pdf_worker", path, str(settings.MAX_PDF_PAGES),
           str(settings.PARSE_MEM_MB), str(settings.PARSE_TIMEOUT_S + 5)]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=settings.PARSE_TIMEOUT_S, cwd=backend_dir,
                           env={**os.environ, "PYTHONPATH": backend_dir})
    except subprocess.TimeoutExpired:
        raise PDFError("PDF parsing timed out (sandbox)")
    try:
        out = json.loads(r.stdout.decode("utf-8", "replace"))
    except Exception:
        raise PDFError("PDF parser crashed (sandbox): " + r.stderr.decode("utf-8", "replace")[-200:])
    if "err" in out:
        raise PDFError(f"PDF parse failed: {out['err']}")
    return out["ok"]


# ---------- 3. cleaning ----------
def clean_pages(pages: list[list[dict]]) -> list[list[dict]]:
    """Drop repeated headers/footers and page numbers, keep reading order."""
    n = len(pages)
    counts = Counter()
    for lines in pages:
        for ln in set(re.sub(r"\d+", "#", l["text"]).strip().lower() for l in lines):
            counts[ln] += 1
    out = []
    for lines in pages:
        keep = []
        for l in lines:
            norm = re.sub(r"\d+", "#", l["text"]).strip().lower()
            if re.fullmatch(r"[\-–\s#]*#[\-–\s#]*|page #( of #)?", norm):
                continue
            if n >= 3 and counts[norm] >= max(3, 0.5 * n) and len(norm) < 90:
                continue
            keep.append(l)
        out.append(keep)
    return out


def _dehyphenate_join(lines: list[str]) -> str:
    text = ""
    for ln in lines:
        if text.endswith("-") and ln[:1].islower():
            text = text[:-1] + ln
        else:
            text += (" " if text else "") + ln
    return re.sub(r"\s+", " ", text).strip()


# ---------- 4. section detection ----------
def detect_sections(pages: list[list[dict]], body_size: float) -> list[dict]:
    """Returns [{section, page, text}] using heading heuristics (font size/bold + known names)."""
    sections: list[dict] = []
    cur = {"section": "front", "page": 1, "lines": []}
    for lines in pages:
        for l in lines:
            sec = classify_heading(l["text"]) if (l["bold"] or l["size"] >= body_size * 1.05 or len(l["text"]) < 40) else None
            if sec and len(l["text"]) < 70:
                sections.append(cur)
                cur = {"section": sec, "page": l["page"], "lines": []}
                continue
            # abstract often inline: "Abstract— text" / "Abstract. text"
            m = re.match(r"^\s*abstract\s*[—\-–:.]\s*(.+)", l["text"], re.I)
            if m and cur["section"] == "front":
                sections.append(cur)
                cur = {"section": "abstract", "page": l["page"], "lines": [m.group(1)]}
                continue
            cur["lines"].append(l["text"])
            cur.setdefault("last_page", l["page"])
            cur["last_page"] = l["page"]
    sections.append(cur)
    out = []
    for s in sections:
        text = _dehyphenate_join(s["lines"])
        if text:
            out.append({"section": s["section"], "page": s["page"], "text": text})
    return out


# ---------- 5. metadata ----------
def extract_metadata(pages: list[list[dict]], pdf_meta: dict, sections: list[dict]) -> dict:
    first = pages[0] if pages else []
    title, authors = "", []
    if pdf_meta.get("title") and len(pdf_meta["title"]) > 8 and not pdf_meta["title"].lower().endswith(".pdf"):
        title = pdf_meta["title"].strip()
    if first:
        biggest = max(l["size"] for l in first)
        tl = [l for l in first if abs(l["size"] - biggest) < 0.6][:4]
        guess = " ".join(l["text"] for l in tl).strip()
        if not title or len(guess) > len(title) * 0.6:
            title = guess or title
        idx = first.index(tl[-1]) + 1 if tl else 0
        for l in first[idx:idx + 4]:
            t = l["text"]
            if re.search(r"abstract|university|institute|department|@|college|synthetic", t, re.I):
                continue
            if re.fullmatch(r"[A-Z][A-Za-z.\-' ]+(?:,| and | & )[A-Za-z.,\-' &]+", t) and len(t) < 160:
                authors = [a.strip() for a in re.split(r",| and | & ", t) if len(a.strip()) > 2]
                break
    if not authors and pdf_meta.get("author"):
        authors = [a.strip() for a in re.split(r";|,| and ", pdf_meta["author"]) if a.strip()]
    full = " ".join(s["text"] for s in sections[:3])
    year = None
    m = re.search(r"\b(19[89]\d|20[0-3]\d)\b", " ".join(l["text"] for l in first[:40]))
    if m:
        year = int(m.group(1))
    elif pdf_meta.get("creationDate"):
        mm = re.search(r"(20\d\d|19\d\d)", pdf_meta["creationDate"])
        year = int(mm.group(1)) if mm else None
    doi = re.search(r"\b10\.\d{4,9}/[^\s\"<>]+", full + " ".join(l["text"] for l in first))
    venue = None
    vm = re.search(r"(?:Proceedings of|Journal of|Transactions on|IEEE|ACM|Springer|arXiv)[^.\n]{0,80}", " ".join(l["text"] for l in first[:25]))
    if vm:
        venue = vm.group(0).strip()
    return {"title": title, "authors": authors, "year": year, "doi": doi.group(0).rstrip(".,;") if doi else None,
            "venue": venue}


def crossref_lookup(title: str) -> dict:
    """Cross-check metadata against Crossref (best effort; silent on failure/offline)."""
    if not settings.ENABLE_METADATA_LOOKUP or len(title) < 12:
        return {}
    try:
        r = httpx.get("https://api.crossref.org/works", params={"query.bibliographic": title, "rows": 1}, timeout=4)
        it = r.json()["message"]["items"][0]
        ct = (it.get("title") or [""])[0].lower()
        if ct and (ct in title.lower() or title.lower() in ct):
            yr = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
            return {"doi": it.get("DOI"), "year": yr, "venue": (it.get("container-title") or [None])[0]}
    except Exception:
        pass
    return {}


# ---------- 6. chunking ----------
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def chunk_sections(sections: list[dict]) -> list[dict]:
    """Section-aware sliding window of whole sentences (~CHUNK_WORDS words, CHUNK_OVERLAP overlap)."""
    chunks, idx = [], 0
    max_w, ov = settings.CHUNK_WORDS, settings.CHUNK_OVERLAP
    for s in sections:
        if s["section"] == "references":
            continue
        sents = split_sentences(s["text"])
        cur, cur_w = [], 0
        for sent in sents:
            w = len(sent.split())
            if cur and cur_w + w > max_w:
                chunks.append({"section": s["section"], "page": s["page"], "text": " ".join(cur), "chunk_index": idx})
                idx += 1
                # carry over trailing sentences for ~overlap
                carry, cw = [], 0
                for x in reversed(cur):
                    if cw >= max_w * ov:
                        break
                    carry.insert(0, x); cw += len(x.split())
                cur, cur_w = carry, cw
            cur.append(sent); cur_w += w
        if cur:
            chunks.append({"section": s["section"], "page": s["page"], "text": " ".join(cur), "chunk_index": idx})
            idx += 1
    for c in chunks:
        c["text"], c["suspicious"] = sanitize_document_text(c["text"])
    return chunks


# ---------- optional GROBID path ----------
def grobid_sections(path: str) -> dict | None:
    if not settings.GROBID_URL:
        return None
    try:
        httpx.get(f"{settings.GROBID_URL}/api/isalive", timeout=2).raise_for_status()  # fail fast, don't wait on a dead URL
    except Exception:
        log.warning("GROBID_URL is set but %s is not reachable; using PyMuPDF", settings.GROBID_URL)
        return None
    try:
        from xml.etree import ElementTree as ET
        with open(path, "rb") as f:
            r = httpx.post(f"{settings.GROBID_URL}/api/processFulltextDocument", files={"input": f},
                           data={"consolidateHeader": "0"}, timeout=120)
        r.raise_for_status()
        ns = {"t": "http://www.tei-c.org/ns/1.0"}
        root = ET.fromstring(r.text)
        title = (root.findtext(".//t:titleStmt/t:title", namespaces=ns) or "").strip()
        authors = []
        for a in root.findall(".//t:sourceDesc//t:author/t:persName", ns):
            nm = " ".join(x.text for x in a if x.text)
            if nm:
                authors.append(nm)
        abstract = " ".join("".join(p.itertext()) for p in root.findall(".//t:profileDesc/t:abstract//t:p", ns))
        secs = [{"section": "abstract", "page": 1, "text": abstract}] if abstract else []
        from .pdf_worker import SECTION_PATTERNS
        for div in root.findall(".//t:body/t:div", ns):
            head = (div.findtext("t:head", namespaces=ns) or "").strip().lower()
            text = " ".join("".join(p.itertext()) for p in div.findall("t:p", ns))
            sec = "body"
            for name, pat in SECTION_PATTERNS:
                if re.search(pat, head):
                    sec = name; break
            if text:
                secs.append({"section": sec, "page": 1, "text": text})
        refs = [" ".join("".join(b.itertext()).split()) for b in root.findall(".//t:listBibl/t:biblStruct", ns)]
        if refs:
            secs.append({"section": "references", "page": 1, "text": "\n".join(refs)})
        dm = root.findtext(".//t:publicationStmt/t:date", namespaces=ns) or ""
        ym = re.search(r"(19|20)\d\d", dm or "")
        return {"sections": secs, "meta": {"title": title, "authors": authors, "year": int(ym.group(0)) if ym else None}}
    except Exception as e:
        log.warning("GROBID failed (%s); falling back to PyMuPDF", e)
        return None


def process_pdf(path: str) -> dict:
    """Full parse. Returns {meta, sections, chunks, references_text, page_count}."""
    g = grobid_sections(path)
    page_count = None
    if g:
        sections, meta = g["sections"], g["meta"]
    else:
        parsed = parse_sandboxed(path)
        page_count = parsed["page_count"]
        pages = clean_pages(parsed["pages"])
        sections = detect_sections(pages, parsed["body_size"])
        meta = extract_metadata(pages, parsed["pdf_meta"], sections)
    cr = crossref_lookup(meta.get("title", ""))
    for k, v in cr.items():
        if v and not meta.get(k):
            meta[k] = v
    if not sections or sum(len(s["text"]) for s in sections) < 200:
        raise PDFError("No extractable text (scanned PDFs need OCR — out of scope)")
    refs = next((s["text"] for s in sections if s["section"] == "references"), "")
    abstract = next((s["text"] for s in sections if s["section"] == "abstract"), "")
    if not abstract:  # fall back to first body paragraph
        first = next((s for s in sections if s["section"] in ("front", "introduction")), sections[0])
        abstract = " ".join(split_sentences(first["text"])[:5])
    meta["abstract"] = abstract
    return {"meta": meta, "sections": sections, "chunks": chunk_sections(sections),
            "references_text": refs, "page_count": page_count}

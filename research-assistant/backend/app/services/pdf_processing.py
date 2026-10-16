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
    """Parse in a child process with memory/CPU limits and a wall-clock timeout. Never executes PDF JS."""
    backend_dir = str(Path(__file__).resolve().parent.parent.parent)
    cmd = [sys.executable, "-m", "app.services.pdf_worker", path, str(settings.MAX_PDF_PAGES),
           str(settings.PARSE_MEM_MB), str(settings.PARSE_TIMEOUT_S + 5)]
    job = None
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=backend_dir,
                                   env={**os.environ, "PYTHONPATH": backend_dir})
        if os.name == "nt":
            try:
                job = _limit_windows_parser_process(process, settings.PARSE_MEM_MB, settings.PARSE_TIMEOUT_S + 5)
            except Exception as e:
                process.kill()
                process.communicate()
                raise PDFError(f"Could not apply Windows PDF parser resource limits: {e}") from e
        try:
            stdout, stderr = process.communicate(timeout=settings.PARSE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise PDFError("PDF parsing timed out (sandbox)")
    except subprocess.TimeoutExpired:
        raise PDFError("PDF parsing timed out (sandbox)")
    finally:
        if job is not None:
            _close_windows_job(job)
    try:
        out = json.loads(stdout.decode("utf-8", "replace"))
    except Exception:
        raise PDFError("PDF parser crashed (sandbox): " + stderr.decode("utf-8", "replace")[-200:])
    if "err" in out:
        raise PDFError(f"PDF parse failed: {out['err']}")
    return out["ok"]


def _limit_windows_parser_process(process: subprocess.Popen, memory_mb: int, cpu_seconds: int):
    """Apply Windows Job Object memory and CPU-time limits to the parser child."""
    import ctypes
    from ctypes import wintypes

    class BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in
                    ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                     "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimitInformation),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    info = ExtendedLimitInformation()
    # JOB_OBJECT_LIMIT_PROCESS_TIME | JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    info.BasicLimitInformation.LimitFlags = 0x00000002 | 0x00000100 | 0x00002000
    info.BasicLimitInformation.PerProcessUserTimeLimit = cpu_seconds * 10_000_000
    info.ProcessMemoryLimit = memory_mb * 1024 * 1024
    try:
        if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())
        return job
    except Exception:
        kernel32.CloseHandle(job)
        raise


def _close_windows_job(job):
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    if not kernel32.CloseHandle(job):
        log.warning("Failed to close parser Job Object handle: %s", ctypes.WinError(ctypes.get_last_error()))


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
            if l["page"] != cur["page"]:
                sections.append(cur)
                cur = {"section": cur["section"], "page": l["page"], "lines": []}
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
    if (pdf_meta.get("title") and len(pdf_meta["title"]) > 8
            and not re.search(r"\.pdf$|^untitled$|^arxiv:", pdf_meta["title"], re.I)):
        title = pdf_meta["title"].strip()
    if first:
        eligible = [line for line in first if not re.search(r"arxiv\s*:|https?://|^copyright|^©", line["text"], re.I)]
        biggest = max((line["size"] for line in eligible), default=0)
        tl = [line for line in eligible if abs(line["size"] - biggest) < 0.6][:4]
        guess = " ".join(l["text"] for l in tl).strip()
        if not title:
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
    # Prefer the explicit arXiv revision date, after removing the numeric identifier (e.g. 2005.14647).
    arxiv_dates = [re.sub(r"arxiv\s*:\s*\S+", "", line["text"], flags=re.I)
                   for line in first if re.search(r"arxiv\s*:", line["text"], re.I)]
    m = re.search(r"\b((?:19|20)\d\d)\b", " ".join(arxiv_dates))
    if not m:
        front = []
        for line in first:
            if re.match(r"\s*(?:abstract|(?:\d+\.?\s+)?introduction)\b", line["text"], re.I):
                break
            front.append(re.sub(r"arxiv\s*:\s*\S+", "", line["text"], flags=re.I))
        m = re.search(r"\b(19[89]\d|20[0-3]\d)\b", " ".join(front))
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
        if venue.lower().startswith("arxiv"):
            venue = "arXiv"
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
def _grobid_tei_sections(tei: str) -> dict:
    from xml.etree import ElementTree as ET
    from .pdf_worker import SECTION_PATTERNS

    ns = {"t": "http://www.tei-c.org/ns/1.0"}
    root = ET.fromstring(tei)
    sections = []

    def add_element(element, section):
        # Sentence coordinates preserve the start page when a paragraph spans multiple pages.
        for part in element.findall(".//t:s", ns) or [element]:
            separator = " " if part.tag.rsplit("}", 1)[-1] == "figure" else ""
            text = " ".join(separator.join(part.itertext()).split())
            if not text:
                continue
            coordinates = part.get("coords") or element.get("coords")
            if not coordinates:
                raise ValueError("GROBID text has no source-page coordinates")
            page = int(coordinates.split(",", 1)[0])
            if page < 1:
                raise ValueError("GROBID returned an invalid source page")
            if page > settings.MAX_PDF_PAGES:
                continue
            if sections and sections[-1]["section"] == section and sections[-1]["page"] == page:
                sections[-1]["text"] += "\n" + text
            else:
                sections.append({"section": section, "page": page, "text": text})

    for paragraph in root.findall(".//t:profileDesc/t:abstract//t:p", ns):
        add_element(paragraph, "abstract")

    def visit(container, inherited="body"):
        heading = container.find("t:head", ns)
        head = " ".join("".join(heading.itertext()).split()).lower() if heading is not None else ""
        section = next((name for name, pattern in SECTION_PATTERNS if re.search(pattern, head)), inherited)
        for child in container:
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "div":
                visit(child, section)
            elif tag == "p":
                add_element(child, section)
            elif tag == "figure":
                # GROBID uses figure elements for both figure captions and table content.
                add_element(child, section)

    body = root.find(".//t:body", ns)
    if body is not None:
        visit(body)
    for reference in root.findall(".//t:listBibl/t:biblStruct", ns):
        add_element(reference, "references")

    title = (root.findtext(".//t:titleStmt/t:title", namespaces=ns) or "").strip()
    authors = [" ".join("".join(author.itertext()).split())
               for author in root.findall(".//t:sourceDesc//t:author/t:persName", ns)]
    dates = root.findall(".//t:sourceDesc//t:date", ns)
    publication_date = next((date for date in dates if date.get("type") == "published"), dates[0] if dates else None)
    date_text = (publication_date.get("when") or " ".join(publication_date.itertext())) if publication_date is not None else ""
    year = re.search(r"(?:19|20)\d\d", date_text)
    doi = root.findtext(".//t:sourceDesc//t:idno[@type='DOI']", namespaces=ns)
    venue = root.findtext(".//t:sourceDesc//t:monogr/t:title[@level='j']", namespaces=ns)
    return {"sections": sections, "meta": {"title": title, "authors": [name for name in authors if name],
                                           "year": int(year.group(0)) if year else None,
                                           "doi": doi.strip() if doi else None, "venue": venue.strip() if venue else None}}


def grobid_sections(path: str) -> dict | None:
    if not settings.GROBID_URL:
        return None
    try:
        httpx.get(f"{settings.GROBID_URL}/api/isalive", timeout=2).raise_for_status()  # fail fast, don't wait on a dead URL
    except Exception:
        log.warning("GROBID_URL is set but %s is not reachable; using PyMuPDF", settings.GROBID_URL)
        return None
    try:
        with open(path, "rb") as f:
            r = httpx.post(f"{settings.GROBID_URL}/api/processFulltextDocument", files={"input": f},
                           data={"consolidateHeader": "0", "segmentSentences": "1",
                                 "teiCoordinates": ["p", "s", "head", "biblStruct", "figure"]}, timeout=120)
        r.raise_for_status()
        return _grobid_tei_sections(r.text)
    except Exception as e:
        log.warning("GROBID failed (%s); falling back to PyMuPDF", e)
        return None


def process_pdf(path: str) -> dict:
    """Full parse. Returns {meta, sections, chunks, references_text, page_count}."""
    # Always validate and bound parsing before sending a document to the optional service.
    parsed = parse_sandboxed(path)
    g = grobid_sections(path)
    page_count = parsed["page_count"]
    if g:
        sections, meta = g["sections"], g["meta"]
        fallback_meta = extract_metadata(clean_pages(parsed["pages"]), parsed["pdf_meta"], sections)
        for key, value in fallback_meta.items():
            if value and not meta.get(key):
                meta[key] = value
    else:
        pages = clean_pages(parsed["pages"])
        sections = detect_sections(pages, parsed["body_size"])
        meta = extract_metadata(pages, parsed["pdf_meta"], sections)
    cr = crossref_lookup(meta.get("title", ""))
    for k, v in cr.items():
        if v and not meta.get(k):
            meta[k] = v
    if not sections or sum(len(s["text"]) for s in sections) < 200:
        raise PDFError("No extractable text (scanned PDFs need OCR — out of scope)")
    refs = "\n".join(s["text"] for s in sections if s["section"] == "references")
    abstract = " ".join(s["text"] for s in sections if s["section"] == "abstract")
    if not abstract:  # fall back to first body paragraph
        first = next((s for s in sections if s["section"] in ("front", "introduction")), sections[0])
        abstract = " ".join(split_sentences(first["text"])[:5])
    meta["abstract"] = abstract
    return {"meta": meta, "sections": sections, "chunks": chunk_sections(sections),
            "references_text": refs, "page_count": page_count}

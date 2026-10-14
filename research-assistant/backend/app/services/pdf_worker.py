"""Sandboxed PDF parser. Runs in a separate process with CPU-time + memory caps (Part 23).
Never executes embedded JavaScript: PyMuPDF only extracts text/blocks."""
import re
import statistics
import sys

SECTION_PATTERNS = [
    ("abstract", r"abstract"),
    ("introduction", r"introduction|background|motivation"),
    ("related_work", r"related work|literature review|prior work|previous work"),
    ("method", r"method(s|ology)?|materials and methods|approach|proposed (method|system|model|framework)|model architecture|system design"),
    ("experiments", r"experiment(s|al)?( setup| settings?)?|datasets?|data collection|implementation details|evaluation (protocol|setup)"),
    ("results", r"results?( and discussion| and analysis)?|findings|performance (evaluation|analysis)"),
    ("discussion", r"discussion|analysis"),
    ("limitations", r"limitations?|threats to validity"),
    ("future_work", r"future (work|directions?|research|scope)"),
    ("conclusion", r"conclusions?( and future work)?|concluding remarks|summary"),
    ("references", r"references|bibliography"),
]
_HEAD_RE = re.compile(r"^\s*(?:(?:\d+|[IVX]+)[\.\)]?\s+|(?:\d+\.\d+\.?\s+))?([A-Za-z][A-Za-z \-&,/']{2,60})\s*:?\s*$")


def classify_heading(line: str):
    m = _HEAD_RE.match(line.strip())
    if not m:
        return None
    label = m.group(1).strip().lower()
    for sec, pat in SECTION_PATTERNS:
        if re.fullmatch(pat, label) or (len(label.split()) <= 4 and re.match(rf"^(?:{pat})\b", label)):
            return sec
    return None


def _limit_resources(mem_mb: int, cpu_s: int):
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
    except Exception:
        pass  # non-POSIX (Windows): timeout in parent still applies


def parse_pdf(path: str, max_pages: int = 80) -> dict:
    import pymupdf
    doc = pymupdf.open(path)
    if doc.needs_pass:
        raise ValueError("Encrypted PDF")
    n_pages = min(doc.page_count, max_pages)
    pages = []
    font_sizes = []
    for pno in range(n_pages):
        page = doc[pno]
        lines = []
        for b in page.get_text("dict")["blocks"]:
            if b.get("type") != 0:
                continue
            for ln in b["lines"]:
                txt = "".join(s["text"] for s in ln["spans"]).strip()
                if not txt:
                    continue
                size = max((s["size"] for s in ln["spans"]), default=10)
                bold = any("bold" in s["font"].lower() for s in ln["spans"])
                font_sizes.append(size)
                lines.append({"text": txt, "size": round(size, 1), "bold": bold, "y": ln["bbox"][1], "page": pno + 1})
        pages.append(lines)
    body = statistics.median(font_sizes) if font_sizes else 10
    meta = {k: v for k, v in (doc.metadata or {}).items() if v}
    return {"pages": pages, "body_size": body, "pdf_meta": meta, "page_count": doc.page_count}


def main(argv):
    """CLI: python -m app.services.pdf_worker <pdf> <max_pages> <mem_mb> <cpu_s>  -> JSON on stdout."""
    import json
    path, max_pages, mem_mb, cpu_s = argv[1], int(argv[2]), int(argv[3]), int(argv[4])
    _limit_resources(mem_mb, cpu_s)
    try:
        sys.stdout.write(json.dumps({"ok": parse_pdf(path, max_pages)}))
    except BaseException as e:  # noqa
        sys.stdout.write(json.dumps({"err": f"{type(e).__name__}: {e}"}))


if __name__ == "__main__":
    main(sys.argv)

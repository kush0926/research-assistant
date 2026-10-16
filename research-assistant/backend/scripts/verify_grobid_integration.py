"""Check live GROBID parsing and compare it with the PyMuPDF fallback on local PDFs."""
import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--url", default=os.environ.get("GROBID_URL", "http://127.0.0.1:8070"))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.corpus.glob("*.pdf"))[:args.limit]
    if args.limit < 1 or not paths:
        parser.error("Choose a positive limit and a corpus directory containing PDFs.")
    os.environ.update({"GROBID_URL": args.url.rstrip("/"), "ENABLE_METADATA_LOOKUP": "false"})
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.services import pdf_processing as pdf

    selected = []
    original = pdf.grobid_sections

    def checked(path):
        result = original(path)
        selected.append(bool(result))
        return result

    pdf.grobid_sections = checked
    rows = []
    try:
        for path in paths:
            started = time.monotonic()
            result = pdf.process_pdf(str(path))
            elapsed = time.monotonic() - started
            assert selected[-1], f"GROBID fell back for {path.name}; this does not verify the live parser."
            assert result["chunks"] and result["meta"]["title"] and result["page_count"]
            assert all(1 <= chunk["page"] <= result["page_count"] for chunk in result["chunks"])
            raw = pdf.parse_sandboxed(str(path))
            pages = pdf.clean_pages(raw["pages"])
            fallback_sections = pdf.detect_sections(pages, raw["body_size"])
            fallback_meta = pdf.extract_metadata(pages, raw["pdf_meta"], fallback_sections)
            rows.append({"filename": path.name, "grobid_seconds": round(elapsed, 2), "grobid_selected": True,
                         "page_count": result["page_count"],
                         "chunk_pages": sorted({chunk["page"] for chunk in result["chunks"]}),
                         "chunks": len(result["chunks"]),
                         "sections": sorted({section["section"] for section in result["sections"]}),
                         "title": result["meta"]["title"], "year": result["meta"].get("year"),
                         "authors": len(result["meta"].get("authors", [])),
                         "references_chars": len(result["references_text"]),
                         "fallback_title": fallback_meta.get("title"), "fallback_year": fallback_meta.get("year"),
                         "fallback_sections": sorted({section["section"] for section in fallback_sections})})
            print(f"GROBID verified {path.name}: {len(result['chunks'])} chunks in {elapsed:.2f}s", file=sys.stderr, flush=True)
    finally:
        pdf.grobid_sections = original

    report = {"status": "passed", "grobid_url": args.url, "papers": rows,
              "note": "Integration and source-page checks; extraction accuracy still requires reviewed labels."}
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

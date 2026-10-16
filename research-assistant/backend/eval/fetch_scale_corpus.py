"""Build a unique public Parkinson/speech load corpus from a saved arXiv feed.

Uses one connection and at least three seconds between outbound requests.
Existing pilot files are reused. PDFs are checked in the application's parser
sandbox before being admitted; downloaded content is never treated as instructions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.fetch_parkvoice_corpus import fetch


def candidates(feed):
    ns = {"a": "http://www.w3.org/2005/Atom"}
    result = []
    for entry in ET.fromstring(feed.read_bytes()).findall("a:entry", ns):
        get = lambda key: " ".join((entry.findtext("a:" + key, namespaces=ns) or "").split())
        identifier = get("id").rsplit("/abs/", 1)[-1]
        if not re.fullmatch(r"\d{4}\.\d{4,5}v\d+", identifier):
            continue
        title, abstract = get("title"), get("summary")
        text = (title + " " + abstract).lower()
        if not re.search(r"parkinson|dysarthri", text) or not re.search(r"speech|voice|audio|vocal|phonat|acoustic", text):
            continue
        result.append({"source": "arxiv", "external_id": identifier, "title": title,
                       "year": int(get("published")[:4]), "abstract": abstract,
                       "authors": [" ".join((a.findtext("a:name", namespaces=ns) or "").split())
                                   for a in entry.findall("a:author", ns)],
                       "url": f"https://arxiv.org/abs/{identifier}",
                       "pdf_url": f"https://arxiv.org/pdf/{identifier}",
                       "filename": "arxiv_" + identifier.replace(".", "_") + ".pdf"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feed", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--destination", type=Path, default=ROOT / "data_store/parkvoice_scale_corpus")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    from app.services.pdf_processing import parse_sandboxed, validate_pdf_bytes
    args.destination.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    pilot = json.loads((ROOT / "eval/parkvoice_manifest.json").read_text(encoding="utf-8"))
    seen_ids, seen_hashes, accepted = set(), set(), []
    report = {"status": "running", "requested": args.count, "accepted": accepted,
              "excluded": [], "source_feed": str(args.feed), "repeated_files": False}
    last_request = 0.0

    def save():
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (args.destination / "manifest.json").write_text(json.dumps(accepted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    try:
        for entry in pilot + candidates(args.feed):
            base_id = re.sub(r"v\d+$", "", entry["external_id"])
            if base_id in seen_ids:
                continue
            seen_ids.add(base_id)
            target = args.destination / entry["filename"]
            cached = ROOT / "data_store/parkvoice_arxiv_corpus" / entry["filename"]
            try:
                if not target.exists() and cached.is_file():
                    shutil.copy2(cached, target)
                if not target.exists():
                    time.sleep(max(0.0, 3.1 - (time.monotonic() - last_request)))
                    last_request = time.monotonic()
                outcome = fetch(entry, args.destination, False)
                data = target.read_bytes()
                validate_pdf_bytes(data, "application/pdf")
                digest = hashlib.sha256(data).hexdigest()
                if digest in seen_hashes:
                    raise ValueError("Duplicate PDF content")
                parsed = parse_sandboxed(str(target.resolve()))
                if not 1 <= parsed["page_count"] <= 80:
                    raise ValueError("Outside application page limit")
                if not any(parsed["pages"]):
                    raise ValueError("No extractable text")
                seen_hashes.add(digest)
                accepted.append({**entry, "sha256": digest, "bytes": len(data), "pages": parsed["page_count"]})
                print(f"[{len(accepted)}/{args.count}] {outcome}: {entry['filename']}", flush=True)
            except Exception as error:
                report["excluded"].append({"filename": entry["filename"], "error": str(error)[:200]})
                print(f"Excluded {entry['filename']}: {type(error).__name__}", flush=True)
            save()
            if len(accepted) >= args.count:
                break
        report["status"] = "passed" if len(accepted) == args.count else "insufficient_valid_papers"
        report["unique_pdf_hashes"] = len(seen_hashes)
        report["unique_arxiv_ids"] = len(accepted)
        report["total_bytes"] = sum(e["bytes"] for e in accepted)
        save()
        if report["status"] != "passed":
            raise RuntimeError(f"Only {len(accepted)}/{args.count} valid unique papers available")
    finally:
        save()


if __name__ == "__main__":
    main()

"""Fetch the pinned open-access arXiv PDF set used by the ParkVoice pilot evaluation."""
import argparse
import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("parkvoice_manifest.json")
DEFAULT_DESTINATION = ROOT / "data_store" / "parkvoice_arxiv_corpus"
ALLOWED_HOSTS = {"arxiv.org", "export.arxiv.org"}
MAX_PDF_BYTES = 50 * 1024 * 1024


def fetch(entry: dict, destination: Path, force: bool) -> str:
    filename = entry["filename"]
    if Path(filename).name != filename or not filename.endswith(".pdf"):
        raise ValueError(f"Unsafe corpus filename: {filename!r}")
    target = destination / filename
    if target.is_file() and target.stat().st_size and not force:
        with target.open("rb") as stream:
            if stream.read(5) == b"%PDF-":
                return "kept"

    parsed = urlparse(entry["pdf_url"])
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"Unexpected PDF source URL for {filename}: {entry['pdf_url']!r}")

    req = Request(entry["pdf_url"], headers={"User-Agent": "research-assistant-evaluation/1.0"})
    with urlopen(req, timeout=45) as response:
        final = urlparse(response.geturl())
        if final.scheme != "https" or final.hostname not in ALLOWED_HOSTS:
            raise ValueError(f"Unexpected redirect target for {filename}: {response.geturl()!r}")
        data = response.read(MAX_PDF_BYTES + 1)
    if len(data) > MAX_PDF_BYTES:
        raise ValueError(f"PDF exceeds {MAX_PDF_BYTES} bytes: {filename}")
    if not data.startswith(b"%PDF-"):
        raise ValueError(f"Download is not a PDF: {filename}")

    fd, tmp_name = tempfile.mkstemp(prefix=filename + ".", suffix=".part", dir=destination)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    return "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--force", action="store_true", help="replace already downloaded PDFs")
    args = parser.parse_args()
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    args.destination.mkdir(parents=True, exist_ok=True)
    counts = {"downloaded": 0, "kept": 0}
    for index, entry in enumerate(entries, start=1):
        outcome = fetch(entry, args.destination, args.force)
        counts[outcome] += 1
        print(f"[{index}/{len(entries)}] {outcome}: {entry['filename']} — {entry['title']}")
    (args.destination / "manifest.json").write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Corpus ready: {len(entries)} papers ({counts['downloaded']} downloaded, {counts['kept']} already present)")


if __name__ == "__main__":
    main()

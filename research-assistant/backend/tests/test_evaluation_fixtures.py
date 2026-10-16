import json
import subprocess
import sys
import tempfile
from pathlib import Path


def test_synthetic_evaluation_files_match_harness_schema():
    backend = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix="ra-eval-fixture-") as temp_dir:
        tmp_path = Path(temp_dir)
        subprocess.run([sys.executable, str(backend / "scripts" / "make_sample_papers.py"), str(tmp_path)], check=True)

        gaps = json.loads((tmp_path / "gold_gaps.json").read_text(encoding="utf-8"))
        ie = json.loads((tmp_path / "ie_gold.json").read_text(encoding="utf-8"))
        queries = json.loads((tmp_path / "retrieval_queries.json").read_text(encoding="utf-8"))

        assert all({"problem", "dataset", "method"} <= set(item) for item in gaps)
        assert all({"methods", "datasets"} <= set(item) for item in ie)
        expected_files = {f"paper_{index:02d}.pdf" for index in range(1, 17)}
        assert {name for item in queries for name in item["relevant_files"]} == expected_files

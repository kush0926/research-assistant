import os
import subprocess
import sys

import pytest

from app.services.pdf_processing import _close_windows_job, _limit_windows_parser_process


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Objects are Windows-specific")
def test_parser_job_object_enforces_process_memory_limit():
    script = "buf=bytearray(256*1024*1024);\nfor i in range(0,len(buf),4096): buf[i]=1\nprint('allocation completed')"
    process = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    job = _limit_windows_parser_process(process, memory_mb=64, cpu_seconds=10)
    try:
        stdout, _ = process.communicate(timeout=8)
        assert process.returncode != 0, "child allocated more memory than its Job Object limit"
        assert b"allocation completed" not in stdout
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
        _close_windows_job(job)

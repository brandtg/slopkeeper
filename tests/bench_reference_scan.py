# SPDX-FileCopyrightText: 2026 Greg Brandt <brandt.greg@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Bench: scan_references_stream on a generated 50k-file tree.

Bench-only (reference-rewrites PRD Phase 3, step 4): excluded from the
default suite by filename (not test_*) AND by the `bench` marker; run
it directly, once, and record the number in the PRD's §6 — one
measurement, no tuning (PRD §6: scan cost is a non-issue, a rare
monthly operation).

    uv run python tests/bench_reference_scan.py
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.bench

N_DIRS = 100
FILES_PER_DIR = 500  # 100 × 500 = 50_000 files


def build_tree(root: Path) -> None:
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "api.md").write_text("# api\n", encoding="utf-8")
    for d in range(N_DIRS):
        bulk = root / f"bulk/d{d:03}"
        bulk.mkdir(parents=True)
        for i in range(FILES_PER_DIR):
            (bulk / f"f{i:04}.txt").write_text(
                "cite docs/api.md here\n", encoding="utf-8"
            )


def main() -> None:
    from slopkeeper.references import scan_references_stream

    tmp = Path(tempfile.mkdtemp(prefix="sk-bench-"))
    try:
        t0 = time.monotonic()
        build_tree(tmp)
        gen_s = time.monotonic() - t0

        t0 = time.monotonic()
        out = scan_references_stream(tmp)
        elapsed = time.monotonic() - t0

        stats = out["stats"]
        n = N_DIRS * FILES_PER_DIR + 1
        print(f"generated {n} files in {gen_s:.1f}s")
        print(
            f"scan_references_stream: {elapsed:.2f}s | "
            f"files_scanned={stats['files_scanned']} hits={stats['hits']}"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

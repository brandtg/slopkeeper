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

P3 (0.2.8) added `--scale`, the same generator at 5k with the other
corpus-side commands timed beside it. That is the shape the slow smoke
(tests/test_scale_smoke.py) asserts; the numbers below are the 0.2.7
baselines it has to beat, on one dev box (5k files, ext4, warm cache):

    corpus_fingerprint  0.124s      inventory --fast  0.332s
    doctor              0.624s      refs stream        0.301s
    refs stream, 5k DISTINCT basenames (the alternation case) 2.226s
    clusters at the 3k cap (collect + collect_similarity)     6.859s

    uv run python tests/bench_reference_scan.py --scale
"""

from __future__ import annotations

import shutil
import sys
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


def build_scale_tree(root: Path, n_dirs: int = 100, per_dir: int = 50) -> list[str]:
    """5k files with DISTINCT basenames, each citing three others.

    The shape that punishes a basename alternation (every candidate is
    tried at every position), which is what the P3 prefilter was for.
    """
    names = [
        f"dossier_{d:03d}_{i:03d}.md" for d in range(n_dirs) for i in range(per_dir)
    ]
    for d in range(n_dirs):
        bulk = root / f"bulk/d{d:03d}"
        bulk.mkdir(parents=True)
        for i in range(per_dir):
            cites = ", ".join(
                names[(d * per_dir + i + k * 37) % len(names)] for k in (1, 2, 3)
            )
            (bulk / names[d * per_dir + i]).write_text(
                f"# note {d}-{i}\n\nSee {cites} and docs/api.md for context.\n",
                encoding="utf-8",
            )
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "api.md").write_text("# api\n", encoding="utf-8")
    return names


def scale(reps: int = 3) -> dict[str, float]:
    """Time the corpus-side commands on the 5k tree (the slow smoke's shape)."""
    from slopkeeper.doctor import doctor
    from slopkeeper.inventory import collect
    from slopkeeper.models import corpus_fingerprint
    from slopkeeper.references import scan_references_stream
    from slopkeeper.similar import collect_similarity

    tmp = Path(tempfile.mkdtemp(prefix="sk-scale-"))
    timings: dict[str, float] = {}
    try:
        build_scale_tree(tmp)

        def run(label: str, fn) -> None:
            best = None
            for _ in range(reps):
                t0 = time.monotonic()
                fn()
                dt = time.monotonic() - t0
                best = dt if best is None else min(best, dt)
            timings[label] = best
            print(f"{label:28s} {best:6.3f}s")

        run("corpus_fingerprint", lambda: corpus_fingerprint(tmp))
        run("inventory --fast", lambda: collect(tmp, fast=True))
        run("scan_references_stream", lambda: scan_references_stream(tmp))
        run("doctor", lambda: doctor(tmp))
        # 3k files: the clusters command at its own cap (the cap is
        # P4's to raise; the smoke measures what runs under it).
        small = tmp / "docs"
        run(
            "clusters at the cap (3k)",
            lambda: collect_similarity(
                tmp,
                [
                    {"path": p}
                    for p in [
                        f"bulk/d{d:03d}/dossier_{d:03d}_{i:03d}.md"
                        for d in range(60)
                        for i in range(50)
                    ]
                ],
                with_dupes=False,
            ),
        )
        del small
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return timings


def main() -> None:
    if "--scale" in sys.argv:
        scale()
        return
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

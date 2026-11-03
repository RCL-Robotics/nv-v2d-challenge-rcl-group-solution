#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""lfs_fill.py: make the Git LFS files under <worktree>/<subdir> real files, and verify them.

  lfs_fill.py --worktree W [--subdir robotic_grounding/flash_chord/src/flash_chord/assets] (--source S | --check-only)
              --report OUT.tsv

The list of LFS files and their (oid, size) comes from the worktree's HEAD tree (`git ls-tree` + `git cat-file
--batch`), so the check holds whether the working file is still a pointer or not. For each one:
  - the working file already has the pointer's sha256 and size -> ok;
  - --source S: copy S/<path> (or S/<path without 'robotic_grounding/flash_chord/'>, for a flash_chord-only checkout)
    only if its size and sha256 equal the pointer's, through a temp file and an atomic rename -> filled;
  - otherwise -> missing (exit 1).
Nothing is written outside <worktree>/<subdir> except the report. Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HEADER = b"version https://git-lfs.github.com/spec/v1"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pointers(wt: Path, subdir: str) -> list[tuple[str, str, int]]:
    ls = subprocess.run(["git", "-C", str(wt), "ls-tree", "-r", "-z", "-l", "HEAD", "--", subdir],
                        check=True, capture_output=True).stdout.split(b"\0")
    cands = []
    for rec in ls:
        if not rec:
            continue
        meta, path = rec.split(b"\t", 1)
        mode, typ, blob, size = meta.split()
        if typ == b"blob" and size.isdigit() and int(size) < 1024:
            cands.append((blob.decode(), path.decode()))
    if not cands:
        return []
    inp = "".join(f"{b}\n" for b, _ in cands).encode()
    out = subprocess.run(["git", "-C", str(wt), "cat-file", "--batch"], input=inp, check=True,
                         capture_output=True).stdout
    res, pos = [], 0
    for _, path in cands:
        nl = out.index(b"\n", pos)
        size = int(out[pos:nl].split()[2])
        body = out[nl + 1: nl + 1 + size]
        pos = nl + 1 + size + 1
        if body.startswith(HEADER):
            oid = sz = None
            for line in body.decode().splitlines():
                if line.startswith("oid sha256:"):
                    oid = line.split(":", 1)[1].strip()
                elif line.startswith("size "):
                    sz = int(line.split()[1])
            res.append((path, oid, sz))
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worktree", type=Path, required=True)
    ap.add_argument("--subdir", default="robotic_grounding/flash_chord/src/flash_chord/assets")
    ap.add_argument("--source", type=Path)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args()
    if not a.source and not a.check_only:
        ap.error("give --source or --check-only")
    wt = a.worktree.resolve()
    rows, counts = [], {"ok": 0, "filled": 0, "missing": 0}
    for path, oid, size in pointers(wt, a.subdir):
        dst = wt / path
        status = "missing"
        if dst.is_file() and not dst.is_symlink() and dst.stat().st_size == size and sha256(dst) == oid:
            status = "ok"
        elif a.source:
            short = path[len("robotic_grounding/flash_chord/"):] if path.startswith("robotic_grounding/flash_chord/") else path
            for src in (a.source / path, a.source / short):
                if src.is_file() and src.stat().st_size == size and sha256(src) == oid:
                    tmp = dst.with_name(dst.name + ".lfsfill.tmp")
                    shutil.copyfile(src, tmp)
                    os.chmod(tmp, 0o644)
                    if sha256(tmp) == oid:
                        os.replace(tmp, dst)
                        status = "filled"
                    else:
                        tmp.unlink()
                    break
        counts[status] += 1
        rows.append(f"{path}\t{oid}\t{size}\t{status}")
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text("path\toid_sha256\tsize\tstatus\n" + "".join(r + "\n" for r in rows))
    print(f"[lfs_fill] {wt}: {len(rows)} LFS files under {a.subdir}: {counts['ok']} already ok, {counts['filled']} filled, "
          f"{counts['missing']} missing (report {a.report})")
    return 1 if counts["missing"] else 0


if __name__ == "__main__":
    sys.exit(main())

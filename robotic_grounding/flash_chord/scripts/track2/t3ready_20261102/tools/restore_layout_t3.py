#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""restore_layout_t3.py: put the files of this record back at the paths they ran from (SOURCES.tsv), so the drivers and
prep scripts, which name absolute paths under the training server's work root, run unchanged.

  python3 restore_layout_t3.py --me /root/autodl-tmp/haoran.zhang [--han <dir for Han's release copies>] [--rewrite] [--dry]

  --me       the work root to restore into (<ME> in SOURCES.tsv). On the training server this is
             /root/autodl-tmp/haoran.zhang; elsewhere any empty directory.
  --han      optional: also restore the copies of Han Zhou's release files (<HAN>/tier3/release/...) under this root.
  --rewrite  in text files, replace the training server's root /root/autodl-tmp/haoran.zhang by --me (needed only when
             --me differs; the record's sha256 pins then no longer match the rewritten files, by design).
  --dry      list what would be written.
Every file is checked against its SOURCES.tsv sha256 before it is copied; an existing file is never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent.parent
SERVER_ME = "/root/autodl-tmp/haoran.zhang"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--me", required=True)
    p.add_argument("--han")
    p.add_argument("--rewrite", action="store_true")
    p.add_argument("--dry", action="store_true")
    a = p.parse_args()
    roots = {"<ME>": Path(a.me)}
    if a.han:
        roots["<HAN>"] = Path(a.han)
    n, skipped, bad = 0, 0, []
    for line in (HERE / "SOURCES.tsv").read_text().splitlines()[1:]:
        bundle_path, source, want, _ = line.split("\t")
        ph = source.split("/", 1)[0]
        if ph not in roots:
            continue
        src = HERE / bundle_path
        if sha256(src) != want:
            bad.append(f"sha256 mismatch: {bundle_path}")
            continue
        dst = roots[ph] / source.split("/", 1)[1]
        if dst.exists():
            skipped += 1
            continue
        n += 1
        if a.dry:
            print(f"{bundle_path} -> {dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if a.rewrite and str(roots["<ME>"]) != SERVER_ME:
            try:
                dst.write_text(src.read_text().replace(SERVER_ME, str(roots["<ME>"])))
                shutil.copymode(src, dst)
                continue
            except UnicodeDecodeError:
                pass
        shutil.copy2(src, dst)
    print(f"{'would write' if a.dry else 'wrote'} {n} files; {skipped} existed (kept); {len(bad)} problems")
    for b in bad:
        print(b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

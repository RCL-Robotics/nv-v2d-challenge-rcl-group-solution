#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""verify_transfer.py: check a copied transfer payload against its MANIFEST.tsv (size and sha256 of every file).

The manifest is written next to the payload on the source machine (it is not in Git). Its columns are
  group  rel_path  size  sha256  source  note
with rel_path relative to the payload root. Usage, on the new machine:
  python verify_transfer.py --manifest <root>/MANIFEST.tsv --root <root> [--groups G1,G2,... | --groups all] [--quick]
Without --groups it checks the default copy: every group except the optional D_metrics and
H_stock_runtime_optional and the unstaged X_teammate_ok. --quick compares sizes only. Exit 0 when every listed file
of the selected groups is present and equal.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

sys.dont_write_bytecode = True
NOT_DEFAULT = {"D_metrics", "H_stock_runtime_optional", "X_teammate_ok"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--groups", help="comma list of groups to check, or 'all' (default: all but D, H and X)")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    groups = None if a.groups == "all" else set(a.groups.split(",")) if a.groups else None
    n = ok = 0
    bad = []
    with a.manifest.open() as fp:
        for row in csv.DictReader((line for line in fp if not line.startswith("#")), delimiter="\t"):
            if groups and row["group"] not in groups:
                continue
            if groups is None and a.groups != "all" and row["group"] in NOT_DEFAULT:
                continue
            n += 1
            f = a.root / row["rel_path"]
            if not f.is_file():
                bad.append(f"missing {row['rel_path']}")
            elif f.stat().st_size != int(row["size"]):
                bad.append(f"size {f.stat().st_size} != {row['size']}: {row['rel_path']}")
            elif not a.quick and sha256(f) != row["sha256"]:
                bad.append(f"sha256 differs: {row['rel_path']}")
            else:
                ok += 1
    for b in bad[:50]:
        print("BAD", b)
    print(f"{ok}/{n} files ok{' (sizes only)' if a.quick else ''}; {len(bad)} problem(s)")
    return 0 if not bad and n else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""compare_exports.py: are two exported (stock-format) checkpoints the same policy?

Exports are not byte-deterministic (the safetensors header order varies), so their sha256 differ between runs of
the exporter. This compares what matters: every tensor (name, dtype, shape and bytes) and the parsed metadata,
including the resolved config the evaluator reads. Use it to check a re-export, or an export made by another tree's
exporter, against the evaluated file.

  python compare_exports.py A.safetensors B.safetensors [--ignore-meta key1,key2]
Exit 0 when tensors and metadata are equal (after the ignored keys), 1 otherwise. Needs only the standard library.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys

sys.dont_write_bytecode = True


def read(path: str) -> tuple[dict, dict, int]:
    with open(path, "rb") as fp:
        n = struct.unpack("<Q", fp.read(8))[0]
        header = json.loads(fp.read(n))
        base = 8 + n
        meta = header.pop("__metadata__", {}) or {}
        tensors = {}
        for name, info in header.items():
            start, end = info["data_offsets"]
            fp.seek(base + start)
            tensors[name] = (info["dtype"], tuple(info["shape"]), fp.read(end - start))
    parsed = {}
    for k, v in meta.items():
        try:
            parsed[k] = json.loads(v)
        except (TypeError, ValueError):
            parsed[k] = v
    return tensors, parsed, n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--ignore-meta", default="", help="comma list of metadata keys to leave out")
    args = p.parse_args()
    ta, ma, _ = read(args.a)
    tb, mb, _ = read(args.b)
    problems = []
    if set(ta) != set(tb):
        problems.append(f"tensor names differ: only A {sorted(set(ta) - set(tb))[:5]}, only B {sorted(set(tb) - set(ta))[:5]}")
    for name in sorted(set(ta) & set(tb)):
        if ta[name][:2] != tb[name][:2]:
            problems.append(f"{name}: dtype/shape {ta[name][:2]} != {tb[name][:2]}")
        elif ta[name][2] != tb[name][2]:
            problems.append(f"{name}: values differ")
    ignore = {k for k in args.ignore_meta.split(",") if k}
    for k in sorted((set(ma) | set(mb)) - ignore):
        if ma.get(k) != mb.get(k):
            problems.append(f"metadata {k} differs")
    print(f"tensors: {len(ta)} vs {len(tb)}; metadata keys: {len(ma)} vs {len(mb)}")
    for x in problems[:40]:
        print("DIFF", x)
    print("SAME POLICY" if not problems else f"{len(problems)} difference(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())

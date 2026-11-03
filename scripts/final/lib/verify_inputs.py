#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""verify_inputs.py: check that this machine's input files are the ones the counted file's policies were trained and
evaluated on (sha256 against the set record). CPU only, read-only, standard library.

  verify_inputs.py --tier 2|3 [--pack-frame-refs DIR]

Tier 2 (set_items_r9.json): each item's own-input task.parquet (raw / K3 / K3fix / G2 root) and its frame reference
  (<frame_reference dir>/episode_NNNNNN.npz; for raw items the released episode <dir>/data/chunk-000/episode_NNNNNN.parquet;
  else the byte copy in --pack-frame-refs, the packed folder's frame_refs/).
Tier 3 (set_items.json): each item's f0 task.parquet, frame reference and the root's manifest.
Placeholders <ME>, <ZX>, <HAN> map to $ROOT_ME, $ROOT_ZX, $ROOT_HAN. Exit 1 if a parquet is missing or differs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
TR = REPO / "robotic_grounding/flash_chord/scripts/track2"


def sha(p: Path) -> str | None:
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", type=int, choices=(2, 3), required=True)
    ap.add_argument("--pack-frame-refs", type=Path)
    a = ap.parse_args()
    R = {"<ME>": os.environ.get("ROOT_ME", "/root/autodl-tmp/haoran.zhang"),
         "<ZX>": os.environ.get("ROOT_ZX", "/root/autodl-tmp/zuxing.lu"),
         "<HAN>": os.environ.get("ROOT_HAN", "/root/autodl-tmp/han.zhou")}

    def m(s: str) -> Path:
        for k, v in R.items():
            s = s.replace(k, v)
        return Path(s)

    if a.tier == 2:
        items = json.loads((TR / "t2r9_20261104/set/set_items_r9.json").read_text())["episodes"]
        rows = [(int(e["episode"]), e["input"], m(e["parquet"]), e["parquet_sha256"],
                 m(e["evaluated"]["frame_reference"]) / f"episode_{int(e['episode']):06d}.npz",
                 e["evaluated"]["frame_reference_file_sha256"]) for e in items]
    else:
        items = json.loads((TR / "t3ready_20261102/set/set_items.json").read_text())["episodes"]
        rows = [(int(e["episode"]), "f0", m(e["input"]["parquet"]), e["input"]["parquet_sha256"],
                 m(e["input"]["frame_ref"]), e["input"]["frame_ref_sha256"]) for e in items]
    bad_parquet = bad_ref = 0
    print(f"{'ep':>3} {'input':6} parquet           frame reference")
    for ep, inp, pq, pq_sha, fr, fr_sha in rows:
        got = sha(pq)
        ps = "ok" if got == pq_sha else ("MISSING" if got is None else f"DIFFERS {got[:12]}")
        bad_parquet += ps != "ok"
        # raw Tier-2 items: the frame reference is the released episode itself (<dir>/data/chunk-000/episode_N.parquet)
        cands = [fr, fr.parent / "data/chunk-000" / f"episode_{ep:06d}.parquet"]
        cands += [a.pack_frame_refs / f"episode_{ep:06d}.npz"] if a.pack_frame_refs else []
        fs = "not found here"
        for c in cands:
            g = sha(c)
            if g is not None:
                fs = ("ok" if g == fr_sha else f"DIFFERS {g[:12]}") + (" (packed copy)" if a.pack_frame_refs and c.parent == a.pack_frame_refs else "")
                break
        bad_ref += not fs.startswith("ok")
        print(f"{ep:>3} {inp:6} {ps:17} {fs}")
    print(f"[verify_inputs] tier {a.tier}: {len(rows)} items; parquets not ok: {bad_parquet}; frame references not ok "
          f"or not found: {bad_ref}")
    return 1 if bad_parquet else 0


if __name__ == "__main__":
    sys.exit(main())

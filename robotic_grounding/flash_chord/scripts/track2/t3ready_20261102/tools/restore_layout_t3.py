#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""restore_layout_t3.py: put the files of this record back at the paths they ran from (SOURCES.tsv), so the drivers and
prep scripts, which name absolute paths under the training server's work root, run unchanged.

  python3 restore_layout_t3.py --me /root/autodl-tmp/haoran.zhang [--han <root for Han's release>] [--rewrite]
                               [--records-dir <dir>] [--with-data-records] [--dry]

  --me        the work root to restore into (<ME> in SOURCES.tsv). On the training server this is
              /root/autodl-tmp/haoran.zhang; elsewhere any empty directory.
  --han       optional: the root that holds Han Zhou's releases (<HAN>, i.e. <han>/tier3/release/...). The copies of
              his release documents are restored under it. The release arrays themselves are not in git; the prep
              steps need tier3_opus_20261021 and tier3_frame0_supports_20261024 there.
  --rewrite   in text files, replace the training server's roots by --me and (with --han) by --han. Needed only when the
              roots differ; the record's sha256 pins then no longer match the rewritten files, by design.
  --records-dir
              where the committed write-once records of the two data roots go (default <me>/t3ready_expected). They
              are the EXPECTED outputs of prep steps 0 and A-F (READY, FILES/SOURCE_FILES/SUPPORT_FILES.sha256,
              track2_manifest.json, SCENE_SUPPORT.json): compare the rebuilt roots against them afterwards.
  --with-data-records
              old behaviour, for audits on the training server only: restore those records inside data/ itself.
              A restored READY makes t3f0_build_root.py and t3opus_build_data.py exit 0 without building ("READY
              exists") and t3f0_retarget.sh refuse; a records-only data/human_motion_data_tier3_opus shadows the
              real opus root.
  --dry       list what would be written.
Every file is checked against its SOURCES.tsv sha256 before it is copied; an existing file is never overwritten.

Not restored by this tool (see data_prep/README_tier3.md, "Rebuild order"):
  - the retarget code: t3f0_retarget.sh / t3opus_retarget.sh run soma_to_g1.py from <me>/code/nv-v2d-challenge-codebase
    at f2923d61817e61eb37d939e0f75e5b4a7aaa5cbf with a clean tree. That commit is an ancestor of this record, so from a
    clone of this repo:  git worktree add <me>/code/nv-v2d-challenge-codebase f2923d61817e61eb37d939e0f75e5b4a7aaa5cbf
  - the environments (conda_envs/v2d, flashchord, v2dkit) and <me>/activate.sh.
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
SERVER_HAN = "/root/autodl-tmp/han.zhou"
NV_PIN = "f2923d61817e61eb37d939e0f75e5b4a7aaa5cbf"
# write-once records of the two data roots: expected outputs, never inputs
DATA_RECORDS = ("<ME>/data/human_motion_data_tier3_f0/", "<ME>/data/human_motion_data_tier3_opus/")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--me", required=True)
    p.add_argument("--han")
    p.add_argument("--rewrite", action="store_true")
    p.add_argument("--records-dir")
    p.add_argument("--with-data-records", action="store_true")
    p.add_argument("--dry", action="store_true")
    a = p.parse_args()
    me = Path(a.me)
    roots = {"<ME>": me}
    if a.han:
        roots["<HAN>"] = Path(a.han)
    subs = []
    if a.rewrite:
        if str(me) != SERVER_ME:
            subs.append((SERVER_ME, str(me)))
        if a.han and str(Path(a.han)) != SERVER_HAN:
            subs.append((SERVER_HAN, str(Path(a.han))))
    records_dir = Path(a.records_dir) if a.records_dir else me / "t3ready_expected"
    n, skipped, moved, bad = 0, 0, 0, []
    for line in (HERE / "SOURCES.tsv").read_text().splitlines()[1:]:
        bundle_path, source, want, _ = line.split("\t")
        ph = source.split("/", 1)[0]
        if ph not in roots:
            continue
        src = HERE / bundle_path
        if sha256(src) != want:
            bad.append(f"sha256 mismatch: {bundle_path}")
            continue
        rel = source.split("/", 1)[1]
        dst = roots[ph] / rel
        if source.startswith(DATA_RECORDS) and not a.with_data_records:
            dst = records_dir / rel  # expected output, kept out of data/
            moved += 1
        if dst.exists():
            skipped += 1
            continue
        n += 1
        if a.dry:
            print(f"{bundle_path} -> {dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if subs:
            try:
                text = src.read_text()
                for old, new in subs:
                    text = text.replace(old, new)
                dst.write_text(text)
                shutil.copymode(src, dst)
                continue
            except UnicodeDecodeError:
                pass
        shutil.copy2(src, dst)
    print(f"{'would write' if a.dry else 'wrote'} {n} files; {skipped} existed (kept); {len(bad)} problems")
    if moved:
        print(f"{moved} write-once data-root records placed under {records_dir} (expected outputs; compare after the "
              f"rebuild). Use --with-data-records only for audits on the training server.")
    nv = me / "code/nv-v2d-challenge-codebase"
    print(f"retarget code: {nv} must be a clean checkout at {NV_PIN} "
          f"(git worktree add {nv} {NV_PIN}); present: {nv.is_dir()}")
    for b in bad:
        print(b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

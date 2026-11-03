#!/usr/bin/env python3
"""stage_r9.py: copy the round-9 record files (text only, byte-identical) into the C8 stage folder
<stage>/t2r9_20261104/selection/round9_20261103/ and list what is not copied. Usage: stage_r9.py <R9 root> <stage root>
(round 8's tools/stage_r8.py with round 9's file names; dev/ holds the development copies of the scripts, which are not
pinned and not used, PREREG_R9_PINS 1.5, so it is not copied, like round 8's superseded_build1/)."""
import sys

sys.dont_write_bytecode = True
import hashlib  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path  # noqa: E402

R9, STAGE = Path(sys.argv[1]), Path(sys.argv[2])
DST = STAGE / "t2r9_20261104/selection/round9_20261103"
TOP = """COPIES_r9.tsv DRYRUN_R9.md LOCK_R9.sha256 POOL_R9.json POOL_R9.sha256 PREREG_R9.md PREREG_R9.sha256
PREREG_R9_PINS.md PREREG_R9_PINS.sha256 R9_DRIVER_HEARTBEAT.json RELOCATION_R9.json RELOCATION_R9.sha256 RELOCATION_R9.tsv
SELECTION_R9.json SELECTION_R9.sha256 SELECTION_S9.sha256 TELL_ZUXING_r9.txt eval_r9_driver.log
eval_r9_driver_stdout_f9.log eval_r9_driver_stdout_s9.log eval_r9_heartbeat_f9.json eval_r9_heartbeat_s9.json
final_r9.json manifest_f9.csv manifest_f9.json manifest_s9.csv manifest_s9.json own_input_evals_f9.csv
own_input_evals_s9.csv r9_driver.log r9_driver.py r9_driver_stdout.log r9_eval.py r9_launch.sh r9_pool.py
r9_pool_stdout.log r9_select.py r9_select_stdout.log seeds_f9.tsv seeds_s9.tsv selection_r9.md
selection_r9_episodes.csv selection_s9.json selection_s9.md selection_s9.txt selection_s9_candidates.csv
summary_f9_own_input.csv summary_f9_own_input.json summary_f9_own_input.md summary_r9_trials.csv
summary_s9_own_input.csv summary_s9_own_input.json summary_s9_own_input.md""".split()
DIRS_ALL = ["inputs_r9", "integrity", "preflight", "scores_f9", "scores_s9", "selftest_r9", "dryrun", "zuxing_copies_r9"]
BIN_EXT = (".safetensors", ".parquet", ".npz", ".pt", ".pkl")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fp:
        for c in iter(lambda: fp.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def is_text(p):
    with open(p, "rb") as fp:
        return b"\0" not in fp.read(8000)


if DST.exists():
    raise SystemExit(f"{DST} exists")
take, skip = [], []
for rel in TOP:
    p = R9 / rel
    if not p.is_file():
        raise SystemExit(f"missing {p}")
    take.append(rel)
for d in DIRS_ALL:
    for p in sorted((R9 / d).rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(R9))
        (skip if p.name.endswith(BIN_EXT) or not is_text(p) else take).append(rel)
for p in sorted((R9 / "exports").rglob("*")):
    if p.is_file():
        rel = str(p.relative_to(R9))
        (take if p.name == "export.json" and p.parent.parent.name == "exports" else skip).append(rel)
listed = set(take) | set(skip)
for p in sorted(R9.rglob("*")):  # everything else of the folder: not copied (counted by top-level entry)
    if p.is_file() and str(p.relative_to(R9)) not in listed:
        skip.append(str(p.relative_to(R9)))
for rel in take:
    src, dst = R9 / rel, DST / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    os.chmod(dst, 0o644)
    if sha(src) != sha(dst):
        raise SystemExit(f"copy differs: {rel}")
with open(STAGE / "copied_r9.tsv", "w") as fp:
    for rel in take:
        fp.write(f"{sha(R9 / rel)}\t{rel}\n")
from collections import Counter  # noqa: E402

c = Counter(r.split("/")[0] if "/" in r else "(top)" for r in skip)
with open(STAGE / "not_copied_r9.tsv", "w") as fp:
    for rel in sorted(skip):
        p = R9 / rel
        fp.write(f"{os.path.getsize(p)}\t{rel}\n")
print(f"copied {len(take)} files; not copied {len(skip)}: {dict(c)}")

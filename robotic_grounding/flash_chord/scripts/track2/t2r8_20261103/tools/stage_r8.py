#!/usr/bin/env python3
"""stage_r8.py: copy the round-8 record files (text only, byte-identical) into the C7 stage folder
<stage>/t2r8_20261103/selection/round8_20261103/ and list what is not copied. Usage: stage_r8.py <R8 root> <stage root>"""
import sys

sys.dont_write_bytecode = True
import hashlib  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path  # noqa: E402

R8, STAGE = Path(sys.argv[1]), Path(sys.argv[2])
DST = STAGE / "t2r8_20261103/selection/round8_20261103"
TOP = """AMENDMENTS_R8.md AMENDMENTS_R8.sha256 COPIES_r8.tsv DRYRUN_R8.md LAUNCH_R8.md LOCK_R8.sha256 POOL_R8.json
POOL_R8.sha256 PREREG_R8.md PREREG_R8.sha256 PREREG_R8_CHANGES.md PREREG_R8_PINS.md PREREG_R8_PINS.sha256
R8_DRIVER_HEARTBEAT.json RELOCATION_R8.json RELOCATION_R8.sha256 RELOCATION_R8.tsv SELECTION_R8.json SELECTION_R8.sha256
SELECTION_S8.sha256 TELL_ZUXING_r8.txt eval_r8_driver.log eval_r8_driver_stdout_f8.log eval_r8_driver_stdout_s8.log
eval_r8_heartbeat_f8.json eval_r8_heartbeat_s8.json final_r8.json manifest_f8.csv manifest_f8.json manifest_s8.csv
manifest_s8.json own_input_evals_f8.csv own_input_evals_s8.csv r8_driver.log r8_driver.py r8_driver_stdout.log r8_eval.py
r8_launch.sh r8_pool.py r8_pool_stdout.log r8_select.py r8_select_stdout.log seeds_f8.tsv seeds_s8.tsv selection_r8.md
selection_r8_episodes.csv selection_s8.json selection_s8.md selection_s8.txt selection_s8_candidates.csv
summary_f8_own_input.csv summary_f8_own_input.json summary_f8_own_input.md summary_r8_trials.csv summary_s8_own_input.csv
summary_s8_own_input.json summary_s8_own_input.md""".split()
DIRS_ALL = ["inputs_r8", "integrity", "preflight", "scores_f8", "scores_s8", "selftest_r8", "superseded_a1_20261103",
            "dryrun", "zuxing_copies_r8"]
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
    p = R8 / rel
    if not p.is_file():
        raise SystemExit(f"missing {p}")
    take.append(rel)
for d in DIRS_ALL:
    for p in sorted((R8 / d).rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(R8))
        (skip if p.name.endswith(BIN_EXT) or not is_text(p) else take).append(rel)
for p in sorted((R8 / "exports").rglob("*")):
    if p.is_file():
        rel = str(p.relative_to(R8))
        (take if p.name == "export.json" and p.parent.parent.name == "exports" else skip).append(rel)
listed = set(take) | set(skip)
for p in sorted(R8.rglob("*")):  # everything else of the folder: not copied (counted by top-level entry)
    if p.is_file() and str(p.relative_to(R8)) not in listed:
        skip.append(str(p.relative_to(R8)))
for rel in take:
    src, dst = R8 / rel, DST / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    os.chmod(dst, 0o644)
    if sha(src) != sha(dst):
        raise SystemExit(f"copy differs: {rel}")
with open(STAGE / "copied_r8.tsv", "w") as fp:
    for rel in take:
        fp.write(f"{sha(R8 / rel)}\t{rel}\n")
from collections import Counter  # noqa: E402

c = Counter(r.split("/")[0] if "/" in r else "(top)" for r in skip)
with open(STAGE / "not_copied_r8.tsv", "w") as fp:
    for rel in sorted(skip):
        p = R8 / rel
        fp.write(f"{os.path.getsize(p)}\t{rel}\n")
print(f"copied {len(take)} files; not copied {len(skip)}: {dict(c)}")

#!/usr/bin/env python3
"""t3final_r_build.py: builds FINAL_r, the retry-updated Tier-3 full file, exactly as pre-registered.

Governing texts (read-only here):
  fc_runs/submissions/t3_f0_final/PREREG.md (e4e3a5da), its AMENDMENT_1.md (e2a43438: items (a), reuse of chosen_run (b),
  build time (c), pack and checks (d), Final mark (e)) and AMENDMENT_2.md (names this file's sha256; written before the
  real build), tmp_analysis/tier3/f0_retry/PREREG_retry.md (8b40cff0: acceptance (i)/(ii), pick rule) and PREREG_v3.md
  (e62c39b2, section 7).

RULE (AMENDMENT_1 (a)/(b)), per episode NN:
  retry item  iff, at the build time, f0_retry/SUBMISSION_CHOICE.json lists NN with use = retry_final, f0_retry/eval/epNN/
              post.json has status done, a decision starting with REPLACE and the same retry_final_sha256, and the pinned
              PREREG_retry / RETRY_FILES (v2) and PREREG_v3 / RETRY_FILES_v3 hashes still match. Its rollout = one of the
              retry post's 5 official own-input evaluations of the retry final (f0_retry/eval/epNN/retry/rK_*), re-checked
              against every validity condition of t3_f0_final PREREG section 2 (pinned file = that run's own exported
              checkpoint; export source = the retry final.json checkpoint); among the passing runs, the one whose own-input
              ADD-AUC is closest to their mean (exact rational arithmetic on the float64 values; ties -> lower run index).
              It must agree with SUBMISSION_CHOICE chosen_run, else the build STOPS (a human decides). No passing run ->
              the episode keeps its t3_f0_final item (logged). Frame reference: the episode's own frame_references_f0 npz.
  otherwise   t3_f0_final's frozen item + frozen selection for NN, i.e. the same rollout as in FINAL_f0 (byte copy;
              checked against pack/FINAL/file.json).
BUILD TIME (AMENDMENT_1 (c)): the first of (1) every retry candidate (f0_retry/recipes/epNN.json, = the frozen 14) has an
  outcome (POST_DONE = post.json status done; LATE or CRASHED marker; never started by the training cut-off 2026-11-02
  12:00 CST) and (2) 2026-11-02 18:00 CST. A candidate without a done post at the build time keeps its f0 item; retry
  results that appear after the build time are ignored. FINAL_f0 must exist. No retry item -> FINAL_r is not built.
PACK (AMENDMENT_1 (d) = t3_f0_final PREREG section 4): the kit packer unchanged (v2dkit python -I -B, --track
  track_2_tier3), byte copies of each episode's own frame reference, placeholder commit URL, input-reference residual
  < 1e-4 m / 1e-3 rad, the uploader's validate_submission() (offline, through the pinned t3final_kit.py) and provenance
  (uniform digests + per-episode checkpoint digest = the item's eval file / the chosen retry run's exported file).
  The packer's stdout carries its own clean cross-check ("locally computed"): it goes only to
  pack_LOCAL_NOT_FOR_UPLOAD.log, and only non-clean lines (residual, rows, episode count) are parsed. No metric_code here.
NO-CLEAN GUARD: every decision read goes through guard(), which refuses paths naming clean / LOCAL_NOT_FOR_UPLOAD /
  tier_1_multiview_caption / track_2/baseline (the t3_f0_final rule). `report` (after the lock only) is the single
  report-only exception and decides nothing.
WRITES ONLY: fc_runs/submissions/t3_f0_final/pack/FINAL_r/ (or FINAL_r.FAILED_<stamp>), t3_f0_final/FINAL_r.sha256,
  t3_f0_final/upload_r/, tmp_analysis/tier3/f0_retry/{final_r_test/, final_r_tmp/, final_r_build.log}.
  No GPU, no process is signalled, nothing is uploaded or pushed.

  python -I -B t3final_r_build.py selftest      CPU only; never writes FINAL_r. Acceptance test (all 30 episodes on their
                                               t3_f0_final items -> pose columns == FINAL_f0's) + the composition as it
                                               stands now + the repack path with a dummy URL + rule unit tests.
  python -I -B t3final_r_build.py when          build-time state (exit 0 iff the build time is reached, records consistent)
  python -I -B t3final_r_build.py build         the real build, once (needs AMENDMENT_2 naming this file's sha256)
  python -I -B t3final_r_build.py report        after the lock only: LOCAL report-only clean score of FINAL_r and FINAL_f0
  python -I -B t3final_r_build.py repack --commit-url URL   FINAL_r with the real commit -> upload_r/<sha7>/ (no upload)
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import ast  # noqa: E402
import csv  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import shlex  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from fractions import Fraction  # noqa: E402
from pathlib import Path  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
F = ME / "fc_runs/submissions/t3_f0_final"
RA = ME / "tmp_analysis/tier3/f0_retry"
F0A = ME / "tmp_analysis/tier3/f0"
RUNS_R = ME / "fc_runs/C3r2_tier3_f0_retry"
KIT = ME / "v2d_submission_kit_src/v2d_submission_kit"
STOCK = ME / "code/video_to_data_8c778f9/robotic_grounding/flash_chord"
DATASET = ME / "data/video_to_data_challenge"
F0_DATA = ME / "data/human_motion_data_tier3_f0"
PACK_PYTHON = ME / "conda_envs/v2dkit/bin/python"
KITHELPER = F / "t3final_kit.py"
PACKER = KIT / "tools/pack_submission.py"
UPLOADER = KIT / "submit/v2d_submit.py"
EVAL_SCRIPT = KIT / "flash_chord/scripts/evaluate_policy.py"
F0_LOCK = F0A / "retarget/parquet_lock.json"
FINAL_F0 = F / "pack/FINAL"
FINAL_R = F / "pack/FINAL_r"
LOCK_FILE = F / "FINAL_r.sha256"
AMEND2, AMEND2_SHA = F / "AMENDMENT_2.md", F / "AMENDMENT_2.sha256"
CSV_NAME = "packed_LOCAL_NOT_FOR_UPLOAD.csv"  # the t3_f0_final naming: placeholder commit, not an upload file
REPORT_DIR_NAME = "report_LOCAL_NOT_FOR_UPLOAD"
TMP = RA / "final_r_tmp"
TEST = RA / "final_r_test"
RUNLOG = RA / "final_r_build.log"

PREREG_F_SHA = "e4e3a5daf239f5b5858e32442c43eb708947b38c20eae76e12af757a747bd0eb"
AMEND1_SHA = "e2a434388be698163fe110057831490c0900192ee90846b127ed1f7c27a24faa"
PREREG_RETRY_SHA = "8b40cff0accec12bb404aedec80cc17cdceeb921e920a348826966bb9ad488c3"
PREREG_V3_SHA = "e62c39b2e092a0abce903168c74d64fcdfa3ce0c57a22f05153469ff1ea79db7"
KITHELPER_SHA = "1c8f94b96a05934f71a3893d2910702e1e5fc186248fcc4f0ab4ad2451beebc7"
PACKER_SHA = "47a4589574ff8284ba4376ef31ac5548c8fe87ead753be2d3b43d6c87041e5be"
UPLOADER_SHA = "fe5cb1dc86d12abd01eab9dd5dd98b51569b5d8513a2d00d4e79da42a5f11718"
EVAL_SCRIPT_SHA = "160d24f5fbeb70ccd569ca4192056f1892b2a9f2999066db9d8279461648fdf1"
EVAL_CODE_SHA = "b1b3ebe3ae4629d1687a83f420149e3f6fdfc154d70acad55c12fc8fa69811a1"
KIT_MANIFEST_SHA = "398947a7783490245a47d70ec68d6894ecef7636e712532710ec448dc0ca1c3e"
KIT_TRACKS_SHA = "5799f424a4cd5453f8a6cfceaeff76e7cc7f8a78ba7d22132108827b1c2a09c7"
METRIC_SHA = {"AUC": "58b437697c32c86497ff5a306fe586014c3429b0d2c18e1d3094a6f503479339",
              "SP-SR": "d206042872acd1c1de6120f96966272495fbbd71a50315f49f5192fea1aaa5bf",
              "MP-SR": "a1d72216eb7745a13ed7db6a55b9efb31492b35149113181a6c42ffc231fff11",
              "MPPE": "41815bc1e8870fb377036750976dcf135818e029f06096ccef5177090fe5e033"}  # fmt: skip
F0_LOCK_SHA = "bdb83b0ee34f05a8408a871cba3d8ca2bb4cef6ec0e67ca435a57232b3e0b0d7"
F0_FILES_SHA = "f9c94e8550e03223c96e101e155038537efeedda3b5213d96cecc4dcbafaba49"
FINAL_F0_CSV_SHA = "6f73715a9a6f046acfedb6e59f1936a4e507694a889e4e4685bde80b05b48cc6"  # pack/FINAL/validate.json
FINAL_F0_POSE_SHA = "778b808369df51fa3d176982e2b584de21bf7d87f6579808459a03763c66fefa"  # pack/FINAL/validate.json
FROZEN_CANDIDATES = (1, 2, 4, 5, 11, 13, 14, 16, 17, 19, 21, 23, 25, 28)  # PREREG_v3 section 2
F0_ITER, R_ITER, RUNS_PER_ARM = 44922, 90000, 5
TZ8 = dt.timezone(dt.timedelta(hours=8))
TRAIN_CUTOFF = dt.datetime(2026, 11, 2, 12, 0, 0, tzinfo=TZ8).timestamp()
BUILD_DEADLINE = dt.datetime(2026, 11, 2, 18, 0, 0, tzinfo=TZ8).timestamp()
TRACK = "track_2_tier3"
EPISODES = tuple(range(30))
EXPORT_TOLS = (1e-4, 2e-4)
COMMIT_PLACEHOLDER = "https://github.com/local-placeholder/not-uploaded/commit/0000000000000000000000000000000000000000"
COMMIT_URL_RE = re.compile(r"^https://github\.com/[^/\s]+/[^/\s]+/commit/[0-9a-f]{40}$")
SELFTEST_URL = "https://github.com/selftest-owner/selftest-repo/commit/" + "1" * 40  # selftest of the repack path only
FORBIDDEN = ("clean", "local_not_for_upload", "tier_1_multiview_caption", "/track_2/baseline")
FALLBACK_KEYS = {"auc": "add_auc", "sp_sr": "spider_sr_uncentered", "mp_sr": "maniptrans_sr", "mppe": "mean_per_point_error_cm"}
PACK_TIMEOUT = 3600.0
ENTRY_KEYS = ("run", "json", "json_sha256", "parquet", "parquet_sha256", "episode", "kind", "eval_sha256", "frame_ref",
              "frame_ref_sha256", "object")  # the t3_f0_final composition record


# ================================================================================================ small helpers
def now() -> str:
    return dt.datetime.now(TZ8).strftime("%Y-%m-%d %H:%M:%S")


def stamp() -> str:
    return dt.datetime.now(TZ8).strftime("%Y%m%d_%H%M%S")


def ts_of(text) -> float | None:
    try:
        return dt.datetime.strptime(str(text), "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ8).timestamp()
    except (TypeError, ValueError):
        return None


def fnum(v):
    try:
        x = None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None
    return x if x is None or math.isfinite(x) else None


def ep2(e: int) -> str:
    return f"{e:02d}"


def sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def guard(path) -> Path:
    """Tier-3 rule (t3_f0_final): no clean-reference / LOCAL file is ever read for a decision."""
    if any(x in str(path).lower() for x in FORBIDDEN):
        raise SystemExit(f"guard: refusing to read {path} (clean / LOCAL file; Tier-3 decisions are noisy-only)")
    return Path(path)


def read_json(path) -> dict | None:
    try:
        d = json.loads(guard(path).read_text())
    except (OSError, ValueError, TypeError):
        return None
    return d if isinstance(d, dict) else None


def read_raw(path) -> tuple[bytes | None, dict | None]:
    try:
        b = guard(path).read_bytes()
    except OSError:
        return None, None
    try:
        d = json.loads(b)
    except ValueError:
        return b, None
    return b, d if isinstance(d, dict) else None


def realpath(p) -> str:
    return os.path.realpath(str(p)) if p else ""


def check_write(target: Path) -> Path:
    t = os.path.abspath(str(target))
    ok = (t == str(LOCK_FILE) or t == str(RUNLOG) or t.startswith(str(FINAL_R)) or t.startswith(str(F / "upload_r") + "/")
          or t.startswith(str(TEST) + "/") or t.startswith(str(TMP)))
    if not ok:
        raise SystemExit(f"refusing to write {target}: outside this builder's output places")
    return target


def write_atomic(path: Path, text: str) -> None:
    check_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_bytes_new(path: Path, data: bytes) -> None:
    check_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as fp:
        fp.write(data)


def write_once(path: Path, text: str) -> None:
    check_write(path)
    with path.open("x") as fp:
        fp.write(text)
    os.chmod(path, 0o444)


def dump(d) -> str:
    return json.dumps(d, indent=1, default=str) + "\n"


def copy_verified(src: Path, dst: Path, want: str) -> str:
    check_write(dst)
    h = sha256(src)
    if h != want:
        raise SystemExit(f"{src}: sha256 {h[:16]} != expected {want[:16]}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp")
    shutil.copyfile(src, tmp)
    if sha256(tmp) != h:
        raise SystemExit(f"copy of {src} differs")
    os.replace(tmp, dst)
    return h


def self_sha() -> str:
    return sha256(Path(__file__).resolve())


class Logger:
    def __init__(self, label: str):
        self.label, self.lines = label, []

    def __call__(self, msg: str) -> None:
        line = f"[{now()}] {msg}"
        self.lines.append(line)
        print(line, flush=True)
        try:
            with check_write(RUNLOG).open("a") as fp:
                fp.write(f"[{self.label}] {line}\n")
        except OSError:
            pass

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def kit_metric_keys() -> tuple[dict, str]:
    """ours -> results-JSON key, from the kit's REPORT_METRIC_KEY (parsed with ast, as t3final_driver.py does)."""
    try:
        for node in ast.parse((KIT / "v2dlb/report_schema.py").read_text()).body:
            targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
            if any(getattr(t, "id", None) == "REPORT_METRIC_KEY" for t in targets):
                table = ast.literal_eval(node.value)
                names = {"auc": "add_auc", "sp_sr": "spider_sr", "mp_sr": "maniptrans_sr", "mppe": "mppe_cm"}
                return {ours: table[name] for ours, name in names.items()}, "kit v2dlb/report_schema.py"
        raise KeyError("REPORT_METRIC_KEY not found")
    except Exception as error:  # noqa: BLE001
        return dict(FALLBACK_KEYS), f"built-in fallback ({error!r})"


def select_closest(aucs: dict[int, float]) -> int | None:
    """t3_f0_final PREREG section 3 / AMENDMENT_1 (b): own-input ADD-AUC closest to the mean of the candidate runs,
    exact rational arithmetic on the float64 values; ties -> lower run index."""
    if not aucs:
        return None
    exact = {int(k): Fraction(float(v)) for k, v in aucs.items()}
    mean = sum(exact.values(), Fraction(0)) / len(exact)
    return min(exact, key=lambda k: (abs(exact[k] - mean), k))


# ================================================================================================ pins and gates
def sha_file_token(p: Path) -> str:
    try:
        return guard(p).read_text().split()[0]
    except (OSError, IndexError):
        return "missing"


def check_listing(listing: Path, base: Path) -> list[str]:
    bad = []
    try:
        lines = guard(listing).read_text().splitlines()
    except OSError:
        return [f"{listing} missing"]
    n = 0
    for line in lines:
        f = line.split(None, 1)
        if len(f) != 2:
            continue
        n += 1
        p = base / f[1].strip()
        got = sha256(p) if p.is_file() else "missing"
        if got != f[0]:
            bad.append(f"{p}: {got[:16]} != {f[0][:16]}")
    if n == 0:
        bad.append(f"{listing}: empty")
    return bad


def pins() -> tuple[list[str], list[str]]:
    """Every pin this build relies on. fatal = refuse."""
    fatal, info = [], []

    def pin(label, path, want):
        got = sha256(path) if Path(path).is_file() else "missing"
        (info if got == want else fatal).append(f"{label} {got[:16]} " + ("ok" if got == want else f"!= {want[:16]}"))

    pin("t3_f0_final/PREREG.md", F / "PREREG.md", PREREG_F_SHA)
    (info if sha_file_token(F / "PREREG.sha256") == PREREG_F_SHA else fatal).append("t3_f0_final/PREREG.sha256 names e4e3a5da")
    pin("t3_f0_final/AMENDMENT_1.md", F / "AMENDMENT_1.md", AMEND1_SHA)
    pin("f0_retry/AMENDMENT_1_t3_f0_final.md (staged)", RA / "AMENDMENT_1_t3_f0_final.md", AMEND1_SHA)
    (info if sha_file_token(F / "AMENDMENT_1.sha256") == AMEND1_SHA else fatal).append("AMENDMENT_1.sha256 names e2a43438")
    pin("PREREG_retry.md (v2)", RA / "PREREG_retry.md", PREREG_RETRY_SHA)
    (info if sha_file_token(RA / "PREREG_retry.sha256") == PREREG_RETRY_SHA else fatal).append("PREREG_retry.sha256 names 8b40cff0")
    pin("PREREG_v3.md", RA / "PREREG_v3.md", PREREG_V3_SHA)
    (info if sha_file_token(RA / "PREREG_v3.sha256") == PREREG_V3_SHA else fatal).append("PREREG_v3.sha256 names e62c39b2")
    for lst in ("RETRY_FILES.sha256", "RETRY_FILES_v3.sha256"):
        bad = check_listing(RA / lst, RA)
        (fatal if bad else info).append(f"{lst}: " + ("all pinned files unchanged" if not bad else f"MISMATCH {bad[:4]}"))
    go = RA / "RETRY_GO"
    try:
        gtext = guard(go).read_text()
    except OSError:
        gtext = ""
    (info if PREREG_V3_SHA in gtext and "tier3_board_scores_seen: none" in gtext else fatal).append(
        "RETRY_GO present, names PREREG_v3, tier3_board_scores_seen: none")
    (fatal if (RA / "RETRY_NOGO").exists() else info).append("RETRY_NOGO " + ("PRESENT" if (RA / "RETRY_NOGO").exists() else "absent"))
    first = ME / "fc_runs/submissions/TIER3_FIRST_UPLOAD"
    if first.exists():
        older = go.is_file() and go.stat().st_mtime < first.stat().st_mtime
        (info if older else fatal).append(f"TIER3_FIRST_UPLOAD present; RETRY_GO older: {older}")
    else:
        info.append("TIER3_FIRST_UPLOAD absent (no Tier-3 file of ours uploaded)")
    pin("t3final_kit.py", KITHELPER, KITHELPER_SHA)
    pin("kit tools/pack_submission.py", PACKER, PACKER_SHA)
    pin("kit submit/v2d_submit.py", UPLOADER, UPLOADER_SHA)
    pin("kit flash_chord/scripts/evaluate_policy.py", EVAL_SCRIPT, EVAL_SCRIPT_SHA)
    pin("stock scripts/evaluate_policy.py", STOCK / "scripts/evaluate_policy.py", EVAL_SCRIPT_SHA)
    pin("kit KIT_MANIFEST.json", KIT / "KIT_MANIFEST.json", KIT_MANIFEST_SHA)
    pin("kit config/tracks.json", KIT / "config/tracks.json", KIT_TRACKS_SHA)
    for m, want in METRIC_SHA.items():
        pin(f"kit metric_code/track_2/{m}.py", KIT / f"metric_code/track_2/{m}.py", want)
    try:
        man = json.loads((KIT / "KIT_MANIFEST.json").read_text())["files"]
        bad = [p for p, hh in man.items() if not (KIT / p).is_file() or sha256(KIT / p) != hh]
        (fatal if bad else info).append(f"KIT_MANIFEST: {len(man)} files, {len(bad)} mismatched {bad[:5]}")
    except (OSError, ValueError, KeyError) as error:
        fatal.append(f"KIT_MANIFEST unreadable ({error!r})")
    pin("f0 retarget parquet_lock.json", F0_LOCK, F0_LOCK_SHA)
    pin("f0 data root FILES.sha256", F0_DATA / "FILES.sha256", F0_FILES_SHA)
    for need in (PACK_PYTHON,):
        (info if need.exists() else fatal).append(f"{need} " + ("present" if need.exists() else "MISSING"))
    # FINAL_f0 exists, packed and valid (non-clean records only; its CSV is not opened here)
    st = read_json(FINAL_F0 / "pack_status.json") or {}
    v = read_json(FINAL_F0 / "validate.json") or {}
    fj = read_json(FINAL_F0 / "file.json") or {}
    ok = ((st.get("pack") or {}).get("exit") == 0 and st.get("input_reference_ok") is True
          and (st.get("validate") or {}).get("ok") is True and v.get("ok") is True and not v.get("problems")
          and v.get("uploader_validate_submission") == "pass" and v.get("csv_sha256") == FINAL_F0_CSV_SHA
          and v.get("pose_columns_sha256") == FINAL_F0_POSE_SHA and (st.get("validate") or {}).get("pose_columns_sha256") == FINAL_F0_POSE_SHA
          and sorted(int(e) for e in (fj.get("episodes") or {})) == list(EPISODES) and fj.get("partial_episodes") is None)
    (info if ok else fatal).append(f"FINAL_f0 packed + validated (csv {FINAL_F0_CSV_SHA[:8]}, pose columns {FINAL_F0_POSE_SHA[:8]}, "
                                   f"30 episodes): {ok}")
    return fatal, info


def amend2_status() -> str | None:
    """None iff t3_f0_final/AMENDMENT_2.md exists, matches AMENDMENT_2.sha256 and names this builder's sha256."""
    if not AMEND2.is_file() or not AMEND2_SHA.is_file():
        return "AMENDMENT_2.md / AMENDMENT_2.sha256 missing"
    want = sha_file_token(AMEND2_SHA)
    got = sha256(AMEND2)
    if got != want:
        return f"AMENDMENT_2.md sha256 {got[:16]} != AMENDMENT_2.sha256 {want[:16]}"
    text = AMEND2.read_text()
    me = self_sha()
    if me not in text:
        return f"AMENDMENT_2.md does not name this builder's sha256 {me[:16]}"
    if KITHELPER_SHA not in text:
        return "AMENDMENT_2.md does not name the t3final_kit.py sha256"
    return None


# ================================================================================================ records snapshot
def snapshot() -> dict:
    """One read of every retry record the decision uses (raw bytes kept: the build stores them as its inputs)."""
    s = {"taken_at": now(), "taken_ts": time.time(), "raw": {}, "post": {}}
    for name in ("SUBMISSION_CHOICE.json", "VERDICT_retry.json"):
        b, d = read_raw(RA / name)
        s["raw"][name] = b
        s[name] = d or {}
    for e in FROZEN_CANDIDATES:
        b, d = read_raw(RA / "eval" / f"ep{ep2(e)}" / "post.json")
        if b is not None:
            s["raw"][f"post_ep{ep2(e)}.json"] = b
        s["post"][e] = d or {}
    try:
        b = guard(RA / "pids.tsv").read_bytes()
        s["raw"]["pids.tsv"] = b
        s["launched"] = {int(line.split("\t")[0]) for line in b.decode().splitlines() if line.strip()}
    except (OSError, ValueError):
        s["launched"] = set()
    s["markers"] = {e: [m for m in ("CRASHED", "LATE", "STARTFAIL") if (RA / f"{m}_ep{ep2(e)}").exists()] for e in FROZEN_CANDIDATES}
    s["recipes"] = sorted(int(m.group(1)) for p in (RA / "recipes").glob("ep*.json") if (m := re.match(r"^ep(\d+)\.json$", p.name)))
    return s


def candidate_outcomes(s: dict, at_ts: float) -> dict[int, str | None]:
    out = {}
    for e in FROZEN_CANDIDATES:
        post = s["post"].get(e) or {}
        if post.get("status") == "done":
            out[e] = "POST_DONE"
        elif "CRASHED" in s["markers"][e]:
            out[e] = "CRASHED"
        elif "LATE" in s["markers"][e]:
            out[e] = "LATE"
        elif at_ts >= TRAIN_CUTOFF and e not in s["launched"]:
            out[e] = "NOT_STARTED_BY_CUTOFF"
        else:
            out[e] = None
    return out


def record_consistency(s: dict, e: int) -> list[str]:
    """For a POST_DONE candidate: post.json, VERDICT_retry.json and SUBMISSION_CHOICE.json say the same thing."""
    post = s["post"].get(e) or {}
    ver = ((s["VERDICT_retry.json"].get("episodes") or {}).get(ep2(e))) or {}
    ch = ((s["SUBMISSION_CHOICE.json"].get("episodes") or {}).get(ep2(e))) or {}
    out = []
    dec = str(post.get("decision") or "")
    if ver.get("decision") != dec or ver.get("post") != "done":
        out.append(f"VERDICT_retry decision {ver.get('decision')!r} != post.json {dec!r}")
    want_use = "retry_final" if dec.startswith("REPLACE") else "prereg_f0"
    if ch.get("use") != want_use:
        out.append(f"SUBMISSION_CHOICE use {ch.get('use')!r} != {want_use!r} for post decision {dec[:40]!r}")
    if want_use == "retry_final" and (ch.get("sha256") != post.get("retry_final_sha256") or not ch.get("chosen_run")):
        out.append("SUBMISSION_CHOICE retry sha256 / chosen_run does not match post.json")
    return out


def build_time_state(s: dict) -> dict:
    t = s["taken_ts"]
    oc = candidate_outcomes(s, t)
    all_out = all(v is not None for v in oc.values())
    deadline = t >= BUILD_DEADLINE
    problems = []
    if s["recipes"] != list(FROZEN_CANDIDATES):
        problems.append(f"recipes/ {s['recipes']} != the frozen 14 {list(FROZEN_CANDIDATES)}")
    incons = {e: record_consistency(s, e) for e, v in oc.items() if v == "POST_DONE"}
    incons = {e: v for e, v in incons.items() if v}
    if s["SUBMISSION_CHOICE.json"].get("episodes") is None or s["VERDICT_retry.json"].get("episodes") is None:
        problems.append("SUBMISSION_CHOICE.json / VERDICT_retry.json unreadable")
    reached = all_out or deadline
    # before the deadline the records must agree (verdict() rewrites them right after a post); at the deadline the
    # literal AMENDMENT_1 (a) test decides and an inconsistent clip simply keeps its f0 item
    ready = reached and not problems and (deadline or not incons)
    return {"at": s["taken_at"], "outcomes": {ep2(e): v for e, v in oc.items()}, "all_candidates_have_outcome": all_out,
            "deadline_passed": deadline, "build_time_reached": reached,
            "build_time_rule": "first of: every candidate has an outcome (POST_DONE / LATE / CRASHED / not started by "
                               "2026-11-02 12:00 CST) or 2026-11-02 18:00 CST (AMENDMENT_1 (c))",
            "inconsistent_records": {ep2(e): v for e, v in incons.items()}, "problems": problems, "ready": ready}


# ================================================================================================ items
def f0_entry(e: int, final_fj: dict, keys: dict) -> tuple[dict | None, dict, list[str]]:
    """t3_f0_final's frozen item + frozen selection for e (= the FINAL_f0 rollout, checked against pack/FINAL/file.json)."""
    probs: list[str] = []
    it = read_json(F / "items" / f"ep{ep2(e)}.json") or {}
    sel = read_json(F / "selection" / f"ep{ep2(e)}.json") or {}
    if it.get("state") not in ("f0", "opus_v2"):
        return None, {}, [f"ep{ep2(e)}: t3_f0_final item state {it.get('state')}"]
    if not sel or sel.get("eval_sha256") != it.get("eval_sha256"):
        return None, {}, [f"ep{ep2(e)}: t3_f0_final selection missing / not of the item"]
    k = int(sel["chosen_run"])
    if k == 0:
        src = {"json": it["post_results_json"], "json_sha256": it["post_results_json_sha256"],
               "parquet": it["post_results_parquet"], "parquet_sha256": it["post_results_parquet_sha256"]}
    else:
        r = read_json(F / "runs" / f"ep{ep2(e)}" / f"r{k}" / "run.json") or {}
        if r.get("status") != "ok" or r.get("checkpoint_sha256") != it.get("eval_sha256"):
            return None, {}, [f"ep{ep2(e)}: t3_f0_final run {k} is not ok / not of the item"]
        src = {"json": r["results_json"], "json_sha256": r["json_sha256"], "parquet": r["results_parquet"],
               "parquet_sha256": r["parquet_sha256"]}
    entry = {"run": k, **src, "episode": e, "kind": it["state"], "eval_sha256": it["eval_sha256"], "frame_ref": it["frame_ref"],
             "frame_ref_sha256": it["frame_ref_sha256"], "object": it["object"]}
    fin = ((final_fj.get("episodes") or {}).get(ep2(e))) or {}
    diff = [x for x in ENTRY_KEYS if fin.get(x) != entry.get(x)]
    if diff:
        probs.append(f"ep{ep2(e)}: t3_f0_final item/selection differs from pack/FINAL/file.json in {diff}")
    js = read_json(entry["json"]) or {}
    m = js.get("metrics") or {}
    info = {"own": {x: fnum(m.get(src_key)) for x, src_key in keys.items()}, "arm_mean_auc": sel.get("mean_auc"),
            "arm_n_ok": sel.get("n_ok"), "sequence_id": it.get("sequence_id"), "parquet": it.get("parquet"),
            "parquet_sha256_input": it.get("parquet_sha256")}
    if not Path(entry["json"]).is_file() or sha256(entry["json"]) != entry["json_sha256"]:
        probs.append(f"ep{ep2(e)}: f0 results JSON missing / changed")
    if not Path(entry["parquet"]).is_file() or sha256(entry["parquet"]) != entry["parquet_sha256"]:
        probs.append(f"ep{ep2(e)}: f0 rollout parquet missing / changed")
    if entry["kind"] == "f0" and not f0_frame_ref_listed(entry["frame_ref"], entry["frame_ref_sha256"]):
        probs.append(f"ep{ep2(e)}: frame reference not in the f0 FILES.sha256")
    return entry, info, probs


def f0_frame_ref_listed(path: str, want: str) -> bool:
    try:
        rel = Path(path).relative_to(F0_DATA).as_posix()
    except ValueError:
        return False
    if not Path(path).is_file() or sha256(path) != want:
        return False
    for line in (F0_DATA / "FILES.sha256").read_text().splitlines():
        f = line.split(None, 1)
        if len(f) == 2 and f[1].strip().lstrip("./") in (rel, rel.split("/", 1)[-1]) and f[0] == want:
            return True
    return False


def parquet_checks(pp: Path, want_sha: str, step_count) -> list[str]:
    import pyarrow.parquet as pq

    f = pq.ParquetFile(pp)
    md = {k.decode(errors="replace"): v.decode(errors="replace") for k, v in (f.schema_arrow.metadata or {}).items()
          if not k.startswith(b"pandas")}  # fmt: skip
    out = []
    if md.get("flash_chord.checkpoint_sha256") != want_sha:
        out.append("parquet checkpoint_sha256 != the pinned file")
    if md.get("flash_chord.eval_script_sha256") != EVAL_SCRIPT_SHA:
        out.append(f"parquet eval_script_sha256 {md.get('flash_chord.eval_script_sha256')} is not 160d24f5")
    if md.get("flash_chord.eval_code_sha256") != EVAL_CODE_SHA:
        out.append(f"parquet eval_code_sha256 {md.get('flash_chord.eval_code_sha256')} is not b1b3ebe3")
    if str(md.get("flash_chord.step_count")) != str(step_count):
        out.append(f"parquet step_count {md.get('flash_chord.step_count')} != JSON {step_count}")
    if md.get("flash_chord.world_count") not in (None, "1"):
        out.append(f"parquet world_count {md.get('flash_chord.world_count')} != 1")
    valid = pq.read_table(pp, columns=["environment_valid"]).column("environment_valid").to_pylist()
    if not valid or not all(valid):
        out.append("environment_valid=false rows")
    return out


def retry_final_check(e: int, post: dict) -> tuple[Path | None, list[str]]:
    rdir = RUNS_R / f"ep{ep2(e)}"
    fin = read_json(rdir / "final.json") or {}
    probs = []
    ck = rdir / f"model_{R_ITER}.safetensors"
    if not (fin.get("iteration") == R_ITER and fin.get("reason") == "max_iterations" and fin.get("start_iteration") == F0_ITER
            and fin.get("checkpoint") == ck.name and ck.is_file()):
        probs.append(f"retry final.json not (90000, max_iterations, start 44922, {ck.name}): {fin}")
        return None, probs
    if realpath(post.get("retry_final")) != realpath(ck):
        probs.append(f"post.json retry_final {post.get('retry_final')} != {ck}")
    if sha256(ck) != post.get("retry_final_sha256"):
        probs.append("retry final checkpoint sha256 != post.json retry_final_sha256")
    return ck, probs


def check_retry_run(e: int, rec: dict, k: int, final_ck: Path, final_sha: str, lock: dict, keys: dict) -> tuple[list[str], dict]:
    """AMENDMENT_1 (b): t3_f0_final PREREG section-2 validity of one retry-arm run (pinned file = the run's own exported
    checkpoint; export source = the retry final.json checkpoint)."""
    probs: list[str] = []
    info: dict = {"run": k}
    if rec.get("run") != k or rec.get("arm") != "retry":
        return [f"post.json retry record {k} is run {rec.get('run')} / arm {rec.get('arm')}"], info
    tol = fnum(rec.get("tolerance"))
    if tol not in EXPORT_TOLS:
        return [f"tolerance {rec.get('tolerance')}"], info
    w = RA / "eval" / f"ep{ep2(e)}" / "retry" / f"r{k}_tol{tol:g}_try{rec.get('attempt')}"
    info["dir"] = str(w)
    rj = read_json(w / "run.json")
    if rj != rec:
        probs.append("run.json differs from the post.json record")
    if not (rec.get("ok") is True and rec.get("exit") == 0 and rec.get("eval_status") == "evaluated"):
        probs.append(f"run not ok / exit {rec.get('exit')} / eval_status {rec.get('eval_status')}")
    pinned = Path(str(rec.get("evaluated_checkpoint") or ""))
    if not pinned.is_file() or not realpath(pinned).startswith(realpath(w / "export") + "/"):
        return probs + [f"evaluated checkpoint {pinned} missing / not this run's export"], info
    psha = sha256(pinned)
    info.update(pinned=str(pinned), pinned_sha256=psha)
    if realpath(rec.get("source_checkpoint")) != realpath(final_ck):
        probs.append(f"source checkpoint {rec.get('source_checkpoint')} is not the retry final")
    rep = read_json(w / "eval_report.json") or {}
    row = next((r for r in rep.get("episodes", []) if str(r.get("episode")) == str(e)), None)
    if row is None:
        probs.append("no eval_report.json row")
    elif (row.get("status") != "evaluated" or row.get("checkpoint_sha256") != psha
          or realpath(row.get("evaluated_checkpoint")) != realpath(pinned) or row.get("iteration") != R_ITER):
        probs.append(f"eval_report row status {row.get('status')} / sha256 / checkpoint / iteration {row.get('iteration')}")
    er = read_json(pinned.parent / "export_report.json") or {}
    stock = er.get("stock_runtime") or {}
    rel = fnum(er.get("max_rel_diff"))
    if er.get("status") != "ok" or fnum(er.get("tolerance")) != tol or rel is None or rel > tol:
        probs.append(f"export report status {er.get('status')} tolerance {er.get('tolerance')} max_rel_diff {er.get('max_rel_diff')}")
    if stock.get("verified") is not True or realpath(stock.get("path")) != realpath(STOCK):
        probs.append(f"export not verified on the stock runtime ({stock.get('verified')}, {stock.get('path')})")
    if stock.get("eval_code_sha256") not in (None, EVAL_CODE_SHA):
        probs.append(f"stock eval_code_sha256 {stock.get('eval_code_sha256')}")
    if stock.get("curriculum_stage_voc_scale") not in (None, 0, 0.0):
        probs.append(f"curriculum_stage_voc_scale {stock.get('curriculum_stage_voc_scale')} (help) != 0")
    if er.get("destination_sha256") != psha or realpath(er.get("destination")) != realpath(pinned):
        probs.append("export destination != the evaluated file")
    if realpath(er.get("source")) != realpath(final_ck) or er.get("source_sha256") != final_sha:
        probs.append(f"export source {er.get('source')} ({str(er.get('source_sha256'))[:16]}) != the retry final")
    jp, pp = w / "results" / f"episode_{e:06d}.json", w / "results" / f"episode_{e:06d}.parquet"
    js = read_json(jp)
    if js is None or not pp.is_file():
        return probs + ["results JSON / rollout parquet missing"], info
    if js.get("checkpoint_sha256") != psha or realpath(js.get("checkpoint")) != realpath(pinned):
        probs.append("results JSON checkpoint is not the pinned file")
    ref = Path(str(lock.get("parquet") or ""))
    if realpath(js.get("parquet")) != realpath(ref) or not ref.is_file() or sha256(ref) != lock.get("sha256"):
        probs.append(f"reference parquet {js.get('parquet')} is not the locked f0 input (path + sha256)")
    for key, want in (("start_frame", 0), ("motion_start_frame", 0), ("world_count", 1), ("reset_mode", "explicit")):
        if js.get(key) != want:
            probs.append(f"{key}={js.get(key)!r}, expected {want!r}")
    if js.get("non_finite_worlds"):
        probs.append(f"non_finite_worlds {js.get('non_finite_worlds')}")
    m = js.get("metrics") or {}
    own = {x: fnum(m.get(src)) for x, src in keys.items()}
    if any(v is None for v in own.values()):
        probs.append(f"metrics missing {own}")
    if own.get("auc") != fnum(rec.get("auc")) or own.get("sp_sr") != fnum(rec.get("sp_sr")) or own.get("mp_sr") != fnum(rec.get("mp_sr")):
        probs.append("results JSON metrics != the post.json record")
    try:
        probs += parquet_checks(pp, psha, js.get("step_count"))
    except Exception as error:  # noqa: BLE001
        probs.append(f"parquet unreadable ({error!r})")
    if sha256(pinned) != psha:
        probs.append("the pinned file CHANGED")
    info.update(own=own, json=str(jp), json_sha256=sha256(jp), parquet=str(pp), parquet_sha256=sha256(pp))
    return probs, info


def retry_entry(e: int, s: dict, f0e: dict, keys: dict, log) -> tuple[dict | None, dict, list[str], list[str]]:
    """(entry or None (= keep f0), info, problems (-> keep f0, logged), fatal (-> STOP: a human decides))."""
    post = s["post"].get(e) or {}
    ch = ((s["SUBMISSION_CHOICE.json"].get("episodes") or {}).get(ep2(e))) or {}
    final_ck, probs = retry_final_check(e, post)
    if final_ck is None or probs:
        return None, {}, probs, []
    lock = (read_json(F0_LOCK) or {}).get(ep2(e)) or {}
    rows = (post.get("runs") or {}).get("retry") or []
    runs, passing = {}, {}
    for k in range(1, RUNS_PER_ARM + 1):
        rec = rows[k - 1] if len(rows) >= k else {}
        p, info = check_retry_run(e, rec, k, final_ck, post["retry_final_sha256"], lock, keys)
        runs[k] = {"problems": p, **info}
        if not p:
            passing[k] = info
    info = {"runs": {k: {"ok": not v["problems"], "problems": v["problems"], "own": v.get("own"), "dir": v.get("dir")}
                     for k, v in runs.items()},
            "n_passing": len(passing), "retry_final": str(final_ck), "retry_final_sha256": post["retry_final_sha256"],
            "decision": post.get("decision")}
    if not passing:
        return None, info, [f"ep{ep2(e)}: no retry run passes the validity check: {[v['problems'] for v in runs.values()]}"], []
    k = select_closest({kk: v["own"]["auc"] for kk, v in passing.items()})
    pk = passing[k]
    info["arm_mean_auc"] = float(sum((Fraction(v["own"]["auc"]) for v in passing.values()), Fraction(0)) / len(passing))
    info["chosen_run"] = k
    cr = ch.get("chosen_run") or {}
    agree = (cr.get("run") == k and realpath(cr.get("results_json")) == realpath(pk["json"])
             and cr.get("results_json_sha256") == pk["json_sha256"] and realpath(cr.get("results_parquet")) == realpath(pk["parquet"])
             and cr.get("results_parquet_sha256") == pk["parquet_sha256"]
             and realpath(cr.get("evaluated_checkpoint")) == realpath(pk["pinned"])
             and cr.get("evaluated_checkpoint_sha256") == pk["pinned_sha256"] and fnum(cr.get("auc")) == pk["own"]["auc"])
    if not agree:
        return None, info, [], [f"ep{ep2(e)}: recomputed chosen run {k} ({pk['json_sha256'][:12]}) disagrees with "
                                f"SUBMISSION_CHOICE chosen_run {cr.get('run')} ({str(cr.get('results_json_sha256'))[:12]}): STOP"]
    entry = {"run": k, "json": pk["json"], "json_sha256": pk["json_sha256"], "parquet": pk["parquet"],
             "parquet_sha256": pk["parquet_sha256"], "episode": e, "kind": "retry", "eval_sha256": pk["pinned_sha256"],
             "frame_ref": f0e["frame_ref"], "frame_ref_sha256": f0e["frame_ref_sha256"], "object": f0e["object"],
             "retry_final_sha256": post["retry_final_sha256"], "evaluated_checkpoint": pk["pinned"], "retry_dir": pk["dir"]}
    info["own"] = pk["own"]
    return entry, info, [], []


def compose(s: dict, log, mode: str) -> tuple[dict[int, dict], list[dict], list[str]]:
    """mode 'rule' (AMENDMENT_1 (a)/(b) at the snapshot's time) or 'all_f0' (acceptance test). -> comp, table, fatal."""
    keys, keys_from = kit_metric_keys()
    log(f"metric keys {keys} ({keys_from})")
    final_fj = read_json(FINAL_F0 / "file.json") or {}
    state = build_time_state(s)
    t = s["taken_ts"]
    comp, table, fatal = {}, [], []
    for e in EPISODES:
        f0e, f0i, p = f0_entry(e, final_fj, keys)
        if p or f0e is None:
            fatal += p or [f"ep{ep2(e)}: no t3_f0_final entry"]
            continue
        row = {"episode": e, "object": f0e["object"], "candidate": e in FROZEN_CANDIDATES, "item": "f0",
               "decision": "", "why": "", "chosen_run": f0e["run"], "arm": "t3_f0_final (f0 final, 5 runs)",
               "arm_mean_auc": f0i["arm_mean_auc"], "arm_n_ok": f0i["arm_n_ok"],
               **{f"own_{x}": f0i["own"][x] for x in ("auc", "sp_sr", "mp_sr", "mppe")},
               **{f"round1_own_{x}": f0i["own"][x] for x in ("auc", "sp_sr", "mp_sr", "mppe")},
               "eval_sha256": f0e["eval_sha256"], "results_json_sha256": f0e["json_sha256"],
               "parquet_sha256": f0e["parquet_sha256"], "frame_ref_sha256": f0e["frame_ref_sha256"]}  # fmt: skip
        entry = f0e
        if mode == "all_f0":
            row["why"] = "acceptance test: every episode on its t3_f0_final item"
        elif e not in FROZEN_CANDIDATES:
            row["why"] = "not a retry candidate (own-input SP = MP = 1)"
            ch = ((s["SUBMISSION_CHOICE.json"].get("episodes") or {}).get(ep2(e))) or {}
            if ch.get("use") not in (None, "prereg_f0"):
                fatal.append(f"ep{ep2(e)}: non-candidate with SUBMISSION_CHOICE use {ch.get('use')}")
        else:
            post = s["post"].get(e) or {}
            oc = state["outcomes"][ep2(e)]
            row["decision"] = str(post.get("decision") or oc or "no outcome")
            if oc != "POST_DONE":
                row["why"] = f"candidate without a done post at the build time ({oc or 'pending'}): keeps its f0 item"
            elif not state["all_candidates_have_outcome"] and (ts_of(post.get("finished_at")) or 0) > BUILD_DEADLINE:
                row["why"] = "post finished after the build time (2026-11-02 18:00 CST): ignored, keeps its f0 item"
            elif record_consistency(s, e) and not str(post.get("decision") or "").startswith("REPLACE"):
                row["why"] = f"KEEP decision; records inconsistent {record_consistency(s, e)}: keeps its f0 item"
            elif not str(post.get("decision") or "").startswith("REPLACE"):
                row["why"] = "acceptance rule: KEEP_F0 (PREREG_retry section 6)"
            elif record_consistency(s, e):
                row["why"] = f"AMENDMENT_1 (a) not met ({record_consistency(s, e)}): keeps its f0 item"
                log(f"ep{ep2(e)}: WARN {row['why']}")
            else:
                r_entry, r_info, r_probs, r_fatal = retry_entry(e, s, f0e, keys, log)
                if r_fatal:
                    fatal += r_fatal
                    continue
                if r_entry is None:
                    row["why"] = f"REPLACE, but no valid retry run: keeps its f0 item ({r_probs})"
                    log(f"ep{ep2(e)}: {row['why']}")
                else:
                    entry = r_entry
                    row.update({"item": "retry", "why": "AMENDMENT_1 (a): REPLACE, records agree; (b) chosen run re-validated",
                                "chosen_run": r_entry["run"], "arm": "retry final model_90000 (5 runs)",
                                "arm_mean_auc": r_info["arm_mean_auc"], "arm_n_ok": r_info["n_passing"],
                                **{f"own_{x}": r_info["own"][x] for x in ("auc", "sp_sr", "mp_sr", "mppe")},
                                "eval_sha256": r_entry["eval_sha256"], "results_json_sha256": r_entry["json_sha256"],
                                "parquet_sha256": r_entry["parquet_sha256"]})
                    log(f"ep{ep2(e)}: RETRY item: run {r_entry['run']} of {r_info['n_passing']} valid "
                        f"({r_entry['retry_dir'].rsplit('/', 1)[-1]}), exported sha256 {r_entry['eval_sha256'][:12]}")
        comp[e] = entry
        table.append(row)
    if sorted(comp) != list(EPISODES) and not fatal:
        fatal.append(f"composition has {len(comp)} episodes")
    if mode == "all_f0":
        diff = [e for e in comp if {k: comp[e].get(k) for k in ENTRY_KEYS} != {k: ((final_fj.get('episodes') or {}).get(ep2(e)) or {}).get(k) for k in ENTRY_KEYS}]
        if diff:
            fatal.append(f"acceptance composition differs from FINAL_f0 file.json for {diff}")
    return comp, table, fatal


def totals(table: list[dict], prefix: str) -> dict:
    vals = {x: [r[f"{prefix}{x}"] for r in table] for x in ("auc", "sp_sr", "mp_sr", "mppe")}
    if any(v is None for vv in vals.values() for v in vv) or not table:
        return {"incomplete": True}
    n = len(table)
    return {"episodes": n, "mean_auc": sum(vals["auc"]) / n, "sp_sr_sum": sum(vals["sp_sr"]), "mp_sr_sum": sum(vals["mp_sr"]),
            "sp_sr_mean": sum(vals["sp_sr"]) / n, "mp_sr_mean": sum(vals["mp_sr"]) / n, "mean_mppe_cm": sum(vals["mppe"]) / n}


TABLE_COLS = ["episode", "object", "candidate", "item", "chosen_run", "own_auc", "own_sp_sr", "own_mp_sr", "own_mppe",
              "arm", "arm_mean_auc", "arm_n_ok", "decision", "why", "round1_own_auc", "round1_own_sp_sr", "round1_own_mp_sr",
              "round1_own_mppe", "eval_sha256", "results_json_sha256", "parquet_sha256", "frame_ref_sha256"]


def write_table(d: Path, table: list[dict], tot: dict, tot1: dict, name: str) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=TABLE_COLS, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in table:
        w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    write_atomic(d / f"{name}.csv", buf.getvalue())

    def f(v, n=4):
        return "" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))

    md = [f"# {name}: per-episode choice (own-input numbers only)", "",
          "Own input = each packed rollout against its own Tier-3 training input (the official evaluator's numbers in the "
          "packed results JSON). No clean number appears here.", "",
          "| ep | object | item | run | own AUC | SP | MP | MPPE cm | arm mean AUC | round-1 own AUC | decision |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        md.append(f"| {r['episode']:02d} | {r['object']} | {r['item']} | {r['chosen_run']} | {f(r['own_auc'])} | "
                  f"{f(r['own_sp_sr'], 0)} | {f(r['own_mp_sr'], 0)} | {f(r['own_mppe'], 2)} | {f(r['arm_mean_auc'])} | "
                  f"{f(r['round1_own_auc'])} | {(r['decision'] or r['why'])[:70]} |")
    md += ["", "| totals (30 episodes, own input) | mean AUC | SP-SR = 1 | MP-SR = 1 | mean MPPE cm |", "|---|---|---|---|---|"]
    for label, t in ((name, tot), ("round-1 FINAL_f0 (all 30 episodes on their t3_f0_final items)", tot1)):
        md.append(f"| {label} | {f(t.get('mean_auc'))} | {f(t.get('sp_sr_sum'), 0)} | {f(t.get('mp_sr_sum'), 0)} | "
                  f"{f(t.get('mean_mppe_cm'), 2)} |")
    write_atomic(d / f"{name}.md", "\n".join(md) + "\n")


# ================================================================================================ pack (kit, unchanged)
def kit_env() -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")
           and not k.startswith("CUDA_MPS_")}
    TMP.mkdir(parents=True, exist_ok=True)
    env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               TMPDIR=str(TMP), CUDA_VISIBLE_DEVICES="")
    return env


def run_packer(d: Path, url: str, out_csv: Path) -> dict:
    cmd = ["nice", "-n", "15", str(PACK_PYTHON), "-I", "-B", str(PACKER), "--track", TRACK, "--results", str(d / "results"),
           "--dataset-root", str(DATASET), "--frame-references", str(d / "frame_refs"), "--code-commit", url,
           "--eval-script", str(EVAL_SCRIPT), "-o", str(check_write(out_csv))]  # fmt: skip
    log_file = d / "pack_LOCAL_NOT_FOR_UPLOAD.log"
    try:
        p = subprocess.run(cmd, cwd=d, env=kit_env(), capture_output=True, text=True, timeout=PACK_TIMEOUT)
        code, stdout, stderr = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        code, stdout, stderr = -9, "", "timeout"
    write_atomic(log_file, f"$ {shlex.join(cmd)}\n{stdout}\n{stderr}")  # clean cross-check lines live only here
    out = {"exit": code, "csv": str(out_csv), "log": str(log_file), "command": shlex.join(cmd)}
    m = re.search(r"input reference check: .*?maximum residual ([0-9.eE+-]+) m, ([0-9.eE+-]+) rad", stdout)
    if m:
        out["input_reference_residual_m"], out["input_reference_residual_rad"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"^wrote .*? \(([\d,]+) rows", stdout, re.MULTILINE)
    if m:
        out["rows"] = int(m.group(1).replace(",", ""))
    m = re.search(r"^converted (\d+) Track 2 episode", stdout, re.MULTILINE)
    out["converted_episodes"] = int(m.group(1)) if m else 0
    out["error_tail"] = "" if code == 0 else " | ".join(
        x for x in (stderr or stdout).splitlines()[-3:] if "locally computed" not in x and "clean" not in x.lower())[:600]
    if code == 0 and out_csv.is_file():
        out["csv_sha256"] = sha256(out_csv)
    return out


def kit_validate(d: Path, csv_file: Path, expect: Path) -> dict:
    cmd = ["nice", "-n", "15", str(PACK_PYTHON), "-I", "-B", str(KITHELPER), "validate", "--csv", str(csv_file), "--track",
           TRACK, "--out", str(check_write(d / "validate.json")), "--expect", str(expect)]
    try:
        p = subprocess.run(cmd, cwd=d, env=kit_env(), capture_output=True, text=True, timeout=PACK_TIMEOUT)
        code, text = p.returncode, f"{p.stdout}\n{p.stderr}"
    except subprocess.TimeoutExpired:
        code, text = -9, "timeout"
    write_atomic(d / "validate.log", f"$ {shlex.join(cmd)}\n{text}")
    v = read_json(d / "validate.json") or {}
    return {"exit": code, "uploader_validate_submission": v.get("uploader_validate_submission"), "problems": v.get("problems"),
            "pose_columns_sha256": v.get("pose_columns_sha256"), "rows": v.get("rows"), "episodes": v.get("episodes"),
            "csv_sha256": v.get("csv_sha256"), "ok": v.get("ok")}


def pack_dir(d: Path, comp: dict[int, dict], url: str, meta: dict, log) -> dict:
    """t3_f0_final pack_one without metric_code: byte copies, the kit packer, expect.json, the uploader validation."""
    check_write(d)
    (d / "results").mkdir(parents=True, exist_ok=True)
    (d / "frame_refs").mkdir(exist_ok=True)
    for e, s in sorted(comp.items()):
        copy_verified(Path(s["json"]), d / "results" / f"episode_{e:06d}.json", s["json_sha256"])
        copy_verified(Path(s["parquet"]), d / "results" / f"episode_{e:06d}.parquet", s["parquet_sha256"])
        copy_verified(Path(s["frame_ref"]), d / "frame_refs" / f"episode_{e:06d}.npz", s["frame_ref_sha256"])
    write_atomic(d / "file.json", dump({**meta, "track": TRACK, "code_commit_url": url, "partial_episodes": None,
                                        "episodes": {ep2(e): s for e, s in sorted(comp.items())}, "created_at": now(),
                                        "builder_sha256": self_sha()}))
    pk = run_packer(d, url, d / CSV_NAME)
    res = {"pack": pk, "n_episodes": len(comp)}
    res["input_reference_ok"] = bool(pk.get("input_reference_residual_m") is not None and pk["input_reference_residual_m"] < 1e-4
                                     and pk.get("input_reference_residual_rad", 1.0) < 1e-3)
    if pk["exit"] == 0 and pk.get("csv_sha256"):
        expect = {"episodes": {str(e): s["eval_sha256"] for e, s in comp.items()}, "code_commit_url": url,
                  "eval_script_sha256": EVAL_SCRIPT_SHA, "eval_code_sha256": EVAL_CODE_SHA,
                  "packer_script_sha256": PACKER_SHA, "uploader_script_sha256": UPLOADER_SHA}  # fmt: skip
        write_atomic(d / "expect.json", dump(expect))
        res["validate"] = kit_validate(d, d / CSV_NAME, d / "expect.json")
    v = res.get("validate") or {}
    res["ok"] = bool(pk["exit"] == 0 and res["input_reference_ok"] and v.get("ok") is True and not v.get("problems")
                     and v.get("uploader_validate_submission") == "pass" and v.get("episodes") == list(EPISODES)
                     and pk.get("converted_episodes") == 30 and v.get("csv_sha256") == pk.get("csv_sha256"))
    write_atomic(d / "pack_status.json", dump(res))
    log(f"{d.name}: packer exit {pk['exit']}, rows {pk.get('rows')}, converted {pk.get('converted_episodes')}, input reference "
        f"residual {pk.get('input_reference_residual_m')} m / {pk.get('input_reference_residual_rad')} rad, uploader validate "
        f"{v.get('uploader_validate_submission')}, provenance problems {len(v.get('problems') or [])}, pose columns "
        f"{str(v.get('pose_columns_sha256'))[:16]}, csv {str(pk.get('csv_sha256'))[:16]}, ok {res['ok']}"
        + (f"; error {pk.get('error_tail')}" if pk["exit"] else ""))
    return res


# ================================================================================================ commands
def show_pins(log) -> list[str]:
    fatal, info = pins()
    for x in info:
        log(f"pin: {x}")
    for x in fatal:
        log(f"pin FATAL: {x}")
    return fatal


def cmd_when(a) -> int:
    st = build_time_state(snapshot())
    print(json.dumps(st, indent=1))
    return 0 if st["ready"] else 3


def cmd_selftest(a) -> int:
    base = TEST / f"selftest_{stamp()}"
    check_write(base)
    base.mkdir(parents=True)
    log = Logger("selftest")
    log(f"selftest {base} (builder sha256 {self_sha()}); never writes {FINAL_R}")
    bad = show_pins(log)
    log(f"AMENDMENT_2 gate (needed only by build / repack): {amend2_status() or 'ok'}")
    for runs, want in (({1: 0.5, 2: 0.6, 3: 0.7, 4: 0.6, 5: 0.5}, 2), ({1: 0.9, 2: 0.9, 3: 0.9, 4: 0.9, 5: 0.9}, 1),
                       ({1: 0.1, 3: 0.3}, 1), ({1: 0.2, 2: 0.4, 3: 0.3, 4: 0.3, 5: 0.6}, 2), ({2: 0.8, 3: 0.2, 4: 0.5}, 4),
                       ({1: 0.31, 2: 0.29, 3: 0.30, 4: 0.33, 5: 0.27}, 3), ({}, None)):
        got = select_closest(runs)
        log(f"select_closest {runs} -> {got} (want {want})")
        if got != want:
            bad.append(f"select_closest {runs} -> {got} != {want}")
    s = snapshot()
    st = build_time_state(s)
    log(f"build-time state now: reached {st['build_time_reached']}, ready {st['ready']}, outcomes {st['outcomes']}, "
        f"inconsistent {st['inconsistent_records']}, problems {st['problems']}")
    # every ok retry-arm run of every decided candidate (REPLACE and KEEP) through the validity check
    keys, _ = kit_metric_keys()
    checked = {}
    for e in FROZEN_CANDIDATES:
        post = s["post"].get(e) or {}
        if post.get("status") != "done":
            continue
        ck, p = retry_final_check(e, post)
        if ck is None or p:
            bad.append(f"ep{ep2(e)}: retry final check {p}")
            continue
        lock = (read_json(F0_LOCK) or {}).get(ep2(e)) or {}
        res = {}
        for k, rec in enumerate((post.get("runs") or {}).get("retry") or [], start=1):
            pr, _ = check_retry_run(e, rec, k, ck, post["retry_final_sha256"], lock, keys)
            res[k] = pr
        checked[ep2(e)] = {k: (not v) for k, v in res.items()}
        nbad = sum(1 for v in res.values() if v)
        log(f"ep{ep2(e)} ({str(post.get('decision'))[:12]}): retry runs passing the AMENDMENT_1 (b) check "
            f"{len(res) - nbad}/{len(res)}" + (f"; problems {[v for v in res.values() if v]}" if nbad else ""))
        if record_consistency(s, e):
            bad.append(f"ep{ep2(e)}: records inconsistent {record_consistency(s, e)}")
    # acceptance test (AMENDMENT_1 (d)): every episode on its t3_f0_final item -> FINAL_f0's pose columns exactly
    comp0, table0, fatal0 = compose(s, log, "all_f0")
    bad += fatal0
    if not fatal0:
        r0 = pack_dir(base / "acceptance_all_f0", comp0, COMMIT_PLACEHOLDER, {"file": "SELFTEST_ACCEPTANCE_ALL_F0"}, log)
        pose0 = (r0.get("validate") or {}).get("pose_columns_sha256")
        acc = r0["ok"] and pose0 == FINAL_F0_POSE_SHA
        log(f"ACCEPTANCE TEST: pose columns {pose0} vs FINAL_f0 {FINAL_F0_POSE_SHA}: {'PASS' if acc else 'FAIL'}; csv sha256 "
            f"{r0['pack'].get('csv_sha256')} {'==' if r0['pack'].get('csv_sha256') == FINAL_F0_CSV_SHA else '!='} FINAL_f0's")
        if not acc:
            bad.append("acceptance test failed")
    # the composition as the rule gives it now (candidates without a done post stay on their f0 item)
    comp, table, fatal = compose(s, log, "rule")
    bad += fatal
    if not fatal:
        tot, tot1 = totals(table, "own_"), totals(table, "round1_own_")
        write_table(base, table, tot, tot1, "choices_selftest")
        log(f"selftest composition: retry items {[r['episode'] for r in table if r['item'] == 'retry']}; own-input totals "
            f"{tot}; round-1 {tot1}")
        r1 = pack_dir(base / "composition_now", comp, COMMIT_PLACEHOLDER, {"file": "SELFTEST_COMPOSITION_NOW"}, log)
        if not r1["ok"]:
            bad.append("composition pack / validate failed")
        r2 = pack_dir(base / "repack_dummy_url", comp, SELFTEST_URL, {"file": "SELFTEST_REPACK_DUMMY_URL"}, log)
        same = (r2.get("validate") or {}).get("pose_columns_sha256") == (r1.get("validate") or {}).get("pose_columns_sha256")
        log(f"repack path (dummy well-formed commit URL): ok {r2['ok']}, pose columns equal: {same}")
        if not (r2["ok"] and same):
            bad.append("repack path test failed")
    write_atomic(base / "selftest.json", dump({"at": now(), "builder_sha256": self_sha(), "problems": bad, "state": st,
                                               "retry_runs_checked": checked}))
    write_atomic(base / "selftest.log", log.text())
    log(f"selftest: {len(bad)} problems" + (f": {bad}" if bad else "") + f"; {FINAL_R} exists: {FINAL_R.exists()}")
    return 0 if not bad else 1


def cmd_build(a) -> int:
    log = Logger("build")
    me = self_sha()
    log(f"build: builder {Path(__file__).resolve()} sha256 {me}")
    if FINAL_R.exists() or LOCK_FILE.exists():
        log(f"build refused: {FINAL_R} or {LOCK_FILE} already exists (FINAL_r is built once)")
        return 2
    fatal = show_pins(log)
    why = amend2_status()
    log(f"AMENDMENT_2 gate: {why or 'ok'}")
    if fatal or why:
        log("build refused (pins / AMENDMENT_2)")
        return 2
    s = snapshot()
    st = build_time_state(s)
    log(f"build-time state: {json.dumps(st)}")
    if not st["ready"]:
        log("build: not yet (build time not reached or records not yet consistent); nothing written")
        return 3
    comp, table, cfatal = compose(s, log, "rule")
    if cfatal:
        log(f"build STOPPED (a human decides): {cfatal}")
        return 1
    n_retry = sum(1 for r in table if r["item"] == "retry")
    if n_retry == 0:
        log("no retry item: FINAL_r would be identical to FINAL_f0 and is not built (AMENDMENT_1 (c))")
        return 0
    os.mkdir(check_write(FINAL_R))
    try:
        return build_into(s, st, comp, table, me, log)
    except BaseException as error:  # noqa: BLE001  (a half-built FINAL_r is never left under its final name)
        if LOCK_FILE.exists():
            log(f"error after the lock ({error!r}); FINAL_r stays locked as built")
            return 1
        log(f"build FAILED with {error!r}: FINAL_r is not valid; a human decides")
        try:
            write_atomic(FINAL_R / "build.log", log.text())
        finally:
            os.rename(FINAL_R, check_write(F / "pack" / f"FINAL_r.FAILED_{stamp()}"))
        return 1


def build_into(s: dict, st: dict, comp: dict, table: list[dict], me: str, log) -> int:
    for name, b in sorted(s["raw"].items()):
        if b is not None:
            write_bytes_new(FINAL_R / "inputs" / name, b)
    inputs_sha = {name: sha_bytes(b) for name, b in s["raw"].items() if b is not None}
    res = pack_dir(FINAL_R, comp, COMMIT_PLACEHOLDER, {"file": "FINAL_r", "note": "placeholder commit URL: NOT an upload "
                   "file; repack --commit-url <real commit> later (user's OK needed for any upload)"}, log)
    tot, tot1 = totals(table, "own_"), totals(table, "round1_own_")
    write_table(FINAL_R, table, tot, tot1, "choices")
    v = res.get("validate") or {}
    build = {"file": "FINAL_r", "built_at": now(), "builder": str(Path(__file__).resolve()), "builder_sha256": me,
             "amendment_2_sha256": sha256(AMEND2), "amendment_1_sha256": AMEND1_SHA, "prereg_v3_sha256": PREREG_V3_SHA,
             "prereg_retry_sha256": PREREG_RETRY_SHA, "t3_f0_final_prereg_sha256": PREREG_F_SHA,
             "kit_helper_sha256": KITHELPER_SHA, "build_time_state": st, "inputs_sha256": inputs_sha,
             "retry_items": [r["episode"] for r in table if r["item"] == "retry"],
             "f0_items": [r["episode"] for r in table if r["item"] == "f0"], "own_input_totals": tot,
             "round1_own_input_totals": tot1, "pack_ok": res["ok"], "csv": str(FINAL_R / CSV_NAME),
             "csv_sha256": res["pack"].get("csv_sha256"), "pose_columns_sha256": v.get("pose_columns_sha256"),
             "rows": res["pack"].get("rows"), "input_reference_residual_m": res["pack"].get("input_reference_residual_m"),
             "input_reference_residual_rad": res["pack"].get("input_reference_residual_rad"),
             "uploader_validate_submission": v.get("uploader_validate_submission"), "provenance_problems": v.get("problems"),
             "commit_url": COMMIT_PLACEHOLDER, "note": "LOCAL file: placeholder commit; never uploaded here"}
    write_atomic(FINAL_R / "build.json", dump(build))
    if not res["ok"]:
        log("build FAILED the pack checks (AMENDMENT_1 (d)): FINAL_r is not valid; a human decides")
        write_atomic(FINAL_R / "build.log", log.text())
        failed = F / "pack" / f"FINAL_r.FAILED_{stamp()}"
        os.rename(FINAL_R, check_write(failed))
        return 1
    log(f"FINAL_r built: retry items {build['retry_items']}; csv sha256 {build['csv_sha256']}; pose columns "
        f"{build['pose_columns_sha256']}; own-input totals {tot}")
    write_atomic(FINAL_R / "build.log", log.text() + f"[{now()}] lock: FILES.sha256 + {LOCK_FILE}\n")
    files = sorted(p for p in FINAL_R.rglob("*") if p.is_file())
    listing = "".join(f"{sha256(p)}  {p.relative_to(FINAL_R).as_posix()}\n" for p in files)
    write_once(FINAL_R / "FILES.sha256", listing)
    for p in files:
        os.chmod(p, 0o444)
    write_once(LOCK_FILE, f"{build['csv_sha256']}  pack/FINAL_r/{CSV_NAME}\n{sha256(FINAL_R / 'FILES.sha256')}  pack/FINAL_r/FILES.sha256\n"
                          f"# builder {me}  AMENDMENT_2 {build['amendment_2_sha256']}  locked {now()}\n")
    log(f"LOCKED: {LOCK_FILE} ({len(files)} files in FILES.sha256)")
    return 0


def verify_lock() -> str | None:
    if not LOCK_FILE.is_file() or not (FINAL_R / "FILES.sha256").is_file():
        return "FINAL_r is not locked"
    lines = LOCK_FILE.read_text().splitlines()
    if f"{sha256(FINAL_R / 'FILES.sha256')}  pack/FINAL_r/FILES.sha256" not in lines:
        return "FILES.sha256 does not match FINAL_r.sha256"
    bad = check_listing(FINAL_R / "FILES.sha256", FINAL_R)
    return f"locked files changed: {bad[:3]}" if bad else None


def read_report_only(path: Path) -> dict:
    """The one unguarded read: a LOCAL clean score, after the lock, for the report only (decides nothing)."""
    return json.loads(path.read_text())


def cmd_report(a) -> int:
    log = Logger("report")
    why = verify_lock()
    if why:
        log(f"report refused: {why}")
        return 2
    d = FINAL_R / REPORT_DIR_NAME
    os.mkdir(check_write(d))
    final_csv = FINAL_F0 / CSV_NAME
    if sha256(final_csv) != FINAL_F0_CSV_SHA:
        log("report refused: FINAL_f0 csv sha256 changed")
        return 2
    b = read_json(FINAL_R / "build.json") or {}
    out = {"note": "LOCAL_NOT_FOR_UPLOAD. Official kit metric_code against the clean reference. REPORT ONLY, computed after "
                   "FINAL_r was built and hash-locked; it changes no decision (file choice and Final mark are fixed by "
                   "AMENDMENT_1 (e) and are never made by a clean or board score).", "at": now(),
           "builder_sha256": self_sha(), "files": {}}
    for label, csvp in (("FINAL_r", FINAL_R / CSV_NAME), ("FINAL_f0", final_csv)):
        res = d / f"{label}_metric_code_LOCAL_NOT_FOR_UPLOAD.json"
        cmd = ["nice", "-n", "15", str(PACK_PYTHON), "-I", "-B", str(KITHELPER), "score", "--csv", str(csvp), "--track", TRACK,
               "--episodes", "all", "--out", str(check_write(res))]
        p = subprocess.run(cmd, cwd=d, env=kit_env(), capture_output=True, text=True, timeout=PACK_TIMEOUT)
        write_atomic(d / f"{label}_score_LOCAL_NOT_FOR_UPLOAD.log", f"$ {shlex.join(cmd)}\n{p.stdout}\n{p.stderr}")
        if p.returncode != 0 or not res.is_file():
            log(f"report: metric_code for {label} failed (exit {p.returncode})")
            return 1
        out["files"][label] = {"csv": str(csvp), "csv_sha256": sha256(csvp), "metric_code": read_report_only(res)}
    retry_eps = set(b.get("retry_items") or [])
    pe_r = out["files"]["FINAL_r"]["metric_code"].get("per_episode_float32") or {}
    pe_f = out["files"]["FINAL_f0"]["metric_code"].get("per_episode_float32") or {}
    M = ("AUC", "SP-SR", "MP-SR", "MPPE")
    out["per_episode"] = {str(e): {"item": "retry" if e in retry_eps else "f0", "FINAL_r": pe_r.get(str(e)), "FINAL_f0": pe_f.get(str(e))}
                          for e in EPISODES}
    write_atomic(d / "report_LOCAL_NOT_FOR_UPLOAD.json", dump(out))

    def g(x, m, n=4):
        v = (x or {}).get(m)
        return "" if v is None else f"{v:.{n}f}"

    L = ["# Tier-3 FINAL_r vs FINAL_f0: LOCAL clean score (REPORT ONLY, NOT FOR ANY DECISION)", "",
         f"Generated {now()} after FINAL_r was built and hash-locked (t3_f0_final/FINAL_r.sha256). Official kit metric_code "
         "(t3final_kit.py score) against the clean reference. Nothing here chooses a file, an item, a run or the Final mark "
         "(AMENDMENT_1 (e), AMENDMENT_2).", "", "| file | csv sha256 | AUC | SP-SR | MP-SR | MPPE cm | (float64 AUC) |",
         "|---|---|---|---|---|---|---|"]
    for label, x in out["files"].items():
        ov, ov64 = x["metric_code"].get("overall_float32") or {}, x["metric_code"].get("overall_float64") or {}
        L.append(f"| {label} | {x['csv_sha256'][:16]} | " + " | ".join(g(ov, m, 2 if m == "MPPE" else 4) for m in M)
                 + f" | {g(ov64, 'AUC')} |")
    L += ["", "| ep | item | FINAL_r AUC | FINAL_f0 AUC | d AUC | FINAL_r SP/MP | FINAL_f0 SP/MP | FINAL_r MPPE | FINAL_f0 MPPE |",
          "|---|---|---|---|---|---|---|---|---|"]
    for e in EPISODES:
        r, f0 = pe_r.get(str(e)) or {}, pe_f.get(str(e)) or {}
        dauc = "" if r.get("AUC") is None or f0.get("AUC") is None else f"{r['AUC'] - f0['AUC']:+.4f}"
        L.append(f"| {e:02d} | {'retry' if e in retry_eps else 'f0'} | {g(r, 'AUC')} | {g(f0, 'AUC')} | {dauc} | "
                 f"{g(r, 'SP-SR', 0)}/{g(r, 'MP-SR', 0)} | {g(f0, 'SP-SR', 0)}/{g(f0, 'MP-SR', 0)} | {g(r, 'MPPE', 2)} | {g(f0, 'MPPE', 2)} |")
    write_atomic(d / "report_LOCAL_NOT_FOR_UPLOAD.md", "\n".join(L) + "\n")
    log(f"report written: {d} (report only)")
    return 0


def cmd_repack(a) -> int:
    log = Logger("repack")
    url = a.commit_url.strip()
    if not COMMIT_URL_RE.match(url):
        print(f"--commit-url must be https://github.com/<owner>/<repo>/commit/<40 hex>: {url!r}")
        return 2
    why = verify_lock() or amend2_status()
    if why:
        log(f"repack refused: {why}")
        return 2
    fj = read_json(FINAL_R / "file.json") or {}
    b = read_json(FINAL_R / "build.json") or {}
    sha7 = url.rsplit("/", 1)[1][:7]
    d = F / "upload_r" / sha7
    if d.exists():
        log(f"repack refused: {d} exists")
        return 1
    d.mkdir(parents=True)
    comp = {int(e): s for e, s in fj["episodes"].items()}
    res = pack_dir(d, comp, url, {"file": "FINAL_r (repack with the real commit URL)"}, log)
    same = (res.get("validate") or {}).get("pose_columns_sha256") == b.get("pose_columns_sha256")
    if res["ok"] and same:
        out = d / f"t3_final_r_closest_{sha7}.csv"
        shutil.copyfile(d / CSV_NAME, out)
        write_atomic(d / "UPLOAD_CANDIDATE.json", dump({"csv": str(out), "sha256": sha256(out), "commit_url": url,
                                                        "pose_columns_equal_FINAL_r": same, "at": now(),
                                                        "note": "candidate only; the user decides; nothing uploaded"}))
        log(f"repack: {out} ({sha256(out)[:16]}), pose columns equal FINAL_r: {same}")
        return 0
    log(f"repack FAILED: ok {res['ok']}, pose columns equal {same}")
    return 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("selftest", "when", "build", "report"):
        sub.add_parser(name)
    r = sub.add_parser("repack")
    r.add_argument("--commit-url", required=True)
    a = p.parse_args()
    if not sys.flags.dont_write_bytecode or not sys.flags.isolated:
        print("run with python -I -B")
        return 2
    return {"selftest": cmd_selftest, "when": cmd_when, "build": cmd_build, "report": cmd_report, "repack": cmd_repack}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())

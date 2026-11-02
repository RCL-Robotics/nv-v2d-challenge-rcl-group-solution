#!/usr/bin/env python3
"""t2cons_consensus.py: steps S1 (selftest, own-input validation), S2 (references), S4 (consensus scores) and S5
(selection) of fc_runs/submissions/t2_consensus_20261024. It implements PREREG.md sections 2, 3, 4 and 5 unchanged.

Noisy-only. Every file this script opens goes through guard() first, which refuses (case-insensitive, on the path as
given and as resolved) any path containing: clean, LOCAL_NOT_FOR_UPLOAD, packed_, tier_1_multiview_caption,
track_2/baseline, /data/human_motion_data/, /data/v2d_challenge_data_public (PREREG 5.7). The rollout parquet of each
ok try is the only file opened inside a run folder. The packer's main() / crosscheck() and
v2dlb.dataset.participant_reference_root() are never called (they read the clean Tier-1 reference).

Kit code is imported unchanged; its bytes must equal inputs/KIT.sha256:
  tools/pack_submission.py: input_frame_reference, convert_track2_frame
  v2dlb.trajectory.read_episode, v2dlb.submission_parquet.read_trajectory / provenance,
  v2dlb.object_pose.episode_metric / _normalise_quaternions, v2dlb.vertices.mesh_path_for / sample_vertices,
  config/tracks.json (track_2_tier2: align_initial_pose False).

Run with the kit Python, isolated, no bytecode:
  conda_envs/v2dkit/bin/python -I -B scripts/t2cons_consensus.py selftest   S1  PREREG 3.5.1  -> selftest/selftest.json
  conda_envs/v2dkit/bin/python -I -B scripts/t2cons_consensus.py validate   S1  PREREG 3.5.2  -> scores/own_input_validation.*
  conda_envs/v2dkit/bin/python -I -B scripts/t2cons_consensus.py refs       S2  PREREG 2      -> refs/{A,B}/, refs/build.json
  conda_envs/v2dkit/bin/python -I -B scripts/t2cons_consensus.py score      S4  PREREG 3      -> scores/consensus_*.{csv,json}
  conda_envs/v2dkit/bin/python -I -B scripts/t2cons_consensus.py select     S5  PREREG 5      -> selection*, summary_consensus.*,
                                                                                               SELECTION.sha256 (last)
refs / score / select refuse to run unless: PREREG.md == PREREG.sha256; every PREREG section 11 pin holds;
inputs/SNAPSHOT.sha256 and inputs/KIT.sha256 hold; PREREG_PINS.md == PREREG_PINS.sha256 and pins this file, the selftest
result and the validation result, and both say PASS. select refuses once SELECTION.sha256 exists.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
ROOT = ME / "fc_runs/submissions/t2_consensus_20261024"
B5 = ME / "fc_runs/submissions/t2_best5_20261024"  # read only: exports/<job>/export.json and the ok tries' parquets
ZU = Path("/root/autodl-tmp/zuxing.lu")  # read only: frame_refs_g2 (variant B)
KIT = ME / "v2d_submission_kit_src/v2d_submission_kit"
STOCK_REPO = ME / "code/video_to_data_8c778f9"
SNAP = ROOT / "inputs"
PREREG, PREREG_SHA = ROOT / "PREREG.md", ROOT / "PREREG.sha256"
PINS, PINS_SHA = ROOT / "PREREG_PINS.md", ROOT / "PREREG_PINS.sha256"
SEL_SHA = ROOT / "SELECTION.sha256"
TZ8 = dt.timezone(dt.timedelta(hours=8))

T2 = ME / "data/video_to_data_challenge/track_2/tier_2_synthetic_noise"
K3FIX_REFS = ME / "tmp_analysis/tier2/k3fix_pilot/frame_refs_k3fix"
K3_REFS = ME / "tmp_analysis/tier2/kalman/eval/frame_refs_k3"
K3CA_REFS = ME / "tmp_analysis/tier2/k3ca/frame_refs_k3ca"
G2_REFS = ZU / "data/human_motion_data_tier2_g2-2026-10-24/frame_refs_g2"
TRACKS = {"R": T2, "F": K3FIX_REFS, "C": K3CA_REFS, "G": G2_REFS}  # PREREG 2.2, in de-duplication order
VARIANTS = {"A": ("R", "F", "C"), "B": ("R", "F", "C", "G")}  # PREREG 2.5
INPUT_REF = {"raw": T2, "k3fix": K3FIX_REFS, "k3": K3_REFS, "k3ca": K3CA_REFS, "g2": G2_REFS}  # PREREG 3.2 table
GATE_FAIL = (0, 5, 15, 19, 22, 26, 28)
EXPECTED_USED = {  # PREREG 2.5 table (inventory pass at writing); a mismatch stops the build
    "A": {e: (("R", "F", "C") if e in GATE_FAIL else ("R", "F")) for e in range(30)},
    "B": {e: (("R", "F", "C", "G") if e in (19, 28) else ("R", "F", "G") if e in (1, 4, 9, 21)
              else (("R", "F", "C") if e in GATE_FAIL else ("R", "F"))) for e in range(30)},
}  # fmt: skip
A_CHALLENGERS = {  # PREREG 4.1 table: number of variant-A challengers per episode (consistency check)
    **{e: 0 for e in (0, 15, 19, 22, 28)}, **{e: 1 for e in (4, 8, 13, 17, 20, 21, 23, 24, 25, 27)},
    **{e: 2 for e in (2, 6, 7, 10, 11, 12, 14, 16, 18, 29, 9, 5, 26)}, 1: 3, 3: 4,
}  # fmt: skip
METRICS = {"auc": "add_auc", "sp_sr": "spider_sr", "mp_sr": "maniptrans_sr", "mppe": "mppe_cm"}
NRUNS, MARGIN, FLOOR = 5, 0.01, 4  # PREREG 5.3 / 5.4
GATE = {"max_abs_d_auc": 0.02, "max_abs_d_mppe": 0.5, "min_sp_equal_frac": 0.95, "min_mp_equal_frac": 0.95}  # 3.5.2
EVAL_SCRIPT_SHA = "160d24f5fbeb70ccd569ca4192056f1892b2a9f2999066db9d8279461648fdf1"
EVAL_CODE_SHA = "b1b3ebe3ae4629d1687a83f420149e3f6fdfc154d70acad55c12fc8fa69811a1"
FORBIDDEN = ("clean", "local_not_for_upload", "packed_", "tier_1_multiview_caption", "track_2/baseline",
             "/data/human_motion_data/", "/data/v2d_challenge_data_public")  # PREREG 5.7  # fmt: skip
KIT_FILES = ("tools/pack_submission.py", "v2dlb/object_pose.py", "v2dlb/trajectory.py", "v2dlb/submission_parquet.py",
             "v2dlb/vertices.py", "v2dlb/dataset.py", "v2dlb/report_schema.py", "config/tracks.json")  # fmt: skip
N_T2BEST5, N_T2BEST5_A, N_NEW, N_NEW_A = 75, 71, 5, 2  # PREREG 4.1 / 4.2


# ================================================================================================ helpers
def now() -> str:
    return dt.datetime.now(TZ8).strftime("%Y-%m-%d %H:%M:%S")


def stamp() -> str:
    return dt.datetime.now(TZ8).strftime("%Y%m%d_%H%M%S")


def guard(p) -> Path:
    """PREREG 5.7: refuse, before opening, any forbidden path (as given and as resolved)."""
    for s in (str(p), os.path.realpath(str(p))):
        low = s.lower()
        hit = [tok for tok in FORBIDDEN if tok in low]
        if hit:
            raise PermissionError(f"path guard (PREREG 5.7): refusing {p} (contains {hit})")
    return Path(p)


def sha256_file(file) -> str:
    h = hashlib.sha256()
    with guard(file).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(p):
    return json.loads(guard(p).read_text())


def read_csv(p) -> list[dict]:
    with guard(p).open(newline="") as fp:
        return list(csv.DictReader(fp))


def write_bytes_atomic(p: Path, data: bytes) -> None:
    p = guard(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, p)


def write_text(p: Path, text: str) -> None:
    write_bytes_atomic(p, text.encode())


def dump(obj) -> str:
    return json.dumps(obj, indent=1, sort_keys=False, default=_json_default) + "\n"


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def csv_text(rows: list[dict], cols: list[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    return buf.getvalue()


def fnum(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def f4(v) -> str:
    return "" if v is None else f"{float(v):.4f}"


def ep_file(root: Path, e: int) -> Path:
    """The file input_frame_reference() opens for episode e under root."""
    if root == T2:
        return T2 / "data/chunk-000" / f"episode_{e:06d}.parquet"
    return root / f"episode_{e:06d}.npz"


# ================================================================================================ pins / integrity
def git_head(repo: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                              env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}).stdout.strip()  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return ""


def hash_list(p: Path) -> dict[str, str]:
    out = {}
    for line in guard(p).read_text().splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and len(parts[0]) == 64:
            out[parts[1].strip()] = parts[0]
    return out


def check_pins(need_pins: bool) -> tuple[list[str], list[str]]:
    fatal, info = [], []
    want = (PREREG_SHA.read_text().split() or [""])[0] if PREREG_SHA.is_file() else ""
    got = sha256_file(PREREG) if PREREG.is_file() else ""
    if not want or got != want:
        return [f"PREREG.md sha256 {got[:16]} != PREREG.sha256 {want[:16]}"], info
    info.append(f"PREREG.md sha256 {got} == PREREG.sha256")
    for line in PREREG.read_text().split("## 11. Pins", 1)[-1].splitlines():
        f = line.split()
        if len(f) == 3 and f[0] == "pin":
            p, label = (ROOT / f[1] if f[1].startswith("inputs/") else ME / f[1]), f[1]
        elif len(f) == 4 and f[0] == "pin" and f[1] == "kit":
            p, label, f = KIT / f[2], "kit:" + f[2], ["pin", f[2], f[3]]
        elif len(f) == 3 and f[0] == "commit":
            head = git_head(ME / f[1])
            (info if head == f[2] else fatal).append(f"commit {f[1]}: HEAD {head[:12]} (pinned {f[2][:12]})")
            continue
        else:
            continue
        try:
            guard(p)
        except PermissionError:
            info.append(f"pin {label}: not opened by S1-S5 (path guard, PREREG 5.7); checked at S6")
            continue
        h = sha256_file(p) if p.is_file() else "missing"
        (info if h == f[2] else fatal).append(f"pin {label}: {h[:16]} (pinned {f[2][:16]})")
    n = 0
    for rel, h0 in hash_list(SNAP / "SNAPSHOT.sha256").items():
        p = SNAP / rel
        if p.name == "SNAPSHOT.sha256":
            continue
        n += 1
        h = sha256_file(p) if p.is_file() else "missing"
        if h != h0:
            fatal.append(f"inputs/{rel}: {h[:16]} != SNAPSHOT.sha256 {h0[:16]}")
    info.append(f"inputs/SNAPSHOT.sha256: {n} files checked")
    kit = hash_list(SNAP / "KIT.sha256")
    for rel in KIT_FILES:
        h = sha256_file(KIT / rel) if (KIT / rel).is_file() else "missing"
        if kit.get(rel) != h:
            fatal.append(f"kit {rel}: {h[:16]} != inputs/KIT.sha256 {str(kit.get(rel))[:16]}")
    info.append(f"kit files used ({len(KIT_FILES)}) == inputs/KIT.sha256")
    mine = sha256_file(Path(__file__).resolve())
    info.append(f"scripts/t2cons_consensus.py sha256 {mine}")
    if need_pins:
        pw = (PINS_SHA.read_text().split() or [""])[0] if PINS_SHA.is_file() else ""
        pg = sha256_file(PINS) if PINS.is_file() else ""
        if not pw or pg != pw:
            fatal.append(f"PREREG_PINS.md sha256 {pg[:16]} != PREREG_PINS.sha256 {pw[:16]}")
            return fatal, info
        text = PINS.read_text()
        if f"pin scripts/t2cons_consensus.py {mine}" not in text:
            fatal.append(f"PREREG_PINS.md does not pin this script's sha256 {mine[:16]}")
        for line in text.splitlines():
            f = line.split()
            if len(f) == 3 and f[0] == "pin":
                p = ME / f[1] if f[1].startswith("scripts/") else ROOT / f[1]
                h = sha256_file(p) if p.is_file() else "missing"
                (info if h == f[2] else fatal).append(f"PINS pin {f[1]}: {h[:16]} (pinned {f[2][:16]})")
        for rel in ("selftest/selftest.json", "scores/own_input_validation.json"):
            if f"pin {rel} " not in text:
                fatal.append(f"PREREG_PINS.md does not pin {rel}")
                continue
            res = read_json(ROOT / rel)
            if res.get("pass") is not True:
                fatal.append(f"{rel}: pass is {res.get('pass')!r}")
        info.append(f"PREREG_PINS.md sha256 {pg[:16]} == PREREG_PINS.sha256; pins this script, selftest PASS, validation PASS")
    return fatal, info


def tree_listing(root: Path) -> dict[str, list]:
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for name in filenames:
            p = Path(dirpath) / name
            try:
                st = p.lstat()
            except OSError:
                continue
            out[str(p.relative_to(root))] = [st.st_size, st.st_mtime_ns]
    return out


def integrity() -> dict:
    return {"kit": tree_listing(KIT), "stock": tree_listing(STOCK_REPO)}


def integrity_diff(a: dict, b: dict) -> list[str]:
    out = []
    for key in ("kit", "stock"):
        x, y = a.get(key, {}), b.get(key, {})
        for p in sorted(set(x) | set(y)):
            if x.get(p) != y.get(p):
                out.append(f"{key}: {p} " + ("new" if p not in x else "removed" if p not in y else "changed"))
    return out


# ================================================================================================ kit
def load_kit() -> SimpleNamespace:
    kit_sha = hash_list(SNAP / "KIT.sha256")
    for rel in KIT_FILES:
        if sha256_file(KIT / rel) != kit_sha.get(rel):
            raise SystemExit(f"kit {rel} differs from inputs/KIT.sha256")
    if str(KIT) not in sys.path:
        sys.path.insert(0, str(KIT))
    from v2dlb import object_pose as op
    from v2dlb.submission_parquet import provenance, read_trajectory
    from v2dlb.trajectory import read_episode
    from v2dlb.vertices import mesh_path_for, sample_vertices

    for mod, rel in ((op, "v2dlb/object_pose.py"),):
        if Path(mod.__file__).resolve() != (KIT / rel).resolve():
            raise SystemExit(f"{mod.__name__} imported from {mod.__file__}, not the kit")
    spec = importlib.util.spec_from_file_location("kit_pack_submission", KIT / "tools/pack_submission.py")
    pk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pk)
    import pandas as pd

    config = read_json(KIT / "config/tracks.json")
    tspec = config["tracks"]["track_2_tier2"]
    align = bool(tspec.get("align_initial_pose", False))
    if align is not False or list(tspec["metrics"]) != list(METRICS.values()):
        raise SystemExit(f"track_2_tier2 config differs from PREREG 3.3: align {align}, metrics {tspec['metrics']}")
    return SimpleNamespace(op=op, read_episode=read_episode, read_trajectory=read_trajectory, provenance=provenance,
                           mesh_path_for=mesh_path_for, sample_vertices=sample_vertices, pk=pk, pd=pd, align=align,
                           input_frame_reference=pk.input_frame_reference,
                           convert_track2_frame=pk.convert_track2_frame, verts={},
                           tracks_sha=hash_list(SNAP / "TRACKS.sha256"))  # fmt: skip


def grid(K, e: int):
    """PREREG 2.1: the visible frames of the released Tier-2 input, and its object order."""
    r = K.read_episode(guard(T2 / "data/chunk-000" / f"episode_{e:06d}.parquet"), e)
    return np.asarray(r.frame_index, dtype=np.int64), list(r.object_names)


def read_track(K, root: Path, e: int, frames, names):
    """(T, 7) on the grid via the kit's own input_frame_reference(root, e, grid, 1, names); None if no file."""
    f = guard(ep_file(guard(root), e))
    if not f.is_file():
        return None
    return K.input_frame_reference(root, e, frames, 1, names)


def vertices(K, name: str) -> np.ndarray:
    if name not in K.verts:
        mesh = guard(K.mesh_path_for(guard(T2 / "mesh"), name))
        h = sha256_file(mesh)
        if K.tracks_sha.get(str(mesh)) != h:
            raise SystemExit(f"{mesh} sha256 {h[:16]} != inputs/TRACKS.sha256")
        K.verts[name] = (K.sample_vertices(mesh), mesh)
    return K.verts[name][0]


# ================================================================================================ combination (PREREG 2.3 / 2.4)
def dedup(named: list[tuple[str, np.ndarray]]):
    kept, dropped = [], []
    for tid, arr in named:
        same = next((k for k, a in kept if np.array_equal(a, arr)), None)
        if same is None:
            kept.append((tid, arr))
        else:
            dropped.append([tid, same])
    return kept, dropped


def normalise(op, pose: np.ndarray) -> np.ndarray:
    return op._normalise_quaternions(np.asarray(pose, dtype=np.float64), "track", ValueError)


def combine(op, tracks: list[np.ndarray], sign_ref: np.ndarray | None = None):
    """Componentwise median position; chordal L2 mean rotation (top eigenvector of sum q q^T, float64, eigh)."""
    norm = [normalise(op, t) for t in tracks]
    P = np.stack([t[:, :3] for t in norm])
    Q = np.stack([t[:, 3:] for t in norm])
    pos = np.median(P, axis=0)
    M = np.einsum("ntk,ntl->tkl", Q, Q)
    w, v = np.linalg.eigh(M)
    q = v[:, :, -1].copy()
    ref = Q[0] if sign_ref is None else normalise(op, sign_ref)[:, 3:]
    d = np.sum(q * ref, axis=-1)
    flip = (d < 0) | ((d == 0) & (q[:, 3] < 0))
    q[flip] *= -1.0
    gap = w[:, -1] - w[:, -2] if w.shape[1] > 1 else w[:, -1]
    return np.concatenate([pos, q], axis=1), float(gap.min())


def pair_distance(op, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a, b = normalise(op, a), normalise(op, b)
    dp = float(np.linalg.norm(a[:, :3] - b[:, :3], axis=1).max())
    ang = float(op._quaternion_geodesic_angle(a[:, 3:], b[:, 3:]).max())
    return dp, ang


# ================================================================================================ jobs / evaluations
def load_jobs() -> dict[str, dict]:
    rows = [(r, "t2_best5") for r in read_json(SNAP / "t2_best5/manifest.json")["rows"]]
    rows += [(json.loads(ln), "t2_best5") for ln in guard(SNAP / "t2_best5/manifest_stageB.jsonl").read_text().splitlines()
             if ln.strip()]  # fmt: skip
    rows += [(r, "new") for r in read_json(ROOT / "manifest_new.json")["rows"]]
    jobs = {}
    for r, src in rows:
        if r["job"] in jobs:
            raise SystemExit(f"duplicate job {r['job']}")
        kind = r.get("root_kind")
        if kind not in INPUT_REF:
            raise SystemExit(f"{r['job']}: unknown input kind {kind}")
        fr = r.get("frame_refs")
        if (kind == "raw" and fr is not None) or (kind != "raw" and Path(str(fr)) != INPUT_REF[kind]):
            raise SystemExit(f"{r['job']}: manifest frame_refs {fr} != PREREG 3.2 input reference for {kind}")
        base = B5 if src == "t2_best5" else ROOT
        exp = read_json(base / "exports" / r["job"] / "export.json")
        jobs[r["job"]] = {
            "job": r["job"], "episode": int(r["episode"]), "source": src, "status": r.get("status"), "input": kind,
            "input_reference": str(INPUT_REF[kind]), "family": r.get("family"), "owner": r.get("owner"),
            "iteration": r.get("iteration"), "provisional": bool(r.get("provisional")), "object": r.get("object"),
            "parquet_sha256": r.get("parquet_sha256"), "variants": ("B",) if kind == "g2" else ("A", "B"),
            "export_status": exp.get("status"), "export_sha256": exp.get("sha256"),
        }  # fmt: skip
    return jobs


def load_evals(jobs: dict[str, dict]) -> dict[str, dict[int, dict]]:
    """job -> run -> the ok own-input row (evaluator values) + the rollout parquet path of that ok try (PREREG 3.1)."""
    out: dict[str, dict[int, dict]] = {}
    for src, file in (("t2_best5", SNAP / "t2_best5/own_input_evals.csv"), ("new", ROOT / "own_input_evals_new.csv")):
        for r in read_csv(file):
            if r.get("status") != "ok":
                continue
            j = jobs.get(r["job"])
            if j is None or j["source"] != src:
                raise SystemExit(f"{file.name}: ok row of unknown job {r['job']}")
            k, t, e = int(r["run"]), int(r["try"]), j["episode"]
            if int(r["episode"]) != e:
                raise SystemExit(f"{r['job']} run {k}: episode {r['episode']} != {e}")
            base = B5 if src == "t2_best5" else ROOT
            pq = base / "runs" / r["job"] / f"run{k}" / f"try{t}" / "results" / f"episode_{e:06d}.parquet"
            if src == "new" and Path(r["results_parquet"]) != pq:
                raise SystemExit(f"{r['job']} run {k}: results_parquet {r['results_parquet']} != {pq}")
            if not str(j["export_sha256"] or "").startswith(r["export_sha256"]) or len(r["export_sha256"]) < 16:
                raise SystemExit(f"{r['job']} run {k}: export_sha256 {r['export_sha256']} != export.json")
            if k in out.setdefault(r["job"], {}):
                raise SystemExit(f"{r['job']} run {k}: two ok rows")
            out[r["job"]][k] = {"run": k, "try": t, "parquet": str(pq),
                                **{m: fnum(r[m]) for m in METRICS}}  # fmt: skip
    return out


def candidates(jobs, evals) -> list[str]:
    """PREREG 4.1 / 4.2: every job with 5 ok official evaluations (t2_best5 snapshot + the 5 new jobs)."""
    cand = [j for j in sorted(jobs) if jobs[j]["status"] == "ok" and jobs[j]["export_status"] == "ok"
            and sorted(evals.get(j, {})) == list(range(1, NRUNS + 1))]  # fmt: skip
    n5 = [j for j in cand if jobs[j]["source"] == "t2_best5"]
    nn = [j for j in cand if jobs[j]["source"] == "new"]
    counts = (len(n5), sum("A" in jobs[j]["variants"] for j in n5), len(nn), sum("A" in jobs[j]["variants"] for j in nn))
    if counts != (N_T2BEST5, N_T2BEST5_A, N_NEW, N_NEW_A):
        raise SystemExit(f"candidate counts {counts} != PREREG 4.1/4.2 {(N_T2BEST5, N_T2BEST5_A, N_NEW, N_NEW_A)}")
    return cand


# ================================================================================================ scoring (PREREG 3)
def rollout_on_grid(K, job: dict, rec: dict, frames, names):
    """The packer's own ordering, checks and frame conversion. Returns (converted (T,7), info) or raises ValueError."""
    traj = K.read_trajectory(guard(rec["parquet"]))
    e = job["episode"]
    if traj.episode_index != e:
        raise ValueError(f"trajectory episode {traj.episode_index} != {e}")
    prov = K.provenance(traj)
    if prov.get("checkpoint_sha256") != job["export_sha256"]:
        raise ValueError(f"checkpoint_sha256 {str(prov.get('checkpoint_sha256'))[:16]} != export {job['export_sha256'][:16]}")
    if prov.get("eval_script_sha256") != EVAL_SCRIPT_SHA:
        raise ValueError(f"eval_script_sha256 {prov.get('eval_script_sha256')}")
    if prov.get("eval_code_sha256", EVAL_CODE_SHA) != EVAL_CODE_SHA:
        raise ValueError(f"eval_code_sha256 {prov.get('eval_code_sha256')}")
    pd = K.pd
    if not traj.world_valid.all():
        raise ValueError("environment_valid false")
    wanted = pd.MultiIndex.from_product([range(1), frames.tolist(), range(len(names))],
                                        names=["world_index", "frame_index", "object_slot"])  # fmt: skip
    got = pd.MultiIndex.from_arrays([traj.world_index, traj.frame_index, traj.object_slot],
                                    names=["world_index", "frame_index", "object_slot"])  # fmt: skip
    if got.duplicated().any():
        raise ValueError("duplicate (world, frame, object) rows")
    absent = wanted.difference(got)
    if len(absent):
        raise ValueError(f"missing {len(absent)} of {len(wanted)} grid rows")
    order = pd.Series(np.arange(len(got)), index=got).loc[wanted].to_numpy()
    pose = traj.pose_xyzw[order]
    if traj.object_name is not None:
        sub = np.asarray(traj.object_name, dtype=object)[order]
        exp = np.tile(np.array(names, dtype=object), len(frames))
        if not np.array_equal(sub, exp) and not (len(names) == 1 and np.all(sub == "object")):
            raise ValueError("object_name does not follow the reference object order")
    if traj.reference_xyzw is None:
        raise ValueError("no evaluator reference poses")
    sref = traj.reference_xyzw[order]
    inref = read_track(K, Path(job["input_reference"]), e, frames, names)
    if inref is None:
        raise ValueError("input reference missing")
    converted, pos_err, ang_err, changed = K.convert_track2_frame(pose, sref, inref)  # ValueError = conversion_failed
    info = {"rows": int(len(wanted)), "rollout_rows": int(len(got)), "conv_pos_residual_m": pos_err,
            "conv_ang_residual_rad": ang_err, "converted": bool(changed),
            "eval_code_recorded": "eval_code_sha256" in prov}  # fmt: skip
    return converted, inref, info


def metrics_vs(K, converted: np.ndarray, reference: np.ndarray, names) -> dict:
    T, B = len(converted) // len(names), len(names)
    achieved = K.op._normalise_quaternions(converted.reshape(T, 1, B, 7), "submission", ValueError)
    ref = K.op._normalise_quaternions(np.asarray(reference, dtype=np.float64).reshape(T, 1, B, 7), "reference", ValueError)
    verts = np.stack([vertices(K, n) for n in names])
    ids = np.arange(B, dtype=np.int64)
    return {m: float(K.op.episode_metric(key, achieved, ref, ids, verts, align_initial_pose=K.align))
            for m, key in METRICS.items()}  # fmt: skip


def stats(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=np.float64)
    return {"mean": float(a.mean()), "var": float(a.var(ddof=0)), "min": float(a.min()), "max": float(a.max())}


# ================================================================================================ selection (PREREG 5)
def floor_count(own: dict[int, dict]) -> int:
    return sum(1 for r in own.values() if r.get("sp_sr") == 1.0 and r.get("mp_sr") == 1.0)


def select_episode(default: str, challengers: list[str], cons: dict[str, dict], own: dict[str, dict[int, dict]]):
    """cons[job] = {"n": scored runs, "auc": [5 values], "mppe": [...]} for this variant; own[job] = ok own-input runs.
    Returns (pick, reason, flags, rows)."""
    flags, rows = [], []
    d = cons.get(default, {"n": 0})
    if d["n"] != NRUNS:
        return default, f"D kept: INCOMPLETE ({d['n']}/{NRUNS} consensus-scored runs)", ["INCOMPLETE"], rows
    d_mean = stats(d["auc"])["mean"]
    for c in sorted(challengers):
        x = cons.get(c, {"n": 0})
        n_own = len(own.get(c, {}))
        fl = floor_count(own.get(c, {}))
        why = []
        if x["n"] != NRUNS:
            why.append(f"{x['n']}/{NRUNS} consensus-scored runs")
        if n_own != NRUNS:
            why.append(f"{n_own}/{NRUNS} ok own-input runs")
        row = {"job": c, "n_scored": x["n"], "n_own_ok": n_own, "floor_count": fl}
        if x["n"] == NRUNS:
            st, sm = stats(x["auc"]), stats(x["mppe"])
            row.update({"cons_auc": list(x["auc"]), "cons_auc_mean": st["mean"], "cons_auc_var": st["var"],
                        "cons_mppe_mean": sm["mean"], "delta_vs_D": st["mean"] - d_mean})  # fmt: skip
            if not why:
                if not st["mean"] >= d_mean + MARGIN:
                    why.append(f"cons_mean_AUC {st['mean']:.4f} < D {d_mean:.4f} + {MARGIN}")
                if not fl >= FLOOR:
                    why.append(f"stability floor {fl}/{NRUNS} own-input runs with SP-SR = MP-SR = 1 (< {FLOOR})")
        row.update({"qualifies": not why, "why_not": why})
        rows.append(row)
    q = [r for r in rows if r["qualifies"]]
    if not q:
        return default, "D kept (no qualifying challenger)", flags, rows
    q.sort(key=lambda r: (-r["cons_auc_mean"], r["cons_auc_var"], r["cons_mppe_mean"], r["job"]))
    best = q[0]
    tie = len(q) > 1 and q[1]["cons_auc_mean"] == best["cons_auc_mean"]
    reason = (f"replaced by {best['job']} (cons_mean_AUC {best['cons_auc_mean']:.4f} >= D {d_mean:.4f} + {MARGIN}; "
              f"floor {best['floor_count']}/{NRUNS})" + ("; exact AUC tie broken by variance/MPPE/name" if tie else ""))  # fmt: skip
    return best["job"], reason, flags, rows


# ================================================================================================ S1: selftest
def cmd_selftest(a) -> int:
    K = load_kit()
    op = K.op
    rng = np.random.default_rng(12345)
    res, problems = [], []

    def check(name, ok, detail=""):
        res.append({"test": name, "ok": bool(ok), "detail": detail})
        if not ok:
            problems.append(f"{name}: {detail}")

    def rand_pose(T):
        q = rng.normal(size=(T, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        return np.concatenate([rng.normal(size=(T, 3)), q], axis=1)

    def same_rot(q1, q2, tol=1e-9):
        return np.all(np.abs(np.abs(np.sum(q1 * q2, axis=-1)) - 1.0) < tol)

    T = 50
    R, C = rand_pose(T), rand_pose(T)
    kept, dropped = dedup([("R", R), ("F", R.copy()), ("C", C), ("G", C.copy())])
    check("dedup drops identical tracks, keeps order", [k for k, _ in kept] == ["R", "C"]
          and dropped == [["F", "R"], ["G", "C"]], f"kept {[k for k, _ in kept]} dropped {dropped}")  # fmt: skip
    near = R.copy()
    near[0, 0] = np.nextafter(near[0, 0], np.inf)
    kept2, _ = dedup([("R", R), ("F", near)])
    check("dedup keeps a track that differs in one element", len(kept2) == 2)
    out1, _ = combine(op, [R])
    check("n=1 returns the (normalised) track", np.allclose(out1, normalise(op, R), atol=1e-12, rtol=0))
    # n = 2: midpoint and geodesic midpoint
    A2 = rand_pose(T)
    B2 = A2.copy()
    axis = rng.normal(size=(T, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    ang = rng.uniform(0.0, 2.5, size=T)
    dq = np.concatenate([axis * np.sin(ang / 2)[:, None], np.cos(ang / 2)[:, None]], axis=1)
    B2[:, 3:] = op._quaternion_multiply(dq, A2[:, 3:])
    B2[:, :3] = rng.normal(size=(T, 3))
    half = np.concatenate([axis * np.sin(ang / 4)[:, None], np.cos(ang / 4)[:, None]], axis=1)
    mid_q = op._quaternion_multiply(half, A2[:, 3:])
    out2, _ = combine(op, [A2, B2])
    check("n=2 position is the midpoint", np.allclose(out2[:, :3], (A2[:, :3] + B2[:, :3]) / 2, atol=1e-12, rtol=0))
    check("n=2 rotation is the geodesic midpoint", same_rot(out2[:, 3:], mid_q, 1e-9),
          f"max |1-|dot|| {float(np.max(np.abs(1 - np.abs(np.sum(out2[:, 3:] * mid_q, axis=1))))):.2e}")
    # n = 3: componentwise median
    A3, B3, C3 = rand_pose(T), rand_pose(T), rand_pose(T)
    out3, _ = combine(op, [A3, B3, C3])
    med = np.median(np.stack([A3[:, :3], B3[:, :3], C3[:, :3]]), axis=0)
    check("n=3 position is the componentwise median", np.array_equal(out3[:, :3], med))
    manual = np.array([[sorted([A3[t, i], B3[t, i], C3[t, i]])[1] for i in range(3)] for t in range(T)])
    check("n=3 median equals the middle value per component", np.array_equal(out3[:, :3], manual))
    q3 = A3.copy()
    q3[:, :3] = B3[:, :3]
    q3b = A3.copy()
    q3b[:, 3:] *= -1
    out3s, _ = combine(op, [A3, q3, q3b])
    check("n=3 identical rotations (any sign) give that rotation", same_rot(out3s[:, 3:], A3[:, 3:], 1e-12))
    # n = 4: mean of the middle two
    A4 = [rand_pose(T) for _ in range(4)]
    out4, _ = combine(op, A4)
    srt = np.sort(np.stack([x[:, :3] for x in A4]), axis=0)
    check("n=4 position is the mean of the middle two", np.allclose(out4[:, :3], (srt[1] + srt[2]) / 2, atol=1e-12, rtol=0))
    # sign invariance
    S = [rand_pose(T) for _ in range(3)]
    base, _ = combine(op, S)
    flipped = [x.copy() for x in S]
    for x in flipped[1:]:
        x[rng.random(T) < 0.5, 3:] *= -1
    alt, _ = combine(op, flipped, sign_ref=S[0])
    check("q* invariant to the sign of each input", np.allclose(base, alt, atol=1e-12, rtol=0),
          f"max diff {float(np.abs(base - alt).max()):.2e}")
    check("q* sign: dot with q_R >= 0", bool(np.all(np.sum(base[:, 3:] * normalise(op, S[0])[:, 3:], axis=1) >= 0)))
    check("q* unit norm", np.allclose(np.linalg.norm(base[:, 3:], axis=1), 1.0, atol=1e-12))
    # chordal L2 mean = argmax over unit q of sum_i (q . q_i)^2: q* beats 4000 random unit vectors and every input
    Q = np.stack([normalise(op, x)[:, 3:] for x in S])

    def objective(qq):
        return np.sum(np.einsum("tk,ntk->nt", qq, Q) ** 2, axis=0)

    f_star = objective(base[:, 3:])
    beaten = True
    for _ in range(4000):
        r = rng.normal(size=(T, 4))
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        beaten &= bool(np.all(objective(r) <= f_star + 1e-12))
    for i in range(len(S)):
        beaten &= bool(np.all(objective(Q[i]) <= f_star + 1e-12))
    check("chordal mean maximises sum (q . q_i)^2 (random search + inputs)", beaten)
    # path guard
    bad = ["/x/clean_LOCAL_NOT_FOR_UPLOAD.csv", "/x/Clean/y", "/x/CLEAN", "/x/summary_LOCAL_NOT_FOR_UPLOAD.md",
           "/x/local_not_for_upload/y", "/x/packed_x.csv", "/x/PACKED_x", "/d/track_2/tier_1_multiview_caption/e.parquet",
           "/d/track_2/TIER_1_MULTIVIEW_CAPTION", "/d/video_to_data_challenge/track_2/baseline/data/x.parquet",
           "/d/track_2/Baseline/mesh", str(ME / "data/human_motion_data/whole_body/x.parquet"),
           str(ME / "data/v2d_challenge_data_public/x"), str(ME / "data/V2D_CHALLENGE_DATA_PUBLIC")]  # fmt: skip
    refused = []
    for p in bad:
        try:
            guard(p)
        except PermissionError:
            refused.append(p)
    check("path guard refuses every forbidden pattern", refused == bad, f"not refused: {sorted(set(bad) - set(refused))}")
    good = [str(T2 / "data/chunk-000/episode_000000.parquet"), str(K3FIX_REFS / "episode_000000.npz"),
            str(K3CA_REFS / "episode_000000.npz"), str(G2_REFS / "episode_000001.npz"),
            str(ME / "data/human_motion_data_tier2/x"), str(B5 / "runs/j/run1/try1/results/episode_000000.parquet"),
            str(T2 / "mesh/iron/iron.glb")]  # fmt: skip
    passed = []
    for p in good:
        try:
            guard(p)
            passed.append(p)
        except PermissionError:
            pass
    check("path guard passes the noisy-only inputs", passed == good, f"refused: {sorted(set(good) - set(passed))}")
    # metric plumbing: achieved == reference; a rigid frame change is undone by the packer's conversion
    name = "iron"
    ref = rand_pose(T)
    ref[:, :3] *= 0.3
    m0 = metrics_vs(K, ref.copy(), ref, [name])
    check("identical poses: AUC 1, SP 1, MP 1, MPPE 0", m0["auc"] == 1.0 and m0["sp_sr"] == 1.0 and m0["mp_sr"] == 1.0
          and abs(m0["mppe"]) < 1e-9, str(m0))  # fmt: skip
    gq = rng.normal(size=4)
    gq /= np.linalg.norm(gq)
    G = op._rotation_matrix(gq[None])[0]
    off = rng.normal(size=3)
    sim = ref.copy()
    sim[:, :3] = ref[:, :3] @ G.T + off
    sim[:, 3:] = op._quaternion_multiply(np.broadcast_to(gq, ref[:, 3:].shape), ref[:, 3:])
    ach_sim = sim.copy()
    ach_sim[:, :3] += 0.004
    conv, pe, ae, _ = K.convert_track2_frame(ach_sim, sim, ref)
    m1 = metrics_vs(K, conv, ref, [name])
    m1_sim = metrics_vs(K, ach_sim, sim, [name])
    check("conversion residual ~0 for an exact rigid frame change", pe < 1e-9 and ae < 1e-6, f"{pe:.2e} m {ae:.2e} rad")
    check("metrics invariant to the frame change", all(abs(m1[m] - m1_sim[m]) < 1e-9 for m in METRICS), f"{m1} {m1_sim}")
    # selection rule on synthetic records
    own_ok = {k: {"sp_sr": 1.0, "mp_sr": 1.0} for k in range(1, 6)}
    own_3 = {k: {"sp_sr": 1.0, "mp_sr": 1.0 if k <= 3 else 0.0} for k in range(1, 6)}
    own_4 = {k: {"sp_sr": 1.0 if k != 2 else 0.0, "mp_sr": 1.0} for k in range(1, 6)}

    def rec(xs, mppe=1.0):
        return {"n": len(xs), "auc": list(xs), "mppe": [mppe] * len(xs)}

    cons = {"D": rec([0.5] * 5), "c_margin": rec([0.51] * 5), "c_below": rec([0.5099999] * 5), "c_floor3": rec([0.9] * 5),
            "c_floor4": rec([0.6] * 5), "c_4runs": rec([0.9] * 4)}  # fmt: skip
    own = {"D": own_ok, "c_margin": own_ok, "c_below": own_ok, "c_floor3": own_3, "c_floor4": own_4, "c_4runs": own_ok}
    pick, why, _, rows = select_episode("D", ["c_below", "c_floor3", "c_4runs"], cons, own)
    check("below margin / floor 3 of 5 / 4 scored runs do not qualify", pick == "D", f"{pick} {why}")
    d_exact = 0.5 + MARGIN
    cons["c_exact"] = rec([d_exact] * 5)
    own["c_exact"] = own_ok
    pick, why, _, _ = select_episode("D", ["c_exact"], cons, own)
    check("exactly D + 0.01 qualifies (>=)", pick == "c_exact", f"{pick} {why}")
    pick, why, _, _ = select_episode("D", ["c_margin", "c_floor4", "c_below"], cons, own)
    check("floor 4 of 5 qualifies; highest mean wins", pick == "c_floor4", f"{pick} {why}")
    # exactly representable values, so the four means tie exactly (0.625)
    cons.update({"t_a": {"n": 5, "auc": [0.625] * 5, "mppe": [2.0] * 5},
                 "t_b": {"n": 5, "auc": [0.625] * 5, "mppe": [1.0] * 5},
                 "t_c": {"n": 5, "auc": [0.5, 0.75, 0.625, 0.625, 0.625], "mppe": [0.5] * 5},
                 "t_d": {"n": 5, "auc": [0.625] * 5, "mppe": [1.0] * 5}})  # fmt: skip
    for j in ("t_a", "t_b", "t_c", "t_d"):
        own[j] = own_ok
    tie_means = {j: stats(cons[j]["auc"])["mean"] for j in ("t_a", "t_b", "t_c", "t_d")}
    pick, why, _, _ = select_episode("D", ["t_a", "t_b", "t_c", "t_d"], cons, own)
    check("exact ties: lower variance, then lower MPPE, then name", len(set(tie_means.values())) == 1 and pick == "t_b",
          f"{pick} {why} means {tie_means}")  # fmt: skip
    pick, why, _, _ = select_episode("D", ["t_d", "t_b"], cons, own)
    check("exact tie on AUC, variance and MPPE: ASCII job name", pick == "t_b", f"{pick} {why}")
    cons["D4"] = rec([0.5] * 4)
    pick, why, flags, _ = select_episode("D4", ["c_floor4"], cons, own)
    check("INCOMPLETE default is kept, no replacement", pick == "D4" and flags == ["INCOMPLETE"], f"{pick} {flags}")
    out = {"written_at": now(), "script_sha256": sha256_file(Path(__file__).resolve()), "pass": not problems,
           "n_tests": len(res), "problems": problems, "tests": res}  # fmt: skip
    write_text(ROOT / "selftest/selftest.json", dump(out))
    for r in res:
        print(("ok   " if r["ok"] else "FAIL ") + r["test"] + (f"  [{r['detail']}]" if r["detail"] and not r["ok"] else ""))
    print(f"selftest: {len(res)} tests, {len(problems)} problems -> {ROOT / 'selftest/selftest.json'}")
    return 0 if not problems else 1


# ================================================================================================ S1: own-input validation
def cmd_validate(a) -> int:
    fatal, info = check_pins(need_pins=False)
    for x in info:
        print("  " + x)
    if fatal:
        print("FATAL:\n  " + "\n  ".join(fatal))
        return 2
    st = read_json(ROOT / "selftest/selftest.json")
    if st.get("pass") is not True or st.get("script_sha256") != sha256_file(Path(__file__).resolve()):
        print("FATAL: selftest/selftest.json is not a PASS of this script")
        return 2
    before = integrity()
    K = load_kit()
    jobs = load_jobs()
    evals = load_evals(jobs)
    cand = candidates(jobs, evals)
    rows, problems = [], []
    grids = {}
    for job in cand:
        j = jobs[job]
        e = j["episode"]
        if e not in grids:
            grids[e] = grid(K, e)
        frames, names = grids[e]
        for k in range(1, NRUNS + 1):
            rec = evals[job][k]
            row = {"job": job, "episode": e, "run": k, "try": rec["try"], "input": j["input"], "source": j["source"]}
            try:
                conv, inref, inf = rollout_on_grid(K, j, rec, frames, names)
                mine = metrics_vs(K, conv, inref, names)
            except ValueError as err:
                row.update({"status": "conversion_failed" if "rigid transform" in str(err) else "failed", "problem": str(err)})
                problems.append(f"{job} run {k}: {err}")
                rows.append(row)
                continue
            row.update({"status": "ok", **inf})
            for m in METRICS:
                row[f"eval_{m}"] = rec[m]
                row[f"mine_{m}"] = mine[m]
                row[f"d_{m}"] = None if rec[m] is None else mine[m] - rec[m]
            rows.append(row)
    ok = [r for r in rows if r["status"] == "ok"]
    d_auc = max((abs(r["d_auc"]) for r in ok), default=None)
    d_mppe = max((abs(r["d_mppe"]) for r in ok), default=None)
    sp_eq = sum(r["mine_sp_sr"] == r["eval_sp_sr"] for r in ok) / len(ok) if ok else 0.0
    mp_eq = sum(r["mine_mp_sr"] == r["eval_mp_sr"] for r in ok) / len(ok) if ok else 0.0
    passed = bool(ok) and d_auc <= GATE["max_abs_d_auc"] and d_mppe <= GATE["max_abs_d_mppe"] and \
        sp_eq >= GATE["min_sp_equal_frac"] and mp_eq >= GATE["min_mp_equal_frac"]  # fmt: skip
    after = integrity()
    diff = integrity_diff(before, after)
    worst = sorted(ok, key=lambda r: -abs(r["d_auc"]))[:5]
    out = {
        "written_at": now(), "script_sha256": sha256_file(Path(__file__).resolve()), "prereg": "3.5.2",
        "pass": passed and not diff, "thresholds": GATE, "n_runs": len(rows), "n_scored": len(ok),
        "n_problem_runs": len(rows) - len(ok), "problems": problems, "max_abs_d_auc": d_auc, "max_abs_d_mppe": d_mppe,
        "sp_equal_frac": sp_eq, "mp_equal_frac": mp_eq,
        "mean_d_auc": float(np.mean([r["d_auc"] for r in ok])) if ok else None,
        "mean_abs_d_auc": float(np.mean([abs(r["d_auc"]) for r in ok])) if ok else None,
        "max_conv_pos_residual_m": max((r["conv_pos_residual_m"] for r in ok), default=None),
        "max_conv_ang_residual_rad": max((r["conv_ang_residual_rad"] for r in ok), default=None),
        "sp_mismatch_runs": [f"{r['job']} r{r['run']}" for r in ok if r["mine_sp_sr"] != r["eval_sp_sr"]],
        "mp_mismatch_runs": [f"{r['job']} r{r['run']}" for r in ok if r["mine_mp_sr"] != r["eval_mp_sr"]],
        "worst_auc_runs": [{"job": r["job"], "run": r["run"], "d_auc": r["d_auc"], "d_mppe": r["d_mppe"]} for r in worst],
        "by_input": {kind: {"n": len(xs), "max_abs_d_auc": max(abs(r["d_auc"]) for r in xs),
                            "max_abs_d_mppe": max(abs(r["d_mppe"]) for r in xs)}
                     for kind in sorted({r["input"] for r in ok}) for xs in [[r for r in ok if r["input"] == kind]]},
        "vertex_meshes": {n: {"file": str(v[1]), "sha256": sha256_file(v[1])} for n, v in sorted(K.verts.items())},
        "integrity_kit_stock_unchanged": not diff, "integrity_diff": diff[:50],
        "note": "own-input check only (each run vs its OWN training input); no consensus reference exists or is read",
    }  # fmt: skip
    cols = ["job", "episode", "run", "try", "input", "source", "status", "problem", "rows", "rollout_rows",
            "conv_pos_residual_m", "conv_ang_residual_rad", "converted", "eval_code_recorded"] + \
        [f"{p}_{m}" for m in METRICS for p in ("eval", "mine", "d")]  # fmt: skip
    write_text(ROOT / "scores/own_input_validation.csv", csv_text(rows, cols))
    write_text(ROOT / "scores/own_input_validation.json", dump(out))
    print(json.dumps({k: out[k] for k in ("pass", "n_runs", "n_scored", "n_problem_runs", "max_abs_d_auc", "max_abs_d_mppe",
                                          "sp_equal_frac", "mp_equal_frac", "max_conv_pos_residual_m",
                                          "max_conv_ang_residual_rad", "integrity_kit_stock_unchanged")}, indent=1))  # fmt: skip
    print("by input:", json.dumps(out["by_input"]))
    print("SP mismatches:", out["sp_mismatch_runs"][:20], "MP mismatches:", out["mp_mismatch_runs"][:20])
    for p in problems[:20]:
        print("  problem:", p)
    return 0 if out["pass"] else 1


def require_pins() -> bool:
    fatal, info = check_pins(need_pins=True)
    for x in info:
        print("  " + x)
    if fatal:
        print("FATAL:\n  " + "\n  ".join(fatal))
        return False
    return True


# ================================================================================================ S2: references
def cmd_refs(a) -> int:
    if not require_pins():
        return 2
    if (ROOT / "refs/build.json").exists():
        print("FATAL: refs/build.json exists; references are built once")
        return 2
    before = integrity()
    K = load_kit()
    op = K.op
    tracks_sha = hash_list(SNAP / "TRACKS.sha256")
    build = {"built_at": now(), "script_sha256": sha256_file(Path(__file__).resolve()),
             "prereg_sha256": sha256_file(PREREG), "pins_sha256": sha256_file(PINS),
             "recipe": "PREREG 2: tracks R,F,C(,G) on the released Tier-2 visible-frame grid via the kit's "
                       "input_frame_reference; de-duplicated by numpy.array_equal in order R,F,C,G; quaternions "
                       "normalised (op._normalise_quaternions); position = numpy.median; rotation = top eigenvector of "
                       "sum q q^T (numpy.linalg.eigh, float64), sign q*.q_R >= 0 (w >= 0 on 0); nothing else",
             "tracks": {k: str(v) for k, v in TRACKS.items()}, "variants": {}}  # fmt: skip
    problems = []
    for v, ids in VARIANTS.items():
        build["variants"][v] = {"tracks": list(ids), "episodes": {}}
    for e in range(30):
        gfile = T2 / "data/chunk-000" / f"episode_{e:06d}.parquet"
        frames, names = grid(K, e)
        read = {}
        files = {}
        for tid, root in TRACKS.items():
            f = ep_file(root, e)
            arr = read_track(K, root, e, frames, names)
            if arr is None:
                continue
            h = sha256_file(f)
            if tracks_sha.get(str(f)) != h:
                problems.append(f"ep{e:02d} {tid}: {f} sha256 {h[:16]} != inputs/TRACKS.sha256 {str(tracks_sha.get(str(f)))[:16]}")
            read[tid], files[tid] = arr, {"file": str(f), "sha256": h}
        for v, ids in VARIANTS.items():
            named = [(t, read[t]) for t in ids if t in read]
            kept, dropped = dedup(named)
            used = tuple(t for t, _ in kept)
            if used != EXPECTED_USED[v][e]:
                problems.append(f"variant {v} ep{e:02d}: distinct tracks {used} != PREREG 2.5 {EXPECTED_USED[v][e]}")
            if used[0] != "R":
                problems.append(f"variant {v} ep{e:02d}: first track is not R")
            pose, gap = combine(op, [arr for _, arr in kept])
            pairs = {}
            for i in range(len(kept)):
                for jj in range(i + 1, len(kept)):
                    dp, ang = pair_distance(op, kept[i][1], kept[jj][1])
                    pairs[f"{kept[i][0]}-{kept[jj][0]}"] = {"max_pos_m": dp, "max_angle_rad": ang}
            out = ROOT / "refs" / v / f"episode_{e:06d}.npz"
            buf = io.BytesIO()
            np.savez(buf, episode_index=np.int64(e), coordinate_frame=np.array("released_track2_world"),
                     frame_index=frames.astype(np.int64), pose_xyzw=pose.reshape(len(frames), 1, 7).astype(np.float64))
            write_bytes_atomic(out, buf.getvalue())
            back = K.input_frame_reference(ROOT / "refs" / v, e, frames, 1, names)
            if not np.array_equal(back, pose):
                problems.append(f"variant {v} ep{e:02d}: kit reader does not return the written reference")
            build["variants"][v]["episodes"][f"{e:02d}"] = {
                "grid": {"file": str(gfile), "frames": int(len(frames)), "first": int(frames[0]), "last": int(frames[-1]),
                         "object_names": names},
                "available": [t for t in ids if t in read], "used": list(used), "n": len(used),
                "duplicates_dropped": dropped, "inputs": {t: files[t] for t in ids if t in read},
                "output": {"file": str(out), "sha256": sha256_file(out)}, "min_eigen_gap": gap,
                "pairwise_max_distance_report_only": pairs,
            }  # fmt: skip
    after = integrity()
    diff = integrity_diff(before, after)
    build["problems"] = problems
    build["integrity_kit_stock_unchanged"] = not diff
    build["ok"] = not problems and not diff
    write_text(ROOT / "refs/build.json", dump(build))
    for v in VARIANTS:
        ns = {}
        for e, x in build["variants"][v]["episodes"].items():
            ns.setdefault(x["n"], []).append(e)
        print(f"variant {v}: " + "; ".join(f"n={n}: {len(es)} episodes {','.join(es)}" for n, es in sorted(ns.items())))
    for p in problems + diff[:20]:
        print("  problem:", p)
    print(f"refs/build.json ok={build['ok']}")
    return 0 if build["ok"] else 1


def load_refs(K) -> dict[str, dict[int, np.ndarray]]:
    build = read_json(ROOT / "refs/build.json")
    if not build.get("ok"):
        raise SystemExit("refs/build.json is not ok")
    refs = {}
    for v in VARIANTS:
        refs[v] = {}
        for es, x in build["variants"][v]["episodes"].items():
            if sha256_file(x["output"]["file"]) != x["output"]["sha256"]:
                raise SystemExit(f"{x['output']['file']} changed since the build")
            e = int(es)
            frames, names = grid(K, e)
            refs[v][e] = K.input_frame_reference(ROOT / "refs" / v, e, frames, 1, names)
    return refs


# ================================================================================================ S4: consensus scores
SCORE_COLS = ["job", "episode", "run", "try", "variant", "input", "source", "status", "problem", "auc", "sp_sr", "mp_sr",
              "mppe", "conv_pos_residual_m", "conv_ang_residual_rad", "checkpoint_sha256", "parquet"]  # fmt: skip


def cmd_score(a) -> int:
    if not require_pins():
        return 2
    if (ROOT / "scores/consensus_scores.csv").exists():
        print("FATAL: scores/consensus_scores.csv exists; scores are computed once")
        return 2
    before = integrity()
    K = load_kit()
    refs = load_refs(K)
    jobs = load_jobs()
    evals = load_evals(jobs)
    cand = candidates(jobs, evals)
    rows, problems, grids = [], [], {}
    for job in cand:
        j = jobs[job]
        e = j["episode"]
        if e not in grids:
            grids[e] = grid(K, e)
        frames, names = grids[e]
        for k in range(1, NRUNS + 1):
            rec = evals[job][k]
            base = {"job": job, "episode": e, "run": k, "try": rec["try"], "input": j["input"], "source": j["source"],
                    "checkpoint_sha256": j["export_sha256"], "parquet": rec["parquet"]}  # fmt: skip
            try:
                conv, _, inf = rollout_on_grid(K, j, rec, frames, names)
            except ValueError as err:
                st = "conversion_failed" if "rigid transform" in str(err) else "failed"
                problems.append(f"{job} run {k}: {err}")
                for v in VARIANTS:
                    rows.append({**base, "variant": v, "status": st, "problem": str(err)})
                continue
            for v in VARIANTS:
                m = metrics_vs(K, conv, refs[v][e], names)
                rows.append({**base, "variant": v, "status": "ok", **m, "conv_pos_residual_m": inf["conv_pos_residual_m"],
                             "conv_ang_residual_rad": inf["conv_ang_residual_rad"]})  # fmt: skip
    write_text(ROOT / "scores/consensus_scores.csv", csv_text(rows, SCORE_COLS))
    jstats = {}
    for job in cand:
        jstats[job] = {"episode": jobs[job]["episode"], "input": jobs[job]["input"], "variants": list(jobs[job]["variants"])}
        for v in VARIANTS:
            rs = sorted([r for r in rows if r["job"] == job and r["variant"] == v and r["status"] == "ok"], key=lambda r: r["run"])
            d = {"n": len(rs), "runs": [r["run"] for r in rs]}
            for m in METRICS:
                xs = [r[m] for r in rs]
                d[m] = xs
                if xs:
                    d[m + "_stats"] = stats(xs)
            if rs:
                mu = d["auc_stats"]["mean"]
                d["closest_run"] = min(rs, key=lambda r: (abs(r["auc"] - mu), r["run"]))["run"]
            jstats[job][v] = d
    after = integrity()
    diff = integrity_diff(before, after)
    out = {"written_at": now(), "script_sha256": sha256_file(Path(__file__).resolve()),
           "refs_build_sha256": sha256_file(ROOT / "refs/build.json"),
           "scores_csv_sha256": sha256_file(ROOT / "scores/consensus_scores.csv"), "n_rows": len(rows),
           "n_ok_rows": sum(r["status"] == "ok" for r in rows), "problems": problems,
           "integrity_kit_stock_unchanged": not diff, "integrity_diff": diff[:50], "jobs": jstats}  # fmt: skip
    write_text(ROOT / "scores/consensus_job_stats.json", dump(out))
    print(f"scored {len(cand)} jobs: {out['n_ok_rows']}/{len(rows)} ok rows; problems {len(problems)}; "
          f"kit/stock unchanged {not diff}")
    for p in problems[:20]:
        print("  problem:", p)
    return 0 if not diff else 1


# ================================================================================================ S5: selection
def comp_summary(mapping: dict[int, str], jstats: dict, own: dict[str, dict[int, dict]]) -> dict:
    res = {}
    for label, getter in (("cons_A", lambda j, m: jstats[j]["A"][m]), ("cons_B", lambda j, m: jstats[j]["B"][m]),
                          ("own_input", lambda j, m: [own[j][k][m] for k in sorted(own[j])])):
        d = {}
        for m in METRICS:
            per = [float(np.mean(getter(mapping[e], m))) for e in sorted(mapping)]
            tk = [float(np.mean([getter(mapping[e], m)[k] for e in sorted(mapping)])) for k in range(NRUNS)]
            d[m] = {"average": float(np.mean(per)), "T_min": min(tk), "T_max": max(tk)}
        res[label] = d
    return res


def cmd_select(a) -> int:
    if SEL_SHA.exists():
        print("FATAL: SELECTION.sha256 exists; the selection is frozen")
        return 2
    if not require_pins():
        return 2
    js = read_json(ROOT / "scores/consensus_job_stats.json")
    if js.get("scores_csv_sha256") != sha256_file(ROOT / "scores/consensus_scores.csv") or \
            js.get("refs_build_sha256") != sha256_file(ROOT / "refs/build.json") or not js.get("integrity_kit_stock_unchanged"):  # fmt: skip
        print("FATAL: scores / refs changed since scoring, or integrity flagged")
        return 2
    jobs = load_jobs()
    evals = load_evals(jobs)
    cand = candidates(jobs, evals)
    jstats = js["jobs"]
    if sorted(jstats) != sorted(cand):
        print("FATAL: scored job set != candidate set")
        return 2
    own = {j: evals[j] for j in cand}
    snap_sel = read_json(SNAP / "t2_best5/selection.json")
    by_ep = {int(x["episode"]): x for x in snap_sel["episodes"]}
    out_eps, problems, table = [], [], []
    for e in range(30):
        s = by_ep[e]
        D = s["selected"]
        flags = ["PENDING"] if e == 4 else []
        if D not in jobs or jobs[D]["episode"] != e or "A" not in jobs[D]["variants"]:
            problems.append(f"ep{e:02d}: default {D} is not a variant-A job of this episode")
        if e == 4 and jobs.get(D, {}).get("input") != "raw":
            problems.append("ep04: default is not the raw final")
        ep_rec = {"episode": e, "object": s.get("object"), "default": D, "default_source": "t2_best5 SELECTED (snapshot)",
                  "t2best5_flags": s.get("flags", []), "flags": flags, "variants": {}}  # fmt: skip
        d_own = own.get(D, {})
        for v in ("A", "B"):
            pool = [j for j in cand if jobs[j]["episode"] == e and v in jobs[j]["variants"]]
            chal = [j for j in pool if j != D]
            cons = {j: {"n": jstats[j][v]["n"], "auc": jstats[j][v]["auc"], "mppe": jstats[j][v]["mppe"]} for j in pool}
            pick, reason, fl, rows = select_episode(D, chal, cons, own)
            dst = jstats.get(D, {}).get(v, {})
            ep_rec["variants"][v] = {
                "default": {"job": D, "cons_auc": dst.get("auc"), **({"cons_auc_mean": dst["auc_stats"]["mean"],
                            "cons_auc_var": dst["auc_stats"]["var"], "cons_mppe_mean": dst["mppe_stats"]["mean"],
                            "cons_sp_mean": dst["sp_sr_stats"]["mean"], "cons_mp_mean": dst["mp_sr_stats"]["mean"]}
                            if dst.get("n") else {}), "floor_count": floor_count(d_own), "n_own_ok": len(d_own)},
                "challengers": [{**r, "input": jobs[r["job"]]["input"], "family": jobs[r["job"]]["family"],
                                 "provisional": jobs[r["job"]]["provisional"]} for r in rows],
                "pick": pick, "pick_input": jobs[pick]["input"], "reason": reason, "flags": fl,
                "pick_closest_run_A": jstats[pick]["A"].get("closest_run"),
            }  # fmt: skip
            if v == "A" and len(chal) != A_CHALLENGERS[e]:
                problems.append(f"ep{e:02d}: {len(chal)} variant-A challengers != PREREG 4.1 table {A_CHALLENGERS[e]}")
        A, B = ep_rec["variants"]["A"], ep_rec["variants"]["B"]
        ep_rec["pick_A"], ep_rec["pick_B"] = A["pick"], B["pick"]
        ep_rec["changed_A"], ep_rec["changed_B"] = A["pick"] != D, B["pick"] != D
        ep_rec["closest_run_A"] = A["pick_closest_run_A"]
        out_eps.append(ep_rec)

        def best_ch(V):
            ok = [c for c in V["challengers"] if "cons_auc_mean" in c]
            return max(ok, key=lambda c: c["cons_auc_mean"]) if ok else None

        bA, bB = best_ch(A), best_ch(B)
        own_mean = {m: float(np.mean([d_own[k][m] for k in sorted(d_own)])) for m in METRICS} if d_own else {}
        pa = jstats[A["pick"]]
        table.append({
            "episode": f"{e:02d}", "object": s.get("object"), "default": D, "default_input": jobs[D]["input"],
            "D_consA_auc": A["default"].get("cons_auc_mean"), "D_consB_auc": B["default"].get("cons_auc_mean"),
            "D_own_auc": own_mean.get("auc"), "D_own_sp": own_mean.get("sp_sr"), "D_own_mp": own_mean.get("mp_sr"),
            "D_floor": f"{A['default']['floor_count']}/{NRUNS}", "n_chal_A": len(A["challengers"]),
            "best_chal_A": bA["job"] if bA else "", "best_chal_A_consA_auc": bA["cons_auc_mean"] if bA else None,
            "best_chal_A_floor": f"{bA['floor_count']}/{NRUNS}" if bA else "",
            "pick_A": A["pick"], "pick_A_input": A["pick_input"], "pick_A_consA_auc": pa["A"]["auc_stats"]["mean"],
            "pick_A_consA_sd": math.sqrt(pa["A"]["auc_stats"]["var"]),
            "pick_A_own_auc": float(np.mean([own[A['pick']][k]["auc"] for k in own[A['pick']]])),
            "pick_A_floor": f"{floor_count(own[A['pick']])}/{NRUNS}", "closest_run_A": A["pick_closest_run_A"],
            "reason_A": A["reason"], "n_chal_B": len(B["challengers"]), "best_chal_B": bB["job"] if bB else "",
            "pick_B": B["pick"], "pick_B_consB_auc": jstats[B["pick"]]["B"]["auc_stats"]["mean"], "reason_B": B["reason"],
            "flags": ";".join(flags + A["flags"]),
        })  # fmt: skip
    if problems:
        print("FATAL (consistency with PREREG):\n  " + "\n  ".join(problems))
        return 2
    comps = {"DEFAULT": {e["episode"]: e["default"] for e in out_eps},
             "CONS-A": {e["episode"]: e["pick_A"] for e in out_eps},
             "CONS-B": {e["episode"]: e["pick_B"] for e in out_eps},
             "RAW-ONLY": {int(x["episode"]): x["rawref"] for x in snap_sel["episodes"]},
             "K3FULL": {int(x["episode"]): x["chosen"] for x in snap_sel["episodes"]}}  # fmt: skip
    summ = {name: {"jobs": {f"{e:02d}": j for e, j in sorted(m.items())}, **comp_summary(m, jstats, own)}
            for name, m in comps.items()}  # fmt: skip
    sel = {
        "generated_at": now(), "prereg_sha256": sha256_file(PREREG), "pins_sha256": sha256_file(PINS),
        "script_sha256": sha256_file(Path(__file__).resolve()), "refs_build_sha256": sha256_file(ROOT / "refs/build.json"),
        "consensus_scores_sha256": sha256_file(ROOT / "scores/consensus_scores.csv"),
        "snapshot_selection_sha256": sha256_file(SNAP / "t2_best5/selection.json"),
        "rule": "PREREG 5: default D = t2_best5 SELECTED (snapshot); a challenger qualifies iff cons_mean_AUC >= "
                "cons_mean_AUC(D) + 0.01 (exact floats) AND own-input SP-SR = MP-SR = 1 in >= 4 of 5 runs; pick the "
                "qualifying one with the highest cons_mean_AUC (ties: lower population variance, lower cons_mean MPPE, "
                "job name); variant A binding (no G2 job, reference R/F/C), variant B report only (adds G2)",
        "binding_variant": "A", "noisy_only": True,
        "changed_episodes_A": [f"{e['episode']:02d}" for e in out_eps if e["changed_A"]],
        "changed_episodes_B": [f"{e['episode']:02d}" for e in out_eps if e["changed_B"]],
        "cons_A_closest_runs": {f"{e['episode']:02d}": {"job": e["pick_A"], "run": e["closest_run_A"]} for e in out_eps},
        "compositions": {k: {f"{e:02d}": j for e, j in sorted(m.items())} for k, m in comps.items()},
        "episodes": out_eps,
    }  # fmt: skip
    write_text(ROOT / "selection.json", dump(sel))
    write_text(ROOT / "selection_consensus.json", dump(sel))
    tcols = list(table[0].keys())
    write_text(ROOT / "selection_table.csv", csv_text(table, tcols))
    # text + markdown
    lines = [f"selection {sel['generated_at']} (noisy-only consensus; PREREG {sel['prereg_sha256'][:12]}; "
             f"refs {sel['refs_build_sha256'][:12]}; scores {sel['consensus_scores_sha256'][:12]})"]  # fmt: skip
    for x in out_eps:
        for v in ("A", "B"):
            V = x["variants"][v]
            dd = V["default"]
            lines.append(f"ep{x['episode']:02d} [{v}] D {x['default']} consAUC {f4(dd.get('cons_auc_mean'))} floor "
                         f"{dd['floor_count']}/5 -> {V['pick']} ({V['reason']})" + (f" flags {x['flags']}" if x["flags"] else ""))  # fmt: skip
            for c in V["challengers"]:
                lines.append(f"      {c['job']} [{c['input']}] consAUC {f4(c.get('cons_auc_mean'))} sd "
                             f"{f4(math.sqrt(c['cons_auc_var'])) if 'cons_auc_var' in c else ''} floor {c['floor_count']}/5 "
                             f"qualifies={c['qualifies']} {c['why_not']}")  # fmt: skip
    write_text(ROOT / "selection.txt", "\n".join(lines) + "\n")
    md = ["# t2_consensus_20261024: selection (noisy-only; PREREG section 5)", "",
          f"Generated {sel['generated_at']} CST. Binding: variant A (reference R/F/C, no G2 job). Variant B (adds G2) is "
          "report only. No clean or LOCAL file was opened.", "",
          f"Changed vs DEFAULT under A: {', '.join(sel['changed_episodes_A']) or 'none'}. "
          f"Under B: {', '.join(sel['changed_episodes_B']) or 'none'}.", "",
          "| ep | object | default (D) | D cons-A AUC | D own SP/MP | D floor | A chal. | best A challenger | its cons-A AUC | "
          "its floor | pick A | pick cons-A AUC (sd) | CLOSEST run | reason A | pick B |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for t in table:
        md.append(f"| {t['episode']} | {t['object']} | {t['default']} | {f4(t['D_consA_auc'])} | {t['D_own_sp']:.1f}/{t['D_own_mp']:.1f} "
                  f"| {t['D_floor']} | {t['n_chal_A']} | {t['best_chal_A']} | {f4(t['best_chal_A_consA_auc'])} | "
                  f"{t['best_chal_A_floor']} | **{t['pick_A']}** | {f4(t['pick_A_consA_auc'])} ({f4(t['pick_A_consA_sd'])}) | "
                  f"{t['closest_run_A']} | {t['reason_A']}{' [' + t['flags'] + ']' if t['flags'] else ''} | {t['pick_B']} |")  # fmt: skip
    write_text(ROOT / "selection_table.md", "\n".join(md) + "\n")
    smd = ["# t2_consensus_20261024: composition summary (noisy-only)", "",
           "Averages over the 30 episodes of each episode's 5-run mean. cons_A / cons_B = against the consensus "
           "references; own_input = the evaluator's own-input values. Consensus numbers are not leaderboard estimates "
           "and favour the selected composition by construction (PREREG 9.6).", "",
           "| composition | cons-A AUC | cons-A SP | cons-A MP | cons-A MPPE | cons-B AUC | own AUC | own SP | own MP | own MPPE |",
           "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for name, d in summ.items():
        smd.append(f"| {name} | {f4(d['cons_A']['auc']['average'])} | {f4(d['cons_A']['sp_sr']['average'])} | "
                   f"{f4(d['cons_A']['mp_sr']['average'])} | {f4(d['cons_A']['mppe']['average'])} | "
                   f"{f4(d['cons_B']['auc']['average'])} | {f4(d['own_input']['auc']['average'])} | "
                   f"{f4(d['own_input']['sp_sr']['average'])} | {f4(d['own_input']['mp_sr']['average'])} | "
                   f"{f4(d['own_input']['mppe']['average'])} |")  # fmt: skip
    write_text(ROOT / "summary_consensus.json", dump({"generated_at": sel["generated_at"], "compositions": summ}))
    write_text(ROOT / "summary_consensus.md", "\n".join(smd) + "\n")
    frozen = ["selection.json", "selection_consensus.json", "selection.txt", "selection_table.csv", "selection_table.md",
              "summary_consensus.json", "summary_consensus.md", "scores/consensus_scores.csv",
              "scores/consensus_job_stats.json", "refs/build.json", "scores/own_input_validation.json",
              "selftest/selftest.json", "PREREG_PINS.md", "PREREG.md"]  # fmt: skip
    body = "".join(f"{sha256_file(ROOT / f)}  {f}\n" for f in frozen)
    write_text(SEL_SHA, body)
    for f in frozen[:7] + ["SELECTION.sha256"]:
        os.chmod(ROOT / f, 0o444)
    print("\n".join(smd))
    print()
    print("\n".join(md[4:]))
    print(f"\nSELECTION.sha256 written {now()} (frozen): {sha256_file(SEL_SHA)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("selftest", "validate", "refs", "score", "select", "pins"))
    a = ap.parse_args()
    if not sys.flags.dont_write_bytecode or not sys.flags.isolated:
        print("run with the kit python -I -B")
        return 2
    if a.cmd == "pins":
        fatal, info = check_pins(need_pins=PINS.exists())
        print("\n".join(info + ["FATAL: " + x for x in fatal]))
        return 1 if fatal else 0
    return {"selftest": cmd_selftest, "validate": cmd_validate, "refs": cmd_refs, "score": cmd_score,
            "select": cmd_select}[a.cmd](a)  # fmt: skip


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""k3only_pack.py: step 3 of fc_runs/submissions/t2_k3only_20261102 (Tier 2, K3-only file).

Packs the trial chosen in CHOICE.json (own-input rule, hashed in CHOICE.sha256 before this script ran) from the EXISTING
official rollouts of fc_runs/submissions/t2_best5_20261024 (no evaluation, no export), with the kit packer UNCHANGED and
the d769f0c commit URL, then runs the kit uploader's offline validate_submission (no Kaggle API).

prepare  pack/<name>/results/   byte copies of each episode's chosen run (its ok try): episode parquet + json; checks:
                                run.json ok + try, results JSON checkpoint_sha256 == the job's export sha256 (export.json,
                                file present and re-hashed), JSON own-input metrics == own_input_evals.csv / CHOICE.json,
                                start_frame 0 / world_count 1 / explicit reset / no non-finite world, parquet provenance
                                (episode, checkpoint digest, evaluator digests 160d24f5 / b1b3ebe3)
         pack/<name>/frame_refs/ each episode's OWN policy input on the released grid: K3fix -> byte copy of
                                tmp_analysis/tier2/k3fix_pilot/frame_refs_k3fix npz, K3 -> tmp_analysis/tier2/kalman/eval/
                                frame_refs_k3 npz (sha256 == round-1 inputs/TRACKS.sha256), raw -> npz written from the
                                released Tier-2 parquet (read_episode), checked equal to the packer default on the grid
pack     kit tools/pack_submission.py --track track_2_tier2 (unchanged, run directly) with the d769f0c URL; the packer's
         "locally computed:" line (clean) is withheld from pack.log and kept only in pack_stdout_LOCAL_NOT_FOR_UPLOAD.log;
         kit hashed before/after; results / frame_refs hashed before/after
validate uploader validate_submission (track_2_tier2) + provenance: rows == sample, uniform provenance columns with the
         expected evaluator / packer / uploader digests and commit URL, one checkpoint digest per episode == chosen export
New files only, under this folder. CPU only. Nothing uploaded, nothing signalled.
Run: conda_envs/v2dkit/bin/python -I -B tools/k3only_pack.py prepare|pack|validate
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
W = ME / "fc_runs/submissions/t2_k3only_20261102"
B5 = ME / "fc_runs/submissions/t2_best5_20261024"
KIT = ME / "v2d_submission_kit_src/v2d_submission_kit"
KPY = ME / "conda_envs/v2dkit/bin/python"
DATA = ME / "data/video_to_data_challenge"
T2 = DATA / "track_2/tier_2_synthetic_noise"
STOCK = ME / "code/video_to_data_8c778f9"
FR = {"k3fix": ME / "tmp_analysis/tier2/k3fix_pilot/frame_refs_k3fix", "k3": ME / "tmp_analysis/tier2/kalman/eval/frame_refs_k3"}
TRACKS = ME / "fc_runs/submissions/t2_consensus_20261024/inputs/TRACKS.sha256"  # read only (round-1 frozen hashes)
COMMIT = "d769f0c257c12314c5f53123b8e81de010778c7f"
URL = f"https://github.com/Beat-in-our-hearts/nv-v2d-challenge-codebase/commit/{COMMIT}"
EVAL_SCRIPT_SHA = "160d24f5fbeb70ccd569ca4192056f1892b2a9f2999066db9d8279461648fdf1"
EVAL_CODE_SHA = "b1b3ebe3ae4629d1687a83f420149e3f6fdfc154d70acad55c12fc8fa69811a1"
KIT_PINS = {
    "tools/pack_submission.py": "47a4589574ff8284ba4376ef31ac5548c8fe87ead753be2d3b43d6c87041e5be",
    "submit/v2d_submit.py": "fe5cb1dc86d12abd01eab9dd5dd98b51569b5d8513a2d00d4e79da42a5f11718",
    "metric_code/track_2/AUC.py": "58b437697c32c86497ff5a306fe586014c3429b0d2c18e1d3094a6f503479339",
    "metric_code/track_2/SP-SR.py": "d206042872acd1c1de6120f96966272495fbbd71a50315f49f5192fea1aaa5bf",
    "metric_code/track_2/MP-SR.py": "a1d72216eb7745a13ed7db6a55b9efb31492b35149113181a6c42ffc231fff11",
    "metric_code/track_2/MPPE.py": "41815bc1e8870fb377036750976dcf135818e029f06096ccef5177090fe5e033",
    "flash_chord/scripts/evaluate_policy.py": EVAL_SCRIPT_SHA,
}
METRIC_KEYS = {"auc": "add_auc", "sp_sr": "spider_sr_uncentered", "mp_sr": "maniptrans_sr", "mppe": "mean_per_point_error_cm"}
WITHHELD = "locally computed: [withheld by k3only_pack.py: clean-reference line, see pack_stdout_LOCAL_NOT_FOR_UPLOAD.log]"
TZ8 = dt.timezone(dt.timedelta(hours=8))


def now() -> str:
    return dt.datetime.now(TZ8).strftime("%Y-%m-%d %H:%M:%S")


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rj(p):
    return json.loads(Path(p).read_text())


def dump(o) -> str:
    def d(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (Path, np.ndarray)):
            return str(x) if isinstance(x, Path) else x.tolist()
        raise TypeError(type(x))
    return json.dumps(o, indent=1, default=d) + "\n"


def write_new(p: Path, text: str) -> None:
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, p)


def copy_new(src: Path, dst: Path, want: str | None = None) -> str:
    h = sha(src)
    if want is not None and h != want:
        raise SystemExit(f"{src} sha256 {h[:16]} != expected {want[:16]}")
    if dst.exists():
        raise SystemExit(f"refusing to overwrite {dst}")
    shutil.copyfile(src, dst)
    if sha(dst) != h:
        raise SystemExit(f"copy of {src} differs")
    return h


def digest_dir(d: Path) -> dict:
    out = {}
    for dp, dn, fn in os.walk(d):
        dn[:] = [x for x in dn if x != ".git"]
        for n in fn:
            p = Path(dp) / n
            out[str(p.relative_to(d))] = sha(p) if p.is_file() else "special"
    return out


def listing(d: Path) -> dict:
    out = {}
    for dp, dn, fn in os.walk(d):
        dn[:] = [x for x in dn if x != ".git"]
        for n in fn:
            p = Path(dp) / n
            try:
                st = p.lstat()
                out[str(p.relative_to(d))] = (st.st_size, st.st_mtime_ns)
            except OSError:
                out[str(p.relative_to(d))] = None
    return out


def gate() -> tuple[dict, str, int]:
    """CHOICE.sha256 entries hold; kit pins and KIT_MANIFEST hold."""
    for line in (W / "CHOICE.sha256").read_text().splitlines():
        h0, rel = line.split(None, 1)
        if sha(W / rel.strip()) != h0:
            raise SystemExit(f"CHOICE.sha256: {rel} changed")
    choice = rj(W / "CHOICE.json")
    for rel, want in KIT_PINS.items():
        if sha(KIT / rel) != want:
            raise SystemExit(f"kit {rel} != pin")
    man = rj(KIT / "KIT_MANIFEST.json")["files"]
    bad = [p for p, h in man.items() if not (KIT / p).is_file() or sha(KIT / p) != h]
    if bad:
        raise SystemExit(f"KIT_MANIFEST mismatches {bad[:5]}")
    k = int(choice["chosen_trial"])
    return choice, f"k3only_closest_trial{k:02d}", k


def load_kit():
    if str(KIT) not in sys.path:
        sys.path.insert(0, str(KIT))
    from v2dlb.submission_parquet import provenance, read_trajectory
    from v2dlb.trajectory import read_episode
    spec = importlib.util.spec_from_file_location("kit_pack_submission", KIT / "tools/pack_submission.py")
    pk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pk)
    return read_episode, read_trajectory, provenance, pk


def hash_list(p: Path) -> dict:
    out = {}
    for line in p.read_text().splitlines():
        if line.strip():
            h, path = line.split(None, 1)
            out[path.strip()] = h
    return out


def cmd_prepare() -> int:
    choice, name, k = gate()
    read_episode, read_trajectory, provenance, pk = load_kit()
    d = W / "pack" / name
    if d.exists():
        print(f"{d} exists; skipped")
        return 0
    tmpd = W / "pack" / f".{name}.preparing"
    if tmpd.exists():
        raise SystemExit(f"stale {tmpd}")
    (tmpd / "results").mkdir(parents=True)
    (tmpd / "frame_refs").mkdir()
    tracks = hash_list(TRACKS)
    rows, problems = [], []
    for e in range(30):
        ep = choice["episodes"][f"{e:02d}"]
        job, kind = ep["job"], ep["input"]
        r = ep["runs"][str(k)]
        t = int(r["try"])
        ex = rj(B5 / "exports" / job / "export.json")
        exf = Path(ex["file"])
        exsha = sha(exf) if exf.is_file() else "missing"
        if exsha != ex["sha256"] or not ex["sha256"].startswith(r["export_sha256_16"]):
            problems.append(f"ep{e:02d} {job}: export file {exf} sha256 {exsha[:16]} != export.json {ex['sha256'][:16]}")
        run = rj(B5 / "runs" / job / f"run{k}" / "run.json")
        src = B5 / "runs" / job / f"run{k}" / f"try{t}" / "results"
        pq, js = src / f"episode_{e:06d}.parquet", src / f"episode_{e:06d}.json"
        if run["status"] != "ok" or int(run["try"]) != t or Path(run["results_parquet"]) != pq or run["export_sha256"] != ex["sha256"]:
            problems.append(f"ep{e:02d} {job}: run.json status/try/parquet/export mismatch")
        res = rj(js)
        mt = res["metrics"]
        for m, key in METRIC_KEYS.items():
            tol = 1e-9 * max(1.0, abs(float(r[m])))
            if abs(float(mt[key]) - float(r[m])) > tol or abs(float(run["metrics"][m]) - float(r[m])) > tol:
                problems.append(f"ep{e:02d} {job}: {m} results JSON {mt[key]} / run.json {run['metrics'][m]} != CHOICE {r[m]}")
        if res["checkpoint_sha256"] != ex["sha256"] or res["start_frame"] != 0 or res["world_count"] != 1 or \
                res["reset_mode"] != "explicit" or res["non_finite_worlds"] or res["parquet"] != run["training_parquet"]:  # fmt: skip
            problems.append(f"ep{e:02d} {job}: results JSON setup/checkpoint mismatch")
        traj = read_trajectory(pq)
        prov = provenance(traj)
        if traj.episode_index != e or prov.get("checkpoint_sha256") != ex["sha256"] or \
                prov.get("eval_script_sha256") != EVAL_SCRIPT_SHA or prov.get("eval_code_sha256", EVAL_CODE_SHA) != EVAL_CODE_SHA:  # fmt: skip
            problems.append(f"ep{e:02d} {job}: parquet provenance mismatch {prov}")
        s = {"episode": e, "job": job, "input": kind, "trial": k, "run": k, "try": t, "export_file": str(exf),
             "export_sha256": ex["sha256"], "training_checkpoint_sha256": ex["source_sha256"],
             "rollout_src": str(pq), "json_src": str(js),
             "own_input": {m: float(r[m]) for m in METRIC_KEYS}}
        s["rollout_sha256"] = copy_new(pq, tmpd / "results" / pq.name)
        s["json_sha256"] = copy_new(js, tmpd / "results" / js.name)
        npz = tmpd / "frame_refs" / f"episode_{e:06d}.npz"
        rel = read_episode(T2 / "data/chunk-000" / f"episode_{e:06d}.parquet", e)
        frames = np.asarray(rel.frame_index, dtype=np.int64)
        names = list(rel.object_names)
        if kind == "raw":
            np.savez(npz, episode_index=np.int64(e), coordinate_frame=np.array("released_track2_world"),
                     frame_index=frames, pose_xyzw=np.asarray(rel.pose_xyzw, dtype=np.float64))
            s["frame_ref_kind"] = "written_from_released_tier2"
            s["frame_ref_src"] = str(T2 / "data/chunk-000" / f"episode_{e:06d}.parquet")
            mine = pk.input_frame_reference(tmpd / "frame_refs", e, frames, 1, names)
            dflt = pk.input_frame_reference(T2, e, frames, 1, names)
            s["frame_ref_equals_packer_default"] = bool(np.array_equal(mine, dflt))
            if not s["frame_ref_equals_packer_default"]:
                problems.append(f"ep{e:02d}: raw frame reference != packer default")
        else:
            srcnpz = FR[kind] / f"episode_{e:06d}.npz"
            want = tracks.get(str(srcnpz))
            s["frame_ref_kind"] = "byte_copy"
            s["frame_ref_src"] = str(srcnpz)
            s["frame_ref_src_sha256_frozen"] = want
            copy_new(srcnpz, npz, want)
            mine = pk.input_frame_reference(tmpd / "frame_refs", e, frames, 1, names)
            own = pk.input_frame_reference(FR[kind], e, frames, 1, names)
            s["frame_ref_equals_own_input_on_grid"] = bool(np.array_equal(mine, own))
            if not s["frame_ref_equals_own_input_on_grid"]:
                problems.append(f"ep{e:02d}: frame reference differs from own input on grid")
        s["frame_ref_sha256"] = sha(npz)
        s["grid_frames"] = int(len(frames))
        rows.append(s)
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        return 1
    meta = {"file": name, "kind": "K3-only (K3FULL) closest-to-mean trial", "created_at": now(),
            "script_sha256": sha(Path(__file__).resolve()), "choice_json_sha256": sha(W / "CHOICE.json"),
            "choice_sha256_file_sha256": sha(W / "CHOICE.sha256"), "chosen_trial": k, "episodes": rows,
            "inputs": {x: sum(r["input"] == x for r in rows) for x in sorted({r["input"] for r in rows})}}
    write_new(tmpd / "file.json", dump(meta))
    os.rename(tmpd, d)
    print(f"prepared {d}: 30 episodes, trial {k}, inputs {meta['inputs']}")
    return 0


PACK_ENV = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ.get("HOME", "/root"), "LANG": "C.UTF-8",
            "TZ": "Asia/Shanghai", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2", "CUDA_VISIBLE_DEVICES": ""}


def out_name(k: int) -> str:
    return f"T2_K3only_closest_trial{k:02d}_{COMMIT[:7]}.csv"


def cmd_pack() -> int:
    choice, name, k = gate()
    d = W / "pack" / name
    meta = rj(d / "file.json")
    sub = W / "submission"
    out_csv = sub / out_name(k)
    if out_csv.exists():
        print(f"{out_csv} exists; skipped")
        return 0
    sub.mkdir(parents=True, exist_ok=True)
    kit_before, res_before, fr_before = digest_dir(KIT), digest_dir(d / "results"), digest_dir(d / "frame_refs")
    stock_before = listing(STOCK)
    cmd = ["nice", "-n", "10", str(KPY), "-I", "-B", str(KIT / "tools/pack_submission.py"), "--track", "track_2_tier2",
           "--results", str(d / "results"), "--dataset-root", str(DATA), "--frame-references", str(d / "frame_refs"),
           "--code-commit", URL, "--eval-script", str(KIT / "flash_chord/scripts/evaluate_policy.py"), "-o", str(out_csv)]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(d), env=PACK_ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    raw = "+ " + " ".join(cmd) + "\n" + p.stdout
    write_new(sub / "pack_stdout_LOCAL_NOT_FOR_UPLOAD.log", raw)
    kept = [WITHHELD if ln.startswith("locally computed:") else ln for ln in raw.splitlines()]
    write_new(sub / "pack.log", "\n".join(kept) + "\n")
    kit_after = digest_dir(KIT)
    rec = {"at": now(), "exit": p.returncode, "seconds": round(time.time() - t0, 1), "command": cmd, "commit_url": URL,
           "csv": str(out_csv), "csv_exists": out_csv.is_file(), "csv_sha256": sha(out_csv) if out_csv.is_file() else None,
           "kit_files": len(kit_after), "kit_unchanged": kit_before == kit_after,
           "results_unchanged": digest_dir(d / "results") == res_before,
           "frame_refs_unchanged": digest_dir(d / "frame_refs") == fr_before,
           "stock_runtime_unchanged_listing": listing(STOCK) == stock_before,
           "input_reference_line": next((ln for ln in kept if ln.startswith("input reference check")), None),
           "frame_reference_line": next((ln for ln in kept if ln.startswith("frame conversion reference")), None),
           "file_json_sha256": sha(d / "file.json"), "chosen_trial": k, "n_episodes": len(meta["episodes"])}
    rec["ok"] = bool(p.returncode == 0 and rec["csv_exists"] and rec["kit_unchanged"] and rec["results_unchanged"]
                     and rec["frame_refs_unchanged"] and rec["stock_runtime_unchanged_listing"])
    write_new(sub / "pack_report.json", dump(rec))
    print(json.dumps({x: rec[x] for x in ("exit", "seconds", "csv_sha256", "kit_unchanged", "results_unchanged",
                                         "frame_refs_unchanged", "stock_runtime_unchanged_listing",
                                         "input_reference_line", "ok")}, indent=1))
    return 0 if rec["ok"] else 1


def cmd_validate() -> int:
    choice, name, k = gate()
    d = W / "pack" / name
    meta = rj(d / "file.json")
    sub = W / "submission"
    csvp = sub / out_name(k)
    outp = sub / "validate_report.json"
    if outp.exists():
        print(f"{outp} exists; skipped")
        return 0
    load_kit()
    spec = importlib.util.spec_from_file_location("kit_v2d_submit", KIT / "submit/v2d_submit.py")
    up = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(up)
    from v2dlb import object_pose as op
    from v2dlb import report_schema as rs
    import pandas as pd

    config = rj(KIT / "config/tracks.json")
    man = rj(KIT / "KIT_MANIFEST.json")["files"]
    n_sample = len(pd.read_csv(KIT / "data/track_2_sample_submission.csv", usecols=[rs.ROW_ID_COLUMN], dtype=str))
    res = {"at": now(), "csv": str(csvp), "csv_sha256": sha(csvp), "problems": []}
    try:
        frame = up.validate_submission(csvp, "track_2_tier2", config)
        res["uploader_validate_submission"] = "pass"
    except SystemExit as err:
        res["uploader_validate_submission"] = f"FAIL: {err}"
        res["problems"].append("uploader validate_submission failed")
        frame = pd.read_csv(csvp)
    res["rows"], res["sample_rows"] = int(len(frame)), n_sample
    if res["rows"] != n_sample:
        res["problems"].append(f"{res['rows']} rows != sample {n_sample}")
    uniform = {c: sorted(frame[c].astype(str).str.strip().unique()) for c in rs.UNIFORM_PROVENANCE_COLUMNS}
    res["uniform_columns"] = uniform
    for c, v in uniform.items():
        if len(v) != 1:
            res["problems"].append(f"{c} not uniform")
    want = {"eval_script_sha256": EVAL_SCRIPT_SHA, "eval_code_sha256": EVAL_CODE_SHA,
            "packer_script_sha256": man["tools/pack_submission.py"], "uploader_script_sha256": man["submit/v2d_submit.py"],
            "code_commit_url": URL}
    for c, v in want.items():
        if uniform.get(c, [None])[0] != v:
            res["problems"].append(f"{c} {uniform.get(c)} != {v}")
    eps = frame["row_id"].astype(str).str.split("/").str[1]
    per_ep = frame.groupby(eps)["checkpoint_sha256"].agg(lambda s: sorted(s.astype(str).str.strip().unique()))
    chosen = {f"e{r['episode']:06d}": r["export_sha256"] for r in meta["episodes"]}
    if sorted(per_ep.index) != sorted(chosen):
        res["problems"].append("episode set differs from the plan")
    for ep, vals in per_ep.items():
        if len(vals) != 1 or vals[0] != chosen.get(ep):
            res["problems"].append(f"{ep}: checkpoint digest {vals} != chosen export")
    res["distinct_checkpoints"] = int(len({v[0] for v in per_ep}))
    npose = len(op._POSE_COLUMNS)
    h = hashlib.sha256()
    with csvp.open() as fp:
        header = fp.readline().rstrip("\n").split(",")
        h.update((",".join(header[: 1 + npose]) + "\n").encode())
        for line in fp:
            h.update((",".join(line.rstrip("\n").split(",")[: 1 + npose]) + "\n").encode())
    res["pose_columns_sha256"] = h.hexdigest()
    res["pose_header_ok"] = header[: 1 + npose] == [rs.ROW_ID_COLUMN, *op._POSE_COLUMNS]
    if not res["pose_header_ok"]:
        res["problems"].append("unexpected header")
    res["ok"] = not res["problems"]
    write_new(outp, dump(res))
    print(f"uploader validate_submission {res['uploader_validate_submission']}; rows {res['rows']:,} (sample {n_sample:,}); "
          f"distinct checkpoints {res['distinct_checkpoints']}; csv sha256 {res['csv_sha256']}; pose sha256 "
          f"{res['pose_columns_sha256']}; commit_url {uniform.get('code_commit_url')}; problems {res['problems'] or 'none'}")
    return 0 if res["ok"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prepare", "pack", "validate"))
    a = ap.parse_args()
    if not sys.flags.dont_write_bytecode or not sys.flags.isolated:
        print("run with the kit python -I -B")
        return 2
    return {"prepare": cmd_prepare, "pack": cmd_pack, "validate": cmd_validate}[a.cmd]()


if __name__ == "__main__":
    raise SystemExit(main())

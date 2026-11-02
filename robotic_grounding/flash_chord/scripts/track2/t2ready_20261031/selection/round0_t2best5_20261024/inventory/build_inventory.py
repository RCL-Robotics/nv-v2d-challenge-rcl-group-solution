"""Tier-2 candidate inventory (ours + Zuxing's), READ-ONLY over every run / data folder.

Writes only into fc_runs/submissions/t2_best5_20261024/inventory/:
  inventory.json                                  own-input + provenance (no clean numbers)
  inventory_own_input.csv                         one row per candidate checkpoint, own-input only
  inventory_clean_LOCAL_NOT_FOR_UPLOAD.csv        clean (leaderboard-style, vs Tier-1 reference) numbers, REPORT ONLY
Decides nothing. Own-input AUCs are comparable only between rows with the same parquet sha256.
Zuxing's folder is read with plain file reads only (no git); nothing is copied out of it.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path

import yaml

ME = Path("/root/autodl-tmp/haoran.zhang")
Z = Path("/root/autodl-tmp/zuxing.lu")
OUT = ME / "fc_runs/submissions/t2_best5_20261024/inventory"
T2 = ME / "tmp_analysis/tier2"
ROOTS = {
    "raw": ME / "data/human_motion_data_tier2",
    "k3fix": ME / "data/human_motion_data_tier2_k3fix",
    "k3": ME / "data/human_motion_data_tier2_k3",
    "g2": Z / "data/human_motion_data_tier2_g2-2026-10-24",
}
OURS_CODE = "code/fc_r2 @ 61fa5d31ff80 (branch wp0-rsiv3-r2, worktree clean)"
ZX_CODE = "zuxing worktrees/dev.zuxing.speedup @ bc3b1bd (logged at launch; branch tip bc3b1bdf34f2 now; dirty state not checked: no git in his folder)"

manifest = json.loads((ROOTS["raw"] / "whole_body/soma/track2_manifest.json").read_text())
EP = {int(v["episode_index"]): v for v in manifest.values()}

_sha: dict[str, str] = {}


def sha(p: Path | str | None) -> str | None:
    if p is None:
        return None
    p = str(p)
    if p not in _sha:
        if not os.path.isfile(p):
            return None
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
        _sha[p] = h.hexdigest()
    return _sha[p]


def pq(kind: str, ep: int) -> Path:
    return ROOTS[kind] / f"whole_body/soma/sequence_id={EP[ep]['sequence_id']}/robot_name=g1/data.parquet"


def fnum(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def jload(p: Path) -> dict:
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return {}


def rows(p: Path) -> list[dict]:
    try:
        with open(p) as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def models(run: Path) -> list[int]:
    out = []
    try:
        for n in os.listdir(run):
            m = re.fullmatch(r"model_(\d+)\.safetensors", n)
            if m:
                out.append(int(m.group(1)))
    except OSError:
        pass
    return sorted(out)


def live_trainers() -> dict[str, int]:
    """output_dir -> pid for every running train_rl.py."""
    out = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            cmd = open(f"/proc/{d}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        cmd = [c.decode(errors="replace") for c in cmd]
        if any(c.endswith("train_rl.py") for c in cmd):
            for c in cmd:
                if c.startswith("output_dir="):
                    out[os.path.realpath(c.split("=", 1)[1])] = int(d)
    return out


LIVE = live_trainers()


def is_live(run: Path) -> int | None:
    return LIVE.get(os.path.realpath(run))


def find_key(d, key):
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def hydra_info(run: Path) -> dict:
    cfg = {}
    for p in (run / "hydra/.hydra/config.yaml",):
        if p.is_file():
            cfg = yaml.safe_load(p.read_text()) or {}
    info = {}
    if cfg:
        info["parquet"] = (cfg.get("task") or {}).get("parquet")
        info["support"] = (cfg.get("scene") or {}).get("support")
        info["seed"] = (cfg.get("reset") or {}).get("seed")
        tr = cfg.get("training") or {}
        info["max_iterations"] = tr.get("max_iterations")
        info["early_stop_help_zero_tail"] = tr.get("early_stop_help_zero_tail")
        info["resume_checkpoint"] = tr.get("checkpoint") if tr.get("resume") else None
        info["deadline"] = find_key(cfg.get("reset") or {}, "deadline")
        info["zero_ramp"] = find_key(cfg.get("reset") or {}, "zero_ramp")
        ob = cfg.get("objective") or {}
        if "object_task_rotation_beta" in ob:
            info["object_task_rotation_beta"] = ob.get("object_task_rotation_beta")
    ov = run / "hydra/.hydra/overrides.yaml"
    if ov.is_file():
        o = yaml.safe_load(ov.read_text()) or []
        info["overrides"] = [x for x in o if not x.startswith(("output_dir=", "hydra.", "logging.", "+logging"))]
        for x in o:
            if x.startswith("experiment="):
                info["experiment"] = x.split("=", 1)[1]
    else:
        wc = run / "wandb/latest-run/files/config.yaml"
        if wc.is_file():
            w = yaml.safe_load(wc.read_text()) or {}
            for k in ("max_iterations", "experiment_name"):
                if k in w:
                    info[k] = (w[k] or {}).get("value") if isinstance(w[k], dict) else w[k]
    return info


def best_of(evals: list[dict]) -> dict | None:
    """highest own-input AUC, ties -> earlier iteration."""
    ok = [e for e in evals if e.get("auc") is not None]
    if not ok:
        return None
    return sorted(ok, key=lambda e: (-e["auc"], e["iteration"]))[0]


def newest_of(evals: list[dict]) -> dict | None:
    ok = [e for e in evals if e.get("auc") is not None]
    return max(ok, key=lambda e: e["iteration"]) if ok else None


def ckpt_entry(label: str, run: Path, it: int | None, provisional: bool, own: dict | None = None, note: str = "") -> dict:
    p = run / f"model_{it}.safetensors" if it is not None else None
    return {"label": label, "iteration": it, "path": str(p) if p else None, "sha256": sha(p) if p else None,
            "exists": bool(p and p.is_file()), "provisional": provisional, "own_input": own, "note": note}


# ------------------------------------------------------------------------------------------- our sources
early = {int(r["episode"]): r for r in rows(T2 / "early_scores/early_table.csv") if r["episode"].isdigit()}
rawfinal = {int(r["episode"]): r for r in rows(ME / "fc_runs/eval_tier2/C3r2_tier2_final.csv") if r["episode"].isdigit()}
rawclean1 = {int(r["episode"]): r for r in rows(ME / "fc_runs/eval_tier2/C3r2_tier2_tier2_clean_scores_PARTIAL.csv")
             if r["episode"].isdigit()}
verdict = {r["episode"]: r for r in jload(T2 / "k3full/VERDICT.json").get("episodes", [])}
vclean = {r["episode"]: r for r in jload(T2 / "k3full/VERDICT_clean_LOCAL_NOT_FOR_UPLOAD.json").get("episodes", [])}
gates = {ep: verdict.get(ep, {}).get("gate") for ep in range(30)}

prog = rows(T2 / "t2watch/progress.csv")
progc = rows(T2 / "t2watch/progress_clean_LOCAL_NOT_FOR_UPLOAD.csv")


def t2w_evals(arm: str, run: str) -> list[dict]:
    out = []
    for r in prog:
        if r["arm"] == arm and r["run"] == run and r["status"] == "ok":
            out.append({"iteration": int(r["iteration"]), "auc": fnum(r["own_auc"]), "sp": fnum(r["own_sp_sr"]),
                        "mp": fnum(r["own_mp_sr"]), "mppe": fnum(r["own_mppe"]), "kind": r["kind"],
                        "ref_parquet": r["reference_parquet"], "ckpt_sha": r["checkpoint_sha256"], "src": "t2watch"})
    return out


def t2w_clean(arm: str, run: str) -> dict[int, dict]:
    out = {}
    for r in progc:
        if r["arm"] == arm and r["run"] == run and r["score_status"] == "ok":
            out.setdefault(int(r["iteration"]), []).append(fnum(r["clean_auc"]))
    return {k: {"clean_auc_evals": v, "frame_refs": None} for k, v in out.items()}


def post_of(job: str) -> tuple[dict, dict]:
    for base in (T2 / "k3full/eval", T2 / "k3fix_pilot/eval", T2 / "k3easy/eval"):
        if (base / job / "post.json").is_file():
            return jload(base / job / "post.json"), jload(base / job / "clean_LOCAL_NOT_FOR_UPLOAD.json")
    return {}, {}


t2fix_base = {}
for r in rows(ME / "fc_runs/T2fix/eval_BASE/evals.csv"):
    ep = int(r["episode"])
    t2fix_base.setdefault(ep, []).append({"iteration": int(r["iteration"]), "auc": fnum(r["auc"]), "sp": fnum(r["sp_sr"]),
                                          "mp": fnum(r["mp_sr"]), "purpose": r.get("purpose"), "attempt": r.get("attempt")})
t2fix_arm = {}
for r in rows(ME / "fc_runs/T2fix/eval_ARM/evals.csv"):
    if (r.get("status") or "ok") not in ("ok", ""):
        continue
    t2fix_arm.setdefault(int(r["episode"]), []).append({"iteration": int(r["iteration"]), "auc": fnum(r["auc"]), "sp": fnum(r["sp_sr"]),
                                                        "mp": fnum(r["mp_sr"]), "mppe": fnum(r["mppe"]), "src": "T2fix eval_ARM",
                                                        "purpose": r.get("purpose")})
t2fix_verdict = jload(ME / "fc_runs/T2fix/verdict.json")

cands: list[dict] = []
clean_rows: list[dict] = []


def add_clean(c: dict, label: str, it, val, n, src):
    clean_rows.append({"episode": c["episode"], "owner": c["owner"], "arm": c["arm"], "checkpoint_label": label, "iteration": it,
                       "clean_add_auc": val, "n_evals": n, "source": src})


for ep in range(30):
    seq = EP[ep]["sequence_id"]
    obj = EP[ep]["object"]
    # ---------- A. raw C3r2_tier2
    run = ME / f"fc_runs/C3r2_tier2/ep{ep:02d}"
    fin = jload(run / "final.json")
    rf, et = rawfinal.get(ep, {}), early.get(ep, {})
    own1 = {"iteration": int(rf["iteration"]), "auc": fnum(rf["auc"]), "sp": fnum(rf["sp_sr"]), "mp": fnum(rf["mp_sr"]),
            "n": 1, "src": "fc_runs/eval_tier2/C3r2_tier2_final.csv"} if rf.get("auc") else None
    own3 = {"iteration": int(et["t2train_iter"]) if et.get("t2train_iter") else None,
            "auc": fnum(et.get("t2train_evalref_noisy_mean_add_auc")), "sp": fnum(et.get("t2train_evalref_noisy_mean_spider_sr")),
            "mp": fnum(et.get("t2train_evalref_noisy_mean_maniptrans_sr")), "n": 3,
            "src": "early_table.csv t2train_evalref_noisy_mean_* (3 reps vs own noisy input)"} if et else None
    p = pq("raw", ep)
    c = {"episode": ep, "object": obj, "sequence_id": seq, "owner": "ours", "arm": "raw C3r2_tier2", "run_dir": str(run),
         "status": "finished" if fin else ("running" if is_live(run) else "no final.json"), "pid": is_live(run),
         "input": "raw Tier-2 (data/human_motion_data_tier2)", "input_kind": "raw", "parquet": str(p), "parquet_sha256": sha(p),
         "parquet_at_eval": et.get("eval_reference_parquet"), "code": OURS_CODE + "; wandb metadata commit 61fa5d3",
         "experiment": "g1_recon_body_coord_rsi_r2 (C3r2)", "recipe": hydra_info(run), "seed": 42,
         "newest_model": (models(run) or [None])[-1], "final": fin or None,
         "k3full_rule_choice": verdict.get(ep, {}).get("choice"), "gate_G1_G4": gates.get(ep),
         "checkpoints": [ckpt_entry("final.json", run, fin.get("iteration"), False,
                                    {"final_1eval": own1, "final_mean3": own3})] if fin else []}
    cands.append(c)
    if et.get("t2train_mean_add_auc"):
        add_clean(c, "final.json", fin.get("iteration"), fnum(et["t2train_mean_add_auc"]), 3, "early_table t2train_mean_add_auc")
    if rawclean1.get(ep, {}).get("t2clean_add_auc"):
        add_clean(c, "final.json", fin.get("iteration"), fnum(rawclean1[ep]["t2clean_add_auc"]), 1,
                  "eval_tier2/C3r2_tier2_tier2_clean_scores_PARTIAL.csv")

    # ---------- B/C. K3fix arm (+ k3easy + the ep03 raw seed-7 control)
    jobs = []
    for d in sorted((ME / "fc_runs/C3r2_tier2_k3fix").glob(f"ep{ep:02d}_*")):
        jobs.append(("C3r2_tier2_k3fix", d.name, d))
    if (ME / f"fc_runs/C3r2_tier2_k3easy/ep{ep:02d}").is_dir():
        jobs.append(("C3r2_tier2_k3easy", f"ep{ep:02d}", ME / f"fc_runs/C3r2_tier2_k3easy/ep{ep:02d}"))
    for arm, job, run in jobs:
        fin = jload(run / "final.json")
        hi = hydra_info(run)
        live = is_live(run)
        post, pclean = post_of(job if arm == "C3r2_tier2_k3fix" else f"ep{ep:02d}")
        ev = t2w_evals(arm, job)
        cl = t2w_clean(arm, job)
        tb = jload(T2 / f"t2watch/best/{arm}/{job}/best.json")
        trained_pq = hi.get("parquet")
        kind = "raw" if trained_pq and "/human_motion_data_tier2/" in trained_pq else (
            "k3" if trained_pq and "_tier2_k3/" in trained_pq else "k3fix")
        p = Path(trained_pq) if trained_pq else pq(kind, ep)
        c = {"episode": ep, "object": obj, "sequence_id": seq, "owner": "ours",
             "arm": {"C3r2_tier2_k3easy": "K3 easy-clip (k3easy)"}.get(arm, "raw seed-7 control" if "raw_s7" in job else "K3fix"),
             "run_dir": str(run), "status": "finished" if fin else ("running" if live else "stopped, no final.json"), "pid": live,
             "input": {"raw": "raw Tier-2", "k3": "K3 root (= K3fix bytes for this ep?)", "k3fix": "K3fix (data/human_motion_data_tier2_k3fix)"}[kind],
             "input_kind": kind, "parquet": str(p), "parquet_sha256": sha(p),
             "parquet_sha_k3fix_root": sha(pq("k3fix", ep)), "code": OURS_CODE, "experiment": hi.get("experiment"), "recipe": hi,
             "seed": hi.get("seed"), "newest_model": (models(run) or [None])[-1], "final": fin or None,
             "k3full_rule_choice": verdict.get(ep, {}).get("choice") if "raw_s7" not in job else "n/a (control)",
             "gate_G1_G4": gates.get(ep), "checkpoints": []}
        b, n = best_of(ev), newest_of(ev)
        if fin:
            o1 = {"iteration": post.get("evaluated_iteration"), "auc": fnum(post.get("eval_add_auc_own_input")),
                  "sp": fnum(post.get("eval_sp_sr")), "mp": fnum(post.get("eval_mp_sr")), "n": 1, "src": "post.json",
                  "export_max_rel_diff": fnum(post.get("export_max_rel_diff")), "ref_is_train_input": post.get("eval_reference_is_train_input"),
                  "ref_parquet_sha256": post.get("parquet_sha256")} if post else None
            b3 = [e for e in t2fix_base.get(ep, []) if e["iteration"] == fin.get("iteration")] if arm == "C3r2_tier2_k3fix" and "raw_s7" not in job else []
            m3 = {"iteration": fin.get("iteration"), "auc": sum(e["auc"] for e in b3) / len(b3), "n": len(b3),
                  "sp": sum(e["sp"] for e in b3) / len(b3), "mp": sum(e["mp"] for e in b3) / len(b3),
                  "src": "T2fix eval_BASE (all rows at final iteration)",
                  "t2fix_verdict_base_mean3": (t2fix_verdict.get(next((k for k in t2fix_verdict if k.startswith(f"ep{ep:02d}-")), ""), {}) or {}).get("base_mean3")} if b3 else None
            c["checkpoints"].append(ckpt_entry("final.json", run, fin.get("iteration"), False, {"final_1eval": o1, "final_mean_T2fixBASE": m3}))
            if pclean:
                add_clean(c, "final.json", fin.get("iteration"), fnum(pclean.get("clean_add_auc") or pclean.get("t2clean_add_auc")), 1,
                          "post dir clean_LOCAL_NOT_FOR_UPLOAD.json")
            vc = vclean.get(ep, {})
            if vc.get("k3fix_run_clean_add_auc_1eval") is not None and "raw_s7" not in job and not pclean:
                add_clean(c, "final.json", fin.get("iteration"), vc["k3fix_run_clean_add_auc_1eval"], 1, "k3full VERDICT_clean_LOCAL")
        if b:
            c["checkpoints"].append(ckpt_entry("t2watch best-by-own-input" + ("" if fin else " so far"), run, b["iteration"], not fin,
                                               {"single_eval": b, "best_json_iteration": tb.get("iteration"),
                                                "best_json_auc": fnum(tb.get("own_auc"))},
                                               note="secondary under PREREG_v3 (arm uses final.json)" if fin else "run still training"))
            if b["iteration"] in cl:
                add_clean(c, "t2watch best", b["iteration"], cl[b["iteration"]]["clean_auc_evals"][0], 1, "t2watch progress_clean_LOCAL")
        if not fin and n:
            c["checkpoints"].append(ckpt_entry("newest evaluated (t2watch)", run, n["iteration"], True, {"single_eval": n}))
            if n["iteration"] in cl:
                add_clean(c, "newest evaluated", n["iteration"], cl[n["iteration"]]["clean_auc_evals"][0], 1, "t2watch progress_clean_LOCAL")
        c["n_own_evals"] = len(ev)
        cands.append(c)

    # ---------- D. T2fix arms (K3fix input, forks of the K3fix final)
    arms_tsv = ME / "fc_runs/T2fix/arms.tsv"
    with open(arms_tsv) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if int(r["episode"]) != ep:
                continue
            run = Path(r["run_dir"])
            hi = hydra_info(run)
            live = is_live(Path(r["trainer_output_dir"])) or is_live(run)
            fin = jload(run / "final.json")
            ev = [e for e in t2fix_arm.get(ep, [])]
            launched = (run / "t2fix_launch.json").is_file()
            p = Path(hi.get("parquet")) if hi.get("parquet") else pq("k3fix", ep)
            c = {"episode": ep, "object": obj, "sequence_id": seq, "owner": "ours", "arm": f"T2fix {r['arm']} ({r['kind']})",
                 "run_dir": str(run), "status": "finished" if fin else ("running" if live else ("paused/stopped" if launched else "not launched (T2fix PAUSE)")),
                 "pid": live, "input": "K3fix", "input_kind": "k3fix", "parquet": str(p), "parquet_sha256": sha(p),
                 "code": OURS_CODE, "experiment": r["experiment"],
                 "recipe": {"fork_of": f"{r['source_dir']} model_{r['source_iteration']}", "max_iterations": int(r["max_iterations"]),
                            "overrides": r["overrides"], "early_stop_help_zero_tail": hi.get("early_stop_help_zero_tail"),
                            "support": hi.get("support")},
                 "seed": hi.get("seed"), "newest_model": (models(run) or [None])[-1], "final": fin or None,
                 "t2fix_verdict": (t2fix_verdict.get(r["arm"]) or {}).get("verdict"), "checkpoints": []}
            b, n = best_of(ev), newest_of(ev)
            if b:
                c["checkpoints"].append(ckpt_entry("best-by-own-input so far (eval_ARM scan)", run, b["iteration"], not fin, {"single_eval": b}))
            if n and (not b or n["iteration"] != b["iteration"]):
                c["checkpoints"].append(ckpt_entry("newest evaluated (eval_ARM scan)", run, n["iteration"], True, {"single_eval": n}))
            c["n_own_evals"] = len(ev)
            cands.append(c)

    # ---------- E/F. Zuxing (read-only)
    for R in sorted((Z / "runs").glob("2026-10-24-tier2-*")):
        run = R / f"ep{ep:02d}"
        if not run.is_dir():
            continue
        hi = hydra_info(run)
        trained_pq = hi.get("parquet") or ""
        kind = "g2" if "_g2-" in trained_pq else ("k3fix" if "_tier2_k3fix/" in trained_pq else "?")
        p = Path(trained_pq) if trained_pq else None
        evdir = R / ("eval" if (R / "eval/metrics.csv").is_file() and not (R / f"eval_ep{ep:02d}").is_dir() else f"eval_ep{ep:02d}")
        ev = []
        for x in rows(evdir / "metrics.csv"):
            if int(x["episode"]) == ep and x["status"] == "ok":
                ev.append({"iteration": int(x["iteration"]), "auc": fnum(x["add_auc"]), "sp": fnum(x["sp_sr"]), "mp": fnum(x["mp_sr"]),
                           "mppe": fnum(x["mppe_cm"]), "export_rel_diff": fnum(x["export_max_rel_diff"]),
                           "stock_verified": x.get("stock_verified"), "ckpt_sha": x.get("source_sha256"), "src": str(evdir / "metrics.csv")})
        cl = {int(x["iteration"]): fnum(x["t2clean_add_auc"]) for x in rows(evdir / "tier2_clean.csv") if int(x["episode"]) == ep}
        fin = jload(run / "final.json")
        live = is_live(run)
        log = Z / "logs" / R.name / f"ep{ep:02d}.log"
        c = {"episode": ep, "object": obj, "sequence_id": seq, "owner": "zuxing", "arm": f"{R.name} (rolling start runin)",
             "run_dir": str(run), "status": "finished" if fin else ("running" if live else "stopped, no final.json"), "pid": live,
             "input": {"g2": "Zuxing G2 object track + raw Tier-2 human (zuxing data/human_motion_data_tier2_g2-2026-10-24)",
                       "k3fix": "K3fix (our data/human_motion_data_tier2_k3fix)"}.get(kind, kind),
             "input_kind": kind, "parquet": trained_pq, "parquet_sha256": sha(p), "parquet_sha_k3fix_root": sha(pq("k3fix", ep)),
             "code": ZX_CODE, "experiment": hi.get("experiment"), "recipe": hi, "seed": hi.get("seed"),
             "newest_model": (models(run) or [None])[-1], "final": fin or None, "eval_dir": str(evdir),
             "eval_flow": "zuxing tmp/t2/eval/eval_watch.sh: export with dev.zuxing.speedup (tol 1e-4) -> kit evaluator on stock 8c778f9, world count 1; "
                          "metrics.csv = own training input; tier2_clean.csv = packer vs clean (frame refs of the run's input)",
             "train_log": str(log) if log.is_file() else None, "checkpoints": []}
        b, n = best_of(ev), newest_of(ev)
        if b:
            c["checkpoints"].append(ckpt_entry("best-by-own-input so far (his metrics.csv)", run, b["iteration"], not fin, {"single_eval": b}))
            add_clean(c, "best own so far", b["iteration"], cl.get(b["iteration"]), 1, str(evdir / "tier2_clean.csv"))
        if n and (not b or n["iteration"] != b["iteration"]):
            c["checkpoints"].append(ckpt_entry("newest evaluated", run, n["iteration"], True, {"single_eval": n}))
            add_clean(c, "newest evaluated", n["iteration"], cl.get(n["iteration"]), 1, str(evdir / "tier2_clean.csv"))
        bc = max(cl.items(), key=lambda kv: (kv[1] or -1)) if cl else None
        if bc:
            add_clean(c, "max clean over his evals (LOCAL, report only)", bc[0], bc[1], 1, str(evdir / "tier2_clean.csv"))
        # verify his recorded checkpoint sha against the file for the listed checkpoints
        for k in c["checkpoints"]:
            rec = (k["own_input"] or {}).get("single_eval", {}).get("ckpt_sha")
            k["sha_matches_his_eval_record"] = (rec == k["sha256"]) if rec and k["sha256"] else None
        c["n_own_evals"] = len(ev)
        cands.append(c)

# ---------- non-candidates (listed for completeness)
others = []
for e in ("ep01", "ep03", "ep04"):
    run = ME / f"fc_runs/C3r2_tier2_k3_stretch90k/{e}"
    others.append({"run_dir": str(run), "input": "K3 (not K3fix)", "newest_model": (models(run) or [None])[-1],
                   "final": jload(run / "final.json") or None, "status": "stopped 2026-10-21 by user ('stop it and all k3')"})
others.append({"run_dir": str(ME / "fc_runs/C3r2_tier2_armA_90k"), "status": "never trained (HOLD; logs only)"})
others.append({"run_dir": str(ME / "fc_runs/C3r2_tier2_k3ca"), "status": "not trained yet (K3CA driver paused; fc_cache + logs only)",
               "input": "data/human_motion_data_tier2_k3ca (eps 00 05 15 19 22 26 28)"})
others.append({"run_dir": str(Z / "runs/2026-10-21-help-force-fix, 2026-10-14-patience-runin, 2026-10-14-psi-worst3"),
               "status": "Zuxing Tier-1 runs (input data/human_motion_data = Tier-1 clean): not Tier-2 candidates"})
others.append({"note": "Zuxing packed Tier-2 raw+K3fix data (tmp/transfer/tier2_raw_k3fix.tgz, 2026-10-23 01:20) and the eval flow for a '5090 host'; "
                       "no 5090 Tier-2 results are visible on pro6000"})

doc = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "decides_nothing": True,
       "rule": "own-input AUCs comparable only within the same parquet_sha256; clean numbers only in inventory_clean_LOCAL_NOT_FOR_UPLOAD.csv",
       "roots_ready": {k: (open(v / 'READY').readline().strip()[:120] if (v / 'READY').is_file() else None) for k, v in ROOTS.items()},
       "candidates": cands, "non_candidates": others}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "inventory.json").write_text(json.dumps(doc, indent=1, default=str))

with open(OUT / "inventory_own_input.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["episode", "object", "owner", "arm", "status", "input_kind", "parquet_sha256_16", "experiment", "run_dir", "newest_model",
                "final_iteration", "ckpt_label", "ckpt_iteration", "provisional", "ckpt_sha256_16", "own_auc", "own_sp", "own_mp", "own_n",
                "own_src", "own_mean_extra", "own_mean_extra_n"])
    for c in cands:
        if not c["checkpoints"]:
            w.writerow([c["episode"], c["object"], c["owner"], c["arm"], c["status"], c["input_kind"], (c["parquet_sha256"] or "")[:16],
                        c.get("experiment"), c["run_dir"], c["newest_model"], (c["final"] or {}).get("iteration"), "-", "", "", "", "", "", "", "",
                        "", "", ""])
        for k in c["checkpoints"]:
            o = k["own_input"] or {}
            one = o.get("final_1eval") or o.get("single_eval") or {}
            ext = o.get("final_mean3") or o.get("final_mean_T2fixBASE") or {}
            w.writerow([c["episode"], c["object"], c["owner"], c["arm"], c["status"], c["input_kind"], (c["parquet_sha256"] or "")[:16],
                        c.get("experiment"), c["run_dir"], c["newest_model"], (c["final"] or {}).get("iteration"), k["label"], k["iteration"],
                        k["provisional"], (k["sha256"] or "")[:16], one.get("auc"), one.get("sp"), one.get("mp"), one.get("n", 1),
                        one.get("src"), ext.get("auc"), ext.get("n")])
with open(OUT / "inventory_clean_LOCAL_NOT_FOR_UPLOAD.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["episode", "owner", "arm", "checkpoint_label", "iteration", "clean_add_auc", "n_evals", "source"])
    w.writeheader()
    for r in clean_rows:
        w.writerow(r)
print(f"wrote {OUT}: {len(cands)} candidate rows, {len(clean_rows)} LOCAL clean rows, live trainers {len(LIVE)}")

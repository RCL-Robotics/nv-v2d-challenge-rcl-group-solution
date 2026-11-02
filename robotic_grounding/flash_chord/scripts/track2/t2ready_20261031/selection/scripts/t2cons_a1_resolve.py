#!/usr/bin/env python3
"""t2cons_a1_resolve.py: AMENDMENTS.md A3 of fc_runs/submissions/t2_consensus_20261024.

Resolves, at the cut-off T_a written in AMENDMENTS.md, the designated checkpoint of every added run (A3.2) by own-input
values only, and maps each onto an existing job (same file sha256) or a new job. Writes inputs_a1/ (new files only):
  inputs_a1/zuxing_metrics/<run>_<evaldir>__metrics.csv   byte copies of Zuxing's own-input metrics.csv
  inputs_a1/candidates_a1.json / .txt                     the resolution, new / reused jobs, challenger counts
  inputs_a1/SNAPSHOT_A1.sha256                            written last

Noisy-only. Every path goes through the PREREG 5.7 guard imported from t2cons_consensus.py (refuses any path containing
clean, LOCAL_NOT_FOR_UPLOAD, packed_, ...), so tier2_clean.csv and every clean file stay unopened. Zuxing's folder is
read only (files only, no git). Nothing is evaluated or scored here.

Run: conda_envs/v2dkit/bin/python -I -B scripts/t2cons_a1_resolve.py
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import datetime as dt  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
spec = importlib.util.spec_from_file_location("t2cons_consensus", ME / "scripts/t2cons_consensus.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

ROOT = C.ROOT
AMEND, AMEND_SHA = ROOT / "AMENDMENTS.md", ROOT / "AMENDMENTS.sha256"
OUT = ROOT / "inputs_a1"
ZU = C.ZU
TZ8 = dt.timezone(dt.timedelta(hours=8))
G2_ROOT = ZU / "data/human_motion_data_tier2_g2-2026-10-24"
K3FIX_ROOT = ME / "data/human_motion_data_tier2_k3fix"
K3CA_ROOT = ME / "data/human_motion_data_tier2_k3ca"
ROOT_KIND = {str(G2_ROOT): "g2", str(K3FIX_ROOT): "k3fix", str(K3CA_ROOT): "k3ca"}

# AMENDMENTS A3.2 table: id, Zuxing run folder, {episode: eval dir}, expected input kind, job tag
ZRUNS = [
    ("Z1", "2026-10-24-tier2-runin-g2", {1: "eval_ep01", 4: "eval_ep04", 19: "eval_ep19", 28: "eval_ep28"}, "g2", "g2"),
    ("Z2", "2026-10-25-tier2-runin-hard-g2", {1: "eval_ep01", 19: "eval_ep19"}, "g2", "g2hard"),
    ("Z3", "2026-10-24-tier2-runin-k3fix-ep05-ep26", {5: "eval_ep05", 26: "eval_ep26"}, "k3fix", "k3fix"),
    ("Z4", "2026-10-24-tier2-runin-k3fix-ep04", {4: "eval"}, "k3fix", "k3fix"),
]  # fmt: skip
OURS = [
    ("O1", 26, ME / "fc_runs/C3r2_tier2_k3ca/ep26_k3ca_s42", "k3ca", "final.json"),
    ("O2", 4, ME / "fc_runs/C3r2_tier2_k3fix/ep04_k3fix_s42", "k3fix", "final.json"),
    ("O3", 1, ME / "fc_runs/T2fix/T2L/ep01", "k3fix", "T2fix runin_final.json best"),
]  # fmt: skip


def die(msg: str) -> None:
    raise SystemExit(f"FATAL: {msg}")


def file_sha(p: Path) -> str:
    return C.sha256_file(p)


def hydra_parquet(run_dir: Path) -> str:
    for line in C.guard(run_dir / "hydra/.hydra/overrides.yaml").read_text().splitlines():
        m = re.match(r"^- task\.parquet='?([^']+)'?$", line.strip())
        if m:
            return m.group(1)
    for line in C.guard(run_dir / "hydra/.hydra/config.yaml").read_text().splitlines()[:10]:
        m = re.match(r"^\s*parquet:\s*(\S+)$", line)
        if m:
            return m.group(1)
    die(f"{run_dir}: no task.parquet in hydra config")
    return ""


def kind_of(parquet: str) -> str:
    for root, kind in ROOT_KIND.items():
        if parquet.startswith(root + "/"):
            return kind
    return "unknown"


def parse_ts(s: str) -> dt.datetime:
    t = dt.datetime.fromisoformat(s.strip())
    return t if t.tzinfo else t.replace(tzinfo=TZ8)


def main() -> int:
    if not sys.flags.dont_write_bytecode or not sys.flags.isolated:
        print("run with the kit python -I -B")
        return 2
    # ---- pins: PREREG and AMENDMENTS
    want = (C.PREREG_SHA.read_text().split() or [""])[0]
    if file_sha(C.PREREG) != want:
        die("PREREG.md != PREREG.sha256")
    if not AMEND_SHA.is_file():
        die("AMENDMENTS.sha256 missing")
    amend_sha = file_sha(AMEND)
    if amend_sha != (AMEND_SHA.read_text().split() or [""])[0]:
        die("AMENDMENTS.md != AMENDMENTS.sha256")
    text = AMEND.read_text()
    m = re.search(r"\*\*Cut-off T_a = (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) CST\.\*\*", text)
    if not m or "@@T_A@@" in text:
        die("no T_a in AMENDMENTS.md")
    t_a_str = m.group(1)
    t_a = dt.datetime.strptime(t_a_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ8)
    if (OUT / "candidates_a1.json").exists() or (OUT / "SNAPSHOT_A1.sha256").exists():
        die("inputs_a1/ already resolved (new files only)")

    # ---- completeness: every Tier-2 run folder of Zuxing's at T_a is in the A3.2 table
    listing = sorted(p.name for p in C.guard(ZU / "runs").iterdir() if p.is_dir() and "tier2" in p.name.lower())
    if listing != sorted(r[1] for r in ZRUNS):
        die(f"Zuxing Tier-2 run folders {listing} != AMENDMENTS A3.2 {[r[1] for r in ZRUNS]}")

    # ---- existing jobs (snapshot t2_best5 + this folder's 5 new jobs), by checkpoint sha256
    jobs = C.load_jobs()
    evals = C.load_evals(jobs)
    cand = C.candidates(jobs, evals)
    rows_by_job = {}
    for r in C.read_json(C.SNAP / "t2_best5/manifest.json")["rows"]:
        rows_by_job[r["job"]] = r
    for ln in C.guard(C.SNAP / "t2_best5/manifest_stageB.jsonl").read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            rows_by_job[r["job"]] = r
    for r in C.read_json(ROOT / "manifest_new.json")["rows"]:
        rows_by_job[r["job"]] = r
    by_sha: dict[str, list[str]] = {}
    for job, r in rows_by_job.items():
        for key in ("training_sha256", "expected_sha256"):
            h = r.get(key)
            if h and job not in by_sha.get(h, []):
                by_sha.setdefault(h, []).append(job)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "zuxing_metrics").mkdir(exist_ok=True)
    resolutions, copies = [], {}

    # ---- rule Z
    for zid, run, evdirs, want_kind, tag in ZRUNS:
        run_root = ZU / "runs" / run
        eps_present = sorted(int(p.name[2:]) for p in C.guard(run_root).iterdir()
                             if p.is_dir() and re.fullmatch(r"ep\d\d", p.name))  # fmt: skip
        if eps_present != sorted(evdirs):
            die(f"{run}: episode folders {eps_present} != A3.2 {sorted(evdirs)}")
        for e, evdir in sorted(evdirs.items()):
            src = run_root / evdir / "metrics.csv"
            data = C.guard(src).read_bytes()
            dst = OUT / "zuxing_metrics" / f"{run}_{evdir}__metrics.csv"
            if dst.exists():
                die(f"{dst} exists")
            C.write_bytes_atomic(dst, data)
            copies[str(dst.relative_to(ROOT))] = C.sha256_bytes(data)
            rows = C.read_csv(dst)
            eligible, excluded = [], {"status": 0, "stock_verified": 0, "after_T_a": 0, "episode": 0, "bad": 0}
            for r in rows:
                try:
                    it, auc, ts = int(r["iteration"]), C.fnum(r["add_auc"]), parse_ts(r["finished_at"])
                except (KeyError, ValueError, TypeError):
                    excluded["bad"] += 1
                    continue
                if int(r.get("episode", e)) != e:
                    excluded["episode"] += 1
                elif r.get("status") != "ok":
                    excluded["status"] += 1
                elif r.get("stock_verified") != "True":
                    excluded["stock_verified"] += 1
                elif ts > t_a:
                    excluded["after_T_a"] += 1
                elif auc is None:
                    excluded["bad"] += 1
                else:
                    eligible.append((auc, it, r))
            eligible.sort(key=lambda x: (-x[0], -x[1]))  # highest add_auc; exact ties -> newest iteration
            ep_dir = run_root / f"ep{e:02d}"
            parquet = hydra_parquet(ep_dir)
            kind = kind_of(parquet)
            if kind != want_kind:
                die(f"{run} ep{e:02d}: training parquet {parquet} is {kind}, A3.2 says {want_kind}")
            pick, skipped = None, []
            for auc, it, r in eligible:
                ck = ep_dir / f"model_{it}.safetensors"
                if not ck.is_file():
                    skipped.append({"iteration": it, "add_auc": auc, "why": "checkpoint missing"})
                    continue
                h = file_sha(ck)
                if h != r.get("source_sha256"):
                    skipped.append({"iteration": it, "add_auc": auc, "why": f"sha256 {h[:16]} != source_sha256"})
                    continue
                pick = (auc, it, r, ck, h)
                break
            rec = {"id": zid, "owner": "zuxing", "run": run, "episode": e, "input": kind, "metrics_csv": str(src),
                   "metrics_copy": str(dst), "metrics_copy_sha256": copies[str(dst.relative_to(ROOT))],
                   "n_rows": len(rows), "n_eligible": len(eligible), "excluded": excluded, "skipped": skipped,
                   "training_parquet": parquet, "designation": "rule Z (AMENDMENTS A3.2)",
                   "top5_own_input": [{"iteration": it, "add_auc": auc, "sp_sr": C.fnum(r.get("sp_sr")),
                                       "mp_sr": C.fnum(r.get("mp_sr")), "finished_at": r.get("finished_at")}
                                      for auc, it, r in eligible[:5]]}  # fmt: skip
            if pick is None:
                rec.update({"status": "no_designation", "job": None})
            else:
                auc, it, r, ck, h = pick
                rsi = ep_dir / f"rsi_{it}.safetensors"
                rec.update({"status": "resolved", "iteration": it, "checkpoint": str(ck), "sha256": h,
                            "rsi": str(rsi), "rsi_exists": rsi.is_file(),
                            "row": {k: r.get(k) for k in ("iteration", "add_auc", "sp_sr", "mp_sr", "mppe_cm", "chord_sr",
                                                          "export_max_rel_diff", "stock_verified", "source_sha256",
                                                          "finished_at", "attempt")},
                            "default_name": f"ep{e:02d}_zx_{tag}_{it}"})  # fmt: skip
            resolutions.append(rec)

    # ---- our runs
    k3ca_v = C.read_json(ME / "tmp_analysis/tier2/k3ca/train/VERDICT.json")
    k3full_v = C.read_json(ME / "tmp_analysis/tier2/k3full/VERDICT.json")
    t2fix_v = C.read_json(ME / "fc_runs/T2fix/verdict.json")
    for oid, e, run_dir, want_kind, how in OURS:
        final = C.read_json(run_dir / "final.json")
        parquet = hydra_parquet(run_dir)
        kind = kind_of(parquet)
        if kind != want_kind:
            die(f"{run_dir}: training parquet {parquet} is {kind}, A3.2 says {want_kind}")
        rec = {"id": oid, "owner": "ours", "run": str(run_dir), "episode": e, "input": kind, "training_parquet": parquet,
               "designation": how, "final_json": {k: final.get(k) for k in ("checkpoint", "iteration", "reason",
                                                                              "help_zero_since", "finished_at")}}  # fmt: skip
        if oid in ("O1", "O2"):
            it = int(final["iteration"])
            ck = run_dir / final["checkpoint"]
            if ck.name != f"model_{it}.safetensors":
                die(f"{run_dir}: final.json checkpoint {ck.name} != iteration {it}")
            name = f"ep{e:02d}_k3ca_final{it}" if oid == "O1" else f"ep{e:02d}_k3fix_final{it}"
            if oid == "O1":
                v = next(x for x in k3ca_v["episodes"] if int(x["episode"]) == e)
                rec["verdict"] = {"file": "tmp_analysis/tier2/k3ca/train/VERDICT.json", "written": k3ca_v.get("written"),
                                  **{k: v.get(k) for k in ("choice", "settled", "state", "k3ca_gates_pass", "final_iteration",
                                                           "own_auc", "own_sp", "own_mp", "exported")}}  # fmt: skip
            else:
                v = next(x for x in k3full_v["episodes"] if int(x["episode"]) == e)
                rec["verdict"] = {"file": "tmp_analysis/tier2/k3full/VERDICT.json", "written": k3full_v.get("written"),
                                  **{k: v.get(k) for k in ("gate", "finished", "final_iteration", "exported")},
                                  "note": "not regenerated after ep04 finished (05:43 verdict run crashed); "
                                          "designation is final.json regardless"}  # fmt: skip
        else:
            rf = C.read_json(ME / "fc_runs/T2fix/eval_ARM/ep01/runin_final.json")
            it = int(rf["best_iteration"])
            ck = run_dir / f"model_{it}.safetensors"
            linked = Path(os.path.realpath(rf["best_checkpoint"]))
            if linked != ck.resolve():
                die(f"runin_final best_checkpoint {rf['best_checkpoint']} -> {linked} != {ck}")
            name = f"ep{e:02d}_t2fixL_designated{it}"
            v = t2fix_v["ep01-L"]
            best = C.read_json(ME / "fc_runs/T2fix/eval_ARM/ep01/best.json")
            rec["verdict"] = {"file": "fc_runs/T2fix/verdict.json",
                              **{k: v.get(k) for k in ("verdict", "arm_mean3", "arm_best_iteration", "arm_sp_mp",
                                                       "base", "base_mean3", "threshold")},
                              "runin_final": {k: rf.get(k) for k in ("best_iteration", "mean_auc", "mean_sp_sr",
                                                                     "mean_mp_sr", "n_evals", "finished_at")},
                              "best_json_checkpoint_sha256": best.get("checkpoint_sha256")}  # fmt: skip
        h = file_sha(ck)
        rsi = run_dir / f"rsi_{it}.safetensors"
        rec.update({"status": "resolved", "iteration": it, "checkpoint": str(ck), "sha256": h, "rsi": str(rsi),
                    "rsi_exists": rsi.is_file(), "default_name": name})  # fmt: skip
        if oid == "O3":
            bj = rec["verdict"]["best_json_checkpoint_sha256"]
            rec["verdict"]["best_json_sha_matches_file"] = bj == h
        resolutions.append(rec)

    # ---- map onto existing jobs / new jobs
    new_jobs, reused = [], []
    for rec in resolutions:
        if rec["status"] != "resolved":
            continue
        hit = by_sha.get(rec["sha256"], [])
        if hit:
            if len(hit) != 1:
                die(f"{rec['default_name']}: sha256 matches several jobs {hit}")
            job = hit[0]
            j = jobs.get(job)
            if j is None or j["episode"] != rec["episode"] or j["input"] != rec["input"]:
                die(f"{rec['default_name']}: existing job {job} differs in episode / input")
            rec.update({"job": job, "job_status": "reused",
                        "is_candidate_already": job in cand, "was_provisional": bool(j.get("provisional")),
                        "settled_now": bool(j.get("provisional"))})  # fmt: skip
            reused.append(job)
        else:
            if rec["default_name"] in jobs:
                die(f"{rec['default_name']}: job name exists with another checkpoint")
            rec.update({"job": rec["default_name"], "job_status": "new"})
            new_jobs.append({"job": rec["default_name"], "episode": rec["episode"], "input": rec["input"],
                             "owner": rec["owner"], "run": rec["run"], "iteration": rec["iteration"],
                             "checkpoint": rec["checkpoint"], "sha256": rec["sha256"], "rsi": rec["rsi"],
                             "rsi_exists": rec["rsi_exists"], "training_parquet": rec["training_parquet"],
                             "frame_refs": str(C.INPUT_REF[rec["input"]]),
                             "variants": "B" if rec["input"] == "g2" else "A,B"})  # fmt: skip

    # ---- challenger counts per episode and variant (D = t2_best5 snapshot SELECTED, unchanged)
    snap_sel = C.read_json(C.SNAP / "t2_best5/selection.json")
    D = {int(x["episode"]): x["selected"] for x in snap_sel["episodes"]}
    counts = {}
    for v in ("A", "B"):
        counts[v] = {}
        for e in range(30):
            pool = [j for j in cand if jobs[j]["episode"] == e and v in jobs[j]["variants"]]
            pool += [n["job"] for n in new_jobs if n["episode"] == e and v in n["variants"].split(",")]
            counts[v][f"{e:02d}"] = {"before": len([j for j in cand if jobs[j]["episode"] == e and v in jobs[j]["variants"]
                                                    and j != D[e]]),
                                     "after": len([j for j in pool if j != D[e]])}  # fmt: skip
    bad = [e for e in range(30) if counts["A"][f"{e:02d}"]["before"] != C.A_CHALLENGERS[e]]
    if bad:
        die(f"variant-A challenger counts before A3 differ from PREREG 4.1 on {bad}")

    out = {
        "written_at": C.now(), "amendment": "A3", "T_a": f"{t_a_str} CST", "amendments_sha256": amend_sha,
        "prereg_sha256": want, "script_sha256": file_sha(Path(__file__).resolve()),
        "consensus_script_sha256": file_sha(ME / "scripts/t2cons_consensus.py"),
        "zuxing_tier2_run_folders": listing, "resolutions": resolutions, "new_jobs": new_jobs,
        "reused_jobs": reused, "defaults": {f"{e:02d}": D[e] for e in range(30)}, "challenger_counts": counts,
        "note": "own-input values only; no clean file opened (PREREG 5.7 path guard on every path)",
    }  # fmt: skip
    C.write_text(OUT / "candidates_a1.json", C.dump(out))
    lines = [f"A3 resolution at T_a = {t_a_str} CST (AMENDMENTS sha256 {amend_sha[:16]}), written {out['written_at']}",
             "own-input values only; no clean file opened", ""]  # fmt: skip
    for rec in resolutions:
        if rec["status"] != "resolved":
            lines.append(f"{rec['id']} {rec['run']} ep{rec['episode']:02d}: NO DESIGNATION ({rec['n_eligible']} eligible rows)")
            continue
        extra = ""
        if rec["owner"] == "zuxing":
            r = rec["row"]
            extra = (f"own-input add_auc {float(r['add_auc']):.4f} SP/MP {r['sp_sr']}/{r['mp_sr']} finished {r['finished_at']}; "
                     f"{rec['n_eligible']} eligible of {rec['n_rows']} rows; skipped {len(rec['skipped'])}")  # fmt: skip
        else:
            extra = f"verdict {json.dumps(rec.get('verdict'))[:300]}"
        lines.append(f"{rec['id']} ep{rec['episode']:02d} {rec['input']:5s} it {rec['iteration']:>6} sha {rec['sha256'][:16]} -> "
                     f"{rec['job']} [{rec['job_status']}{', settled now' if rec.get('settled_now') else ''}] | {extra}")  # fmt: skip
    lines += ["", f"new jobs ({len(new_jobs)}): " + ", ".join(n["job"] for n in new_jobs),
              f"reused jobs ({len(reused)}): " + ", ".join(reused), "",
              "challengers per episode (before -> after), D = t2_best5 snapshot SELECTED:"]  # fmt: skip
    for e in range(30):
        a, b = counts["A"][f"{e:02d}"], counts["B"][f"{e:02d}"]
        lines.append(f"  ep{e:02d} D {D[e]:32s} A {a['before']} -> {a['after']}   B {b['before']} -> {b['after']}")
    C.write_text(OUT / "candidates_a1.txt", "\n".join(lines) + "\n")
    shas = dict(copies)
    for rel in ("inputs_a1/candidates_a1.json", "inputs_a1/candidates_a1.txt"):
        shas[rel] = file_sha(ROOT / rel)
    C.write_text(OUT / "SNAPSHOT_A1.sha256", "".join(f"{h}  {rel}\n" for rel, h in sorted(shas.items())))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

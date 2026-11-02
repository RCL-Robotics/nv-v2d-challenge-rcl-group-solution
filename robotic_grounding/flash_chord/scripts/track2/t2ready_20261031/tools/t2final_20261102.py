#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""t2final_20261102.py: evidence for the training commit of Zuxing Lu's remote RTX 5090 checkpoints in the Tier-2 set
R6-CLOSEST, and the record update for the final Tier-2 commit (on top of a94614a, dev.haoran.t2ready_20261031).

Steps (CPU only; reads anything; writes only under W and, for `update`/`sums`, the record folder STAGE):
  copy      copy read-only from Zuxing's folder; for every file the sha256 of the source before, of the copy and of the
            source after must agree (COPIES_t2final.tsv):
              - <ZX>/runs/2026-11-01-round7-5090-ckpts (his copy of the remote 5090 runs, 2026-11-01): MANIFEST.tsv,
                SHA256SUMS and, of the 8 run folders behind the 6 remote checkpoints (ep19 has 3 stages), train.log,
                final.json, hydra/train_rl.log and hydra/.hydra/{config,overrides,hydra}.yaml;
              - the training logs of his 4 pro6000 checkpoints in the set (ep04, ep05, ep26, ep28).
  evidence  per remote checkpoint: the training log's first line (commit=...), the MANIFEST row, the run's Hydra record
            against the committed record, the run's final.json, the checkpoint's stored resolved config against the
            run's saved config and against the stored config of every checkpoint kept in the run folder, iteration and
            environment steps, the sha256 pins (and the exact file when the copy keeps it), the relocated file's tensor
            bytes; ep19's three stages and their resume links; the 4 pro6000 checkpoints; no robolab/ws031 checkpoint
            in the set; the Hydra config tree of the logged commit against this tree. Writes evidence/EVIDENCE_5090.json.
  update    put the copies and the evidence into STAGE (records/zuxing_runs/, records/hydra/ep19.../s1,
            verification/training_commit_evidence.json), update set/set_items.json and set/set_manifest.tsv (training
            commit per checkpoint, the ep19 stage-1 record) and SOURCES.tsv.
  sums      rewrite STAGE/SHA256SUMS (every file but itself).

  <flashchord python> -I -B t2final_20261102.py copy|evidence|update|sums
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

ME = Path("/root/autodl-tmp/haoran.zhang")
ZX = Path("/root/autodl-tmp/zuxing.lu")
ZX5090 = "/mnt/gfs/home/zuxinglu/workspace"
W = ME / "fc_runs/submissions/t2final_20261102"
WT = ME / "code/v2d_t2ready"
STAGE = WT / "robotic_grounding/flash_chord/scripts/track2/t2ready_20261031"
ZC = W / "zuxing_copies_t2final"
COPIES = W / "COPIES_t2final.tsv"
EVID = W / "evidence/EVIDENCE_5090.json"
R7 = "runs/2026-11-01-round7-5090-ckpts"
ROOTS = [(str(ME), "<ME>"), (str(ZX), "<ZX>"), (ZX5090, "<ZX5090>")]
ZX_HARD = "82837c8569c2461f263c28fce406c1b517f40d93"
ZX_SPEEDUP = "bc3b1bdf34f2f80e3f7d56625e3a0e1deca6a53e"

# stage key (job/sN) -> run folder inside his round-7 copy
RUNS = {
    "ep00_zxb_k3fix_60000/s1": "2026-10-24-tier2-runin-k3fix/ep00",
    "ep22_zxb_k3fix_78000/s1": "2026-10-24-tier2-runin-k3fix/ep22",
    "ep16_zxb_g2_34000/s1": "2026-10-28-tier2-runin-a5/ep16",
    "ep25_zxb_k3fix_48000/s1": "2026-10-28-tier2-runin-a5/ep25",
    "ep20_zxb_k3fix_30000/s1": "2026-10-28-tier2-5090-fill/ep20",
    "ep19_zxb_g2hard_248000/s1": "2026-10-25-tier2-runin-hard-g2/ep19",
    "ep19_zxb_g2hard_248000/s2": "2026-10-25-tier2-runin-hard-g2/cont/ep19",
    "ep19_zxb_g2hard_248000/s3": "2026-10-25-tier2-runin-hard-g2/cont2/ep19",
}
RUN_FILES = ("train.log", "final.json", "hydra/train_rl.log", "hydra/.hydra/config.yaml", "hydra/.hydra/overrides.yaml",
             "hydra/.hydra/hydra.yaml")
HY = ("config.yaml", "overrides.yaml", "hydra.yaml")
# his pro6000 checkpoints in the set: job -> (training log, the checkpoint in his run folder)
PRO = {
    "ep04_zxg2_421500": ("logs/2026-10-24-tier2-runin-g2/ep04.log", "runs/2026-10-24-tier2-runin-g2/ep04/model_421500.safetensors"),
    "ep28_zx_g2_102000": ("logs/2026-10-24-tier2-runin-g2/ep28.log", "runs/2026-10-24-tier2-runin-g2/ep28/model_102000.safetensors"),
    "ep05_zx_k3fix_88000": ("logs/2026-10-24-tier2-runin-k3fix-ep05-ep26/ep05.log",
                            "runs/2026-10-24-tier2-runin-k3fix-ep05-ep26/ep05/model_88000.safetensors"),
    "ep26_zx_k3fix_92000": ("logs/2026-10-24-tier2-runin-k3fix-ep05-ep26/ep26.log",
                            "runs/2026-10-24-tier2-runin-k3fix-ep05-ep26/ep26/model_92000.safetensors"),
}
REC_RUNS = "records/zuxing_runs/round7_5090_copy"
REC_PRO = "records/zuxing_runs/pro6000_logs"
COMMIT_RE = re.compile(r"\bcommit=([0-9a-f]{7,40})\b")
WT_RE = re.compile(r"/worktrees/([^/\s]+)/robotic_grounding/")
CONFIGS = "robotic_grounding/flash_chord/src/flash_chord/configs"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def ph(s):
    if isinstance(s, str):
        for a, b in ROOTS:
            s = s.replace(a, b)
    return s


def real(s):
    if isinstance(s, str):
        for a, b in ROOTS:
            s = s.replace(b, a)
    return s


def st_header(p: Path) -> tuple[dict, int]:
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), n


def st_meta(p: Path) -> dict:
    return st_header(p)[0].get("__metadata__", {}) or {}


def st_integrity(p: Path) -> dict:
    h, n = st_header(p)
    end = max((v["data_offsets"][1] for k, v in h.items() if k != "__metadata__"), default=0)
    return {"bytes": p.stat().st_size, "expected_bytes": 8 + n + end, "ok": p.stat().st_size == 8 + n + end}


def data_block_sha(p: Path) -> str:
    _, n = st_header(p)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        f.seek(8 + n)
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def stored_cfg(md: dict) -> dict:
    return json.loads(md.get("user.resolved_config_json") or md["resolved_config_json"])


def flat(d, prefix="") -> dict:
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{prefix}{k}."))
    elif isinstance(d, list) and d and all(isinstance(x, dict) for x in d):
        for i, v in enumerate(d):
            out.update(flat(v, f"{prefix}{i}."))
    else:
        out[prefix[:-1]] = d
    return out


def cfg_diff(a: dict, b: dict) -> list:
    fa, fb = flat({k: v for k, v in a.items() if k != "reference"}), flat({k: v for k, v in b.items() if k != "reference"})
    return [{"key": k, "a": ph(fa.get(k, "<missing>")), "b": ph(fb.get(k, "<missing>"))}
            for k in sorted(set(fa) | set(fb)) if fa.get(k, "<missing>") != fb.get(k, "<missing>")]


def git(*args) -> subprocess.CompletedProcess:
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    return subprocess.run(["git", "-C", str(WT), *args], capture_output=True, text=True, env=env)


def first_line(p: Path) -> str:
    with open(p, "rb") as f:
        return f.readline().decode(errors="replace").rstrip("\n")


# ----------------------------------------------------------------------------------------------------------- copy
def copy_list() -> list[str]:
    out = [f"{R7}/MANIFEST.tsv", f"{R7}/SHA256SUMS"]
    for run in RUNS.values():
        for f in RUN_FILES:
            if (ZX / R7 / run / f).is_file():
                out.append(f"{R7}/{run}/{f}")
            elif f != "final.json":
                raise SystemExit(f"missing {ZX / R7 / run / f}")
    out += [log for log, _ in PRO.values()]
    return out


def do_copy() -> None:
    if COPIES.exists():
        raise SystemExit(f"{COPIES} exists; copies are made once")
    rows = []
    for rel in copy_list():
        src, dst = ZX / rel, ZC / rel
        before = sha256(src)
        st = src.stat()
        if dst.exists():
            raise SystemExit(f"{dst} exists")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)  # content only: nothing is written in his folder, no chmod/utime on the source
        os.chmod(dst, 0o444)
        copy = sha256(dst)
        after = sha256(src)
        ok = before == copy == after
        rows.append({"source": str(src), "copy": str(dst), "bytes": st.st_size,
                     "source_mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
                     "sha256_source_before": before, "sha256_copy": copy, "sha256_source_after": after, "ok": ok,
                     "copied_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        if not ok:
            raise SystemExit(f"sha mismatch for {src}")
    W.mkdir(parents=True, exist_ok=True)
    with open(COPIES, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        wr.writeheader()
        wr.writerows(rows)
    print(f"copied {len(rows)} files, {sum(r['bytes'] for r in rows)} bytes, all ok -> {COPIES}")


# ------------------------------------------------------------------------------------------------------- evidence
def load_yaml(p: Path):
    import yaml  # noqa: PLC0415
    return yaml.safe_load(p.read_text())


def resolve_saved_config(p: Path) -> dict | str:
    from omegaconf import OmegaConf  # noqa: PLC0415
    try:
        return OmegaConf.to_container(OmegaConf.load(p), resolve=True, enum_to_str=True)
    except Exception as err:  # noqa: BLE001
        return f"not resolvable alone: {err!r}"


def manifest_rows() -> dict:
    with open(ZC / R7 / "MANIFEST.tsv", newline="") as f:
        return {r["run"]: r for r in csv.DictReader(f, delimiter="\t")}


def sums_table() -> dict:
    out = {}
    for line in (ZC / R7 / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            h, p = line.split(None, 1)
            out[p.strip().lstrip("*").removeprefix("./")] = h
    return out


def full_commit(short: str) -> dict:
    r = git("rev-parse", "--verify", "--quiet", f"{short}^{{commit}}")
    full = r.stdout.strip() if r.returncode == 0 else None
    out = {"short": short, "full": full}
    if full:
        out["ancestor_of_HEAD"] = git("merge-base", "--is-ancestor", full, "HEAD").returncode == 0
        out["hydra_configs_equal_HEAD"] = git("diff", "--quiet", "--no-renames", full, "HEAD", "--", CONFIGS).returncode == 0
        ns = git("diff", "--no-renames", "--name-status", full, "HEAD", "--", "robotic_grounding/flash_chord/src").stdout.split("\n")
        out["src_files_differing_vs_HEAD"] = [x for x in ns if x.strip()]
    return out


def do_evidence() -> None:
    items = json.loads((STAGE / "set/set_items.json").read_text())
    by_job = {e["job"]: e for e in items["episodes"]}
    man, sums = manifest_rows(), sums_table()
    head = git("rev-parse", "HEAD").stdout.strip()
    ev = {"what": "training-commit evidence for Zuxing Lu's remote RTX 5090 checkpoints of the Tier-2 set R6-CLOSEST",
          "made_at": time.strftime("%Y-%m-%d %H:%M:%S"), "tree_HEAD": head, "source_copy": f"<ZX>/{R7}",
          "copies_table": "COPIES_t2final.tsv", "placeholders": {b: a for a, b in ROOTS}, "checkpoints": [],
          "pro6000": [], "robolab": {}, "failures": []}
    fail = ev["failures"]
    commits = {}
    jobs = sorted({k.split("/")[0] for k in RUNS}, key=lambda j: by_job[j]["episode"])
    for job in jobs:
        ent = by_job[job]
        ck = ent["checkpoint"]
        rel = ck["relocated"]
        stages = ent["train"]["stages"]
        cr = {"episode": ent["episode"], "job": job, "iteration": ck["iteration"], "stages": []}
        as_trained = Path(real(rel["as_trained_copy"]))
        at_sha = sha256(as_trained)
        at_md = st_meta(as_trained)
        at_cfg = stored_cfg(at_md)
        cr["as_trained_copy"] = ph(str(as_trained))
        cr["as_trained_sha256"] = at_sha
        cr["as_trained_sha256_equals_pin"] = at_sha == rel["as_trained_sha256"]
        cr["pin_from_his_bundle_sums"] = rel.get("zuxing_sha256sums_line")
        if not cr["as_trained_sha256_equals_pin"]:
            fail.append(f"{job}: as-trained copy sha256 != pin")
        evaluated_src = Path(real(ck["file"]))
        cr["evaluated_source_checkpoint"] = ph(str(evaluated_src))
        cr["evaluated_source_sha256_equals_pin"] = sha256(evaluated_src) == ck["sha256"]
        cr["relocated_tensor_bytes_equal_as_trained"] = data_block_sha(evaluated_src) == data_block_sha(as_trained)
        rmd = st_meta(evaluated_src)
        cr["relocated_metadata_keys_differing"] = sorted(k for k in set(rmd) | set(at_md) if rmd.get(k) != at_md.get(k))
        cr["relocated_config_diff"] = cfg_diff(at_cfg, stored_cfg(rmd))
        if not (cr["evaluated_source_sha256_equals_pin"] and cr["relocated_tensor_bytes_equal_as_trained"]
                and set(cr["relocated_metadata_keys_differing"]) <= {"user.resolved_config_json"}
                and {d["key"] for d in cr["relocated_config_diff"]} <= {"task.parquet"}):
            fail.append(f"{job}: relocated file differs from the as-trained one beyond the parquet path")
        for st in stages:
            key = f"{job}/s{st['stage']}"
            run = RUNS[key]
            rd = ZC / R7 / run  # text records: our verified copies
            rz = ZX / R7 / run  # checkpoints: read in place in his folder (read only)
            last = st is stages[-1]
            sr = {"stage": st["stage"], "trained_the_checkpoint": last, "run_in_copy": f"<ZX>/{R7}/{run}",
                  "train_log": f"{REC_RUNS}/{run}/train.log"}
            # 1. the training log's first line and the MANIFEST row
            fl = first_line(rd / "train.log")
            m = COMMIT_RE.search(fl)
            sr["train_log_first_line"] = ph(fl)
            sr["train_log_commit"] = full_commit(m.group(1)) if m else None
            text = (rd / "train.log").read_text(errors="replace")
            sr["worktrees_named_in_log"] = sorted(set(WT_RE.findall(text)))
            hyd = load_yaml(rd / "hydra/.hydra/hydra.yaml")
            sr["hydra_runtime_cwd"] = ph(hyd.get("hydra", {}).get("runtime", {}).get("cwd"))
            mrow = man.get(run, {})
            sr["manifest_row"] = {k: ph(v) for k, v in mrow.items()}
            sr["manifest_commit"] = mrow.get("commit")
            ok_commit = bool(m) and sr["train_log_commit"]["full"] == ZX_HARD and mrow.get("commit") == m.group(1) \
                and sr["worktrees_named_in_log"] == ["dev.zuxing.hard_start"] and "worktrees/dev.zuxing.hard_start/" in (sr["hydra_runtime_cwd"] or "")
            sr["commit_ok"] = ok_commit
            if not ok_commit:
                fail.append(f"{key}: training commit evidence inconsistent ({fl!r}, manifest {mrow.get('commit')})")
            if m:
                commits.setdefault(m.group(1), []).append(key)
            # 2. the run's Hydra record vs set_items and the committed record
            ovs = load_yaml(rd / "hydra/.hydra/overrides.yaml")
            sr["overrides_equal_set_items"] = [o for o in ovs if not o.startswith("hydra.")] == [real(o) for o in st["overrides"]]
            if not sr["overrides_equal_set_items"]:
                fail.append(f"{key}: the run's overrides.yaml != set_items overrides")
            if st.get("record"):
                sr["committed_record"] = st["record"]
                sr["committed_record_bytes_equal_run"] = {h: sha256(STAGE / st["record"] / h) == sha256(rd / "hydra/.hydra" / h) for h in HY}
                if not all(sr["committed_record_bytes_equal_run"].values()):
                    fail.append(f"{key}: committed Hydra record != the run's")
            else:
                sr["committed_record"] = None
            out_dir = next((o.split("=", 1)[1] for o in ovs if o.startswith("output_dir=")), None)
            sr["output_dir"] = ph(out_dir)
            sr["manifest_matches_overrides"] = (
                f"experiment={mrow.get('experiment')}" in ovs and f"scene.support={mrow.get('support')}" in ovs
                and any(o.startswith("task.parquet=") and f"/{mrow.get('data_root')}/" in o and f"sequence_id={mrow.get('sequence')}/" in o for o in ovs))
            if not sr["manifest_matches_overrides"]:
                fail.append(f"{key}: MANIFEST row does not match the overrides")
            # 3. final.json, steps per iteration
            fj = rd / "final.json"
            fin = json.loads(fj.read_text()) if fj.is_file() else None
            sr["final_json"] = {k: ph(v) for k, v in fin.items()} if fin else None
            spi = fin["environment_steps"] // fin["iteration"] if fin else None
            # 4. the run's saved config resolved alone, and every checkpoint kept in the run folder
            saved = resolve_saved_config(rd / "hydra/.hydra/config.yaml")
            kept = []
            for f in sorted(rz.glob("model_*.safetensors"), key=lambda x: int(x.stem.split("_")[1])):
                md = st_meta(f)
                c = stored_cfg(md)
                it = int(md["user.iteration"])
                spi = spi or (int(md["user.environment_steps"]) // it if it else None)
                k = {"file": f.name, "iteration": it, "sha256_equals_his_SHA256SUMS": sha256(f) == sums.get(f"{run}/{f.name}"),
                     "integrity": st_integrity(f)["ok"],
                     "stored_config_equals_saved_config": (cfg_diff(c, saved) == []) if isinstance(saved, dict) else saved,
                     "stored_config_equals_checkpoint_config": cfg_diff(c, at_cfg) == [] if last else None,
                     "other_metadata_equal": all(md.get(x) == at_md.get(x) for x in ("format", "version", "user.coordinate_layout_sha256",
                                                                                     "user.policy_schema_json", "user.voc_scale")),
                     "environment_steps_per_iteration": int(md["user.environment_steps"]) / it if it else None}
                if not k["sha256_equals_his_SHA256SUMS"] or not k["integrity"]:
                    fail.append(f"{key}: kept {f.name} sha/integrity")
                if k["stored_config_equals_saved_config"] is not True:
                    fail.append(f"{key}: kept {f.name} stored config != the run's saved config")
                if last and not k["stored_config_equals_checkpoint_config"]:
                    fail.append(f"{key}: kept {f.name} stored config != the set checkpoint's")
                if not k["other_metadata_equal"]:
                    fail.append(f"{key}: kept {f.name} other metadata differ")
                kept.append(k)
            sr["kept_checkpoints"] = kept
            if not kept:
                fail.append(f"{key}: no checkpoint kept in the run folder {rz}")
            sr["environment_steps_per_iteration"] = spi
            if last:
                it = int(at_md["user.iteration"])
                es = int(at_md["user.environment_steps"])
                start = fin.get("start_iteration", 0) if fin else 0
                maxit = (fin or {}).get("max_iterations") or at_cfg["training"]["max_iterations"]
                sr["checkpoint"] = {
                    "iteration": it, "environment_steps": es,
                    "environment_steps_equal_iteration_times_run_rate": es == it * spi,
                    "iteration_in_run_range": start < it <= maxit, "run_range": [start, maxit],
                    "stored_config_equals_saved_config": (cfg_diff(at_cfg, saved) == []) if isinstance(saved, dict) else saved,
                    "stored_output_dir_is_this_run": at_cfg.get("output_dir") == out_dir,
                    "exact_file_in_copy": (rz / ck["name"]).is_file(),
                }
                if (rz / ck["name"]).is_file():
                    s = sha256(rz / ck["name"])
                    sr["checkpoint"]["exact_file_sha256_equals_pin"] = s == rel["as_trained_sha256"]
                    sr["checkpoint"]["exact_file_in_his_SHA256SUMS"] = sums.get(f"{run}/{ck['name']}") == rel["as_trained_sha256"]
                    if s != rel["as_trained_sha256"]:
                        fail.append(f"{key}: the copy's {ck['name']} != the evaluated bundle's as-trained file")
                else:
                    sr["checkpoint"]["note"] = (f"{ck['name']} is not kept in his copy (kept: {', '.join(x['file'] for x in kept)}); "
                                                "its sha256 is pinned by his bundle SHA256SUMS line and our as-trained copy")
                c = sr["checkpoint"]
                if not (c["environment_steps_equal_iteration_times_run_rate"] and c["iteration_in_run_range"]
                        and c["stored_config_equals_saved_config"] is True and c["stored_output_dir_is_this_run"]):
                    fail.append(f"{key}: checkpoint does not match the run ({c})")
            cr["stages"].append(sr)
        ev["checkpoints"].append(cr)

    # ep19: the resume links between its three stages
    st = {s["stage"]: s for s in by_job["ep19_zxb_g2hard_248000"]["train"]["stages"]}
    ov = {n: {o.split("=", 1)[0]: o.split("=", 1)[1] for o in st[n]["overrides"] if "=" in o} for n in st}
    base = ZC / R7 / RUNS["ep19_zxb_g2hard_248000/s1"]
    cont = ZC / R7 / RUNS["ep19_zxb_g2hard_248000/s2"]
    cont2 = ZC / R7 / RUNS["ep19_zxb_g2hard_248000/s3"]
    fb, f2 = json.loads((base / "final.json").read_text()), json.loads((cont2 / "final.json").read_text())
    cont_tail = [ln for ln in (cont / "train.log").read_text(errors="replace").splitlines() if ln.strip()][-3:]
    cont_prog = [ln for ln in (cont / "train.log").read_text(errors="replace").splitlines() if ln.startswith("training progress")]
    zbase, zcont = ZX / R7 / RUNS["ep19_zxb_g2hard_248000/s1"], ZX / R7 / RUNS["ep19_zxb_g2hard_248000/s2"]
    cfg_b = stored_cfg(st_meta(zbase / "model_180000.safetensors"))
    cfg_c = stored_cfg(st_meta(zcont / "model_224000.safetensors"))
    at19 = stored_cfg(st_meta(Path(real(by_job["ep19_zxb_g2hard_248000"]["checkpoint"]["relocated"]["as_trained_copy"]))))
    chain = {
        "s1_final": {"checkpoint": fb["checkpoint"], "iteration": fb["iteration"], "reason": fb["reason"]},
        "s2_resumes_from_s1_final": ov[2].get("training.checkpoint") == f"{ov[1]['output_dir']}/{fb['checkpoint']}",
        "s3_resumes_from_s2_224000": ov[3].get("training.checkpoint") == f"{ov[2]['output_dir']}/model_224000.safetensors",
        "s3_final_json_resumed_from": ph(f2.get("resumed_from")), "s3_start_iteration": f2.get("start_iteration"),
        "s3_final_json_resumed_from_equals_override": ph(f2.get("resumed_from")) == ov[3].get("training.checkpoint"),
        "s2_last_kept_checkpoint": "model_224000.safetensors",
        "s2_last_progress_line": cont_prog[-1] if cont_prog else None,
        "s2_log_tail": cont_tail,
        "s2_model_224000_integrity": st_integrity(zcont / "model_224000.safetensors"),
        "config_diff_s1_final_vs_set_checkpoint": cfg_diff(cfg_b, at19),
        "config_diff_s2_224000_vs_set_checkpoint": cfg_diff(cfg_c, at19),
        "checkpoint_248000_in_s3_range": f2["start_iteration"] < 248000 <= f2["max_iterations"],
    }
    allowed = {"output_dir", "training.resume", "training.checkpoint", "training.max_iterations"}
    chain["diffs_only_in_resume_keys"] = ({d["key"] for d in chain["config_diff_s1_final_vs_set_checkpoint"]} <= allowed
                                          and {d["key"] for d in chain["config_diff_s2_224000_vs_set_checkpoint"]} <= allowed)
    ok = (chain["s2_resumes_from_s1_final"] and chain["s3_resumes_from_s2_224000"] and chain["s3_final_json_resumed_from_equals_override"]
          and chain["s3_start_iteration"] == 224000 and chain["checkpoint_248000_in_s3_range"] and chain["diffs_only_in_resume_keys"]
          and chain["s2_model_224000_integrity"]["ok"])
    chain["ok"] = ok
    if not ok:
        fail.append("ep19 stage chain inconsistent")
    ev["ep19_chain"] = chain

    # his 4 pro6000 checkpoints: training-log first line, checkpoint = his run file
    for job, (log, ckrel) in PRO.items():
        ent = by_job[job]
        fl = first_line(ZC / log)
        m = COMMIT_RE.search(fl)
        his = ZX / ckrel
        r = {"episode": ent["episode"], "job": job, "train_log": f"{REC_PRO}/{log.split('logs/', 1)[1]}", "train_log_first_line": ph(fl),
             "train_log_commit": full_commit(m.group(1)) if m else None, "set_items_commit": ent["train"]["code_commit"],
             "his_run_checkpoint": ph(str(his)), "his_run_checkpoint_exists": his.is_file()}
        if his.is_file():
            r["his_run_checkpoint_sha256_equals_set_checkpoint"] = sha256(his) == ent["checkpoint"]["sha256"]
        r["ok"] = bool(m) and r["train_log_commit"]["full"] == ZX_SPEEDUP == ent["train"]["code_commit"] and r.get(
            "his_run_checkpoint_sha256_equals_set_checkpoint", False)
        if not r["ok"]:
            fail.append(f"{job}: pro6000 evidence inconsistent")
        ev["pro6000"].append(r)

    # robolab / ws031: none of those checkpoints is in the set
    set_shas = set()
    for e in items["episodes"]:
        set_shas |= {e["checkpoint"]["sha256"], e["export"]["sha256"]}
        if e["checkpoint"].get("relocated"):
            set_shas.add(e["checkpoint"]["relocated"]["as_trained_sha256"])
    other = []
    for rec in sorted((ME / "fc_runs/submissions").glob("t2_consensus_r*/RELOCATION_R*.json")):
        for row in json.loads(rec.read_text()).get("rows", []):
            src = f"{row.get('sha256sums_zuxing_line', '')} {row.get('original_copy', '')}"
            if re.search(r"robolab|ws031", src, re.I):
                other.append({"record": ph(str(rec)), "job": row["job"], "original_sha256": row["original_sha256"],
                              "relocated_sha256": row["relocated_sha256"], "source": ph(src.strip())})
    hit = [o for o in other if {o["original_sha256"], o["relocated_sha256"]} & set_shas]
    txt = (STAGE / "set/set_items.json").read_text()
    ev["robolab"] = {"robolab_or_ws031_checkpoints_in_earlier_rounds": other, "in_set": hit,
                     "set_items_mentions_robolab_or_ws031": len(re.findall(r"robolab|ws031", txt, re.I)),
                     "machines_in_set": sorted({e["train"]["machine"] for e in items["episodes"]}),
                     "ok": not hit and not re.search(r"robolab|ws031", txt, re.I)}
    if not ev["robolab"]["ok"]:
        fail.append("a robolab/ws031 checkpoint is in the set")
    ev["commits_seen_in_5090_logs"] = commits
    ev["ok"] = not fail
    EVID.parent.mkdir(parents=True, exist_ok=True)
    EVID.write_text(json.dumps(ev, indent=1) + "\n")
    # summary
    for cr in ev["checkpoints"]:
        last = cr["stages"][-1]
        print(f"ep{cr['episode']:02d} {cr['job']}: commit {[s['train_log_commit']['short'] if s['train_log_commit'] else None for s in cr['stages']]} "
              f"manifest {[s['manifest_commit'] for s in cr['stages']]} kept {len(last['kept_checkpoints'])} "
              f"cfg==run {last['checkpoint']['stored_config_equals_saved_config']} steps {last['checkpoint']['environment_steps_equal_iteration_times_run_rate']} "
              f"exact_in_copy {last['checkpoint']['exact_file_in_copy']} pin {cr['as_trained_sha256_equals_pin']} reloc_tensors {cr['relocated_tensor_bytes_equal_as_trained']}")
    print("ep19 chain ok:", chain["ok"])
    for r in ev["pro6000"]:
        print(f"ep{r['episode']:02d} {r['job']}: {r['train_log_commit']['short'] if r['train_log_commit'] else None} ok {r['ok']}")
    print("robolab ok:", ev["robolab"]["ok"], "| earlier robolab/ws031 rows:", len(other))
    print("commits:", {k: len(v) for k, v in commits.items()})
    print("ALL OK" if ev["ok"] else f"FAILURES: {fail}")


# --------------------------------------------------------------------------------------------------------- update
def put(src: Path, rel: str, rows: list) -> None:
    dst = STAGE / rel
    if dst.exists():
        if sha256(dst) != sha256(src):
            raise SystemExit(f"{rel} exists with other content")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        os.chmod(dst, 0o644)
    rows.append((rel, ph(str(src)), sha256(dst), dst.stat().st_size))


def do_update() -> None:
    ev = json.loads(EVID.read_text())
    if not ev["ok"]:
        raise SystemExit("evidence has failures; not updating")
    rows = []
    for rel in copy_list():
        src = ZC / rel
        if rel.startswith(R7 + "/"):
            put(src, f"{REC_RUNS}/{rel[len(R7) + 1:]}", rows)
        else:
            put(src, f"{REC_PRO}/{rel.split('logs/', 1)[1]}", rows)
    put(COPIES, "records/zuxing_runs/COPIES_t2final.tsv", rows)
    base = ZC / R7 / RUNS["ep19_zxb_g2hard_248000/s1"]
    for h in HY:
        put(base / "hydra/.hydra" / h, f"records/hydra/ep19_zxb_g2hard_248000/s1/{h}", rows)
    put(base / "final.json", "records/hydra/ep19_zxb_g2hard_248000/s1/final.json", rows)
    put(EVID, "verification/training_commit_evidence.json", rows)
    put(Path(__file__).resolve(), "tools/t2final_20261102.py", rows)

    # set_items.json
    p = STAGE / "set/set_items.json"
    items = json.loads(p.read_text())
    evj = {c["job"]: c for c in ev["checkpoints"]}
    evp = {r["job"]: r for r in ev["pro6000"]}
    for e in items["episodes"]:
        t = e["train"]
        if e["job"] in evj:
            c = evj[e["job"]]
            lines = "; ".join(f"stage {s['stage']}: '{s['train_log_first_line']}' ({s['train_log']})" for s in c["stages"])
            t["code_commit_until_20261031"] = t["code_commit"]
            t["code_commit"] = ZX_HARD
            t["code_commit_source"] = (
                f"first line of his RTX 5090 training log(s), copied 2026-11-02 from his run copy <ZX>/{R7}: {lines}. "
                "The same commit is in his MANIFEST.tsv; the log's code paths and Hydra runtime.cwd are under "
                "worktrees/dev.zuxing.hard_start (verification/training_commit_evidence.json). The launcher prints "
                "`git log -1` of that worktree, so uncommitted edits would not show.")
            t["code_in_this_history"] = "82837c8 (dev.zuxing.hard_start, merged 8c8106f); this tree's Hydra configs equal 82837c8's"
            for s in t["stages"]:
                se = next(x for x in c["stages"] if x["stage"] == s["stage"])
                s["run_copy"] = se["run_in_copy"]
                s["train_log"] = se["train_log"]
                s["train_log_first_line"] = se["train_log_first_line"]
                if s["record"] is None:  # ep19 stage 1: now the run's own record
                    s["record"] = "records/hydra/ep19_zxb_g2hard_248000/s1"
                    s["record_kind"] = "hydra"
                    s["record_source"] = f"<ZX>/{R7}/{RUNS['ep19_zxb_g2hard_248000/s1']}/hydra/.hydra"
                    s["record_note"] = ("until 2026-10-31 reconstructed from tmp/t2/run_g2_runin_hard.sh; the run's saved "
                                        "overrides.yaml equals the reconstruction")
            e["train_evidence"] = "verification/training_commit_evidence.json"
        elif e["job"] in evp:
            r = evp[e["job"]]
            t["train_log"] = r["train_log"]
            t["train_log_first_line"] = r["train_log_first_line"]
    items.setdefault("updates", []).append({
        "date": "2026-11-02", "tool": "tools/t2final_20261102.py update",
        "what": "training commit of the 6 remote RTX 5090 checkpoints (ep00, 16, 19, 20, 22, 25) evidenced from Zuxing's "
                "run copy (train.log first line commit=82837c8); ep19 stage-1 Hydra record added; training logs of his "
                "4 pro6000 checkpoints added. No checkpoint, evaluated file or trial changed."})
    p.write_text(json.dumps(items, indent=1) + "\n")
    with open(STAGE / "set/set_manifest.tsv", "w") as fp:
        fp.write("episode\tjob\towner\tinput\ttrain_commit\tmachine\tstages\tcheckpoint\tcheckpoint_sha256\texport_sha256\texporter_tree\trollout_sha256\n")
        for it in items["episodes"]:
            fp.write("\t".join(str(x) for x in (
                f"{it['episode']:02d}", it["job"], it["owner"], it["input"], (it["train"]["code_commit"] or "UNKNOWN")[:12],
                it["train"]["machine"], len(it["train"]["stages"]), it["checkpoint"]["name"], it["checkpoint"]["sha256"],
                it["export"]["sha256"], it["export"]["tree"], it["evaluated"]["rollout_sha256"])) + "\n")
    # SOURCES.tsv: add the new rows
    sp = STAGE / "SOURCES.tsv"
    lines = sp.read_text().splitlines()
    hdr, old = lines[0], [tuple(ln.split("\t")) for ln in lines[1:] if ln.strip()]
    have = {r[0] for r in old}
    new = [tuple(str(x) for x in r) for r in rows if r[0] not in have]
    with open(sp, "w") as fp:
        fp.write(hdr + "\n")
        for r in sorted(old + new):
            fp.write("\t".join(r) + "\n")
    print(f"update: {len(rows)} files placed ({len(new)} new SOURCES rows); set_items.json and set_manifest.tsv rewritten")


def do_sums() -> None:
    files = sorted(str(f.relative_to(STAGE)) for f in STAGE.rglob("*") if f.is_file() and f != STAGE / "SHA256SUMS")
    with open(STAGE / "SHA256SUMS", "w") as fp:
        for rel in files:
            fp.write(f"{sha256(STAGE / rel)}  {rel}\n")
    print(f"SHA256SUMS: {len(files)} files")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    {"copy": do_copy, "evidence": do_evidence, "update": do_update, "sums": do_sums}.get(step, lambda: sys.exit(__doc__))()

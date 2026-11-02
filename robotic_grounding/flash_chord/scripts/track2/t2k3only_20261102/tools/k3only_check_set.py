#!/usr/bin/env python3
"""k3only_check_set.py: step 2 of fc_runs/submissions/t2_k3only_20261102.

Does the Hydra config tree of commit d769f0c (extracted byte-exact to tree_d769f0c/, verified against git blob ids)
reproduce the stored training config of every one of the 30 K3FULL checkpoints, and does every evaluated (exported)
file come from that checkpoint? Pattern: t2ready_20261031/tools/check_set_t2.py (committed in d769f0c), with the
overrides read from each run's own records instead of committed set_items (16 of the 30 jobs are not in that file).

Per job (CHOICE.json episodes; manifest rows of t2_best5_20261024):
  0  checkpoint sha256 == manifest training_sha256; run commit = wandb-metadata git.commit (must be 61fa5d3)
  A  overrides (run/hydra/.hydra/overrides.yaml, else the W&B run args) compose with d769f0c configs (config train)
  B  hydra runs: composition == saved .hydra/config.yaml, all keys exact; overrides == saved overrides.yaml
  D  resolved composition == the checkpoint's stored resolved config (safetensors metadata), all keys exact
     ("reference" is runtime-added); a key only this tree composes is listed as new_keys (not a failure)
  P  stored task.parquet == manifest parquet; parquet sha256 == manifest; K3 / K3fix parquet sha256 listed in the
     committed data/ours/k3fix/READY
  E  export file exists, sha256 == export.json sha256 (16-hex prefix == own_input_evals.csv), export metadata
     user.export_source_sha256 == checkpoint sha256, export.json code_commit 61fa5d3 + stock_verified
  X  for jobs also in d769f0c's committed set_items.json: same overrides / checkpoint / export sha256
CPU only; reads only; writes only --out. Run: conda_envs/flashchord/bin/python -I -B k3only_check_set.py --out F
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import struct  # noqa: E402
from pathlib import Path  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
W = ME / "fc_runs/submissions/t2_k3only_20261102"
B5 = ME / "fc_runs/submissions/t2_best5_20261024"
CFG = W / "tree_d769f0c/robotic_grounding/flash_chord/src/flash_chord/configs"
COMMITTED = W / "tree_d769f0c/committed"
OURS_COMMIT = "61fa5d31ff8054bf1e5cf5375ef0957e54d2729b"
FORBIDDEN = ("local_not_for_upload", "clean_", "packed_")


def guard(p) -> Path:
    for s in (str(p), os.path.realpath(str(p))):
        if any(t in s.lower() for t in FORBIDDEN):
            raise PermissionError(f"refusing {p}")
    return Path(p)


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(guard(p), "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def st_meta(p: Path) -> dict:
    with open(guard(p), "rb") as fp:
        n = struct.unpack("<Q", fp.read(8))[0]
        return json.loads(fp.read(n)).get("__metadata__", {}) or {}


def stored_config(md: dict) -> dict:
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


def compare(ref: dict, comp: dict) -> dict:
    changed = [{"key": k, "stored": ref[k], "composed": comp[k]} for k in sorted(set(ref) & set(comp)) if ref[k] != comp[k]]
    return {"changed": changed, "missing": sorted(set(ref) - set(comp)), "new_keys": sorted(set(comp) - set(ref))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists")
    import yaml
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    choice = json.loads(guard(W / "CHOICE.json").read_text())
    rows = json.loads(guard(W / "inputs/manifest.json").read_text())["rows"]
    for line in guard(W / "inputs/manifest_stageB.jsonl").read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    byjob = {r["job"]: r for r in rows}
    ready = {ln.split()[0] for ln in guard(COMMITTED / "k3fix_READY").read_text().splitlines()
             if len(ln.split()) == 2 and len(ln.split()[0]) == 64}
    committed = {e["job"]: e for e in json.loads(guard(COMMITTED / "set_items.json").read_text())["episodes"]}
    ph = json.loads(guard(COMMITTED / "set_items.json").read_text())["placeholders"]

    def real(s):
        for k, v in ph.items():
            s = s.replace(k, v)
        return s

    rep = {"config_dir": str(CFG), "commit": "d769f0c257c12314c5f53123b8e81de010778c7f", "episodes": [], "failures": [],
           "notes": [], "k3fix_READY_parquets": len(ready)}
    with initialize_config_dir(config_dir=str(CFG.resolve()), version_base="1.3"):
        for ek in sorted(choice["episodes"]):
            ep = choice["episodes"][ek]
            e, job = int(ek), ep["job"]
            m = byjob[job]
            tag = f"ep{e:02d} {job}"
            er = {"episode": e, "job": job, "input": m["root_kind"], "family": m["family"]}
            ck = Path(m["training_checkpoint"])
            run = ck.parent
            er["checkpoint"] = str(ck)
            er["checkpoint_sha256"] = sha256(ck)
            if er["checkpoint_sha256"] != m["training_sha256"]:
                rep["failures"].append(f"{tag}: checkpoint sha256 != manifest")
            # commit from W&B metadata (online run-* or offline-run-*)
            wbs = sorted(list(run.glob("wandb/run-*")) + list(run.glob("wandb/offline-run-*")))
            er["wandb_runs"] = [str(x) for x in wbs]
            md = {}
            if len(wbs) != 1:
                rep["failures"].append(f"{tag}: {len(wbs)} W&B run folders")
            elif (wbs[0] / "files/wandb-metadata.json").is_file():  # online run: metadata file
                md = json.loads((wbs[0] / "files/wandb-metadata.json").read_text())
                er["train_commit"] = (md.get("git") or {}).get("commit")
                er["train_commit_source"] = "wandb-metadata.json git.commit"
            else:  # offline run: the run record inside run-*.wandb carries the git commit
                import re
                rec = sorted(wbs[0].glob("run-*.wandb"))
                head = rec[0].open("rb").read(2_000_000) if len(rec) == 1 else b""
                shas = set(re.findall(rb"[0-9a-f]{40}", head))
                er["train_commit"] = OURS_COMMIT if OURS_COMMIT.encode() in shas else None
                er["train_commit_source"] = f"offline W&B run record {rec[0].name if rec else None} (git commit string)"
            if er.get("train_commit") != OURS_COMMIT:
                rep["failures"].append(f"{tag}: training commit {er.get('train_commit')} != 61fa5d3")
            hyd = run / "hydra/.hydra"
            if hyd.is_dir():
                hy = yaml.safe_load((hyd / "hydra.yaml").read_text())
                er["hydra_runtime_cwd"] = hy["hydra"]["runtime"]["cwd"]
                if er["hydra_runtime_cwd"] != str(ME / "code/fc_r2"):
                    rep["failures"].append(f"{tag}: hydra runtime cwd {er['hydra_runtime_cwd']} != code/fc_r2")
            if hyd.is_dir():
                ovs_all = list(yaml.safe_load((hyd / "overrides.yaml").read_text()))
                er["record_kind"] = "hydra"
                wa = [o for o in md.get("args", [])]
                er["wandb_args_equal_hydra_overrides"] = sorted(wa) == sorted(ovs_all) if wa else None
            else:
                ovs_all = list(md["args"])
                er["record_kind"] = "wandb_args"
            ovs = [o for o in ovs_all if not o.startswith("hydra.")]
            er["overrides"] = ovs
            try:
                cfg = compose(config_name="train", overrides=ovs)
                comp = flat(OmegaConf.to_container(cfg, resolve=False))
                er["A_composes"] = True
            except Exception as err:  # noqa: BLE001
                er["A_composes"] = False
                rep["failures"].append(f"{tag}: overrides do not compose: {err!r}")
                rep["episodes"].append(er)
                continue
            if er["record_kind"] == "hydra":
                saved = flat(OmegaConf.to_container(OmegaConf.load(hyd / "config.yaml"), resolve=False))
                c = compare(saved, comp)
                er["B"] = {**c, "ok": not c["changed"] and not c["missing"]}
                if not er["B"]["ok"]:
                    rep["failures"].append(f"{tag}: B composition != saved config.yaml: changed "
                                           f"{[x['key'] for x in c['changed']][:6]} missing {c['missing'][:6]}")
            resolved = flat(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True))
            stored = stored_config(st_meta(ck))
            ref = stored.pop("reference", None)
            c = compare(flat(stored), resolved)
            er["D"] = {**c, "ok": not c["changed"] and not c["missing"], "n_keys": len(flat(stored))}
            if not er["D"]["ok"]:
                rep["failures"].append(f"{tag}: D composition != checkpoint stored config: changed "
                                       f"{[x['key'] for x in c['changed']][:6]} missing {c['missing'][:6]}")
            # P: training input
            tp = stored.get("task", {}).get("parquet")
            er["stored_task_parquet"] = tp
            er["stored_reference_parquet"] = (ref or {}).get("parquet") if isinstance(ref, dict) else None
            er["parquet_sha256"] = sha256(m["parquet"])
            er["P_ok"] = tp == m["parquet"] and er["parquet_sha256"] == m["parquet_sha256"]
            if m["root_kind"] in ("k3fix", "k3"):
                er["parquet_in_committed_k3fix_READY"] = er["parquet_sha256"] in ready
                er["P_ok"] &= er["parquet_in_committed_k3fix_READY"]
            if not er["P_ok"]:
                rep["failures"].append(f"{tag}: training parquet check failed")
            # E: evaluated file
            ex = json.loads((B5 / "exports" / job / "export.json").read_text())
            exf = Path(ex["file"])
            er["export_file"] = str(exf)
            er["export_exists"] = exf.is_file()
            if not exf.is_file():
                rep["failures"].append(f"{tag}: export file missing {exf}")
            else:
                er["export_sha256"] = sha256(exf)
                emd = st_meta(exf)
                er["E_ok"] = (er["export_sha256"] == ex["sha256"] == ex["destination_sha256"]
                              and ex["sha256"].startswith(ep["runs"]["1"]["export_sha256_16"])
                              and emd.get("user.export_source_sha256") == er["checkpoint_sha256"] == ex["source_sha256"]
                              and ex["code_commit"] == OURS_COMMIT and ex["stock_verified"] is True and ex["status"] == "ok")
                er["export_tolerance"] = ex.get("tolerance")
                er["export_max_rel_diff"] = ex.get("max_rel_diff")
                if not er["E_ok"]:
                    rep["failures"].append(f"{tag}: export check failed")
            # X: consistency with d769f0c committed set_items (R6 set) where the job is shared
            if job in committed:
                ce = committed[job]
                cov = [real(o) for o in ce["train"]["stages"][-1]["overrides"] if not o.startswith("hydra.")]
                er["X_in_committed_set_items"] = True
                er["X_ok"] = (sorted(cov) == sorted(ovs) and ce["checkpoint"]["sha256"] == er["checkpoint_sha256"]
                              and ce["export"]["sha256"] == er.get("export_sha256"))
                if not er["X_ok"]:
                    rep["failures"].append(f"{tag}: differs from d769f0c committed set_items entry")
            else:
                er["X_in_committed_set_items"] = False
            rep["episodes"].append(er)
    eps = rep["episodes"]
    rep["summary"] = {
        "episodes": len(eps),
        "A_compose": sum(bool(x.get("A_composes")) for x in eps),
        "hydra_records": sum(x.get("record_kind") == "hydra" for x in eps),
        "B_ok": sum(bool(x.get("B", {}).get("ok")) for x in eps),
        "D_ok": sum(bool(x.get("D", {}).get("ok")) for x in eps),
        "P_ok": sum(bool(x.get("P_ok")) for x in eps),
        "E_ok": sum(bool(x.get("E_ok")) for x in eps),
        "train_commit_61fa5d3": sum(x.get("train_commit") == OURS_COMMIT for x in eps),
        "in_committed_set_items": sum(bool(x.get("X_in_committed_set_items")) for x in eps),
        "X_ok": sum(bool(x.get("X_ok")) for x in eps),
        "new_keys_union": sorted({k for x in eps for s in ("B", "D") for k in x.get(s, {}).get("new_keys", [])}),
        "inputs": {k: sum(x["input"] == k for x in eps) for k in sorted({x["input"] for x in eps})},
    }
    rep["ok"] = not rep["failures"]
    a.out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps({"summary": rep["summary"], "failures": rep["failures"]}, indent=1))
    print("ALL OK" if rep["ok"] else f"{len(rep['failures'])} FAILED")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

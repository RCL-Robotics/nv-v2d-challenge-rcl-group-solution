#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""check_set_t3.py: does this checkout's Hydra config tree reproduce the training config of every checkpoint behind the
Tier-3 submission file FINAL_r (set/set_items.json, 30 episodes: 19 f0 items, 11 retry items), and does every evaluated
file come from it?

CPU only; reads only; writes only --out. Run with the FlashCHORD env python (hydra-core, omegaconf, PyYAML):

  python -I -B check_set_t3.py [--root <PLACEHOLDER>=<path> ...] [--files] [--hash] [--out report.json]

For every training stage of every episode (an f0 item has one stage, f0; a retry item has two: f0 to 44922, then the
C90 continuation retry_C90 to 90000, resumed from the f0 checkpoint):
  A  the stage's committed overrides compose with this tree's configs (src/flash_chord/configs, config "train", as
     scripts/train_rl.py does).
  B  the composition equals the run's saved Hydra config.yaml (set/hydra/<tag>/), key by key, all keys, exact; and the
     committed overrides equal the saved overrides.yaml.
  D  the composition, resolved (OmegaConf resolve=True, as train_rl.py stores it), equals the resolved config stored in
     the stage's checkpoint (set/checkpoint_config/; with --files also read from the checkpoint file itself and checked
     equal), all keys, exact ("reference" is added at run time from the loaded motion and is not a Hydra key).
For retry items also:
  R  the retry stage resumes from the f0 stage's checkpoint (fork.json: same sha256; training.checkpoint = the fork
     copy; reset.start_update = the f0 iteration) and its overrides contain the frozen recipe's termination overrides
     (evaluation/f0_retry/recipes/epNN.json).
For the stage that trained the evaluated policy:
  E  the evaluated (exported) file names that checkpoint as its export source (set_items; with --files from the export
     file's own metadata); with --hash, the checkpoints and the evaluated file match their sha256 pins.
A key that only this tree composes would be a new key (listed, not a failure). A changed or missing key is a failure.
Placeholders (<ME>, <HAN>) are replaced with the roots in set_items.json "placeholders" unless --root overrides one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent.parent  # .../scripts/track2/t3ready_20261102
CODE = HERE.parents[2]  # robotic_grounding/flash_chord


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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def st_meta(path: Path) -> dict:
    with open(path, "rb") as fp:
        n = struct.unpack("<Q", fp.read(8))[0]
        return json.loads(fp.read(n)).get("__metadata__", {}) or {}


def stored_config(md: dict) -> dict:
    return json.loads(md.get("user.resolved_config_json") or md["resolved_config_json"])


def compare(ref: dict, comp: dict) -> dict:
    changed = [{"key": k, "stored": ref[k], "composed": comp[k]} for k in sorted(set(ref) & set(comp)) if ref[k] != comp[k]]
    return {"changed": changed, "missing": sorted(set(ref) - set(comp)), "new_keys": sorted(set(comp) - set(ref))}


def ov_dict(ovs: list[str]) -> dict:
    out = {}
    for o in ovs:
        k, _, v = o.partition("=")
        out[k.lstrip("+~")] = v.strip("'\"")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--items", type=Path, default=HERE / "set/set_items.json")
    p.add_argument("--code", type=Path, default=CODE, help="flash_chord dir whose src/flash_chord/configs compose")
    p.add_argument("--root", action="append", default=[], help="<PLACEHOLDER>=<path>, e.g. <ME>=/data/me")
    p.add_argument("--files", action="store_true", help="also read the checkpoint and export files (paths in set_items)")
    p.add_argument("--hash", action="store_true", help="with --files: sha256 the files against the pins")
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    import yaml
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    data = json.loads(a.items.read_text())
    roots = dict(data.get("placeholders", {}))
    for x in a.root:
        k, v = x.split("=", 1)
        roots[k] = v

    def real(s):
        if isinstance(s, str):
            for k, v in roots.items():
                s = s.replace(k, v)
        return s

    cfg_dir = a.code / "src/flash_chord/configs"
    rep = {"items": str(a.items), "items_sha256": sha256(a.items), "config_dir": str(cfg_dir), "episodes": [],
           "failures": [], "notes": []}

    def fail(msg):
        rep["failures"].append(msg)

    with initialize_config_dir(config_dir=str(cfg_dir.resolve()), version_base="1.3"):
        for ent in data["episodes"]:
            e = ent["episode"]
            er = {"episode": e, "kind": ent["kind"], "train_commit": ent["train"]["code_commit"], "stages": []}
            stages = ent["train"]["stages"]
            for i, st in enumerate(stages):
                last = i == len(stages) - 1
                tag = f"ep{e:02d} {ent['kind']} stage {st['stage']}"
                sr = {"stage": st["stage"], "trained_the_evaluated_policy": last}
                ovs = [real(o) for o in st["overrides"] if not o.startswith("hydra.")]
                try:
                    cfg = compose(config_name="train", overrides=ovs)
                    comp = flat(OmegaConf.to_container(cfg, resolve=False))
                    sr["A_composes"], sr["A_n_keys"] = True, len(comp)
                except Exception as error:  # noqa: BLE001
                    sr["A_composes"], sr["A_error"] = False, repr(error)
                    fail(f"{tag}: overrides do not compose: {error!r}")
                    er["stages"].append(sr)
                    continue
                # B: the saved Hydra record
                rd = HERE / st["record"]
                saved = flat(OmegaConf.to_container(OmegaConf.load(rd / "config.yaml"), resolve=False))
                c = compare(saved, comp)
                saved_ovs = [str(o) for o in yaml.safe_load((rd / "overrides.yaml").read_text()) if not str(o).startswith("hydra.")]
                same_ovs = sorted(saved_ovs) == sorted(ovs)
                sr["B"] = {**c, "overrides_equal_saved": same_ovs, "ok": not c["changed"] and not c["missing"] and same_ovs}
                if not sr["B"]["ok"]:
                    fail(f"{tag}: B compose != saved config.yaml: changed {[x['key'] for x in c['changed']][:6]} "
                         f"missing {c['missing'][:6]} overrides_equal {same_ovs}")
                # D: the stage checkpoint's stored resolved config
                resolved = flat(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True))
                ck = st["checkpoint"]
                stored = json.loads((HERE / ck["stored_config"]).read_text())
                if a.files:
                    ckp = Path(real(ck["file"]))
                    if not ckp.is_file():
                        fail(f"{tag}: checkpoint file {ckp} not found")
                    else:
                        sr["stored_config_equals_file"] = stored_config(st_meta(ckp)) == stored
                        if not sr["stored_config_equals_file"]:
                            fail(f"{tag}: committed stored config != the checkpoint file's")
                        if a.hash:
                            sr["checkpoint_sha256_ok"] = sha256(ckp) == ck["sha256"]
                            if not sr["checkpoint_sha256_ok"]:
                                fail(f"{tag}: checkpoint sha256 != pin")
                st2 = dict(stored)
                st2.pop("reference", None)
                c = compare(flat(st2), resolved)
                sr["D"] = {**c, "ok": not c["changed"] and not c["missing"]}
                if not sr["D"]["ok"]:
                    fail(f"{tag}: D compose != stored config: changed {[x['key'] for x in c['changed']][:6]} "
                         f"missing {c['missing'][:6]}")
                fin = json.loads((HERE / st["final_json"]).read_text())
                sr["final_json_names_checkpoint"] = fin.get("checkpoint") == ck["name"] or (not last and ent["kind"] == "retry")
                sr["iteration_stored_vs_name"] = ck["iteration"] == int(ck["name"].split("_")[1].split(".")[0])
                if not sr["iteration_stored_vs_name"]:
                    fail(f"{tag}: stored iteration != checkpoint name")
                # R: the retry continuation
                if st["stage"] == "retry_C90":
                    fork = json.loads((HERE / st["fork_json"]).read_text())
                    prev = stages[i - 1]["checkpoint"]
                    od = ov_dict(ovs)
                    rec = json.loads((HERE / st["recipe"]).read_text())
                    term = rec.get("termination_overrides") or {}
                    if isinstance(term, dict):
                        term_items = [(k, str(v)) for k, v in term.items()]
                    else:
                        term_items = [(t.split("=", 1)[0].lstrip("+"), t.split("=", 1)[1]) for t in term]
                    r = {"fork_sha256_is_f0_checkpoint": fork["files"]["model"]["sha256"] == prev["sha256"],
                         "resume_from_fork_copy": od.get("training.checkpoint") == real(fork["files"]["model"]["dst"]),
                         "resume_true": od.get("training.resume") == "true",
                         "start_update_is_f0_iteration": od.get("reset.start_update") == str(prev["iteration"]),
                         "final_resumed_from_fork": fin.get("resumed_from") == real(fork["files"]["model"]["dst"]),
                         "recipe_termination_overrides_present": all(od.get(k) is not None and float(od[k]) == float(v)
                                                                     for k, v in term_items),
                         "recipe_termination_overrides": term_items}
                    sr["R"] = r
                    bad = [k for k, v in r.items() if v is False]
                    if bad:
                        fail(f"{tag}: R retry continuation check failed: {bad}")
                if not last:
                    er["stages"].append(sr)
                    continue
                # E: the evaluated file
                ex = ent["export"]
                sr["E_export_source_is_checkpoint"] = ex["export_source_sha256"] == ck["sha256"]
                if a.files:
                    exp = Path(real(ex["file"]))
                    if not exp.is_file():
                        fail(f"{tag}: evaluated file {exp} not found")
                        sr["E_export_source_is_checkpoint"] = False
                    else:
                        sr["E_export_source_is_checkpoint"] &= st_meta(exp).get("user.export_source_sha256") == ck["sha256"]
                        if a.hash:
                            sr["export_sha256_ok"] = sha256(exp) == ex["sha256"]
                            if not sr["export_sha256_ok"]:
                                fail(f"{tag}: evaluated file sha256 != pin")
                if not sr["E_export_source_is_checkpoint"]:
                    fail(f"{tag}: the evaluated file's export source is not the checkpoint")
                # the packed rollout names the evaluated file
                res = json.loads((HERE / ent["rollout"]["packed_copy"]).read_text())
                sr["rollout_names_evaluated_file"] = res.get("checkpoint_sha256") == ex["sha256"]
                if not sr["rollout_names_evaluated_file"]:
                    fail(f"{tag}: the packed rollout's checkpoint_sha256 is not the evaluated file")
                sr["rollout_reference_is_locked_input"] = res.get("parquet") == real(ent["input"]["parquet"])
                if not sr["rollout_reference_is_locked_input"]:
                    fail(f"{tag}: the packed rollout's reference parquet is not the training input")
                er["stages"].append(sr)
            rep["episodes"].append(er)

    stages = [s for er in rep["episodes"] for s in er["stages"]]
    pinned = [s for s in stages if s["trained_the_evaluated_policy"]]
    rep["summary"] = {
        "episodes": len(rep["episodes"]),
        "f0_items": sum(er["kind"] == "f0" for er in rep["episodes"]),
        "retry_items": sum(er["kind"] == "retry" for er in rep["episodes"]),
        "stages": len(stages),
        "stages_composing": sum(bool(s.get("A_composes")) for s in stages),
        "B_ok": sum(bool(s.get("B", {}).get("ok")) for s in stages),
        "D_ok": sum(bool(s.get("D", {}).get("ok")) for s in stages),
        "R_ok": sum(all(v is not False for v in s["R"].values()) for s in stages if "R" in s),
        "evaluated_policies": len(pinned),
        "E_ok": sum(bool(s.get("E_export_source_is_checkpoint")) for s in pinned),
        "rollouts_name_evaluated_file": sum(bool(s.get("rollout_names_evaluated_file")) for s in pinned),
        "new_keys_union": sorted({k for s in stages for x in ("B", "D") for k in s.get(x, {}).get("new_keys", [])}),
        "files_read": a.files,
        "hash_checked": a.files and a.hash,
    }
    rep["ok"] = not rep["failures"]
    text = json.dumps(rep, indent=1, default=str) + "\n"
    if a.out:
        a.out.write_text(text)
    print(json.dumps({"summary": rep["summary"], "failures": rep["failures"], "notes": rep["notes"]}, indent=1))
    print("ALL OK" if rep["ok"] else f"{len(rep['failures'])} FAILED")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

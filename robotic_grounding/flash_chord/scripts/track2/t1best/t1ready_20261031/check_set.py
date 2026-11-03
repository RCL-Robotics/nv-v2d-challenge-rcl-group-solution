#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""check_set.py: does this checkout's Hydra config tree reproduce the training config of every checkpoint of the
submitted Tier-1 set (set_items.json, 30 episodes), and does every evaluated file come from that checkpoint?

CPU only; reads only; writes only --out. Run with the FlashCHORD env python (hydra-core, omegaconf):

  python check_set.py --records <RUNS> [--records <more roots>] [--exports <T1_EXPORTS>] [--alias OLD=NEW ...]
                      [--hash] [--out report.json]

For every training stage of every episode's lineage (set_items.json "stages"; the last stage trained the checkpoint):
  A  compose the committed overrides (placeholders <MOTION>, <RUNS>, <ME> substituted) with this tree's configs
     (src/flash_chord/configs, config train) and compare with the run's saved hydra/.hydra/config.yaml.
     Run-local keys (output_dir, logging.*, task.parquet, training.checkpoint) are compared after mapping the
     machine roots (--alias) to the placeholders; the task parquet must name the item's sequence.
  B  compose the run's own saved hydra/.hydra/overrides.yaml (hydra.* dropped) with this tree's configs and compare
     with the saved config.yaml key by key, all keys, exact.
  C  the committed overrides equal the saved overrides.yaml (hydra.* and optional W&B entries dropped, roots mapped).
For the stage that trained the checkpoint, also:
  D  resolve that composition (OmegaConf resolve=True, as train_rl.py stores it) and compare with the
     resolved_config_json stored in the training checkpoint itself (all keys; "reference" is added at run time from
     the loaded motion and is not a Hydra key).
  E  the evaluated (exported) file names the training checkpoint as its export source (user.export_source_sha256);
     with --hash, both files' sha256 equal the pins.
A key that only this tree composes (a new option, at its default, from a later merged commit) is listed as a new
key and is not a failure. A changed or missing key is a failure. Records missing on this machine are reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[3]  # robotic_grounding/flash_chord
RUN_LOCAL = ("output_dir", "logging.", "task.parquet", "training.checkpoint")
SUBST = {"<MOTION>": "/MOTION", "<RUNS>": "/RUNS", "<ME>": "/ME", "<WANDB_MODE>": "disabled"}


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


def committed_overrides(stage: dict) -> list[str]:
    out = []
    for o in stage["overrides"]:
        if o.startswith("[") or o.startswith("hydra."):
            continue  # optional W&B entries of the C3r2 line; hydra.* are not config keys
        for k, v in SUBST.items():
            o = o.replace(k, v)
        out.append(o)
    return out


class Roots:
    def __init__(self, records: list[Path], exports: Path | None, aliases: list[tuple[str, str]]):
        self.records, self.exports, self.aliases = records, exports, aliases

    def find(self, rel: str) -> Path | None:
        for r in self.records:
            p = r / rel
            if p.exists():
                return p
        return None

    def norm(self, v):
        if not isinstance(v, str):
            return v
        for old, new in self.aliases:
            v = v.replace(old, new)
        return v


def compare(saved: dict, comp: dict, roots: Roots, run_local_loose: bool) -> dict:
    changed, run_local = [], []
    for k in sorted(set(saved) & set(comp)):
        if saved[k] == comp[k]:
            continue
        if run_local_loose and k.startswith(RUN_LOCAL):
            same = roots.norm(saved[k]) == roots.norm(comp[k])
            run_local.append({"key": k, "equal_after_root_mapping": same})
            continue
        changed.append({"key": k, "saved": saved[k], "composed": comp[k]})
    return {"changed": changed, "missing": sorted(set(saved) - set(comp)), "new_keys": sorted(set(comp) - set(saved)),
            "run_local_differences": run_local}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--items", type=Path, default=HERE / "set_items.json")
    p.add_argument("--code", type=Path, default=CODE, help="flash_chord dir whose src/flash_chord/configs compose")
    p.add_argument("--records", type=Path, action="append", default=[],
                   help="root holding <run>/hydra/.hydra and <run>/model_<N>.safetensors (repeatable, searched in order)")
    p.add_argument("--exports", type=Path, help="T1_EXPORTS root of the t1best payload (exports/<item>/model_<N>)")
    p.add_argument("--alias", action="append", default=[], help="OLD=NEW root mapping for run-local values")
    p.add_argument("--hash", action="store_true", help="also sha256 the checkpoints and evaluated files")
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    aliases = [tuple(x.split("=", 1)) for x in a.alias] + [(v, k) for k, v in SUBST.items() if k != "<WANDB_MODE>"]
    roots = Roots(a.records, a.exports, aliases)
    data = json.loads(a.items.read_text())
    cfg_dir = a.code / "src/flash_chord/configs"
    report = {"items": str(a.items), "items_sha256": sha256(a.items), "config_dir": str(cfg_dir), "episodes": [],
              "failures": [], "notes": []}

    def comp_of(ovs):
        cfg = compose(config_name="train", overrides=ovs)
        return cfg, flat(OmegaConf.to_container(cfg, resolve=False))

    with initialize_config_dir(config_dir=str(cfg_dir.resolve()), version_base="1.3"):
        for ent in data["episodes"]:
            e, item = ent["episode"], ent["item"]
            er = {"episode": e, "item": item, "stages": []}
            n = len(ent["stages"])
            for i, st in enumerate(ent["stages"], 1):
                last = i == n
                sr = {"stage": i, "run": st["run"], "code": st["code"], "trained_the_checkpoint": last}
                tag = f"ep{e:02d} {item} stage {i} ({st['run']})"
                hyd = roots.find(f"{st['run']}/hydra/.hydra")
                saved = saved_ovs = None
                owner = max(j for j, s2 in enumerate(ent["stages"], 1) if s2["run"] == st["run"])
                if owner != i:
                    # restarted in place: the run folder's single Hydra record was written by a later stage
                    sr["record"] = None
                    sr["record_belongs_to_stage"] = owner
                    report["notes"].append(f"{tag}: run folder restarted in place by stage {owner}; its saved record "
                                           f"is that stage's (compose only here)")
                elif hyd is not None and (hyd / "config.yaml").is_file():
                    sr["record"] = str(hyd)
                    saved = flat(OmegaConf.to_container(OmegaConf.load(hyd / "config.yaml"), resolve=False))
                    if (hyd / "overrides.yaml").is_file():
                        saved_ovs = [o for o in OmegaConf.to_container(OmegaConf.load(hyd / "overrides.yaml"))
                                     if not o.startswith("hydra.")]
                else:
                    sr["record"] = None
                    (report["failures"] if last else report["notes"]).append(f"{tag}: no saved hydra record found")
                # A: committed overrides
                ovs = committed_overrides(st)
                try:
                    _, comp = comp_of(ovs)
                    sr["A_composes"] = True
                    sr["A_n_keys"] = len(comp)
                except Exception as error:  # noqa: BLE001
                    sr["A_composes"] = False
                    sr["A_error"] = repr(error)
                    report["failures"].append(f"{tag}: committed overrides do not compose: {error!r}")
                    comp = None
                if comp is not None and saved is not None:
                    c = compare(saved, comp, roots, run_local_loose=True)
                    seq_ok = ent["sequence_id"] in str(saved.get("task.parquet", ""))
                    rl_ok = all(x["equal_after_root_mapping"] for x in c["run_local_differences"])
                    sr["A"] = {**c, "sequence_ok": seq_ok, "run_local_ok": rl_ok,
                               "ok": not c["changed"] and not c["missing"] and seq_ok}
                    if not sr["A"]["ok"]:
                        report["failures"].append(f"{tag}: A committed-overrides compose != saved config: "
                                                  f"changed {[x['key'] for x in c['changed']][:6]} missing {c['missing'][:6]} seq {seq_ok}")
                    if not rl_ok:
                        report["notes"].append(f"{tag}: run-local values differ after root mapping: "
                                               f"{[x['key'] for x in c['run_local_differences'] if not x['equal_after_root_mapping']]}")
                # B + C: the run's own saved overrides
                cfg_b = None
                if saved_ovs is not None:
                    try:
                        cfg_b, comp_b = comp_of(saved_ovs)
                        c = compare(saved, comp_b, roots, run_local_loose=False)
                        sr["B"] = {**c, "ok": not c["changed"] and not c["missing"]}
                        if not sr["B"]["ok"]:
                            report["failures"].append(f"{tag}: B saved-overrides compose != saved config: "
                                                      f"changed {[x['key'] for x in c['changed']][:6]} missing {c['missing'][:6]}")
                    except Exception as error:  # noqa: BLE001
                        sr["B"] = {"ok": False, "error": repr(error)}
                        report["failures"].append(f"{tag}: saved overrides do not compose: {error!r}")
                    mine = sorted(roots.norm(o) for o in ovs)
                    theirs = sorted(roots.norm(o) for o in saved_ovs if not o.startswith("["))
                    only_c = sorted(set(mine) - set(theirs))
                    only_s = sorted(set(theirs) - set(mine))
                    sr["C"] = {"only_committed": only_c, "only_saved": only_s, "ok": not only_c and not only_s}
                    if not sr["C"]["ok"]:
                        report["notes"].append(f"{tag}: C committed overrides differ from saved overrides.yaml: "
                                               f"committed-only {only_c[:4]} saved-only {only_s[:4]}")
                # D: the checkpoint's own stored resolved config
                if last:
                    ck = ent["checkpoint"]
                    ckp = roots.find(f"{ck['run']}/{ck['model']}")
                    if ckp is None:
                        report["failures"].append(f"{tag}: training checkpoint {ck['run']}/{ck['model']} not found")
                    else:
                        sr["checkpoint_file"] = str(ckp)
                        if a.hash:
                            sr["checkpoint_sha256_ok"] = sha256(ckp) == ck["model_sha256"]
                            if not sr["checkpoint_sha256_ok"]:
                                report["failures"].append(f"{tag}: checkpoint sha256 differs from the pin")
                        md = st_meta(ckp)
                        stored = json.loads(md.get("user.resolved_config_json") or md["resolved_config_json"])
                        stored.pop("reference", None)
                        if cfg_b is None:
                            cfg_b, _ = comp_of(ovs)
                            sr["D_composed_from"] = "committed overrides (no saved overrides.yaml)"
                        else:
                            sr["D_composed_from"] = "saved overrides.yaml"
                        try:
                            resolved = flat(OmegaConf.to_container(cfg_b, resolve=True, enum_to_str=True))
                            c = compare(flat(stored), resolved, roots, run_local_loose=False)
                            sr["D"] = {**c, "iteration": md.get("user.iteration"), "ok": not c["changed"] and not c["missing"]}
                            if not sr["D"]["ok"]:
                                report["failures"].append(f"{tag}: D resolved compose != checkpoint resolved_config_json: "
                                                          f"changed {[x['key'] for x in c['changed']][:6]} missing {c['missing'][:6]}")
                        except Exception as error:  # noqa: BLE001
                            sr["D"] = {"ok": False, "error": repr(error)}
                            report["failures"].append(f"{tag}: D resolve failed: {error!r}")
                    # E: the evaluated file
                    ex = ent["export"]
                    exp = (a.exports / ex["file"]) if ex["root"] == "T1_EXPORTS" and a.exports else roots.find(ex["file"])
                    if exp is None or not exp.is_file():
                        report["failures"].append(f"{tag}: evaluated file {ex['file']} not found")
                    else:
                        sr["export_file"] = str(exp)
                        src = st_meta(exp).get("user.export_source_sha256")
                        sr["E_export_source_is_checkpoint"] = src == ck["model_sha256"]
                        if not sr["E_export_source_is_checkpoint"]:
                            report["failures"].append(f"{tag}: evaluated file's export source {src} != checkpoint pin")
                        if a.hash:
                            sr["export_sha256_ok"] = sha256(exp) == ex["sha256"]
                            if not sr["export_sha256_ok"]:
                                report["failures"].append(f"{tag}: evaluated file sha256 differs from the pin")
                er["stages"].append(sr)
            report["episodes"].append(er)

    stages = [s for er in report["episodes"] for s in er["stages"]]
    pinned = [s for s in stages if s["trained_the_checkpoint"]]
    report["summary"] = {
        "episodes": len(report["episodes"]),
        "stages": len(stages),
        "stages_composing": sum(bool(s.get("A_composes")) for s in stages),
        "stages_with_saved_record": sum(s.get("record") is not None for s in stages),
        "stages_restarted_in_place": sum("record_belongs_to_stage" in s for s in stages),
        "A_ok": sum(bool(s.get("A", {}).get("ok")) for s in stages),
        "B_ok": sum(bool(s.get("B", {}).get("ok")) for s in stages),
        "C_ok": sum(bool(s.get("C", {}).get("ok")) for s in stages),
        "pinned_stages": len(pinned),
        "pinned_A_ok": sum(bool(s.get("A", {}).get("ok")) for s in pinned),
        "pinned_B_ok": sum(bool(s.get("B", {}).get("ok")) for s in pinned),
        "pinned_C_ok": sum(bool(s.get("C", {}).get("ok")) for s in pinned),
        "pinned_D_ok": sum(bool(s.get("D", {}).get("ok")) for s in pinned),
        "pinned_E_ok": sum(bool(s.get("E_export_source_is_checkpoint")) for s in pinned),
        "new_keys_union": sorted({k for s in stages for x in ("A", "B", "D") for k in s.get(x, {}).get("new_keys", [])}),
        "hash_checked": a.hash,
    }
    report["ok"] = not report["failures"]
    text = json.dumps(report, indent=1, default=str) + "\n"
    if a.out:
        a.out.write_text(text)
    print(json.dumps({"summary": report["summary"], "failures": report["failures"], "notes": report["notes"]}, indent=1))
    print("ALL OK" if report["ok"] else f"{len(report['failures'])} FAILED")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

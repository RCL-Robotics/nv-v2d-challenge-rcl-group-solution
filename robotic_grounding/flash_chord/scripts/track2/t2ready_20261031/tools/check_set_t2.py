#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""check_set_t2.py: does this checkout's Hydra config tree reproduce the training config of every checkpoint of the
Tier-2 submission set R6-CLOSEST (set/set_items.json, 30 episodes), and does every evaluated file come from it?

CPU only; reads only; writes only --out. Run with the FlashCHORD env python (hydra-core, omegaconf):

  python -I -B check_set_t2.py [--root <PLACEHOLDER>=<path> ...] [--files] [--hash] [--out report.json]

For every training stage of every episode (set_items.json train.stages; the last stage trained the checkpoint):
  A  the stage's committed overrides compose with this tree's configs (src/flash_chord/configs, config train).
  B  if the run's saved Hydra record is committed (records/hydra/<job>/s<N>/): the composition equals its
     config.yaml key by key, all keys, exact; and the committed overrides equal its overrides.yaml.
For the stage that trained the checkpoint, also:
  D  resolve that composition (OmegaConf resolve=True, as train_rl.py stores it) and compare it with the resolved
     config stored in the training checkpoint (records/checkpoint_config/<job>.resolved_config.json; with --files
     also read from the checkpoint file itself and checked equal), all keys, exact ("reference" is added at run time
     from the loaded motion and is not a Hydra key). For a relocated checkpoint (Zuxing's remote ones), D is exact
     against the as-trained copy's stored config, and against the relocated file's config the only differences
     allowed are task.parquet / reference.parquet = the recorded old -> new path.
  E  the evaluated (exported) file names the checkpoint as its export source (set_items; with --files from the
     export file's metadata); with --hash, the checkpoint, the as-trained copy and the evaluated file match their
     sha256 pins.
A key that only this tree composes (an option of a later merged commit, at its default) is a new key, listed and
not a failure. A changed or missing key is a failure. Placeholders (<ME>, <ZX>, <ZX5090>) are replaced with the
roots in set_items.json "placeholders" unless --root overrides one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent.parent  # .../scripts/track2/t2ready_20261031
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--items", type=Path, default=HERE / "set/set_items.json")
    p.add_argument("--code", type=Path, default=CODE, help="flash_chord dir whose src/flash_chord/configs compose")
    p.add_argument("--root", action="append", default=[], help="<PLACEHOLDER>=<path>, e.g. <ME>=/data/me")
    p.add_argument("--files", action="store_true", help="also read the checkpoint and export files (paths in set_items)")
    p.add_argument("--hash", action="store_true", help="with --files: sha256 the files against the pins")
    p.add_argument("--out", type=Path)
    a = p.parse_args()

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

    def comp_of(ovs):
        cfg = compose(config_name="train", overrides=ovs)
        return cfg, flat(OmegaConf.to_container(cfg, resolve=False))

    with initialize_config_dir(config_dir=str(cfg_dir.resolve()), version_base="1.3"):
        for ent in data["episodes"]:
            e, job = ent["episode"], ent["job"]
            er = {"episode": e, "job": job, "owner": ent["owner"], "train_commit": ent["train"]["code_commit"], "stages": []}
            stages = ent["train"]["stages"]
            for i, st in enumerate(stages):
                last = i == len(stages) - 1
                tag = f"ep{e:02d} {job} stage {st['stage']}"
                sr = {"stage": st["stage"], "record_kind": st["record_kind"], "trained_the_checkpoint": last}
                ovs = [real(o) for o in st["overrides"] if not o.startswith("hydra.")]
                try:
                    cfg, comp = comp_of(ovs)
                    sr["A_composes"] = True
                    sr["A_n_keys"] = len(comp)
                except Exception as error:  # noqa: BLE001
                    sr["A_composes"] = False
                    sr["A_error"] = repr(error)
                    rep["failures"].append(f"{tag}: overrides do not compose: {error!r}")
                    er["stages"].append(sr)
                    continue
                # B: the saved Hydra record
                if st.get("record") and st["record_kind"] == "hydra":
                    rd = HERE / st["record"]
                    saved = flat(OmegaConf.to_container(OmegaConf.load(rd / "config.yaml"), resolve=False))
                    c = compare(saved, comp)
                    saved_ovs = [o for o in OmegaConf.to_container(OmegaConf.load(rd / "overrides.yaml")) if not o.startswith("hydra.")]
                    same_ovs = sorted(saved_ovs) == sorted(ovs)
                    sr["B"] = {**c, "overrides_equal_saved": same_ovs, "ok": not c["changed"] and not c["missing"] and same_ovs}
                    if not sr["B"]["ok"]:
                        rep["failures"].append(f"{tag}: B compose != saved config.yaml: changed {[x['key'] for x in c['changed']][:6]} "
                                               f"missing {c['missing'][:6]} overrides_equal {same_ovs}")
                elif st.get("record") is None:
                    rep["notes"].append(f"{tag}: no saved record ({st['record_kind']}); composed only")
                if not last:
                    er["stages"].append(sr)
                    continue
                # D: the checkpoint's stored resolved config
                resolved = flat(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True))
                ck = ent["checkpoint"]
                stored = json.loads((HERE / ck["stored_config"]).read_text())
                if a.files:
                    ckp = Path(real(ck["file"]))
                    if not ckp.is_file():
                        rep["failures"].append(f"{tag}: checkpoint file {ckp} not found")
                    else:
                        sr["stored_config_equals_file"] = stored_config(st_meta(ckp)) == stored
                        if not sr["stored_config_equals_file"]:
                            rep["failures"].append(f"{tag}: committed stored config != the checkpoint file's")
                        if a.hash:
                            sr["checkpoint_sha256_ok"] = sha256(ckp) == ck["sha256"]
                            if not sr["checkpoint_sha256_ok"]:
                                rep["failures"].append(f"{tag}: checkpoint sha256 != pin")
                rel = ck.get("relocated")
                if rel:
                    trained = json.loads((HERE / rel["as_trained_stored_config"]).read_text())
                    trained.pop("reference", None)
                    c = compare(flat(trained), resolved)
                    sr["D_as_trained"] = {**c, "ok": not c["changed"] and not c["missing"]}
                    if not sr["D_as_trained"]["ok"]:
                        rep["failures"].append(f"{tag}: D compose != as-trained stored config: changed "
                                               f"{[x['key'] for x in c['changed']][:6]} missing {c['missing'][:6]}")
                    st2 = dict(stored)
                    st2.pop("reference", None)
                    c2 = compare(flat(st2), resolved)
                    allowed = {"task.parquet": (real(rel["new_parquet_path"]), real(rel["old_parquet_path"]))}
                    bad = [x for x in c2["changed"] if (x["key"] not in allowed) or
                           (x["stored"], x["composed"]) != allowed[x["key"]]]
                    sr["D_relocated"] = {"changed": c2["changed"], "missing": c2["missing"], "unexpected": bad,
                                         "ok": not bad and not c2["missing"]}
                    if not sr["D_relocated"]["ok"]:
                        rep["failures"].append(f"{tag}: D relocated config differs beyond the parquet path: {[x['key'] for x in bad][:6]}")
                    trained_ref = json.loads((HERE / rel["as_trained_stored_config"]).read_text()).get("reference", {})
                    reloc_ref = stored.get("reference", {})
                    sr["reference_parquet_relocated"] = (trained_ref.get("parquet") == real(rel["old_parquet_path"])
                                                         and reloc_ref.get("parquet") == real(rel["new_parquet_path"]))
                    if a.files and a.hash:
                        sr["as_trained_sha256_ok"] = sha256(Path(real(rel["as_trained_copy"]))) == rel["as_trained_sha256"]
                        if not sr["as_trained_sha256_ok"]:
                            rep["failures"].append(f"{tag}: as-trained copy sha256 != pin")
                    sr["D_ok"] = sr["D_as_trained"]["ok"] and sr["D_relocated"]["ok"]
                else:
                    st2 = dict(stored)
                    st2.pop("reference", None)
                    c = compare(flat(st2), resolved)
                    sr["D"] = {**c, "ok": not c["changed"] and not c["missing"]}
                    sr["D_ok"] = sr["D"]["ok"]
                    if not sr["D_ok"]:
                        rep["failures"].append(f"{tag}: D compose != stored config: changed {[x['key'] for x in c['changed']][:6]} "
                                               f"missing {c['missing'][:6]}")
                sr["iteration_stored_vs_name"] = ck["iteration"] == int(ck["name"].split("_")[1].split(".")[0])
                # E: the evaluated file
                ex = ent["export"]
                sr["E_export_source_is_checkpoint"] = ex["export_source_sha256"] == ck["sha256"]
                if a.files:
                    exp = Path(real(ex["file"]))
                    if not exp.is_file():
                        rep["failures"].append(f"{tag}: evaluated file {exp} not found")
                        sr["E_export_source_is_checkpoint"] = False
                    else:
                        sr["E_export_source_is_checkpoint"] &= st_meta(exp).get("user.export_source_sha256") == ck["sha256"]
                        if a.hash:
                            sr["export_sha256_ok"] = sha256(exp) == ex["sha256"]
                            if not sr["export_sha256_ok"]:
                                rep["failures"].append(f"{tag}: evaluated file sha256 != pin")
                if not sr["E_export_source_is_checkpoint"]:
                    rep["failures"].append(f"{tag}: the evaluated file's export source is not the checkpoint")
                er["stages"].append(sr)
            rep["episodes"].append(er)

    stages = [s for er in rep["episodes"] for s in er["stages"]]
    pinned = [s for s in stages if s["trained_the_checkpoint"]]
    rep["summary"] = {
        "episodes": len(rep["episodes"]),
        "stages": len(stages),
        "stages_composing": sum(bool(s.get("A_composes")) for s in stages),
        "stages_with_saved_hydra_record": sum("B" in s for s in stages),
        "B_ok": sum(bool(s.get("B", {}).get("ok")) for s in stages),
        "pinned_stages": len(pinned),
        "pinned_D_ok": sum(bool(s.get("D_ok")) for s in pinned),
        "pinned_relocated": sum("D_relocated" in s for s in pinned),
        "pinned_E_ok": sum(bool(s.get("E_export_source_is_checkpoint")) for s in pinned),
        "new_keys_union": sorted({k for s in stages for x in ("B", "D", "D_as_trained") for k in s.get(x, {}).get("new_keys", [])}),
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

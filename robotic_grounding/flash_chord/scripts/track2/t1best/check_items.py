#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""check_items.py: CPU-only checks of the best-set configs against this checkout (no training, no GPU).

  python check_items.py                          configs: schema, one set item per episode, digests, placeholders,
                                                 best_set/manifest.tsv and configs/pinned_stages.tsv agree
  python check_items.py --compose                also Hydra-compose the overrides of the stage that trained each
                                                 pinned checkpoint, with this tree's configs (FlashCHORD env python)
  python check_items.py --compose --records <dir>   and compare each composition with the run's saved Hydra config
                                                 <dir>/<run>/hydra/.hydra/config.yaml (the transferred run records,
                                                 normally <T1_RUNS>)
  python check_items.py --forks                  also compose a plain fork (arm C) of every item
  python check_items.py --payload [--hash]       check that the copied payload holds every file a fork or a trial
                                                 reads where the configs say: the pinned pair at
                                                 <T1_RUNS>/<pinned.run>/, the export at <T1_EXPORTS>/<export.file>
                                                 and the fork-origin pair of each T1fix stage (sizes; --hash:
                                                 sha256 of the pinned files and exports)
A key that only this tree composes (a new option with its default, from a merged commit) is listed, not failed. A
value that differs from the saved config fails, except the keys a fork or a new run changes by design (paths,
logging, training length and resume).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1common as C  # noqa: E402

HEX64 = re.compile(r"^[0-9a-f]{64}$")
ALWAYS_DIFF = ("output_dir", "logging.", "training.max_iterations", "training.resume", "training.checkpoint",
               "training.early_stop_help_zero_tail", "task.parquet")


def schema(items: dict) -> list[str]:
    bad = []
    if len(items) != 31:
        bad.append(f"{len(items)} item configs, expected 31")
    try:
        C.set_items(items)
    except SystemExit as error:
        bad.append(str(error))
    for it, c in items.items():
        p, e = c["pinned"], c["export"]
        for key, val in (("model_sha256", p["model_sha256"]), ("rsi_sha256", p["rsi_sha256"]), ("export sha256", e["sha256"])):
            if not HEX64.match(str(val)):
                bad.append(f"{it}: {key} {val!r} is not a sha256")
        if p["model"] != f"model_{p['iteration']}.safetensors" or not e["file"].endswith(p["model"]):
            bad.append(f"{it}: pinned file names do not match iteration {p['iteration']}")
        if c["experiment"] not in ("g1_recon_body_coord_rsi_r2", "g1_recon_body_coord_rsi_r2_runin"):
            bad.append(f"{it}: unexpected experiment {c['experiment']}")
        last = c["stages"][-1]["overrides"]
        if f"experiment={c['experiment']}" not in last:
            bad.append(f"{it}: the last stage does not use experiment {c['experiment']}")
        if not any(f"sequence_id={c['sequence_id']}/" in o for o in last):
            bad.append(f"{it}: the last stage's task.parquet is not sequence {c['sequence_id']}")
        for s in c["stages"]:
            for o in s["overrides"]:
                if re.search(r"(^|[='])/", o.replace("<MOTION>", "M").replace("<RUNS>", "R")):
                    bad.append(f"{it}: absolute path in stage {s['run']}: {o}")
        for o in c["fork"]["inherit"]:
            if o not in last:
                bad.append(f"{it}: inherited override {o} is not in the last stage")
    with (C.BEST_SET / "manifest.tsv").open() as fp:
        man = {r["item"]: r for r in csv.DictReader(fp, delimiter="\t")}
    for it, c in items.items():
        r = man.get(it)
        if r is None:
            bad.append(f"{it}: missing from best_set/manifest.tsv")
            continue
        if (r["checkpoint_sha256"], r["export_sha256"], r["in_set"]) != (c["pinned"]["model_sha256"], c["export"]["sha256"],
                                                                       "yes" if c["in_set"] else "no"):
            bad.append(f"{it}: best_set/manifest.tsv disagrees with configs/items/{it}.json")
    with (C.CONFIGS / "pinned_stages.tsv").open() as fp:
        stages = {}
        for r in csv.DictReader(fp, delimiter="\t"):
            stages[r["item"]] = stages.get(r["item"], 0) + 1
    for it, c in items.items():
        if stages.get(it) != len(c["stages"]):
            bad.append(f"{it}: configs/pinned_stages.tsv has {stages.get(it)} stages, the config {len(c['stages'])}")
    return bad


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


def stage_overrides(stage: dict, motion: str, runs: str) -> list[str]:
    out = []
    for o in stage["overrides"]:
        if o.startswith("[") or o.startswith("hydra."):
            continue  # optional W&B keys of the C3r2 line; hydra.* are not config keys
        o = o.replace("<WANDB_MODE>", "disabled").replace("<MOTION>", motion).replace("<RUNS>", runs)
        out.append(o)
    return out


def compose_all(items: dict, code: Path, records: Path | None, forks: bool) -> list[str]:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    bad = []
    motion, runs = "/motion", "/runs"
    with initialize_config_dir(config_dir=str(code / "src/flash_chord/configs"), version_base="1.3"):
        for it, c in items.items():
            stage = c["stages"][-1]
            ovs = stage_overrides(stage, motion, runs)
            try:
                comp = flat(OmegaConf.to_container(compose(config_name="train", overrides=ovs), resolve=False))
            except Exception as error:  # noqa: BLE001
                bad.append(f"{it}: compose failed: {error!r}")
                continue
            line = f"{it}: composes ({len(comp)} keys)"
            if records is not None:
                saved_path = records / stage["run"] / "hydra/.hydra/config.yaml"
                if not saved_path.is_file():
                    line += f"; no saved config at {saved_path}"
                else:
                    saved = flat(OmegaConf.to_container(OmegaConf.load(saved_path), resolve=False))
                    changed = sorted(k for k in set(saved) & set(comp) if saved[k] != comp[k]
                                     and not k.startswith(ALWAYS_DIFF))
                    only_new = sorted(k for k in set(comp) - set(saved) if not k.startswith(ALWAYS_DIFF))
                    only_old = sorted(k for k in set(saved) - set(comp) if not k.startswith(ALWAYS_DIFF))
                    seq_ok = c["sequence_id"] in str(saved.get("task.parquet", ""))
                    line += (f"; vs saved {saved_path.relative_to(records)}: changed {changed or 'none'}; new keys "
                             f"(defaults) {only_new or 'none'}; missing {only_old or 'none'}; sequence ok {seq_ok}")
                    if changed or only_old or not seq_ok:
                        bad.append(line)
            print(line)
            if forks:
                fk = c["fork"]
                fovs = [f"experiment={fk['experiment']}", f"task.parquet='{motion}/x.parquet'",
                        f"scene.support={str(fk['scene_support']).lower()}", f"reset.seed={fk['reset_seed']}",
                        "training.resume=true", f"training.checkpoint={runs}/model_1.safetensors",
                        "training.max_iterations=2", *fk["inherit"]]
                try:
                    compose(config_name="train", overrides=fovs)
                    print(f"{it}: fork (arm C, inherited {fk['inherit'] or 'none'}) composes")
                except Exception as error:  # noqa: BLE001
                    bad.append(f"{it}: fork compose failed: {error!r}")
    return bad


def payload(items: dict, site, full_hash: bool) -> list[str]:
    bad = []
    n = 0

    def need(path: Path, size, sha, what: str, note=None) -> None:
        nonlocal n
        n += 1
        if not path.is_file():
            if note:
                print(f"note {what}: {path} is not in the payload ({note})")
            else:
                bad.append(f"{what}: missing {path}")
        elif size is not None and path.stat().st_size != int(size):
            bad.append(f"{what}: size {path.stat().st_size} != {size}: {path}")
        elif full_hash and sha and C.sha256(path) != sha:
            bad.append(f"{what}: sha256 differs from the pin: {path}")

    for it, c in items.items():
        p, e = c["pinned"], c["export"]
        run = site.runs / p["run"]
        need(run / p["model"], p.get("model_size"), p["model_sha256"], f"{it} pinned model")
        need(run / p["rsi"], p.get("rsi_size"), p["rsi_sha256"], f"{it} pinned rsi", p.get("rsi_note"))
        need(site.exports / e["file"], e.get("size"), e["sha256"], f"{it} export")
        last = c["stages"][-1]
        fork = bool(last.get("fork_pair"))  # T1fix stages start from a transferred fork pair (group E_fork_origins)
        what = f"{it} fork origin" if fork else f"{it} resume point"
        note = None if fork else "the original run's own resume point, not transferred; forks start from the pinned pair"
        for o in last["overrides"]:
            if o.startswith("training.checkpoint=<RUNS>/"):
                model = Path(C.resolve(o.split("=", 1)[1].strip("'\""), site))
                need(model, None, None, what, note)
                need(model.with_name(model.name.replace("model_", "rsi_", 1)), None, None, f"{what} rsi", note)
    print(f"payload: {n} files under T1_RUNS={site.runs} and T1_EXPORTS={site.exports} "
          f"({'size and sha256' if full_hash else 'presence and size'}), {len(bad)} problem(s)")
    return bad


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_site_arg(p)
    p.add_argument("--compose", action="store_true")
    p.add_argument("--forks", action="store_true")
    p.add_argument("--records", type=Path, help="root with <run>/hydra/.hydra/config.yaml (e.g. T1_RUNS)")
    p.add_argument("--payload", action="store_true", help="check the copied payload at the site's T1_RUNS/T1_EXPORTS")
    p.add_argument("--hash", action="store_true", help="with --payload: also compare sha256 with the pins")
    a = p.parse_args()
    site = C.load_site(a.site)
    items = C.load_items()
    bad = schema(items)
    print(f"schema: {len(items)} item configs, {len(bad)} problem(s)")
    if a.compose or a.forks:
        bad += compose_all(items, site.code, a.records, a.forks)
    if a.payload:
        bad += payload(items, site, a.hash)
    for b in bad:
        print("FAIL", b)
    print("ALL OK" if not bad else f"{len(bad)} FAILED")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())

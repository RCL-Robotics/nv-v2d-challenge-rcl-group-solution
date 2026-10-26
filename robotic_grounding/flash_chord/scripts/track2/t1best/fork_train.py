#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""fork_train.py: continue (fork) the training of a pinned Tier-1 item, as the T1fix experiments did.

Portable version of the server launchers t1fix_launch.py (stage 1), t1fix2_launch.py (stage 2) and t1fix5_launch.py
(ep28 wave 2). Their mechanics are kept: a sha256-checked copy of the source pair (model_<N> + rsi_<N>) outside the
run directory, the source's experiment / task / scene.support / reset.seed, an absolute training.max_iterations of
<N + plus>, training.resume from the copied pair, the help-zero early stop off, W&B off with a local metrics.jsonl,
and (by default) a hull cache seeded with the stock runtime's CoACD hulls so that training sees the evaluator's
collision geometry. Their machine-specific parts (GPU allow-lists, MPS refusal, NUMA slices, queue folders and the
slot sharing with other drivers) are options or left to the user: one command starts one training on one GPU.

LAYOUT  <T1_WORK>/forks/<name>/
  fork/model_<N>.safetensors, fork/rsi_<N>.safetensors   the copied source pair (never overwritten)
  run/                                                   the trainer's output_dir (model_*, rsi_*, metrics.jsonl,
                                                         final.json, hydra/)
  fc_cache/convex/*.npz                                  FLASH_CHORD_CACHE_DIR (with --hull-cache stock)
  fork.json, launch.json, train.log

USAGE
  python fork_train.py list                                   items, pinned checkpoints and arms
  python fork_train.py plan  ep28alt --arm C --plus 10000      print what start would do (writes nothing)
  python fork_train.py check ep28alt --arm C                   hydra-compose the fork command (CPU, no training)
  python fork_train.py start ep25 --arm B --param beta=3.0 --gpu 0 [--cpus 0-7] [--name ep25-B3]
  python fork_train.py start ep21 --arm TB --plus 20000 --gpu 1
  python fork_train.py start ep01 --override objective.object_task_rotation_beta=3.5 --gpu 2 --name ep01-B35
  python fork_train.py start ep28alt --from-run <dir> --from-iteration 62000 --arm C --gpu 3   # fork another pair
  python fork_train.py status                                 every fork: pid alive, newest checkpoint, final.json
Then evaluate its checkpoints with eval_watch.py <name>.
Settings (environment or --site): T1_WORK, T1_RUNS, V2D_MOTION, V2D_RUNTIME, FC_PYTHON, T1_CODE (default: this
checkout), WANDB_MODE (default disabled). The pinned pair of an item is <T1_RUNS>/<pinned.run>/{model,rsi}_<N>.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1common as C  # noqa: E402

STD_KEYS = {"experiment", "task.parquet", "scene.support", "reset.seed", "output_dir", "hydra.run.dir", "logging.mode",
            "logging.local_jsonl", "training.early_stop_help_zero_tail", "training.max_iterations", "training.resume",
            "training.checkpoint"}


def fork_root(site: C.Site, name: str) -> Path:
    return site.work / "forks" / name


def arm_overrides(arm_name: str | None, params: list[str]) -> tuple[list[str], dict]:
    if not arm_name:
        return [], {}
    arms = C.load_arms()
    if arm_name not in arms:
        raise SystemExit(f"unknown arm {arm_name}; known: {', '.join(arms)}")
    arm = arms[arm_name]
    values = dict(p.split("=", 1) for p in params)
    out = []
    for o in arm["overrides"]:
        try:
            out.append(o.format(**values))
        except KeyError as missing:
            raise SystemExit(f"arm {arm_name} needs --param {missing.args[0]}=<value> ({arm.get('params', {})})") from None
    return out, arm


def source_pair(args, cfg: dict, site: C.Site) -> dict:
    if args.from_run:
        run = Path(args.from_run)
        run = run if run.is_absolute() else site.runs / run
        if args.from_iteration is None:
            raise SystemExit("--from-run needs --from-iteration")
        n = int(args.from_iteration)
        return {"dir": run, "iteration": n, "model_sha256": None, "rsi_sha256": None, "pinned": False}
    p = cfg["pinned"]
    return {"dir": site.runs / p["run"], "iteration": int(p["iteration"]), "model_sha256": p["model_sha256"],
            "rsi_sha256": p["rsi_sha256"], "pinned": True, "rsi_note": p.get("rsi_note")}


def build(args, site: C.Site) -> dict:
    items = C.load_items()
    cfg = C.item_of(args.item, items)
    src = source_pair(args, cfg, site)
    n = src["iteration"]
    extra, arm = arm_overrides(args.arm, args.param or [])
    extra = C.merge_overrides(extra, args.override or [])
    bad = sorted({C.override_key(o) for o in extra} & STD_KEYS)
    if bad:
        raise SystemExit(f"--override/--arm may not set {bad}: use --plus/--to/--seed/--name instead")
    inherit = [] if args.no_inherit else list(cfg["fork"]["inherit"])
    task = C.merge_overrides(inherit, extra)
    seed = args.seed if args.seed is not None else int(arm.get("reset_seed", cfg["fork"]["reset_seed"]))
    target = int(args.to) if args.to else n + int(args.plus)
    if target <= n:
        raise SystemExit(f"training.max_iterations {target} must be above the fork iteration {n}")
    name = args.name or f"{cfg['item']}-{args.arm or 'X'}-{n}"
    root = fork_root(site, name)
    run = root / "run"
    model = root / "fork" / f"model_{n}.safetensors"
    parquet = C.task_parquet(cfg, site)
    argv = ["scripts/train_rl.py", f"experiment={cfg['fork']['experiment']}", f"task.parquet='{parquet}'",
            f"scene.support={str(cfg['fork']['scene_support']).lower()}", f"reset.seed={seed}", f"output_dir={run}",
            f"hydra.run.dir={run}/hydra", f"logging.mode={site.get('WANDB_MODE', 'disabled')}", "+logging.local_jsonl=true",
            "training.early_stop_help_zero_tail=0", f"training.max_iterations={target}", "training.resume=true",
            f"training.checkpoint={model}", *task]
    cache = None
    if args.hull_cache == "stock":
        cache = root / "fc_cache"
    elif args.hull_cache not in ("tree",):
        cache = Path(args.hull_cache).absolute()
    return {"item": cfg["item"], "episode": cfg["episode"], "name": name, "root": root, "run": run, "source": src,
            "fork_iteration": n, "max_iterations": target, "arm": args.arm, "arm_overrides": extra,
            "inherited": inherit, "task_overrides": task, "seed": seed, "argv": argv, "hull_cache": args.hull_cache,
            "cache_root": cache, "parquet": parquet, "cfg": cfg}


def shown(plan: dict, site: C.Site, gpu, cpus, threads) -> str:
    env = {"CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": site.code / "src", "HUMAN_MOTION_DATA_DIR": site.motion,
           "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "OMP_NUM_THREADS": threads}
    if plan["cache_root"] is not None:
        env["FLASH_CHORD_CACHE_DIR"] = plan["cache_root"]
    pre = ["taskset", "-c", cpus] if cpus else []
    envs = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items())
    return (f"cd {shlex.quote(str(site.code))} && env {envs} {shlex.join(pre + [site.fc_python] + plan['argv'])} "
            f">> {plan['root'] / 'train.log'} 2>&1")


def copy_checked(src: Path, dst: Path, expect: str | None) -> dict:
    """dst ends up byte-identical to src (and to the pin, if any); an existing dst is verified, never overwritten."""
    if dst.exists():
        got = C.sha256(dst)
        if expect and got != expect:
            raise SystemExit(f"{dst} exists with sha256 {got[:16]}... != pin {expect[:16]}...: not overwritten")
        return {"dest": str(dst), "sha256": got, "note": "already present, verified"}
    if not src.is_file():
        raise SystemExit(f"missing {src}")
    h = C.sha256(src)
    if expect and h != expect:
        raise SystemExit(f"{src}: sha256 {h} != pin {expect}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.part")
    shutil.copyfile(src, tmp)
    if C.sha256(tmp) != h:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"copy {src} -> {dst} not verified")
    os.replace(tmp, dst)
    return {"source": str(src), "dest": str(dst), "sha256": h, "size": dst.stat().st_size, "copied_at": C.now()}


def prepare(plan: dict, site: C.Site) -> dict:
    src = plan["source"]
    n = src["iteration"]
    root = plan["root"]
    rec_path = root / "fork.json"
    rec = C.read_json(rec_path) or {}
    if rec and (rec.get("item") != plan["item"] or rec.get("fork_iteration") != n):
        raise SystemExit(f"{root} belongs to another fork ({rec.get('item')} @ {rec.get('fork_iteration')}): pick --name")
    files = {}
    for kind in ("model", "rsi"):
        s = src["dir"] / f"{kind}_{n}.safetensors"
        if kind == "rsi" and not s.is_file() and src.get("rsi_note"):
            raise SystemExit(f"{s} missing: {src['rsi_note']}")
        files[kind] = copy_checked(s, root / "fork" / f"{kind}_{n}.safetensors", src.get(f"{kind}_sha256"))
    plan["run"].mkdir(parents=True, exist_ok=True)
    busy = [p.name for p in plan["run"].iterdir() if p.name.startswith(("model_", "rsi_")) or p.name == "final.json"]
    if busy and not rec.get("launched"):
        raise SystemExit(f"{plan['run']} already holds {busy[:4]}: not a fresh run dir")
    cache = {}
    if plan["hull_cache"] == "stock":
        stock = site.runtime / "src/flash_chord/assets/.cache/convex"
        hulls = sorted(stock.glob("*.npz"))
        if not hulls:
            raise SystemExit(f"no stock hulls in {stock}: copy the transferred stock cache there first (README)")
        for h in hulls:
            cache[h.stem] = copy_checked(h, plan["cache_root"] / "convex" / h.name, None)["sha256"]
    elif plan["cache_root"] is not None and not (plan["cache_root"] / "convex").is_dir():
        raise SystemExit(f"--hull-cache {plan['cache_root']} has no convex/ directory")
    rec.update({"item": plan["item"], "episode": plan["episode"], "name": plan["name"], "source_dir": str(src["dir"]),
                "source_pinned": src["pinned"], "fork_iteration": n, "max_iterations": plan["max_iterations"],
                "arm": plan["arm"], "arm_overrides": plan["arm_overrides"], "inherited_overrides": plan["inherited"],
                "reset_seed": plan["seed"], "files": files, "hull_cache": plan["hull_cache"],
                "cache_dir": None if plan["cache_root"] is None else str(plan["cache_root"]), "cache_seed": cache,
                "forked_at": rec.get("forked_at") or C.now()})
    C.write_atomic(rec_path, C.dump(rec))
    return rec


def compose(plan: dict, site: C.Site) -> tuple[bool, str]:
    """Hydra-compose the fork's overrides against the training tree's configs (no training, no GPU)."""
    from hydra import compose as hcompose, initialize_config_dir
    from omegaconf import OmegaConf

    ovs = [o for o in plan["argv"][1:] if not o.startswith("hydra.")]
    with initialize_config_dir(config_dir=str(site.code / "src/flash_chord/configs"), version_base="1.3"):
        cfg = OmegaConf.to_container(hcompose(config_name="train", overrides=ovs), resolve=False)

    def get(dotted):
        node = cfg
        for part in dotted.split("."):
            if isinstance(node, list) and part.isdigit() and int(part) < len(node):
                node = node[int(part)]
            elif isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return "<absent>"
        return node

    bad = []
    for o in plan["task_overrides"]:
        key, _, val = o.partition("=")
        key = key.lstrip("+~")
        got = get(key)
        try:
            same = float(got) == float(val)
        except (TypeError, ValueError):
            same = str(got).lower() == val.strip("'\"").lower()
        if not same:
            bad.append(f"{key}: composed {got!r} != override {val!r}")
    facts = (f"experiment {plan['cfg']['fork']['experiment']}, support {get('scene.support')}, seed {get('reset.seed')}, "
             f"max_iterations {get('training.max_iterations')}, rotation beta {get('objective.object_task_rotation_beta')}, "
             f"orientation threshold {get('termination.object_orientation_threshold')}")
    return not bad, facts + ("" if not bad else "; MISMATCH " + "; ".join(bad))


def cmd_list(args, site) -> int:
    items = C.load_items()
    print(f"{'item':8} {'set':3} {'pinned':>8} {'run':28} {'experiment':34} {'20-trial AUC':>12}  inherited overrides")
    for it, c in items.items():
        t = c["evaluation"]["trials20"]
        print(f"{it:8} {'yes' if c['in_set'] else 'no':3} {c['pinned']['iteration']:>8} {c['pinned']['run']:28} "
              f"{c['experiment']:34} {t['auc_mean']:>12.4f}  {' '.join(c['fork']['inherit']) or '-'}")
    print("\narms (configs/arms.json):")
    for name, a in C.load_arms().items():
        print(f"  {name:5} {' '.join(a['overrides']) or '(no override)'}")
    return 0


def describe(plan: dict, site: C.Site, args) -> None:
    src = plan["source"]
    print(f"fork {plan['name']}: {plan['item']} (episode {plan['episode']}) from {src['dir']}/model_{src['iteration']} "
          f"(+ rsi_{src['iteration']}){' [pinned, sha256-checked]' if src['pinned'] else ''} -> "
          f"training.max_iterations {plan['max_iterations']}")
    print(f"  inherited overrides: {' '.join(plan['inherited']) or '-'}")
    print(f"  arm/extra overrides: {' '.join(plan['arm_overrides']) or '-'}")
    print(f"  hull cache: {plan['hull_cache']}{'' if plan['cache_root'] is None else ' -> FLASH_CHORD_CACHE_DIR=' + str(plan['cache_root'])}")
    print(f"  command: {shown(plan, site, args.gpu, args.cpus, args.threads)}")


def cmd_plan(args, site) -> int:
    describe(build(args, site), site, args)
    return 0


def cmd_check(args, site) -> int:
    plan = build(args, site)
    ok, msg = compose(plan, site)
    print(f"{'ok' if ok else 'FAIL'} {plan['name']}: {msg}")
    return 0 if ok else 1


def cmd_start(args, site) -> int:
    if args.gpu is None:
        raise SystemExit("--gpu is required (one training per command)")
    if args.gpu in site.exclude_gpus:
        raise SystemExit(f"gpu {args.gpu} is in EXCLUDE_GPUS")
    if site.refuse_mps and args.gpu in C.mps_gpus(C.smi()):
        raise SystemExit(f"gpu {args.gpu} runs an MPS server and REFUSE_MPS=1")
    plan = build(args, site)
    describe(plan, site, args)
    ok, msg = compose(plan, site)
    if not ok:
        raise SystemExit(f"hydra compose check failed: {msg}")
    rec = prepare(plan, site)
    log = plan["root"] / "train.log"
    env = C.fc_env(site, args.gpu, {"PYTHONPATH": site.code / "src", "OMP_NUM_THREADS": args.threads,
                                    "MKL_NUM_THREADS": args.threads, "OPENBLAS_NUM_THREADS": args.threads,
                                    "PYTHONUNBUFFERED": "1", "T1_FORK": plan["name"]})
    if plan["cache_root"] is not None:
        env["FLASH_CHORD_CACHE_DIR"] = str(plan["cache_root"])
    argv = (["taskset", "-c", args.cpus] if args.cpus else []) + [site.fc_python] + plan["argv"]
    with log.open("a") as fp:
        fp.write(f"[{C.now()}] fork_train: {plan['name']} on gpu {args.gpu}: {shlex.join(argv)}\n")
        fp.flush()
        proc = subprocess.Popen(argv, cwd=site.code, env=env, stdin=subprocess.DEVNULL, stdout=fp,
                                stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    time.sleep(5)
    if proc.poll() is not None:
        raise SystemExit(f"the trainer exited at once (code {proc.returncode}); see {log}")
    launch = {"name": plan["name"], "item": plan["item"], "episode": plan["episode"], "pid": proc.pid, "gpu": args.gpu,
              "cpus": args.cpus, "threads": args.threads, "argv": argv, "cwd": str(site.code),
              "code_head": C.git_head(site.code), "flash_chord_cache_dir": env.get("FLASH_CHORD_CACHE_DIR"),
              "fork_iteration": plan["fork_iteration"], "max_iterations": plan["max_iterations"],
              "task_overrides": plan["task_overrides"], "launched_at": C.now()}
    C.write_atomic(plan["root"] / "launch.json", C.dump(launch))
    rec["launched"] = True
    C.write_atomic(plan["root"] / "fork.json", C.dump(rec))
    print(f"started pid {proc.pid}; log {log}; evaluate with: python eval_watch.py {plan['name']} --gpu {args.gpu}")
    if args.foreground:
        return proc.wait()
    return 0


def alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def cmd_status(args, site) -> int:
    base = site.work / "forks"
    for root in sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []:
        fk, la = C.read_json(root / "fork.json") or {}, C.read_json(root / "launch.json") or {}
        cks = C.checkpoints_in(root / "run")
        fin = C.read_json(root / "run/final.json")
        state = "running" if la.get("pid") and alive(la["pid"]) and not fin else ("ended" if fin else "not running")
        print(f"{root.name:24} {fk.get('item', '?'):8} {state:12} gpu {la.get('gpu', '-')!s:3} model_{fk.get('fork_iteration')} "
              f"-> {fk.get('max_iterations')}  newest {cks[-1] if cks else '-'}  final {None if not fin else fin.get('iteration')}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_site_arg(p)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("status")
    for name in ("plan", "check", "start"):
        s = sub.add_parser(name)
        s.add_argument("item", help="item name (ep00..ep29, ep28alt) or episode number")
        s.add_argument("--arm", help="arm of configs/arms.json (C, B, T, TB, OP, OP2, W2P, W2PD, G)")
        s.add_argument("--param", action="append", help="arm parameter, e.g. beta=3.0")
        s.add_argument("--override", action="append", help="extra Hydra override (replaces one with the same key)")
        s.add_argument("--no-inherit", action="store_true", help="drop the pinned stage's task-specific overrides")
        s.add_argument("--plus", type=int, default=15000, help="iterations after the fork point (default 15000)")
        s.add_argument("--to", type=int, help="absolute training.max_iterations instead of --plus")
        s.add_argument("--seed", type=int, help="reset.seed (default: the item's, or the arm's)")
        s.add_argument("--name", help="fork name (default <item>-<arm>-<fork iteration>)")
        s.add_argument("--from-run", help="fork another pair: run dir (absolute, or relative to T1_RUNS)")
        s.add_argument("--from-iteration", type=int, help="with --from-run: the pair's iteration")
        s.add_argument("--hull-cache", default="stock",
                       help="stock (default: seed with the stock runtime's hulls), tree (the code tree's own cache), "
                            "or a directory that holds convex/")
        s.add_argument("--gpu", type=int)
        s.add_argument("--cpus", help="taskset CPU list for the trainer (optional)")
        s.add_argument("--threads", default="4", help="OMP/MKL/OPENBLAS threads (default 4)")
        if name == "start":
            s.add_argument("--foreground", action="store_true", help="wait for the trainer")
    args = p.parse_args()
    site = C.load_site(args.site)
    return {"list": cmd_list, "status": cmd_status, "plan": cmd_plan, "check": cmd_check, "start": cmd_start}[args.cmd](args, site)


if __name__ == "__main__":
    sys.exit(main())

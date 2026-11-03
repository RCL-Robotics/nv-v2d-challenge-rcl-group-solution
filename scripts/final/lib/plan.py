#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""plan.py: print the shell plan (the exact commands) that scripts/final/{train,eval}_tier{1,2,3}.sh show with
--dry-run and run without it. It reads only the committed set records of the three counted files and prints to
stdout; it writes nothing and needs only the Python standard library.

  plan.py train --tier N --episode E[,E...] [--stage S] [--code training|this] [--runs-root DIR] [--iterations K]
                [--gpu G] [--cpus LIST] [--hull-cache recorded|stock|none|DIR] [--lfs-from DIR]
  plan.py eval  --tier N [--episode all|E[,E...]] [--trials K] [--out DIR] [--gpu G] [--cpus LIST]
                [--reexport [--code training|this] [--lfs-from DIR]]

Set records (the only inputs that decide anything):
  Tier 1  robotic_grounding/flash_chord/scripts/track2/t1best/t1ready_20261031/set_items.json  (set all7, trial 05)
  Tier 2  robotic_grounding/flash_chord/scripts/track2/t2r9_20261104/set/set_items_r9.json     (R9-CLOSEST, trial 3)
  Tier 3  robotic_grounding/flash_chord/scripts/track2/t3ready_20261102/set/set_items.json     (FINAL_r)

Machine paths come from the environment (scripts/final/site.env.example; lib/common.sh sets the defaults):
  V2D_WORK, FC_PYTHON, KIT_PYTHON, V2D_RUNTIME, V2D_KIT, V2D_DATA, PLAN_PY
  ROOT_ME, ROOT_ZX, ROOT_ZX5090, ROOT_ZXWS, ROOT_HAN   where the records' <ME>, <ZX>, <ZX5090>, <ZXWS>, <HAN> live here
  T1_MOTION, T1_RUNS, T1_EXPORTS                      Tier 1's <MOTION>, recorded <RUNS> and payload exports/

train: one block per training stage of the episode's lineage, in order. Code = a git worktree of the stage's
recorded training commit (default) or this checkout (--code this). Overrides = the recorded ones; only the paths
change (inputs -> this machine's copies, outputs -> --runs-root). A stage that resumes gets the recorded resume
pair (model_N + rsi_N) copied into the new run root first, sha256-checked where the record has it. --iterations K
is for smoke tests: scratch stages stop at K, resumed stages at N + K.

eval: per trial and episode, the official kit evaluator (examples/run_policy_evaluation.py + the kit's
flash_chord/scripts/evaluate_policy.py, on the stock runtime) through this checkout's scripts/track2/eval_all.py,
--export no, world count 1, against the policy's OWN input (Tier 2: its raw/K3/K3fix/G2 root; Tier 3: the f0 root).
The evaluated file is the pinned exported checkpoint of the counted file (sha256 checked); with --reexport it is a
fresh CPU export of the training checkpoint by the recorded export commit (or this checkout), which must be the same
policy as the pinned file (compare_exports.py). No packing, no clean reference.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
from pathlib import Path

sys.dont_write_bytecode = True

LIB = Path(__file__).resolve().parent
FINAL = LIB.parent
REPO = FINAL.parents[1]
FC = REPO / "robotic_grounding/flash_chord"
TR = FC / "scripts/track2"
SETS = {
    1: TR / "t1best/t1ready_20261031/set_items.json",
    2: TR / "t2r9_20261104/set/set_items_r9.json",
    3: TR / "t3ready_20261102/set/set_items.json",
}
T3REC = TR / "t3ready_20261102"
ORIG_ME = "/root/autodl-tmp/haoran.zhang"  # the training server's work root, as written in verbatim records

# Training / export commits named in the records -> the commit of this history that holds that code.
FULL = {
    "b39b69b": "b39b69be6c2f561c0123c6aeb8cd13f8b9d542e7",
    "0868a3f": "b39b69be6c2f561c0123c6aeb8cd13f8b9d542e7",  # teammate commit, not on GitHub; tree equal to b39b69b
    "61fa5d3": "f2923d61817e61eb37d939e0f75e5b4a7aaa5cbf",  # local repo fc_r2, not on GitHub; flash_chord code = f2923d6
    "f2923d6": "f2923d61817e61eb37d939e0f75e5b4a7aaa5cbf",
    "7fbc227": "7fbc2273a92baefe652cb11e09c7e0acdd62da00",
    "bc3b1bd": "bc3b1bdf34f2f80e3f7d56625e3a0e1deca6a53e",
    "82837c8": "82837c8569c2461f263c28fce406c1b517f40d93",
}
# Record "hull cache: tree" = the training tree's own convex cache on the training server (git-ignored; payload group
# F_hull_caches). None: the tree was a teammate's worktree that is not on the training server -> stock cache instead.
TREE_CACHE = {
    "b39b69b": "<ME>/code/nv-v2d_runin/robotic_grounding/flash_chord/src/flash_chord/assets/.cache/convex",
    "0868a3f": None,
    "61fa5d3": "<ME>/code/fc_r2/src/flash_chord/assets/.cache/convex",
    "f2923d6": "<ME>/code/fc_r2/src/flash_chord/assets/.cache/convex",
    "7fbc227": None,  # its stages used seeded stock caches (T1fix/ep28w2/fc_cache), recorded per stage
    "bc3b1bd": None,
    "82837c8": None,
}
CKPT_RE = re.compile(r"^(?P<dir>.+)/model_(?P<it>\d+)\.safetensors$")


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default) or default


def q(x) -> str:
    return shlex.quote(str(x))


def roots() -> dict[str, str]:
    me = env("ROOT_ME", ORIG_ME)
    return {
        "<ME>": me,
        "<ZX>": env("ROOT_ZX", "/root/autodl-tmp/zuxing.lu"),
        "<ZX5090>": env("ROOT_ZX5090", "<ZX5090>"),  # Zuxing's RTX 5090 host; unset = not on this machine
        "<ZXWS>": env("ROOT_ZXWS", "<ZXWS>"),  # Zuxing's robolab workstation ws031; unset = not on this machine
        "<HAN>": env("ROOT_HAN", "/root/autodl-tmp/han.zhou"),
        "<MOTION>": env("T1_MOTION", f"{me}/data/human_motion_data"),
    }


def sub(text: str, mapping: dict[str, str]) -> str:
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text


def commit_of(text: str) -> tuple[str, str]:
    m = re.match(r"\s*([0-9a-f]{7,40})", text)
    if not m or m.group(1)[:7] not in FULL:
        raise SystemExit(f"unknown code commit in the record: {text!r}")
    return m.group(1)[:7], FULL[m.group(1)[:7]]


def parse_episodes(text: str) -> list[int] | None:
    if text in ("", "all"):
        return None
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def set_override(ovs: list[str], key: str, value: str) -> list[str]:
    out, done = [], False
    for o in ovs:
        if o.split("=", 1)[0] in (key, "+" + key):
            if not done:
                out.append(f"{o.split('=', 1)[0]}={value}")
                done = True
        else:
            out.append(o)
    if not done:
        out.append(f"{key}={value}")
    return out


def get_override(ovs: list[str], key: str) -> str | None:
    for o in ovs:
        if o.split("=", 1)[0] in (key, "+" + key):
            return o.split("=", 1)[1]
    return None


def unq(v: str) -> str:
    return v[1:-1] if len(v) >= 2 and v[0] == v[-1] == "'" else v


# ---------------------------------------------------------------------------------------------------- the records
def load(tier: int) -> dict:
    return json.loads(SETS[tier].read_text())


def t1_hull(stage: dict, key: str) -> str | None:
    h = stage["hull_cache"]
    runs = env("T1_RUNS", f"{roots()['<ME>']}/fc_runs")
    if h.startswith("tree"):
        p = TREE_CACHE.get(key) if "teammate" not in h else None
        return sub(p, roots()) if p else None
    m = re.match(r"stock:(\S+)", h) or re.match(r"seeded copy (\S+)", h)
    return f"{runs}/{m.group(1)}/convex" if m else None


def stages(tier: int, ent: dict, new_root: str, iterations: int | None) -> list[dict]:
    """The training stages of one episode, mapped to this machine (inputs) and to new_root (outputs)."""
    R = roots()
    out = []
    if tier == 1:
        rec_runs = env("T1_RUNS", f"{R['<ME>']}/fc_runs")
        sts = ent["stages"]
        for i, st in enumerate(sts, 1):
            key, sha = commit_of(st["code"])
            ovs = []
            for o in st["overrides"]:
                if o.startswith("["):  # optional W&B entity/project of the C3r2_wide lines
                    k = re.search(r"<(WANDB_\w+)>", o).group(1)
                    if env(k):
                        ovs.append(o.strip("[]").replace(f"<{k}>", env(k)))
                    continue
                o = o.replace("<WANDB_MODE>", env("WANDB_MODE", "disabled"))
                ovs.append(sub(o.replace("<RUNS>", new_root), R))
            resume = None
            ck = get_override(ovs, "training.checkpoint")
            if ck:
                m = CKPT_RE.match(unq(ck))
                rel = m.group("dir")[len(new_root) + 1:]
                fp = st.get("fork_pair") or {}
                resume = {"src": f"{rec_runs}/{rel}", "dst": m.group("dir"), "it": int(m.group("it")),
                          "model_sha": (fp.get("model") or {}).get("sha256", ""),
                          "rsi_sha": (fp.get("rsi") or {}).get("sha256", "")}
            outdir = unq(get_override(ovs, "output_dir"))
            if not get_override(ovs, "hydra.run.dir"):
                ovs.append(f"hydra.run.dir={outdir}/hydra")
            out.append({"label": f"stage {i}/{len(sts)}: run {st['run']}; start: {st['start']}; recorded end: "
                                 f"{st['end']}", "key": key, "sha": sha, "ovs": ovs, "out": outdir,
                        "motion": R["<MOTION>"], "resume": resume, "hull": t1_hull(st, key), "wandb": "disabled",
                        "name": f"{i}"})
    elif tier == 2:
        tr = ent["train"]
        key, sha = commit_of(tr["code_commit"])
        motion = sub(ent["data_root"], R)
        parquet = sub(ent["parquet"], R)
        job = ent["job"]
        for i, st in enumerate(tr["stages"], 1):
            note = ""
            if "overrides" in st:
                ovs = list(st["overrides"])
            else:  # ep10: no run record on the team server; the checkpoint's embedded configuration + first train_t2 line
                ec = st["embedded_config"]
                exp = re.search(r"exp=(\S+)", tr["train_log_first_line"]).group(1)
                ovs = [f"experiment={exp}", f"task.parquet='{ec['task.parquet']}'",
                       f"scene.support={str(ec['scene.support']).lower()}", f"reset.seed={ec['reset.seed']}",
                       f"output_dir={ec['output_dir']}", "logging.mode=disabled", "+logging.local_jsonl=true",
                       f"scene.world_count={ec['scene.world_count']}", f"training.max_iterations={ec['training.max_iterations']}",
                       f"training.early_stop_help_zero_tail={ec['training.early_stop_help_zero_tail']}"]
                note = " (overrides rebuilt from the checkpoint's embedded configuration and the first train_t2 line)"
            outdir = f"{new_root}/{job}" + (f"/s{i}" if len(tr["stages"]) > 1 else "")
            ovs = set_override(ovs, "task.parquet", f"'{parquet}'")  # this machine's copy (sha256 in the record)
            ovs = set_override(ovs, "output_dir", outdir)
            ovs = set_override(ovs, "hydra.run.dir", f"{outdir}/hydra")
            resume = None
            ck = get_override(ovs, "training.checkpoint")
            if ck:
                m = CKPT_RE.match(sub(unq(ck), R))
                dst = f"{outdir}/resume_from"
                resume = {"src": m.group("dir"), "dst": dst, "it": int(m.group("it")), "model_sha": "", "rsi_sha": ""}
                ovs = set_override(ovs, "training.checkpoint", f"{dst}/model_{m.group('it')}.safetensors")
            hull = sub(TREE_CACHE["61fa5d3"], R) if key == "61fa5d3" else None
            out.append({"label": f"stage {i}/{len(tr['stages'])}: {job} ({ent['owner']}, {tr.get('machine')}); recorded "
                                 f"run {st['run']}{note}", "key": key, "sha": sha, "ovs": ovs, "out": outdir,
                        "motion": motion, "resume": resume, "hull": hull, "wandb": "disabled", "name": f"{i}"})
    else:
        motion = f"{R['<ME>']}/data/human_motion_data_tier3_f0"
        for st in ent["train"]["stages"]:
            key, sha = commit_of(ent["train"]["code_commit"])
            ovs = [sub(o, R) for o in st["overrides"]]
            rec_out = unq(get_override(ovs, "output_dir"))
            outdir = rec_out.replace(f"{R['<ME>']}/fc_runs", new_root, 1)
            ovs = set_override(ovs, "output_dir", outdir)
            ovs = set_override(ovs, "hydra.run.dir", f"{outdir}/hydra")
            resume = None
            if st["stage"] == "retry_C90":
                fork = json.loads((T3REC / st["fork_json"]).read_text())
                fm, fr = fork["files"]["model"], fork["files"]["rsi"]
                src_dir = str(Path(fm["src"].replace(ORIG_ME, R["<ME>"], 1)).parent)
                it = int(fork["fork_iteration"])
                resume = {"src": src_dir, "dst": outdir, "it": it, "model_sha": fm["sha256"], "rsi_sha": fr["sha256"]}
                ovs = set_override(ovs, "training.checkpoint", f"{outdir}/model_{it}.safetensors")
            out.append({"label": f"stage {st['stage']}: recorded run {st['run_dir']} -> {st['checkpoint']['name']} "
                                 f"(sha256 {st['checkpoint']['sha256'][:16]})", "key": key, "sha": sha, "ovs": ovs,
                        "out": outdir, "motion": motion, "resume": resume, "hull": sub(TREE_CACHE["61fa5d3"], R),
                        "wandb": "offline", "name": st["stage"]})
    if iterations:
        for s in out:
            s["ovs"] = set_override(s["ovs"], "training.max_iterations",
                                    str(iterations + (s["resume"]["it"] if s["resume"] else 0)))
    return out


def eval_item(tier: int, ent: dict) -> dict:
    R = roots()
    if tier == 1:
        ex, ck = ent["export"], ent["checkpoint"]
        base = env("T1_RUNS", f"{R['<ME>']}/fc_runs") if ex["root"] == "RUNS" else env(
            "T1_EXPORTS", f"{R['<ME>']}/transfer/t1best_20261026/payload/exports")
        return {"file": f"{base}/{ex['file']}", "sha": ex["sha256"], "it": int(ck["iteration"]), "motion": R["<MOTION>"],
                "ckpt": f"{env('T1_RUNS', R['<ME>'] + '/fc_runs')}/{ck['run']}/{ck['model']}", "ckpt_sha": ck["model_sha256"],
                "export_key": commit_of(ex["exported_by"])[0], "name": ent["item"], "input": "tier 1"}
    if tier == 2:
        ex, ck = ent["export"], ent["checkpoint"]
        return {"file": sub(ex["file"], R), "sha": ex["sha256"], "it": int(ent.get("iteration") or ck["iteration"]), "motion": sub(ent["data_root"], R),
                "ckpt": sub(ck.get("evaluated_file") or ck["file"], R), "ckpt_sha": ck["sha256"],
                "export_key": commit_of(ex["code_commit"])[0], "name": ent["job"], "input": ent["input"]}
    ex, ck = ent["export"], ent["checkpoint"]
    return {"file": sub(ex["file"], R), "sha": ex["sha256"], "it": int(ck["iteration"]),
            "motion": f"{R['<ME>']}/data/human_motion_data_tier3_f0", "ckpt": sub(ck["file"], R), "ckpt_sha": ck["sha256"],
            "export_key": commit_of(ent["train"]["code_commit"])[0], "name": f"ep{ent['episode']:02d} {ent['kind']}",
            "input": "f0"}


# ---------------------------------------------------------------------------------------------------- plan text
def header(title: str) -> list[str]:
    R = roots()
    lines = [
        f"# {title}",
        f"# generated by scripts/final/lib/plan.py from {SETS_NAME} (commit {env('REPO_HEAD', '?')})",
        f"FINAL_LIB={q(LIB)}",
        f"REPO={q(REPO)}",
        f"PLAN_PY={q(env('PLAN_PY', sys.executable))}",
        f"FC_PYTHON={q(env('FC_PYTHON'))}",
        f"KIT_PYTHON={q(env('KIT_PYTHON'))}",
        f"V2D_RUNTIME={q(env('V2D_RUNTIME'))}",
        f"V2D_KIT={q(env('V2D_KIT'))}",
        f"V2D_DATA={q(env('V2D_DATA'))}",
        f"V2D_WORK={q(env('V2D_WORK'))}",
        f"REQUIRE_IDLE_GPU={q(env('REQUIRE_IDLE_GPU', '0'))}",
        "export FINAL_LIB PLAN_PY FC_PYTHON REQUIRE_IDLE_GPU",
        'source "$FINAL_LIB/helpers.sh"',
        "check_tools",
        f"# record roots on this machine: <ME>={R['<ME>']} <ZX>={R['<ZX>']} <MOTION>={R['<MOTION>']}",
    ]
    return lines


def code_dir(a, key: str, sha: str, lines: list[str]) -> str:
    """Emit the worktree setup for commit `sha` (once per plan) and return its flash_chord dir."""
    if a.code == "this":
        return str(FC)
    wt = f"{env('V2D_WORK')}/worktrees/{sha[:7]}"
    if wt not in a._made:
        a._made.add(wt)
        lines.append(f"ensure_worktree \"$REPO\" {q(wt)} {sha}   # recorded {key}")
        if a.need_lfs:
            lines.append(f"fill_lfs {q(wt)} {q(a.lfs_from or '')}")
    return f"{wt}/robotic_grounding/flash_chord"


def train_plan(a) -> list[str]:
    data = load(a.tier)
    eps = parse_episodes(a.episode)
    if not eps:
        raise SystemExit("--episode is required for training (one or more episode numbers)")
    new_root = a.runs_root or f"{env('V2D_WORK')}/runs/tier{a.tier}"
    a._made, a.need_lfs = set(), True
    L = header(f"Tier {a.tier} training: episode(s) {a.episode}"
               + (f", stage {a.stage}" if a.stage else "") + (f", smoke {a.iterations} iterations" if a.iterations else ""))
    L += [f"RUNS_NEW={q(new_root)}", 'mkdir -p "$RUNS_NEW" "$V2D_WORK/tmp" "$V2D_WORK/cache/xdg"']
    by_ep = {int(e["episode"]): e for e in data["episodes"]}
    gpu = a.gpu
    for e in eps:
        ent = by_ep[e]
        sts = stages(a.tier, ent, new_root, a.iterations)
        if a.stage:
            sts = [s for s in sts if s["name"] == a.stage]
            if not sts:
                raise SystemExit(f"episode {e}: no stage {a.stage!r}")
        L.append("")
        L.append(f"# ======== Tier {a.tier} ep{e:02d} {ent.get('object', '')} "
                 f"({ent.get('item') or ent.get('job') or ent.get('kind')}): {len(sts)} stage(s)")
        for s in sts:
            fcd = code_dir(a, s["key"], s["sha"], L)
            L.append(f"# --- {s['label']}")
            L.append(f"# code: recorded {s['key']} -> {'this checkout' if a.code == 'this' else s['sha'][:7] + ' worktree'}")
            L.append(f"OUT={q(s['out'])}")
            L.append('mkdir -p "$OUT"')
            r = s["resume"]
            if r:
                L.append(f"copy_resume_pair {q(r['src'])} {q(r['dst'])} {r['it']} {q(r['model_sha'])} {q(r['rsi_sha'])}")
            hc = a.hull_cache
            if hc == "none":
                L.append('FCCACHE=""   # --hull-cache none: the trainer uses its tree\'s own assets/.cache')
            else:
                src = (s["hull"] if hc == "recorded" else None) if hc in ("recorded", "stock") else hc
                L.append(f"seed_hulls {q(src or '')} \"$OUT/fc_cache\"   # recorded hull cache "
                         f"({'stock runtime cache' if not src else src})")
                L.append('FCCACHE="$OUT/fc_cache"')
            L.append(f"require_idle_gpu {gpu}")
            envs = ['PATH="$(dirname "$FC_PYTHON"):$PATH"', 'LD_LIBRARY_PATH="$(fc_libs)"',
                    f"PYTHONPATH={q(fcd + '/src')}", "PYTHONDONTWRITEBYTECODE=1", "PYTHONUNBUFFERED=1",
                    "XLA_PYTHON_CLIENT_PREALLOCATE=false", f"HUMAN_MOTION_DATA_DIR={q(s['motion'])}",
                    f"WANDB_MODE={s['wandb']}", 'XDG_CACHE_HOME="$V2D_WORK/cache/xdg"', 'TMPDIR="$V2D_WORK/tmp"',
                    '${FCCACHE:+FLASH_CHORD_CACHE_DIR="$FCCACHE"}', "CUDA_DEVICE_ORDER=PCI_BUS_ID",
                    f"CUDA_VISIBLE_DEVICES={gpu}", "OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4", "OPENBLAS_NUM_THREADS=4"]
            run = (f"taskset -c {q(a.cpus)} " if a.cpus else "") + '"$FC_PYTHON" -B scripts/train_rl.py'
            L.append(f"note \"training -> $OUT (log $OUT/train.log)\"")
            L.append(f"( cd {q(fcd)} && env " + " ".join(envs[:6]) + " \\\n    " + " ".join(envs[6:11]) + " \\\n    "
                     + " ".join(envs[11:]) + " \\\n  " + run + " \\\n    " + " ".join(q(o) for o in s["ovs"])
                     + ' ) > "$OUT/train.log" 2>&1')
            L.append('show_final "$OUT"')
    return L


def eval_plan(a) -> list[str]:
    data = load(a.tier)
    eps = parse_episodes(a.episode)
    items = [e for e in data["episodes"] if eps is None or int(e["episode"]) in eps]
    if not items:
        raise SystemExit(f"no episode {a.episode} in the Tier-{a.tier} set")
    out = a.out or f"{env('V2D_WORK')}/eval/tier{a.tier}"
    a._made, a.need_lfs = set(), False  # export and evaluation need no LFS file of the code tree
    L = header(f"Tier {a.tier} official own-input evaluation: episode(s) {a.episode or 'all'}, {a.trials} trial(s)"
               + (", re-exported" if a.reexport else ", pinned evaluated files"))
    L += [f"OUT={q(out)}", 'mkdir -p "$OUT/tmp"', 'check_eval_tools']
    L.append(f"echo {q(json.dumps({'tier': a.tier, 'episodes': a.episode or 'all', 'trials': a.trials, 'reexport': a.reexport, 'set': str(SETS[a.tier].relative_to(REPO))}))} > \"$OUT/plan.json\"")
    files = {}
    for ent in items:
        e = int(ent["episode"])
        it = eval_item(a.tier, ent)
        L.append("")
        L.append(f"# ---- ep{e:02d} {it['name']} (input: {it['input']}; HUMAN_MOTION_DATA_DIR {it['motion']})")
        if a.reexport:
            fcd = code_dir(a, it["export_key"], FULL[it["export_key"]], L)
            dst = f"{out}/reexport/ep{e:02d}"
            L.append(f"check_sha256 {q(it['ckpt'])} {it['ckpt_sha']}   # training checkpoint")
            L.append(f"mkdir -p {q(dst)}")
            L.append("( cd " + q(fcd) + " && env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=" + q(fcd + "/src")
                     + " PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2 HUMAN_MOTION_DATA_DIR="
                     + q(it["motion"]) + ' TMPDIR="$OUT/tmp" \\\n    "$FC_PYTHON" -B -m flash_chord.training.ppo.coordinate.export'
                     + f" --input {q(it['ckpt'])} --output {q(dst + '/model_' + str(it['it']) + '.safetensors')} \\\n"
                     + f'    --stock-runtime "$V2D_RUNTIME" --report {q(dst + "/export_report.json")} --tolerance 0.0001 )'
                     + f" > {q(dst + '/export.log')} 2>&1")
            L.append(f"\"$PLAN_PY\" -I -B {q(TR / 't1best/compare_exports.py')} {q(dst + '/model_' + str(it['it']) + '.safetensors')} "
                     f"{q(it['file'])} | tee {q(dst + '/compare_exports.txt')}   # must say the same policy")
            files[e] = (dst + f"/model_{it['it']}.safetensors", None)
        else:
            L.append(f"check_sha256 {q(it['file'])} {it['sha']}   # pinned evaluated file of the counted CSV")
            files[e] = (it["file"], it["sha"])
        manifest = f"{it['motion']}/whole_body/soma/track2_manifest.json"
        for k in range(1, a.trials + 1):
            arm = f"{out}/t{k:02d}/arm_ep{e:02d}"
            link = f"{arm}/ep{e:02d}/model_{it['it']}.safetensors"
            L.append(f"mkdir -p {q(arm + '/ep%02d' % e)} && ln -sfn {q(files[e][0])} {q(link)}")
            L.append(f"require_idle_gpu {a.gpu}")
            cmd = (f"( cd {q(FC)} && env CUDA_VISIBLE_DEVICES={a.gpu} PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false "
                   f"OMP_NUM_THREADS=2 HUMAN_MOTION_DATA_DIR={q(it['motion'])} TMPDIR=\"$OUT/tmp\" \\\n    "
                   + (f"taskset -c {q(a.cpus)} " if a.cpus else "")
                   + f"\"$FC_PYTHON\" -B {q(TR / 'eval_all.py')} --arm-dir {q(arm)} --arm T{a.tier}final --export no "
                   f"--iteration {it['it']} --episodes {e} --gpus {a.gpu} --jobs-per-gpu 1 --world-count 1 --skip-pack \\\n    "
                   f"--kit-root \"$V2D_KIT\" --stock-runtime \"$V2D_RUNTIME\" --dataset-root \"$V2D_DATA\" --manifest {q(manifest)} "
                   f"--pack-python \"$KIT_PYTHON\" \\\n    --results {q(out + f'/t{k:02d}/results')} --code {q(FC)} "
                   f"--python \"$FC_PYTHON\" ) > {q(arm + '/eval_all.log')} 2>&1")
            L.append(f"note \"trial {k}/{a.trials} ep{e:02d} -> {out}/t{k:02d}/results\"")
            L.append(cmd)
    L.append("")
    L.append(f"\"$PLAN_PY\" -I -B \"$FINAL_LIB/summarize.py\" --tier {a.tier} --out \"$OUT\" --set {q(SETS[a.tier])}"
             + ("" if a.reexport else " --expect-pinned"))
    return L


SETS_NAME = "the committed set records"


def main() -> int:
    global SETS_NAME
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=("train", "eval"))
    p.add_argument("--tier", type=int, required=True, choices=(1, 2, 3))
    p.add_argument("--episode", default="")
    p.add_argument("--stage", default="", help="train: stage number (Tier 1/2) or name f0|retry_C90 (Tier 3)")
    p.add_argument("--code", default="training", choices=("training", "this"))
    p.add_argument("--runs-root", default="")
    p.add_argument("--iterations", type=int, default=0)
    p.add_argument("--gpu", default=env("GPUS", "0").split(",")[0])
    p.add_argument("--cpus", default="")
    p.add_argument("--hull-cache", default="recorded")
    p.add_argument("--lfs-from", default=env("LFS_FROM"))
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--out", default="")
    p.add_argument("--reexport", action="store_true")
    a = p.parse_args()
    SETS_NAME = str(SETS[a.tier].relative_to(REPO))
    if not env("V2D_WORK"):
        raise SystemExit("V2D_WORK is not set (scripts/final/site.env.example)")
    lines = train_plan(a) if a.what == "train" else eval_plan(a)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

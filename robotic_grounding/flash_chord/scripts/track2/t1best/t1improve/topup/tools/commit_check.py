#!/usr/bin/env python3
"""commit_check.py: does the cited commit contain the training and export code of every item of a T1improve set?

Reads only. git runs only in code/v2d_t1push (a full, non-partial clone); the arm trees are hashed file by file with
`git hash-object` (no object written, no git command inside a partial-clone worktree). Hydra composition uses the
cited commit's checkout (v2d_t1push working tree, which must be clean at the commit) and the flashchord env.

  <flashchord python> -I -B commit_check.py --commit 0c7516e... --arms ep14-R,ep28-W3H,... --out commit_check.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
PUSH = ME / "code/v2d_t1push"
FCREL = "robotic_grounding/flash_chord"
ARMS = ME / "fc_runs/T1improve/arms"
TREES = {  # tree label -> (commit whose flash_chord files the arm's working dir must equal, flash_chord dir that ran)
    "b39b69b": ("b39b69be6c2f561c0123c6aeb8cd13f8b9d542e7", ME / "code/nv-v2d_runin/robotic_grounding/flash_chord"),
    "7fbc227": ("7fbc2273a92baefe652cb11e09c7e0acdd62da00", ME / "code/nv-v2d_ep28w2/robotic_grounding/flash_chord"),
    # fc_r2 is its own repo at 61fa5d3; its flash_chord files equal f2923d61817e of the pushed history (README_NEW_MACHINE)
    "61fa5d3": ("f2923d61817e", ME / "code/fc_r2"),
}
ARM_TREE = {"ep14-R": "b39b69b", "ep28-W3H": "7fbc227", "ep10-BT": "b39b69b", "ep19-Bc": "b39b69b", "ep01-Bc": "b39b69b",
            "ep15-B2": "61fa5d3", "ep05-B3": "b39b69b", "ep21-B5": "b39b69b", "ep25-B25": "b39b69b", "ep10-Bc": "b39b69b"}
ALWAYS_DIFF = ("output_dir", "logging.", "training.max_iterations", "training.resume", "training.checkpoint",
               "training.early_stop_help_zero_tail", "task.parquet")
CODE_PATHS = ("src", "scripts/train_rl.py", "scripts/track2/eval_all.py")


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(PUSH), *args], check=True, capture_output=True, text=True).stdout


def ls_tree(commit: str) -> dict[str, str]:
    out = {}
    for line in git("ls-tree", "-r", commit, "--", *(f"{FCREL}/{p}" for p in CODE_PATHS)).splitlines():
        meta, path = line.split("\t", 1)
        out[path[len(FCREL) + 1:]] = meta.split()[2]
    return out


def hash_object(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def lfs_ok(blob: str, path: Path) -> bool | None:
    """None if the blob is not a git-lfs pointer; else whether the working file is that LFS object."""
    data = subprocess.run(["git", "-C", str(PUSH), "cat-file", "blob", blob], check=True, capture_output=True).stdout
    if not data.startswith(b"version https://git-lfs"):
        return None
    oid = next((ln.split(b":", 1)[1].decode() for ln in data.splitlines() if ln.startswith(b"oid sha256:")), None)
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == oid


def tree_matches(label: str) -> dict:
    commit, fcdir = TREES[label]
    want = ls_tree(commit)
    mism, lfs = [], 0
    for p, b in want.items():
        f = fcdir / p
        if f.is_file() and hash_object(f) == b:
            continue
        ok = lfs_ok(b, f)
        if ok:
            lfs += 1
            continue
        mism.append(p)
    extra = sorted(str(p.relative_to(fcdir)) for p in (fcdir / "src").rglob("*.py")
                   if str(p.relative_to(fcdir)) not in want and "__pycache__" not in p.parts)
    return {"tree": label, "commit": commit, "dir": str(fcdir), "n_files": len(want), "lfs_objects_matched": lfs, "mismatched": mism,
            "untracked_py": extra, "ok": not mism and not extra}


def flat(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{prefix}{k}."))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(flat(v, f"{prefix}{i}."))
    else:
        out[prefix[:-1]] = d
    return out


def compose_arms(arms: list[str]) -> list[dict]:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    res = []
    with initialize_config_dir(config_dir=str(PUSH / FCREL / "src/flash_chord/configs"), version_base="1.3"):
        for arm in arms:
            runs = sorted((ARMS / arm / "runs").iterdir())
            run = runs[0]
            hyd = run / "hydra/.hydra"
            ovs = [o for o in OmegaConf.to_container(OmegaConf.load(hyd / "overrides.yaml")) if not o.startswith("hydra.")]
            r = {"arm": arm, "run": str(run), "overrides": ovs}
            try:
                comp = flat(OmegaConf.to_container(compose(config_name="train", overrides=ovs), resolve=False))
            except Exception as error:  # noqa: BLE001
                r.update(ok=False, error=repr(error))
                res.append(r)
                continue
            saved = flat(OmegaConf.to_container(OmegaConf.load(hyd / "config.yaml"), resolve=False))
            changed = sorted(k for k in set(saved) & set(comp) if saved[k] != comp[k])
            only_new = sorted(k for k in set(comp) - set(saved))
            only_old = sorted(k for k in set(saved) - set(comp))
            r.update(n_keys=len(comp), changed=changed, new_keys_defaults=only_new, missing=only_old,
                     ok=not changed and not only_old)
            res.append(r)
    return res


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--commit", required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--arms", required=True)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    arms = [x for x in a.arms.split(",") if x]
    out = {"commit": a.commit, "commit_url": a.url, "arms": arms, "problems": []}
    head = git("rev-parse", "HEAD").strip()
    full = git("rev-parse", a.commit).strip()
    out["push_checkout_head"] = head
    status = git("status", "--porcelain", "--untracked-files=no").strip()
    out["push_checkout_clean"] = not status
    if head != full or status:
        out["problems"].append(f"v2d_t1push checkout is not a clean {full[:12]} (HEAD {head[:12]}, status {status[:200]!r})")
    trees = sorted({ARM_TREE[x] for x in arms})
    out["trees"] = {}
    for t in trees:
        commit = TREES[t][0]
        anc = subprocess.run(["git", "-C", str(PUSH), "merge-base", "--is-ancestor", commit, full]).returncode == 0
        m = tree_matches(t)
        diff = git("diff", "--stat", commit, full, "--", f"{FCREL}/src").strip().splitlines()
        logs = git("log", "--oneline", f"{commit}..{full}", "--", f"{FCREL}/src").strip().splitlines()
        m.update(ancestor_of_cited=anc, src_diff_to_cited=diff[-1] if diff else "no differences", src_diff_files=diff[:-1],
                 commits_touching_src_since=logs)
        out["trees"][t] = m
        if not anc:
            out["problems"].append(f"{t}: {commit[:12]} is not an ancestor of {full[:12]}")
        if not m["ok"]:
            out["problems"].append(f"{t}: the arm working dir differs from {commit[:12]}: {m['mismatched'][:5]} {m['untracked_py'][:5]}")
    exp_rel = "src/flash_chord/training/ppo/coordinate/export.py"
    out["exporter_sha256"] = {t: hashlib.sha256((TREES[t][1] / exp_rel).read_bytes()).hexdigest() for t in TREES}
    out["exporter_sha256"]["cited"] = hashlib.sha256(
        subprocess.run(["git", "-C", str(PUSH), "show", f"{full}:{FCREL}/{exp_rel}"], check=True, capture_output=True).stdout).hexdigest()
    out["compose"] = compose_arms(arms)
    for r in out["compose"]:
        if not r["ok"]:
            out["problems"].append(f"{r['arm']}: compose with the cited tree differs from the saved config: "
                                   f"{r.get('error') or (r['changed'][:5], r['missing'][:5])}")
    out["ok"] = not out["problems"]
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"ok": out["ok"], "problems": out["problems"]}, indent=1))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

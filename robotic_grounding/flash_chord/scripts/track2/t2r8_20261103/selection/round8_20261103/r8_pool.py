#!/usr/bin/env python3
"""r8_pool.py: the round-8 candidate pool (fc_runs/submissions/t2_consensus_r8_20261103; PREREG_R8.md sections 2-3).

Round 7's mechanical pool rule (scripts/t2cons_r7_pool.py 7a08fa6d..., loaded from its verified bytes; its helpers
header, root_kind_of, server_parquet, manifest_episode, ep_frame_ref, picks_of, run_record, scan_pro6000, z5_manifest,
z5_sums, first_train_t2_line, copy_verified, write_excerpt, verify_row and round 5's relocate_bytes / verify_pair are
called unchanged), re-scanned at the round-8 snapshot T_c = 2026-11-03 12:00:00 CST, with the round-8 deltas:
  D1  universe: round 7's (our runs of round 6's list; Zuxing's pro6000 training-run directories; his 5090 run copies)
      plus the copy folders that arrived since (runs/2026-11-02-tier2-r8-handover/<host>/..., runs/2026-11-02-round7-
      robolab-ckpts/<host>/...), each copy run resolved like round 7's R1b (embedded output_dir == the as-trained run
      path, Tier-2-derived data root present on this server, every selected file == its checksum line, a training commit
      named by the run's first train_t2: line or by the host evidence file). A run copied twice counts once (the copy
      with the most iterations). Mechanical picks (round 7's R4, K = 3) only for runs that are new or have more
      iterations than at round 7's snapshot. 5090-hosted runs: only iterations on the 10,000 grid and the run's last
      iteration are pick targets (PREREG_R8 2.4).
  D2  watcher best: per run, from eval_<ep>/metrics.csv next to the run (fallback: the same relative path under
      runs/2026-11-02-tier2-g2-ownsel/inputs_5090), rows with status == ok and finished_at <= T_c whose iteration is on
      the 10,000 grid or equals the run's last ok-evaluated iteration (5090-hosted runs: only when that is also the run's
      last counted iteration L, PREREG_R8 2.4) and whose model file is present at T_c: argmax add_auc, ties -> lower
      iteration. Only the columns episode, iteration, add_auc, status, finished_at are used; the header must hold only
      the watcher's 18 own-input columns; the record copy keeps the 8 own-input columns WATCH_COLS.
  D3  roles (deterministic order): incumbent (== the episode's round-7 pick), barred7 (== a round-7 phase-F participant
      of the episode, or round 7's carried barred pair), reuse7 (== a round-7 phase-S candidate of the episode with the
      same input: its round-7 phase-S results and export are reused), duplicate, screen (new: copy, relocation if needed,
      one export, 3 runs), refused. Identity = file sha256, else the sha256 of the tensor data block (relocated copies).
Only checkpoint headers, checksum files, MANIFEST files, the first train_t2: line of training logs, host evidence files,
final.json / hydra yaml copies and the watcher metrics files named above are opened in Zuxing's folder (round 7's token
guard, then an allowlist). His folder is only read; copies carry sha256 before == copy == after. Record copies of his
MANIFEST / checksum files are excerpts of the lines R8 needs (PREREG_R8 2.8).

  conda_envs/flashchord/bin/python -I -B r8_pool.py dryrun   snapshot NOW, nothing copied: dryrun/pool_dryrun_<stamp>.{json,md}
  conda_envs/flashchord/bin/python -I -B r8_pool.py freeze   at/after T_c only: POOL_R8.json + inputs_r8/, copies,
                                                            relocations, POOL_R8.sha256 (all mode 444); refuses if it exists
  conda_envs/flashchord/bin/python -I -B r8_pool.py trial --limit N   a small frozen pool (snapshot NOW) under dryrun/trial_<stamp>/
                                                            for the plan-only check of r8_eval.py (never used by a decision)
  conda_envs/flashchord/bin/python -I -B r8_pool.py verify   read-only re-check of RELOCATION_R8.tsv and POOL_R8.sha256
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import struct  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
ZU = Path("/root/autodl-tmp/zuxing.lu")
ZR = ZU / "runs"
ROOT = ME / "fc_runs/submissions/t2_consensus_r8_20261103"
R7ROOT = ME / "fc_runs/submissions/t2_consensus_r7_20261102"
PT_FILE, PT_SHA = ME / "scripts/t2cons_r7_pool.py", "7a08fa6d8be91140e139d914b8119352fabb05f58b415eb4a8904f9a02733770"
PREREG_SHA = "bb912c57180dde8255776884de169e12467349f57d584782308b88ff935abf8c"
TZ8 = dt.timezone(dt.timedelta(hours=8))
T_C = dt.datetime(2026, 11, 3, 12, 0, 0, tzinfo=TZ8).timestamp()
SETTLE_S = 300.0
GRID = 10000
HANDOVER = ZR / "2026-11-02-tier2-r8-handover"
ROBOLAB = ZR / "2026-11-02-round7-robolab-ckpts"
OWNSEL_INPUTS = ZR / "2026-11-02-tier2-g2-ownsel/inputs_5090"  # watcher-file fallback only (PREREG_R8 2.6)
NEW_SKIP_TOP = {  # Zuxing's runs/ top folders that are not walked as pro6000 training-run directories (PREREG_R8 2.2)
    "2026-11-02-tier2-r8-handover": "copies of training runs from other hosts: resolved as copy runs (PREREG_R8 2.3)",
    "2026-11-02-round7-robolab-ckpts": "copies of training runs from other hosts: resolved as copy runs (PREREG_R8 2.3)",
    "2026-11-02-tier2-g2-reports-eval": "report / evaluation bundle (not a training-run directory); never entered",
}  # fmt: skip
NEVER_ENTER = ("2026-11-02-tier2-g2-reports-eval", "2026-10-28-tier2-retest", "2026-10-24-5090-best")
HOST_PREFIX = {"5090": "/mnt/gfs/home/zuxinglu/workspace/", "robolab": "/home/zuxinglu/workspace/"}
PRUNED_HOSTS = ("5090",)  # PREREG_R8 2.4: grid + last iteration only (mechanical picks and the watcher best)
WATCH_COLS = ("episode", "iteration", "add_auc", "sp_sr", "mp_sr", "mppe_cm", "status", "finished_at")
WATCH_HEADER = ("episode", "iteration", "add_auc", "sp_sr", "mp_sr", "mppe_cm", "status", "seconds", "log", "object", "attempt",
                "final", "chord_sr", "export_max_rel_diff", "stock_verified", "source_sha256", "finished_at", "note")  # fmt: skip
MANIFEST_KEEP = re.compile(r"(run|source|copied|status|train log|task\.parquet|parquet sha256|models|files)\s")
SUMS_KEEP = re.compile(r"(model|rsi)_\d+\.safetensors|final\.json|train\.log|MANIFEST\.txt|config\.yaml|hydra\.yaml|overrides\.yaml")
Z5_LABELS_EXTRA = {"2026-11-01-tier2-g2-rest": "z5g2rest", "2026-11-02-tier2-g2-redo-5090": "z5g2redo"}
HOST_TAG = {"wsrbl015": "rb15", "ws031": "w31", "ws032": "w32", "ws032_aborted_stream": "w32a"}
GROUP_TAG = {"2026-10-26-tier2-runin-g2-ws": "g2", "2026-10-28-tier2-runin-k3fix-ws": "k3fix"}
SUB_TAG = {"cont": "c", "cont2": "c2", "cont3": "c3", "resume": "r"}


def _load(name: str, path: Path, want: str):
    src = path.read_bytes()
    got = hashlib.sha256(src).hexdigest()
    if got != want:
        raise SystemExit(f"{path} sha256 {got} != pinned {want}")
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(src, str(path), "exec"), mod.__dict__)  # noqa: S102 (verified pinned bytes)
    return mod


PT = _load("t2cons_r7_pool_pinned_r8", PT_FILE, PT_SHA)
PT.ZX_SKIP_TOP = {**PT.ZX_SKIP_TOP, **NEW_SKIP_TOP}  # rebinding only: the pinned scan reads this table at call time
K, MIN_LAST = PT.K, PT.MIN_LAST


# ================================================================================================ guards and helpers
_PT_GUARD = PT.guard  # round 7's token guard (kept for every path outside Zuxing's folder and for directory names)
G2_DATA = ZU / "data/human_motion_data_tier2_g2-2026-10-24"


def tokens_ok(text: str) -> bool:
    """True iff round 7's token guard (t2cons_r7_pool.guard, its list unchanged and not repeated here) accepts the text."""
    try:
        _PT_GUARD(text)
    except (PermissionError, ValueError, TypeError):
        return False
    return True


def zguard(p) -> Path:
    """Allowlist for files opened in Zuxing's folder (PREREG_R8 9): checkpoint / rsi files, checksum and MANIFEST files,
    training logs (first train_t2: line only), host evidence text, final.json, hydra yaml, watcher eval_<ep>/metrics.csv,
    and the G2 data root's parquets, manifest and frame references. Round 7's token guard is applied first to every path
    (in Zuxing's folder: for a watcher file to its eval_<ep> directory, because round 7's list names that file itself)."""
    p = Path(p)
    s = str(p)
    if s.startswith(str(ROOT) + "/") and "/zuxing_copies_r8/" in s:  # our copies of his files: the same allowlist
        q = Path(re.sub(r"\.tmp_r8\w+$", "", s))
        if (q.name == "metrics.csv" and q.parent.name.startswith("eval_")) or q.name == "train_log_first_train_t2_line.txt":
            return p
        return zguard(ZU / "runs" / q.relative_to(Path(s.split("/zuxing_copies_r8/")[0]) / "zuxing_copies_r8")) and p
    if not s.startswith(str(ZU) + "/"):
        return _PT_GUARD(p)
    if any(f"/{x}/" in s or s.endswith(f"/{x}") for x in NEVER_ENTER):
        raise PermissionError(f"r8_pool never opens {p}")
    _PT_GUARD(p.parent if (p.name == "metrics.csv" and p.parent.name.startswith("eval_")) else p)
    if p.is_dir():
        return p
    n = p.name
    if s.startswith(str(G2_DATA) + "/"):
        if n in ("data.parquet", "track2_manifest.json") or re.fullmatch(r"episode_\d{6}\.npz", n):
            return p
        raise PermissionError(f"r8_pool may not open {p} (allowlist)")
    ok = (re.fullmatch(r"(model|rsi)_\d+\.safetensors", n) or n in ("SHA256SUMS", "MANIFEST.tsv", "MANIFEST.txt", "final.json",
          "train.log", "metrics.csv", "config.yaml", "hydra.yaml", "overrides.yaml") or re.fullmatch(r"SHA256SUMS\.[A-Za-z0-9]+", n)
          or (p.parent.name == "evidence" and p.parent.parent == HANDOVER and re.fullmatch(r"[A-Za-z0-9]+\.txt", n))
          or (n.endswith(".log") and "/logs/" in s and not n.startswith("eval")))  # fmt: skip
    if not ok:
        raise PermissionError(f"r8_pool may not open {p} (allowlist)")
    if n == "metrics.csv" and not p.parent.name.startswith("eval_"):
        raise PermissionError(f"r8_pool: {p} is not a watcher file eval_<ep>/metrics.csv")
    return p


PT.guard = zguard  # every helper of the pinned resolver opens files through this guard (rebinding only)


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with zguard(p).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


PT.sha256_file = sha256_file


def data_block_sha(p) -> str:
    """sha256 of the tensor data block of a safetensors file (everything after the header): equal for a file and its
    metadata-only relocated copy."""
    h = hashlib.sha256()
    with zguard(p).open("rb") as fp:
        n = struct.unpack("<Q", fp.read(8))[0]
        fp.seek(8 + n)
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_text(t: float | None = None) -> str:
    return dt.datetime.fromtimestamp(time.time() if t is None else t, TZ8).strftime("%Y-%m-%d %H:%M:%S")


def read_json(p):
    return json.loads(zguard(p).read_text())


def read_sums(p: Path, base_rel: str = "") -> dict[str, str]:
    """checksum file -> {normalised path: sha256}."""
    out = {}
    for ln in zguard(p).read_text().splitlines():
        f = ln.split(None, 1)
        if len(f) == 2 and len(f[0]) == 64:
            out[os.path.normpath(f[1].strip().lstrip("*"))] = f[0]
    return out


def csv_text(rows: list[dict], cols: list[str], delim: str = ",") -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore", delimiter=delim, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({c: (json.dumps(r[c]) if isinstance(r.get(c), (list, dict)) else r.get(c)) for c in cols})
    return buf.getvalue()


def write_new(p: Path, text: str) -> None:
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp_r8p")
    tmp.write_text(text)
    os.replace(tmp, p)
    os.chmod(p, 0o444)


# ================================================================================================ settled files (R3)
def settled_models8(d: Path, t_c: float) -> tuple[dict[int, Path], dict[int, Path]]:
    """Round 7's R3 (mtime <= T_c - 300 s) plus 'present at T_c' (ctime <= T_c: a file copied after T_c never counts)."""
    its = {}
    for f in os.listdir(d):
        m = re.fullmatch(r"model_(\d+)\.safetensors", f)
        if m:
            its[int(m.group(1))] = d / f
    settled = {}
    for i, p in its.items():
        st = os.stat(p)
        if st.st_mtime <= t_c - SETTLE_S and st.st_ctime <= t_c:
            settled[i] = p
    return its, settled


PT.settled_models = settled_models8  # rebinding only: round 7's run_record calls it at call time


def grid_restricted(saved: list[int], L: int) -> list[int]:
    """PREREG_R8 2.4: on a pruned host only grid iterations and the last iteration are pick targets."""
    return sorted(i for i in saved if i % GRID == 0 or i == L)


# ================================================================================================ round-7 records
def load_r7records() -> dict:
    p = ROOT / "inputs_r8/R7_RECORDS_R8.json"
    rec = json.loads(PT.guard(p).read_text())
    return rec


def r7_universe() -> dict[tuple[str, str], dict]:
    """Round 7's eligible runs keyed by (host class, embedded output_dir) with their last iteration L at round 7's T_c."""
    data = json.loads((R7ROOT / "inputs_r7/runs_r7.json").read_text())
    out = {}
    for r in data["runs"]:
        if not r.get("eligible"):
            continue
        host = "5090" if r["machine"] == "5090" else "pro6000"
        key = (host, r["output_dir"] if host == "5090" else os.path.realpath(r["dir"]))
        out[key] = {"dir": r["dir"], "last": int(r["last"]), "picks": r.get("picks"), "label": r.get("label")}
    return out


# ================================================================================================ copy runs (R1c)
def as_trained_kind(parquet: str | None, prefix: str) -> tuple[str | None, str | None, str | None]:
    """(kind, root name, sequence) of an as-trained parquet <prefix>data/<root>/whole_body/soma/..., else Nones."""
    if not parquet or not parquet.startswith(prefix + "data/"):
        return None, None, None
    pp = PT.parquet_parts(parquet)
    if pp is None:
        return None, None, None
    root, seq = pp
    if str(root.parent) + "/" != prefix + "data/":
        return None, None, None
    return PT.Z5_ROOTNAME.get(root.name), root.name, seq


def commit_from_evidence(host: str) -> tuple[str | None, str | None]:
    """A host without git: HEAD from the hand-over evidence file (refs/heads/<branch> value)."""
    p = HANDOVER / "evidence" / f"{host}.txt"
    if not p.is_file():  # a host folder with a suffix (e.g. <host>_aborted_stream): the host's own evidence file
        p = HANDOVER / "evidence" / f"{re.split(r'[^A-Za-z0-9]', host)[0]}.txt"
    if not p.is_file():
        return None, None
    text = zguard(p).read_text(errors="replace")
    m = re.search(r"HEAD: ref: refs/heads/(\S+)", text)
    if not m:
        return None, None
    br = re.escape(m.group(1))
    m2 = re.search(rf"refs/heads/{br} \(common dir [^)]*\): ([0-9a-f]{{40}})", text)
    return (m2.group(1), str(p)) if m2 else (None, None)


def label_for(host: str, group: str, sub: str | None) -> str:
    if re.split(r"[^A-Za-z0-9]", host)[0] == "5090":
        lab = PT.Z5_LABELS.get(group) or Z5_LABELS_EXTRA.get(group) or "z5_" + re.sub(r"[^a-z0-9]", "", group.lower())[-10:]
    else:
        lab = HOST_TAG.get(host, re.sub(r"[^a-z0-9]", "", host.lower())) + \
            GROUP_TAG.get(group, re.sub(r"[^a-z0-9]", "", group.lower())[-8:])  # fmt: skip
    if sub:
        lab += SUB_TAG.get(sub, "_" + re.sub(r"[^a-z0-9]", "", sub.lower()))
    return lab


def copy_runs() -> list[dict]:
    """Every copy-run directory: (directory, host, group, sub, checksum source). Round 7's 5090 folder via its MANIFEST."""
    out = []
    z5 = PT.Z5
    if z5.is_dir():
        sums = PT.z5_sums()
        for row in PT.read_csv(z5 / "MANIFEST.tsv", "\t"):
            run = row["run"]
            parts = Path(run).parts
            mine = {k[len(run) + 1:]: v for k, v in sums.items() if k.startswith(run + "/")}
            out.append({"dir": z5 / run, "host": "5090", "group": parts[0], "sub": parts[1] if len(parts) == 3 else None,
                        "rel": run, "sums": mine, "sums_file": str(z5 / "SHA256SUMS"), "manifest_row": row,
                        "folder": "round7-5090-ckpts", "log": z5 / run / "train.log"})  # fmt: skip
    for base, folder in ((HANDOVER, "r8-handover"), (ROBOLAB, "robolab-ckpts")):
        if not base.is_dir():
            continue
        for hostdir in sorted(p for p in base.iterdir() if p.is_dir() and p.name not in ("evidence",)):
            host = hostdir.name
            root_sums = None
            if folder == "robolab-ckpts":
                sf = base / f"SHA256SUMS.{host}"
                root_sums = (sf, read_sums(sf)) if sf.is_file() else (None, {})
            for dp, dns, fns in os.walk(hostdir):
                dns[:] = sorted(d for d in dns if not d.startswith("eval") and d not in ("hydra", ".hydra", "logs", "wandb"))
                if not any(re.fullmatch(r"model_\d+\.safetensors", f) for f in fns):
                    continue
                d = Path(dp)
                rel = str(d.relative_to(hostdir))
                parts = Path(rel).parts
                group, sub = parts[0], (parts[1] if len(parts) == 3 else None)
                if root_sums is not None:
                    pre = f"{host}/{rel}/"
                    sums = {k[len(pre):]: v for k, v in root_sums[1].items() if k.startswith(pre)}
                    sums_file = str(root_sums[0])
                    log = hostdir / "logs" / group / f"{parts[-1]}{'_' + sub if sub else ''}.log"
                else:
                    sf = d / "SHA256SUMS"
                    sums = read_sums(sf) if sf.is_file() else {}
                    sums_file = str(sf)
                    log = d / "train.log"
                out.append({"dir": d, "host": host, "group": group, "sub": sub, "rel": rel, "sums": sums,
                            "sums_file": sums_file, "manifest_row": None, "folder": folder, "log": log})  # fmt: skip
    return out


def host_parquet_sha(c: dict) -> tuple[str | None, str | None]:
    """The training host's own sha256 of the run's task.parquet, as recorded by the copy (MANIFEST.txt 'parquet sha256'
    line, else the host evidence file's task.parquet table). Recorded only (PREREG_R8 9.4); never an eligibility rule."""
    m = Path(c["dir"]) / "MANIFEST.txt"
    if m.is_file():
        for ln in zguard(m).read_text(errors="replace").splitlines():
            if ln.startswith("parquet sha256"):
                f = ln.split()
                if len(f) >= 3 and len(f[2]) == 64:
                    return f[2], f"{m}"
    host = c["host"]
    ev = HANDOVER / "evidence" / f"{re.split(r'[^A-Za-z0-9]', host)[0]}.txt"
    if ev.is_file():
        for ln in zguard(ev).read_text(errors="replace").splitlines():
            f = [x.strip() for x in ln.split("|")]
            if len(f) == 3 and f[0] == c["rel"] and len(f[1]) == 64:
                return f[1], str(ev)
    return None, None


def copy_run_record(c: dict, t_c: float) -> dict:
    """PREREG_R8 2.3 (round 7's R1b generalised to every copy folder) + R2-R4 (with 2.4 on pruned hosts)."""
    d, host = c["dir"], c["host"]
    hclass = "5090" if re.split(r"[^A-Za-z0-9]", host)[0] == "5090" else "robolab"
    prefix = HOST_PREFIX[hclass]
    rec = {"dir": str(d), "machine": host, "host_class": hclass, "copy_folder": c["folder"], "n_files": 0, "n_settled": 0,
           "eligible": False, "reason": "", "owner": "zuxing", "tree": "b39b69b", "sums_file": c["sums_file"]}  # fmt: skip
    its, settled = settled_models8(d, t_c)
    listed = {i: p for i, p in settled.items() if f"model_{i}.safetensors" in c["sums"]}
    rec.update(n_files=len(its), n_settled=len(settled), n_listed=len(listed),
               not_listed=sorted(i for i in settled if i not in listed))  # fmt: skip
    if not listed:
        rec["reason"] = "no checkpoint present, settled and listed in its checksum file at T_c"
        return rec
    L = max(listed)
    h = PT.header(listed[L])
    kind, rootname, seq = as_trained_kind(h["parquet"], prefix)
    want_out = prefix + "runs/" + c["rel"]
    line, commit = PT.first_train_t2_line(c["log"]) if c["log"] and Path(c["log"]).is_file() else (None, None)
    commit_src = f"first train_t2: line of {c['log']}" if commit else None
    if not commit and c.get("manifest_row") and c["manifest_row"].get("commit"):
        commit, commit_src = c["manifest_row"]["commit"], "round-7 5090 MANIFEST.tsv commit column"
    if not commit and hclass == "robolab":
        commit, src = commit_from_evidence(host)
        commit_src = f"host evidence file {src} (worktree HEAD from refs; the host has no git binary)" if commit else None
    rec.update(last=L, first=min(listed), format=h["format"], output_dir=h["output_dir"], parquet_as_trained=h["parquet"],
               reference_parquet_as_trained=h["reference_parquet"], input=kind, rootname=rootname, sequence_id=seq,
               label=label_for(host, c["group"], c["sub"]), key=[hclass if hclass == "5090" else host, h["output_dir"]],
               newer_unsettled=sorted(i for i in its if i not in settled), train_t2_line=line, commit=commit,
               commit_source=commit_src, final_json_checkpoint=None)  # fmt: skip
    fin = d / "final.json"
    if fin.is_file():
        rec["final_json_checkpoint"] = (read_json(fin) or {}).get("checkpoint")
    if h["output_dir"] != want_out:
        rec["reason"] = f"embedded output_dir {h['output_dir']} != as-trained run path {want_out} (R1c)"
        return rec
    if h["format"] != "learner_state":
        rec["reason"] = f"format {h['format']}"
        return rec
    if kind is None:
        rec["reason"] = f"as-trained parquet {h['parquet']} not in a Tier-2-derived data root (R2)"
        return rec
    if h["reference_parquet"] not in (None, h["parquet"]):
        rec["reason"] = f"reference.parquet {h['reference_parquet']} != task.parquet"
        return rec
    sp = PT.server_parquet(kind, seq)
    if not sp.is_file():
        rec["reason"] = f"relocation target {sp} missing on this server (R2)"
        return rec
    rec["parquet"] = str(sp)
    hsha, hsrc = host_parquet_sha(c)
    rec.update(host_parquet_sha256=hsha, host_parquet_sha256_source=hsrc, server_parquet_sha256=sha256_file(sp))
    rec["host_parquet_matches_server"] = None if hsha is None else hsha == rec["server_parquet_sha256"]
    if L < MIN_LAST:
        rec["reason"] = f"last iteration {L} < {MIN_LAST} (R3)"
        return rec
    e = PT.manifest_episode(PT.ROOTS[kind], seq)
    name_ep = re.fullmatch(r"ep(\d+)", Path(c["rel"]).parts[-1])
    if e is None or (name_ep and int(name_ep.group(1)) != e):
        rec["reason"] = f"sequence {seq} -> episode {e} does not match the run folder {c['rel']}"
        return rec
    if c.get("manifest_row") and (rootname != c["manifest_row"].get("data_root") or seq != c["manifest_row"].get("sequence")
                                  or int(c["manifest_row"]["episode"]) != e):  # fmt: skip
        rec["reason"] = "round-7 5090 MANIFEST row != checkpoint (R1b)"
        return rec
    if not commit:
        rec["reason"] = "missing commit evidence (no commit in the first train_t2: line, MANIFEST or host evidence file)"
        return rec
    rec["episode"] = e
    saved = grid_restricted(list(listed), L) if hclass in PRUNED_HOSTS else sorted(listed)
    picks, slots = PT.picks_of(saved, L)
    rec.update(targets=[round(L * (1 - j / (2 * K)), 1) for j in range(0, K + 1)], picks=picks, slots=slots,
               pick_set="grid + last (pruned host, PREREG_R8 2.4)" if hclass in PRUNED_HOSTS else "every settled file",
               eligible=True, reason="eligible (R1c copy run)")  # fmt: skip
    return rec


# ================================================================================================ watcher best (D2)
def watcher_file(run: dict) -> tuple[Path | None, str]:
    d = Path(run["dir"])
    p = d.parent / f"eval_{d.name}" / "metrics.csv"
    if p.is_file():
        return p, "next to the run"
    if run["machine"] == "5090" or run.get("copy_folder") in ("round7-5090-ckpts", "r8-handover", "robolab-ckpts"):
        if run.get("copy_folder") == "round7-5090-ckpts":
            rel = d.relative_to(PT.Z5)
        elif run.get("copy_folder") == "r8-handover" and run["machine"] == "5090":
            rel = d.relative_to(HANDOVER / run["machine"])
        elif run.get("copy_folder") == "r8-handover":
            rel = Path("robolab") / d.relative_to(HANDOVER / run["machine"])
        else:
            rel = Path("robolab") / d.relative_to(ROBOLAB / run["machine"])
        q = OWNSEL_INPUTS / rel.parent / f"eval_{rel.name}" / "metrics.csv"
        if q.is_file():
            return q, "copy in runs/2026-11-02-tier2-g2-ownsel/inputs_5090 (fallback)"
    return None, "no watcher file"


def parse_finished(s: str) -> float | None:
    try:
        t = dt.datetime.fromisoformat(s.strip())
    except (ValueError, AttributeError):
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=TZ8)
    return t.timestamp()


def copy_stable(src: Path, dst: Path, tries: int = 6) -> dict:
    """A copy of a file another process may still append to: kept only when sha256 before == copy == after."""
    if dst.exists():
        raise SystemExit(f"refusing to overwrite {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    for i in range(tries):
        before = sha256_file(src)
        st = os.stat(src)
        tmp = dst.with_name(dst.name + f".tmp_r8w{i}")
        shutil.copyfile(src, tmp)
        c = sha256_file(tmp)
        after = sha256_file(src)
        if before == c == after:
            os.replace(tmp, dst)
            os.chmod(dst, 0o444)
            return {"source": str(src), "copy": str(dst), "sha256_source_before": before, "sha256_copy": c,
                    "sha256_source_after": after, "equal": True, "size": st.st_size,
                    "source_mtime": now_text(st.st_mtime), "copied_at": now_text()}  # fmt: skip
        tmp.unlink()
        time.sleep(5)
    raise SystemExit(f"{src} kept changing while being copied")


WATCH_COPIES: list[dict] = []
WATCH_TMP = "_tmp_full_watcher_r8"  # our temporary full watcher copies (removed after the parse; never listed)


def remove_watch_tmp(zc: Path) -> None:
    """Removes the (by then empty) temporary directory tree of the full watcher copies: directories only, bottom-up."""
    top = zc / WATCH_TMP
    if not top.is_dir():
        return
    left = [p for p in top.rglob("*") if not p.is_dir()]
    if left:
        raise SystemExit(f"temporary full watcher copies left: {left[:3]}")
    for d in sorted((p for p in top.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)):
        d.rmdir()
    top.rmdir()


def watcher_excerpt(c: dict, rows: list[dict], dst: Path) -> dict:
    """The record copy of a watcher file (PREREG_R8 2.6 / 2.8): only the 8 own-input columns WATCH_COLS of every row. The
    full source was copied first (sha256 before == copy == after) and parsed from that copy; the copy record keeps those
    hashes, and the full verbatim copy (a temporary file) is removed after the parse."""
    if dst.exists():
        raise SystemExit(f"refusing to overwrite {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp_r8x")
    tmp.write_text(csv_text([{k: r.get(k) for k in WATCH_COLS} for r in rows], list(WATCH_COLS)))
    os.replace(tmp, dst)
    os.chmod(dst, 0o444)
    return {**c, "source": f"{c['source']} (excerpt: the columns {','.join(WATCH_COLS)} of all {len(rows)} rows; the "
                           f"full source sha256 {c['sha256_source_before']} == its parsed copy == after)",
            "copy": str(dst), "sha256_copy": sha256_file(dst), "equal": c["equal"], "size": dst.stat().st_size}  # fmt: skip


def watcher_best(run: dict, t_c: float, present: set[int], copy_base: Path | None) -> dict:
    p, how = watcher_file(run)
    out = {"watcher_file": str(p) if p else None, "watcher_source": how, "pick": None}
    if p is None:
        out["why"] = "no watcher file"
        return out
    readp, full = p, None
    if copy_base is not None:  # freeze: parse the verified full copy (temporary), keep the column excerpt as the record
        full = copy_stable(p, copy_base / "zuxing_copies_r8" / WATCH_TMP / p.relative_to(ZR))
        readp = Path(full["copy"])
    before = sha256_file(readp)
    with zguard(readp).open(newline="") as fp:
        rd = csv.DictReader(fp)
        hdr = list(rd.fieldnames or [])
        rows = list(rd)
    after = sha256_file(readp)
    missing = [c for c in WATCH_COLS if c not in hdr]
    if missing:
        raise SystemExit(f"{p}: header lacks {missing} (own-input watcher columns expected: {WATCH_COLS})")
    other = [c for c in hdr if c not in WATCH_HEADER]
    if other or len(set(hdr)) != len(hdr):
        raise SystemExit(f"{p}: header has columns outside the own-input watcher header {WATCH_HEADER}: {other}")
    if full is not None:
        c = watcher_excerpt(full, rows, copy_base / "zuxing_copies_r8" / p.relative_to(ZR))
        os.chmod(readp, 0o644)
        readp.unlink()  # our own temporary full copy (its hashes are kept in the copy record)
        WATCH_COPIES.append(c)
        out["watcher_copy"] = c["copy"]
    e = int(run["episode"])
    ok = []
    for r in rows:
        try:
            ep_r, it, auc = int(r["episode"]), int(r["iteration"]), float(r["add_auc"])
        except (TypeError, ValueError):
            continue
        ft = parse_finished(r.get("finished_at") or "")
        if ep_r == e and r.get("status") == "ok" and ft is not None and ft <= t_c:
            ok.append((it, auc))
    out.update(watcher_sha256=before, watcher_sha256_after=after, n_rows=len(rows), n_ok_rows_by_t_c=len(ok))
    if not ok:
        out["why"] = "no ok row finished by T_c"
        return out
    last = max(it for it, _ in ok)
    pruned = run.get("host_class") in PRUNED_HOSTS
    # PREREG_R8 2.6 (5) with C1 / 2.4: on a pruned host the last ok-evaluated iteration counts off the grid only when it is
    # the run's last counted iteration L
    last_ok = (last == run.get("last")) if pruned else True
    elig = sorted({(it, auc) for it, auc in ok if (it % GRID == 0 or (it == last and last_ok)) and it in present})
    out.update(last_ok_iteration=last, n_eligible_rows=len(elig), pruned_host=pruned,
               last_ok_counts_off_grid=last_ok)  # fmt: skip
    if not elig:
        out["why"] = "no ok row on the 10k grid / last iteration with its model present at T_c"
        return out
    best = min(elig, key=lambda x: (-x[1], x[0]))  # argmax add_auc, ties -> lower iteration
    out.update(pick=best[0], pick_add_auc=best[1], why="argmax add_auc over the eligible rows")
    return out


# ================================================================================================ identity (D3)
class Identity:
    def __init__(self, rec: dict):
        self.rec = rec
        self.db_cache: dict[str, str] = {}

    def db(self, path: str) -> str | None:
        if path not in self.db_cache:
            try:
                self.db_cache[path] = data_block_sha(path) if path and Path(path).is_file() else None
            except (OSError, PermissionError):
                self.db_cache[path] = None
        return self.db_cache[path]

    def same(self, sha: str, cand_path: str, other: dict) -> bool:
        if sha in (other.get("pool_sha256"), other.get("training_sha256")):
            return True
        tc = other.get("training_checkpoint")
        return bool(tc) and self.db(tc) is not None and self.db(tc) == self.db(cand_path)

    def role(self, e: int, sha: str, path: str, parquet_sha: str, frame_sha: str | None, kind: str) -> tuple[str, dict | None, str]:
        k = f"{e:02d}"
        inc = self.rec["incumbents"][k]
        if self.same(sha, path, inc):
            return "incumbent", inc, f"== the round-7 pick {inc['job']} (the round-8 incumbent; not screened)"
        for x in self.rec["r7_phase_f_participants"].get(k, []):
            if self.same(sha, path, x):
                return "barred7", x, (f"== {x['job']}, a round-7 phase-F participant of this episode ({x['outcome']}; its "
                                      f"round-7 phase-F runs do not pass the round-8 test): never a finalist (PREREG_R8 3.3)")
        cands = [x for x in self.rec["r7_phase_s_candidates"].values() if int(x["episode"]) == e]
        for x in sorted(cands, key=lambda x: x["job"]):
            if self.same(sha, path, x):
                if x["job"] in self.rec.get("r7_barred_carried", {}):
                    return "barred7", x, f"== {x['job']}, barred in round 7 against the same incumbent (PREREG_R7 5.4)"
                same_input = x["parquet_sha256"] == parquet_sha and (kind == "raw" or x.get("input_reference_sha256") == frame_sha)
                if same_input:
                    return "reuse7", x, f"== round-7 phase-S candidate {x['job']}: its round-7 phase-S results and export are reused"
                return "screen", x, f"same file as round-7 candidate {x['job']} but its input changed: fresh runs"
        return "screen", None, "new file"


# ================================================================================================ resolve
def episode_of_run(r: dict) -> int:
    return int(r["episode"])


def resolve(t_c: float, log, copy_base: Path | None = None) -> dict:
    t0 = time.time()
    rec7 = load_r7records()
    ident = Identity(rec7)
    univ7 = r7_universe()
    r6_ours = {r["dir"] for r in json.loads(PT.R6_RUNS.read_text())["runs"] if r.get("eligible") and r.get("owner") == "haoran"}
    runs = []
    for d in PT.scan_pro6000():
        r = PT.run_record(d, t_c, r6_ours)
        r["host_class"] = "pro6000"
        r["key"] = ["pro6000", os.path.realpath(str(d))]
        runs.append(r)
    for c in copy_runs():
        runs.append(copy_run_record(c, t_c))
    # one run, several copies: the copy with the most iterations (then more listed files, then path order)
    by_key: dict[tuple, list[dict]] = {}
    for r in runs:
        if r["eligible"]:
            by_key.setdefault(tuple(r["key"]), []).append(r)
    for key, rs in by_key.items():
        rs.sort(key=lambda r: (-r["last"], -r.get("n_listed", r.get("n_settled", 0)), r["dir"]))
        for r in rs[1:]:
            r["eligible"] = False
            r["reason"] = f"duplicate copy of the same run (kept: {rs[0]['dir']}, L {rs[0]['last']})"
    # status vs round 7
    for r in runs:
        if not r["eligible"]:
            continue
        r7 = univ7.get(tuple(r["key"]))
        if r7 is None:
            r["r8_status"], r["r7_last"] = "new", None
        else:
            r["r7_last"] = r7["last"]
            r["r8_status"] = "trained further" if r["last"] > r7["last"] else "unchanged" if r["last"] == r7["last"] else \
                "fewer iterations than at round 7 (treated as unchanged)"  # fmt: skip
            if r.get("label") != r7.get("label") and r7.get("label"):
                r["label"] = r7["label"]  # the same run keeps its round-7 job label
        if r["owner"] == "zuxing" and r["host_class"] == "pro6000":
            rel = Path(r["dir"]).relative_to(ZR)
            line, commit = PT.first_train_t2_line(ZU / "logs" / rel.parts[0] / f"{rel.parts[-1]}.log")
            r.update(train_t2_line=line, commit=commit, commit_source=f"first train_t2: line of logs/{rel.parts[0]}/{rel.parts[-1]}.log")
            if not commit:
                r["eligible"], r["reason"] = False, "missing commit evidence (no commit in the first train_t2: line)"
    for r in runs:  # records (PREREG_R8 2.8): a training-log line that round 7's token guard refuses keeps only its commit
        ln = r.get("train_t2_line")
        if ln and not tokens_ok(str(ln)):
            r["train_t2_line"] = f"train_t2: commit={r.get('commit')} (the other fields of this line are not recorded)"
    elig = [r for r in runs if r["eligible"]]
    log(f"scan at T_c {now_text(t_c)} CST: {len(runs)} directories, {len(elig)} eligible runs "
        f"({sum(1 for r in elig if r['owner'] == 'haoran')} ours, {sum(1 for r in elig if r['owner'] == 'zuxing')} Zuxing's); "
        f"new {sum(1 for r in elig if r['r8_status'] == 'new')}, trained further "
        f"{sum(1 for r in elig if r['r8_status'] == 'trained further')}, unchanged "
        f"{sum(1 for r in elig if r['r8_status'].startswith(('unchanged', 'fewer')))}")  # fmt: skip
    # candidates: mechanical picks of new / further-trained runs + watcher best of every run
    cand: dict[tuple, dict] = {}
    order = sorted(elig, key=lambda r: (episode_of_run(r), r["owner"], r["dir"]))
    watch_rows = []
    for r in order:
        d = Path(r["dir"])
        its, settled = settled_models8(d, t_c)
        present = set(settled)
        if r["host_class"] != "pro6000":
            sums = next((x["sums"] for x in COPY_INDEX if str(x["dir"]) == r["dir"]), {})
            present = {i for i in present if f"model_{i}.safetensors" in sums}
        routes = []
        if r["r8_status"] in ("new", "trained further"):
            routes += [(it, f"R4 {slot}") for it, slot in zip(r["picks"], r["slots"])]
        if r["owner"] == "zuxing":
            wb = watcher_best(r, t_c, present, copy_base)
            wb.update(run=r["dir"], episode=r["episode"], r8_status=r["r8_status"])
            watch_rows.append(wb)
            r["watcher"] = {k: v for k, v in wb.items() if k not in ("run",)}
            if wb.get("pick") is not None:
                routes.append((wb["pick"], "watcher best"))
        for it, route in routes:
            key = (r["dir"], it)
            if key in cand:
                cand[key]["routes"].append(route)
                continue
            cand[key] = {"run": r, "iteration": it, "routes": [route]}
    log(f"candidates before roles: {len(cand)} files ({sum(1 for c in cand.values() if any(x.startswith('R4') for x in c['routes']))} "
        f"mechanical, {sum(1 for c in cand.values() if 'watcher best' in c['routes'])} watcher best)")  # fmt: skip
    pool, seen = [], {}
    names7 = set(rec7["r7_phase_s_candidates"]) | {x["job"] for x in rec7["incumbents"].values()}
    for (rdir, it), c in sorted(cand.items(), key=lambda kv: (episode_of_run(kv[1]["run"]), kv[1]["run"]["owner"], kv[0][0], kv[0][1])):
        r = c["run"]
        e = episode_of_run(r)
        src = Path(rdir) / f"model_{it}.safetensors"
        h = PT.header(src)
        sha = sha256_file(src)
        if r["host_class"] == "pro6000":
            kind, _, seq = PT.root_kind_of(h["parquet"])
            pq = h["parquet"]
        else:
            kind, _, seq = as_trained_kind(h["parquet"], HOST_PREFIX[r["host_class"]])
            pq = str(PT.server_parquet(kind, seq)) if kind else None
        job = f"ep{e:02d}_{r['label']}_{it}"
        row = {"job": job, "episode": e, "input": kind, "owner": r["owner"], "machine": r["machine"], "host_class": r["host_class"],
               "copy_folder": r.get("copy_folder"), "tree": r["tree"], "run_dir": rdir, "run_label": r["label"], "iteration": it,
               "run_last": r["last"], "run_status": r["r8_status"], "routes": c["routes"],
               "slot": "; ".join(c["routes"]), "source_checkpoint": str(src), "sha256": sha, "parquet": pq, "sequence_id": seq,
               "data_root": str(PT.ROOTS[kind]) if kind else None, "commit": r.get("commit"),
               "commit_source": r.get("commit_source")}  # fmt: skip
        if r["host_class"] != "pro6000":
            row.update(parquet_as_trained=h["parquet"], output_dir_as_trained=h["output_dir"])
        if kind != r["input"] or seq != r["sequence_id"]:
            row.update(role="refused", reason=f"checkpoint parquet {h['parquet']} differs from the run's")
            pool.append(row)
            continue
        if r["host_class"] != "pro6000":
            want = next((x["sums"] for x in COPY_INDEX if str(x["dir"]) == rdir), {}).get(f"model_{it}.safetensors")
            row["checksum_line"] = want
            if want != sha:
                row.update(role="refused", reason=f"sha256 {sha[:16]} != checksum line {str(want)[:16]}")
                pool.append(row)
                continue
        row["parquet_sha256"] = sha256_file(Path(pq))
        fr = PT.ep_frame_ref(kind, e)
        row["frame_refs"] = str(PT.FRAME_REFS[kind]) if PT.FRAME_REFS[kind] else None
        row["frame_ref_file"] = str(fr) if fr else None
        row["frame_ref_sha256"] = sha256_file(fr) if fr and fr.is_file() else None
        role, x, why = ident.role(e, sha, str(src), row["parquet_sha256"], row["frame_ref_sha256"], kind)
        if role == "screen" and sha in seen:
            role, why = "duplicate", f"same file as {seen[sha]}"
        row.update(role=role, reason=why)
        if role in ("reuse7", "barred7", "incumbent") and x is not None:
            row["job"] = x["job"]
            row.update(r7_job=x["job"], r7_training_sha256=x["training_sha256"], r7_training_checkpoint=x["training_checkpoint"])
            if role == "reuse7":
                rp = x.get("export_record")
                row.update(r7_export_record=rp, r7_export_record_sha256=PT.sha256_file(Path(rp)) if rp else None,
                           r7_export_sha256=x.get("export_sha256"), r7_export_status=x.get("export_status"),
                           r7_s_eligible=x.get("s_eligible"), r7_cons_mean=x.get("cons_mean"),
                           training_checkpoint=x["training_checkpoint"], training_sha256=x["training_sha256"])  # fmt: skip
        if role == "screen":
            if job in names7:
                row["job"] = job = job + "_r8"
            seen[sha] = job
            if r["owner"] == "zuxing":
                row["copy"] = str(ROOT / "zuxing_copies_r8" / src.relative_to(ZR))
                row["copy_of"] = str(src)
                if r["host_class"] != "pro6000":
                    row["relocated_file"] = str(ROOT / "relocated_r8" / src.relative_to(ZR))
                    row["training_checkpoint"] = row["relocated_file"]
                    row["training_sha256"] = None  # filled from RELOCATION_R8.tsv
                else:
                    row["training_checkpoint"] = row["copy"]
                    row["training_sha256"] = sha
            else:
                row["training_checkpoint"] = str(src)
                row["training_sha256"] = sha
        pool.append(row)
    names = [p["job"] for p in pool if p["role"] in ("screen", "reuse7")]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise SystemExit(f"job names not unique: {dup}")
    log(f"pool: {len(pool)} files: " + ", ".join(f"{k} {sum(1 for p in pool if p['role'] == k)}" for k in
                                                ("screen", "reuse7", "barred7", "incumbent", "duplicate", "refused")))  # fmt: skip
    log(f"resolved in {time.time() - t0:.0f} s")
    return {"runs": runs, "pool": pool, "watch": watch_rows, "t_c": t_c}


COPY_INDEX: list[dict] = []


def scan_copies_index() -> None:
    COPY_INDEX.clear()
    COPY_INDEX.extend(copy_runs())


# ================================================================================================ reports
RUN_COLS = ["dir", "machine", "host_class", "copy_folder", "owner", "label", "input", "episode", "eligible", "reason", "r8_status",
            "r7_last", "first", "last", "n_files", "n_settled", "n_listed", "picks", "slots", "pick_set", "commit",
            "commit_source", "output_dir", "parquet", "parquet_as_trained", "server_parquet_sha256", "host_parquet_sha256",
            "host_parquet_sha256_source", "host_parquet_matches_server"]  # fmt: skip
POOL_COLS = ["job", "episode", "role", "reason", "routes", "input", "owner", "machine", "host_class", "copy_folder", "tree",
             "run_label", "run_dir", "run_status", "iteration", "run_last", "sha256", "checksum_line", "source_checkpoint",
             "training_checkpoint", "training_sha256", "copy", "relocated_file", "parquet", "parquet_sha256",
             "parquet_as_trained", "sequence_id", "data_root", "frame_ref_file", "frame_ref_sha256", "commit", "r7_job",
             "r7_export_status", "r7_s_eligible", "r7_cons_mean"]  # fmt: skip


def md_report(res: dict, rec7: dict, title: str) -> str:
    runs, pool = res["runs"], res["pool"]
    incs = rec7["incumbents"]
    L = [f"## {title}", "", f"Snapshot {now_text(res['t_c'])} CST.", ""]
    L += ["### Candidates per episode", "",
          "| ep | round-8 incumbent (round-7 pick) | new screen | reuse round-7 S | barred | == incumbent | duplicate | refused | evaluated |",
          "|---|---|---:|---:|---:|---:|---:|---:|---|"]  # fmt: skip
    for e in range(30):
        ps = [p for p in pool if p["episode"] == e]
        n = {k: sum(1 for p in ps if p["role"] == k) for k in ("screen", "reuse7", "barred7", "incumbent", "duplicate", "refused")}
        ev = "yes" if n["screen"] + n["reuse7"] else "no (keeps its round-7 pick)"
        L.append(f"| {e:02d} | `{incs[f'{e:02d}']['job']}` | {n['screen']} | {n['reuse7']} | {n['barred7']} | {n['incumbent']} | "
                 f"{n['duplicate']} | {n['refused']} | {ev} |")
    L += ["", f"Totals: {sum(1 for p in pool if p['role'] == 'screen')} new screen candidates (x 3 runs = "
          f"{3 * sum(1 for p in pool if p['role'] == 'screen')} phase-S evaluations), "
          f"{sum(1 for p in pool if p['role'] == 'reuse7')} reusing round-7 phase-S results, "
          f"{sum(1 for p in pool if p['role'] == 'barred7')} barred, {sum(1 for p in pool if p['role'] == 'incumbent')} equal to "
          f"the incumbent, {sum(1 for p in pool if p['role'] == 'duplicate')} duplicates, "
          f"{sum(1 for p in pool if p['role'] == 'refused')} refused.", ""]  # fmt: skip
    L += ["### Candidate list", "", "| ep | job | role | route(s) | input | run (status, L) | note |", "|---|---|---|---|---|---|---|"]
    for p in sorted(pool, key=lambda p: (p["episode"], p["role"], p["job"])):
        rd = p["run_dir"].replace(str(ZR) + "/", "zx:").replace(str(ME / "fc_runs") + "/", "")
        L.append(f"| {p['episode']:02d} | `{p['job']}` | {p['role']} | {', '.join(p['routes'])} | {p['input']} | {rd} "
                 f"({p['run_status']}, L {p['run_last']}) | {p['reason']} |")
    L += ["", "### Runs", "", "| run | host | ep | input | status vs round 7 | L (round 7) | commit | host parquet sha256 == this server's | picks |",
          "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in sorted([r for r in runs if r["eligible"]], key=lambda r: (r["episode"], r["dir"])):
        rd = r["dir"].replace(str(ZR) + "/", "zx:").replace(str(ME / "fc_runs") + "/", "")
        hm = r.get("host_parquet_matches_server")
        hm = "" if r["host_class"] == "pro6000" else ("not recorded" if hm is None else ("yes" if hm else "NO (bytes differ)"))
        L.append(f"| {rd} | {r['machine']} | {r['episode']:02d} | {r['input']} | {r['r8_status']} | {r['last']} "
                 f"({r.get('r7_last')}) | {r.get('commit') or ''} | {hm} | "
                 f"{r.get('picks') if r['r8_status'] in ('new', 'trained further') else '-'} |")
    L += ["", "### Excluded directories and copies (not candidates)", "", "| directory | reason |", "|---|---|"]
    for r in sorted([r for r in runs if not r["eligible"]], key=lambda r: r["dir"]):
        rd = r["dir"].replace(str(ZR) + "/", "zx:").replace(str(ME / "fc_runs") + "/", "")
        L.append(f"| {rd} | {r['reason']} |")
    for k, v in sorted(PT.ZX_SKIP_TOP.items()):
        L.append(f"| zx:{k}/ (not walked) | {v} |")
    L += ["", "### Watcher files (own-input columns only)", "", "| run | episode | status | file | rows ok by T_c | last ok it | pick | why |",
          "|---|---|---|---|---:|---:|---:|---|"]  # fmt: skip
    for w in sorted(res["watch"], key=lambda w: (w["episode"], w["run"])):
        rd = w["run"].replace(str(ZR) + "/", "zx:")
        wf = (w.get("watcher_file") or "").replace(str(ZR) + "/", "zx:")
        L.append(f"| {rd} | {w['episode']:02d} | {w['r8_status']} | {wf} ({w.get('watcher_source')}) | "
                 f"{w.get('n_ok_rows_by_t_c', '')} | {w.get('last_ok_iteration', '')} | {w.get('pick') or '-'} | {w.get('why')} |")
    return "\n".join(L) + "\n"


# ================================================================================================ freeze: copies + relocation
def copy_file(src: Path, dst: Path) -> dict:
    """sha256 before == copy == after (round 7's copy_verified with the allowlist guard)."""
    return PT.copy_verified(src, dst, True)


def sums_line_kept(line: str) -> bool:
    f = line.split(None, 1)
    return len(f) == 2 and len(f[0]) == 64 and bool(SUMS_KEEP.fullmatch(Path(f[1].strip().lstrip("*")).name))


def copy_meta(src: Path, dst: Path) -> dict:
    """The record copy of one of Zuxing's metadata files (PREREG_R8 2.8). MANIFEST.txt: only its run / source / copied /
    status / train log / task.parquet / parquet sha256 / models / files lines; checksum files: only the lines of model,
    rsi, final.json, training-log, MANIFEST.txt and hydra files; in every file, a line that round 7's token guard refuses
    is left out. A file that keeps every line is copied verbatim (sha256 before == copy == after); otherwise the copy is
    an excerpt holding the kept lines, and the record keeps the full source's sha256 before and after the read."""
    if dst.exists():
        raise SystemExit(f"refusing to overwrite {dst}")
    before = sha256_file(src)
    st = os.stat(src)
    lines = zguard(src).read_bytes().decode("utf-8", errors="replace").splitlines(keepends=True)
    n = src.name
    if n == "MANIFEST.txt":
        keep = [ln for ln in lines if MANIFEST_KEEP.match(ln)]
    elif n == "SHA256SUMS" or n.startswith("SHA256SUMS."):
        keep = [ln for ln in lines if sums_line_kept(ln)]
    else:
        keep = lines
    keep = [ln for ln in keep if tokens_ok(ln)]
    if keep == lines:
        c = copy_file(src, dst)
        return c if c["sha256_source_before"] == before else {**c, "equal": False}
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp_r8m")
    tmp.write_text("".join(keep))
    os.replace(tmp, dst)
    os.chmod(dst, 0o444)
    after = sha256_file(src)
    return {"source": f"{src} (excerpt: {len(keep)} of {len(lines)} lines kept, PREREG_R8 2.8)", "copy": str(dst),
            "sha256_source_before": before, "sha256_copy": sha256_file(dst), "sha256_source_after": after,
            "equal": before == after, "size": dst.stat().st_size, "source_mtime": now_text(st.st_mtime),
            "copied_at": now_text()}  # fmt: skip


def freeze_outputs(res: dict, base: Path, log, lines: list[str], limit: int | None = None) -> int:
    """Copies, relocations and the frozen records under base (ROOT for the real freeze, a dryrun/trial_* folder for a
    trial)."""
    pool = res["pool"]
    if limit is not None:  # trial: the first `limit` screen rows, one per (host class) first, for plumbing coverage
        pick, seen_h = [], set()
        for p in pool:
            if p["role"] == "screen" and p["host_class"] not in seen_h:
                pick.append(p["job"])
                seen_h.add(p["host_class"])
        for p in pool:
            if p["role"] == "screen" and p["job"] not in pick and len(pick) < limit:
                pick.append(p["job"])
        pool = [p for p in pool if p["role"] != "screen" or p["job"] in pick]
        res = {**res, "pool": pool}
    copies = []
    zc, rd = base / "zuxing_copies_r8", base / "relocated_r8"
    for p in pool:
        if p["role"] != "screen" or p["owner"] != "zuxing":
            continue
        src = Path(p["copy_of"])
        rel = src.relative_to(ZR)
        p["copy"] = str(zc / rel)
        c = copy_file(src, zc / rel)
        copies.append(c)
        if not c["equal"] or c["sha256_copy"] != p["sha256"]:
            raise SystemExit(f"copy of {src} not verified: {c}")
        rsi = src.with_name(src.name.replace("model_", "rsi_"))
        if rsi.is_file():
            copies.append(copy_file(rsi, zc / rel.with_name(rsi.name)))
        if p["host_class"] != "pro6000":
            p["relocated_file"] = str(rd / rel)
            p["training_checkpoint"] = p["relocated_file"]
        else:
            p["training_checkpoint"] = p["copy"]
    meta_srcs: set[Path] = set()
    for r in res["runs"]:
        if not r["eligible"] or r["owner"] != "zuxing":
            continue
        d = Path(r["dir"])
        for sub in ("hydra/.hydra/config.yaml", "hydra/.hydra/hydra.yaml", "hydra/.hydra/overrides.yaml", "hydra/config.yaml",
                    "hydra/hydra.yaml", "hydra/overrides.yaml", "final.json", "MANIFEST.txt", "SHA256SUMS"):  # fmt: skip
            if (d / sub).is_file():
                meta_srcs.add(d / sub)
        if r.get("sums_file") and Path(r["sums_file"]).is_file():
            meta_srcs.add(Path(r["sums_file"]))
        cs = r.get("commit_source") or ""
        m = re.search(r"host evidence file (\S+)", cs)
        if m:
            meta_srcs.add(Path(m.group(1)))
    copies += WATCH_COPIES  # the watcher files, copied (and parsed from the copy) during the resolve
    if (PT.Z5 / "MANIFEST.tsv").is_file():
        meta_srcs.add(PT.Z5 / "MANIFEST.tsv")
    for src in sorted(meta_srcs):
        if (zc / src.relative_to(ZR)).exists():
            continue
        c = copy_meta(src, zc / src.relative_to(ZR))
        copies.append(c)
        if not c["equal"]:
            raise SystemExit(f"copy of {src} not verified: {c}")
    for r in res["runs"]:
        if r["eligible"] and r.get("train_t2_line") and r["owner"] == "zuxing" and r["host_class"] != "pro6000":
            lg = Path(r["dir"]) / "train.log"
            if lg.is_file():
                copies.append(PT.write_excerpt(lg, zc / Path(r["dir"]).relative_to(ZR) / "train_log_first_train_t2_line.txt",
                                               r["train_t2_line"], True))  # fmt: skip
    bad = [c for c in copies if c.get("equal") is not True]
    if bad:
        raise SystemExit(f"unverified copies: {bad[:3]}")
    remove_watch_tmp(zc)
    # relocation (round 5's relocate_bytes / verify_pair, as round 7's relocate)
    RT = PT.load_rt()
    PT.COPIES, PT.RELOC_DIR = zc, rd  # rebinding only: round 7's verify_row checks these folders
    rrows, rjson = [], []
    for p in pool:
        if p["role"] != "screen" or not p.get("relocated_file"):
            continue
        orig, dst = Path(p["copy"]), Path(p["relocated_file"])
        osha = PT.sha256_file(orig)
        problems = [] if osha == p["sha256"] == p.get("checksum_line") else ["original copy sha256 != pool / checksum line"]
        raw = orig.read_bytes()
        new, info = RT.relocate_bytes(raw, p["parquet"], p["parquet_as_trained"])
        if dst.exists():
            raise SystemExit(f"refusing to overwrite {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp_r8reloc")
        tmp.write_bytes(new)
        os.replace(tmp, dst)
        os.chmod(dst, 0o444)
        v = RT.verify_pair(orig, dst, p["parquet_as_trained"], p["parquet"])
        problems += v["problems"]
        if PT.sha256_file(orig) != osha:
            problems.append("original copy changed during relocation")
        psha = PT.sha256_file(Path(p["parquet"]))
        if psha != p["parquet_sha256"]:
            problems.append("new parquet sha256 != pool")
        row = {"job": p["job"], "episode": p["episode"], "iteration": p["iteration"], "input": p["input"], "original_copy": str(orig),
               "original_sha256": osha, "sha256sums_line": p.get("checksum_line"), "relocated_file": str(dst),
               "relocated_sha256": PT.sha256_file(dst), "fields_rewritten": "task.parquet,reference.parquet",
               "old_parquet_path": p["parquet_as_trained"], "new_parquet_path": p["parquet"], "parquet_sha256_new_path": psha,
               "n_tensors": v.get("n_tensors"), "tensors_identical": v.get("tensors_identical"),
               "safe_open_identical": v.get("safe_open_identical"), "data_block_sha256": v.get("data_block_sha256"),
               "other_metadata_identical": v.get("other_metadata_identical"), "config_diff_paths": v.get("config_diff_paths"),
               "header_len_old": info["header_len_old"], "header_len_new": info["header_len_new"],
               "problems": "; ".join(problems), "status": "ok" if not problems else "failed", "written_at": now_text()}  # fmt: skip
        vr = PT.verify_row(RT, row)
        if vr:
            row["problems"] = "; ".join(problems + vr)
            row["status"] = "failed"
        p["training_sha256"] = row["relocated_sha256"] if row["status"] == "ok" else None
        rrows.append(row)
        rjson.append({**row, "verify": {k: v.get(k) for k in v if k != "problems"}})
        log(f"relocated {p['job']}: {row['status']} {row['relocated_sha256'][:12]}")
    inp = base / "inputs_r8"
    write_new(base / "RELOCATION_R8.tsv", csv_text(rrows, PT.RTSV_COLS, "\t"))
    write_new(base / "RELOCATION_R8.json", json.dumps({"written_at": now_text(), "relocate_code": str(PT.R5_RELOC_TOOL),
                                                       "relocate_code_sha256": PT.R5_RELOC_SHA, "rows": rjson}, indent=1,
                                                      default=str) + "\n")  # fmt: skip
    rlist = [base / "RELOCATION_R8.tsv", base / "RELOCATION_R8.json"] + [Path(r["relocated_file"]) for r in rrows]
    write_new(base / "RELOCATION_R8.sha256", "".join(f"{PT.sha256_file(p)}  {p.relative_to(base)}\n" for p in rlist))
    ccols = ["source", "copy", "sha256_source_before", "sha256_copy", "sha256_source_after", "equal", "size", "source_mtime", "copied_at"]
    write_new(base / "COPIES_r8.tsv", csv_text(copies, ccols, "\t"))
    tracks: dict[str, str] = {}
    for p in pool:
        if p["role"] in ("screen", "reuse7", "incumbent", "barred7") and p.get("parquet_sha256"):
            tracks[p["parquet"]] = p["parquet_sha256"]
            if p.get("frame_ref_file"):
                tracks[p["frame_ref_file"]] = p["frame_ref_sha256"]
    for k, v in PT.ROOTS.items():
        m = v / "whole_body/soma/track2_manifest.json"
        tracks[str(m)] = PT.sha256_file(m)
    write_new(inp / "TRACKS_R8.sha256", "".join(f"{h}  {p}\n" for p, h in sorted(tracks.items())))
    write_new(inp / "runs_r8.json", json.dumps({"t_c": now_text(res["t_c"]), "runs": res["runs"]}, indent=1, default=str) + "\n")
    write_new(inp / "runs_r8.csv", csv_text(res["runs"], RUN_COLS))
    write_new(inp / "pool_r8.csv", csv_text(pool, POOL_COLS))
    wcols = ["run", "episode", "r8_status", "watcher_file", "watcher_source", "watcher_sha256", "watcher_sha256_after", "n_rows",
             "n_ok_rows_by_t_c", "last_ok_iteration", "pruned_host", "last_ok_counts_off_grid", "n_eligible_rows", "pick",
             "pick_add_auc", "why"]  # fmt: skip
    write_new(inp / "WATCHER_R8.tsv", csv_text(res["watch"], wcols, "\t"))
    zrows = [{"run": r["dir"], "machine": r["machine"], "episode": r.get("episode"), "commit": r.get("commit"),
              "commit_source": r.get("commit_source"), "train_t2_line": r.get("train_t2_line")}
             for r in res["runs"] if r["eligible"] and r["owner"] == "zuxing"]  # fmt: skip
    write_new(inp / "ZUXING_CODE_R8.tsv", csv_text(zrows, ["run", "machine", "episode", "commit", "commit_source", "train_t2_line"], "\t"))
    write_new(inp / "resolve_r8.log", "\n".join(lines) + "\n")
    tell = ["Zuxing: round 8 of the Tier-2 noisy-only selection (t2_consensus_r8_20261103) made sha256-verified READ-ONLY copies",
            f"of these files of your runs at the snapshot {now_text(res['t_c'])} CST (nothing in your folder was written; no git",
            "command; none of your scripts was run). Of each training log only the first 'train_t2:' line was read.", ""]  # fmt: skip
    tell += [f"{c['source']} -> {c['copy']} ({c['sha256_copy']})" for c in copies]
    write_new(base / "TELL_ZUXING_r8.txt", "\n".join(tell) + "\n")
    meta = {"t_c": now_text(res["t_c"]) + " CST", "t_c_epoch": res["t_c"], "script": str(Path(__file__).resolve()),
            "script_sha256": PT.sha256_file(Path(__file__).resolve()), "r7_pool_tool_sha256": PT_SHA, "prereg_r8_sha256": PREREG_SHA,
            "K": K, "min_last_iteration": MIN_LAST, "settle_seconds": SETTLE_S, "grid": GRID, "pruned_hosts": list(PRUNED_HOSTS),
            "n_screen": sum(1 for p in pool if p["role"] == "screen"), "rule": __doc__.split("with the round-8 deltas:", 1)[1].split(
                "Only checkpoint headers", 1)[0]}  # fmt: skip
    write_new(base / "POOL_R8.json", json.dumps({**meta, "rows": pool}, indent=1, default=str) + "\n")
    lst = [base / "POOL_R8.json", inp / "pool_r8.csv", inp / "runs_r8.json", inp / "runs_r8.csv", inp / "WATCHER_R8.tsv",
           inp / "TRACKS_R8.sha256", inp / "ZUXING_CODE_R8.tsv", inp / "resolve_r8.log", inp / "R7_RECORDS_R8.json",
           base / "COPIES_r8.tsv", base / "TELL_ZUXING_r8.txt", base / "RELOCATION_R8.tsv", base / "RELOCATION_R8.json",
           base / "RELOCATION_R8.sha256"]  # fmt: skip
    lst += sorted(p for p in zc.rglob("*") if p.is_file()) + sorted(p for p in rd.rglob("*") if p.is_file())
    write_new(base / "POOL_R8.sha256", "".join(f"{PT.sha256_file(p)}  {p.relative_to(base)}\n" for p in lst if p.is_file()))
    bad_rel = [r for r in rrows if r["status"] != "ok"]
    log(f"frozen: {len(pool)} pool rows, {len(copies)} copies, {len(rrows)} relocations ({len(bad_rel)} failed); POOL_R8.sha256 "
        f"{PT.sha256_file(base / 'POOL_R8.sha256')}")  # fmt: skip
    return 0 if not bad_rel else 1


# ================================================================================================ commands
def docs_problems() -> list[str]:
    """PREREG_R8.md == PREREG_R8.sha256 == hard-coded; PREREG_R8_PINS.md == its sha file, pins this script and every pin
    holds; LOCK_R8.sha256 holds (PREREG_R8 10)."""
    out = []
    pre, pre_sha = ROOT / "PREREG_R8.md", ROOT / "PREREG_R8.sha256"
    got = PT.sha256_file(pre) if pre.is_file() else "missing"
    rec = (pre_sha.read_text().split() or [""])[0] if pre_sha.is_file() else ""
    if not got == rec == PREREG_SHA:
        out.append(f"PREREG_R8.md {got[:16]} != sha file {rec[:16]} / hard-coded {PREREG_SHA[:16]}")
    pins, pins_sha = ROOT / "PREREG_R8_PINS.md", ROOT / "PREREG_R8_PINS.sha256"
    if not pins.is_file():
        return out + ["PREREG_R8_PINS.md missing"]
    if PT.sha256_file(pins) != ((pins_sha.read_text().split() or [""])[0] if pins_sha.is_file() else ""):
        out.append("PREREG_R8_PINS.md != PREREG_R8_PINS.sha256")
    mine = PT.sha256_file(Path(__file__).resolve())
    plines = [ln.split() for ln in pins.read_text().splitlines()]
    plist = [(f[1], f[2]) for f in plines if len(f) == 3 and f[0] == "pin" and len(f[2]) == 64]
    if ("r8_pool.py", mine) not in plist:
        out.append(f"PREREG_R8_PINS.md does not pin this script {mine[:16]}")
    for rel, h0 in plist:
        q = Path(rel) if rel.startswith("/") else ROOT / rel
        if (PT.sha256_file(q) if q.is_file() else "missing") != h0:
            out.append(f"pin {rel} changed")
    lock = ROOT / "LOCK_R8.sha256"
    if not lock.is_file():
        out.append("LOCK_R8.sha256 missing")
    else:
        for ln in lock.read_text().splitlines():
            f = ln.split(None, 1)
            if len(f) == 2 and len(f[0]) == 64:
                q = Path(f[1].strip()) if f[1].strip().startswith("/") else ROOT / f[1].strip()
                if (PT.sha256_file(q) if q.is_file() else "missing") != f[0]:
                    out.append(f"LOCK_R8.sha256: {f[1].strip()} changed")
    return out


def make_log(lines: list[str]):
    def log(s: str) -> None:
        line = f"[{now_text()}] {s}"
        print(line, flush=True)
        lines.append(line)

    return log


def cmd_dryrun(a) -> int:
    lines: list[str] = []
    log = make_log(lines)
    t_c = time.time()
    log(f"r8_pool DRY RUN: snapshot NOW ({now_text(t_c)} CST) instead of T_c; nothing is copied, relocated or frozen")
    scan_copies_index()
    res = resolve(t_c, log)
    rec7 = load_r7records()
    stamp = dt.datetime.fromtimestamp(t_c, TZ8).strftime("%Y%m%d_%H%M%S")
    out = ROOT / "dryrun"
    out.mkdir(parents=True, exist_ok=True)
    write_new(out / f"pool_dryrun_{stamp}.json", json.dumps({"snapshot": now_text(t_c), "script_sha256": PT.sha256_file(Path(__file__).resolve()),
                                                             "runs": res["runs"], "pool": res["pool"], "watch": res["watch"],
                                                             "log": lines}, indent=1, default=str) + "\n")  # fmt: skip
    write_new(out / f"pool_dryrun_{stamp}.md", md_report(res, rec7, f"Pool dry run ({stamp})"))
    print(f"written: dryrun/pool_dryrun_{stamp}.json / .md")
    return 0


def cmd_freeze(a) -> int:
    if time.time() < T_C:
        print(f"FATAL: freeze runs at or after T_c {now_text(T_C)} CST only")
        return 2
    if (ROOT / "POOL_R8.json").exists() or (ROOT / "POOL_R8.sha256").exists():
        print("FATAL: POOL_R8.json / POOL_R8.sha256 exists: the round-8 pool is already frozen")
        return 2
    problems = docs_problems()
    if problems:
        print("FATAL: " + "; ".join(problems))
        return 2
    lines: list[str] = []
    log = make_log(lines)
    log(f"r8_pool FREEZE at T_c = {now_text(T_C)} CST (started {now_text()}); script sha256 {PT.sha256_file(Path(__file__).resolve())}")
    scan_copies_index()
    res = resolve(T_C, log, copy_base=ROOT)
    return freeze_outputs(res, ROOT, log, lines)


def cmd_trial(a) -> int:
    lines: list[str] = []
    log = make_log(lines)
    t_c = time.time()
    stamp = dt.datetime.fromtimestamp(t_c, TZ8).strftime("%Y%m%d_%H%M%S")
    base = ROOT / "dryrun" / f"trial_{stamp}"
    log(f"r8_pool TRIAL (plumbing only, snapshot NOW {now_text(t_c)}): {a.limit} screen rows frozen into {base}")
    base.mkdir(parents=True, exist_ok=False)
    (base / "inputs_r8").mkdir()
    scan_copies_index()
    res = resolve(t_c, log, copy_base=base)
    shutil.copyfile(ROOT / "inputs_r8/R7_RECORDS_R8.json", base / "inputs_r8/R7_RECORDS_R8.json")
    return freeze_outputs(res, base, log, lines, limit=a.limit)


def cmd_verify(a) -> int:
    base = Path(a.base) if a.base else ROOT
    RT = PT.load_rt()
    PT.COPIES, PT.RELOC_DIR = base / "zuxing_copies_r8", base / "relocated_r8"
    with (base / "RELOCATION_R8.tsv").open(newline="") as fp:
        rows = list(csv.DictReader(fp, delimiter="\t"))
    bad = 0
    for r in rows:
        pr = PT.verify_row(RT, r)
        bad += bool(pr)
        if pr:
            print(f"{r['job']}: {pr}")
    nl = 0
    for ln in (base / "POOL_R8.sha256").read_text().splitlines():
        f = ln.split(None, 1)
        if len(f) == 2 and PT.sha256_file(base / f[1].strip()) != f[0]:
            nl += 1
            print(f"POOL_R8.sha256: {f[1].strip()} changed")
    print(f"{len(rows)} relocation rows verified, {bad} with problems; POOL_R8.sha256 lines changed: {nl}")
    return 0 if not bad and not nl else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("dryrun", "freeze", "trial", "verify"))
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--base", default=None, help="verify: a trial folder instead of the round-8 folder")
    a = ap.parse_args()
    if not sys.flags.dont_write_bytecode or not sys.flags.isolated:
        print("run with python -I -B")
        return 2
    return {"dryrun": cmd_dryrun, "freeze": cmd_freeze, "trial": cmd_trial, "verify": cmd_verify}[a.mode](a)


if __name__ == "__main__":
    raise SystemExit(main())

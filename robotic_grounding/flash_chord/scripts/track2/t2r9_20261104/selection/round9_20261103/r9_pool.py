#!/usr/bin/env python3
"""r9_pool.py: the round-9 candidate pool (fc_runs/submissions/t2_consensus_r9_20261103; PREREG_R9.md sections 1-2).

Round 8's pool builder r8_pool.py (bebeb013...) is loaded from its verified bytes, and through it round 7's resolver
scripts/t2cons_r7_pool.py (7a08fa6d...); their helpers (header, root / sequence parsing, server_parquet,
manifest_episode, ep_frame_ref, picks_of (R4), run_record, scan_pro6000, copy_runs, copy_run_record, settled_models8,
first_train_t2_line, commit_from_evidence, copy_verified / copy_meta / write_excerpt, data_block_sha, round 5's
relocate_bytes / verify_pair, verify_row) are called unchanged. Round-9 deltas (PREREG_R9):
  D2  candidate pool = (a) every file present and settled at T_c9 in Zuxing's reports/tier2-G2 and reports/tier2-K3fix
      (eligibility: Tier-2-derived as-trained data root present on this server, the file's episode, learner_state format,
      iteration >= 10,000, training-commit evidence; each file's final status at T_c9 is recorded, never a condition),
      plus (b) round 8's mechanical rule (round 7's R1-R4, round 8's R1c copy runs and 5090 grid restriction) re-applied
      at T_c9 to the runs that are new or have more iterations than at round 8's snapshot (runs_r8.json). Round 8's
      watcher-best route is not part of round 9's pool.
  D3  T_c9 = the time the freeze starts (recorded); a file counts iff mtime <= T_c9 - 300 s and ctime <= T_c9.
  D1  roles (deterministic order (episode, owner, source, iteration)): incumbent (== the round-8 pick), barred (round 8's
      barred files and round 8's phase-F participants other than its pick), reuse8 / reuse7 (== a round-8 / round-7
      phase-S candidate of the episode with the same input: its recorded phase-S results and export are reused), duplicate,
      screen (copy, relocation if trained elsewhere, one export, 3 runs), refused. Identity = file sha256, else the
      sha256 of the tensor data block.
Only checkpoint headers, checksum files, MANIFEST files, the first train_t2: line of training logs, host evidence files,
final.json / hydra yaml copies and the report checkpoints are opened in Zuxing's folder (round 7's token guard, then an
allowlist); the four 2026-11-03 bundles and the ownsel / evaluation bundles are never entered. His folder is only read;
copies carry sha256 before == copy == after.

  conda_envs/flashchord/bin/python -I -B r9_pool.py dryrun   snapshot NOW, nothing copied: dryrun/pool_dryrun_<stamp>.{json,md}
  conda_envs/flashchord/bin/python -I -B r9_pool.py freeze   T_c9 = now: POOL_R9.json + inputs_r9/, copies, relocations,
                                                            POOL_R9.sha256 (all mode 444); refuses if it exists
  conda_envs/flashchord/bin/python -I -B r9_pool.py trial --limit N   a small frozen pool (snapshot NOW) under dryrun/trial_<stamp>/
                                                            for the plan-only check of r9_eval.py (never used by a decision)
  conda_envs/flashchord/bin/python -I -B r9_pool.py verify   read-only re-check of RELOCATION_R9.tsv and POOL_R9.sha256
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
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
ROOT = ME / "fc_runs/submissions/t2_consensus_r9_20261103"
R8ROOT = ME / "fc_runs/submissions/t2_consensus_r8_20261103"
P8_FILE, P8_SHA = R8ROOT / "r8_pool.py", "bebeb0132cf45c5320aae060bc50e1d4f1d46e9895f0cdac46cc0797ba0d4628"
PREREG_SHA = "13c3b2a61e39cc761633283ae42bf94683df8d1ffd701d1076e3d61212a4c2a0"
TZ8 = dt.timezone(dt.timedelta(hours=8))
R8_T_C_TEXT = "2026-11-03 12:00:00"
REPORTS = ZU / "reports"
REPORT_DIRS = {"tier2-G2": "G2", "tier2-K3fix": "K3fix"}  # PREREG_R9 D2
REPORT_NAME = re.compile(r"ep(\d{2})-tier2-(G2|K3fix)\.safetensors")
REPORT_MODE = {"tier2-G2": "files", "tier2-K3fix": "files"}  # PREREG_R9 D2 / 2.1 ("files": his files; see PREREG_R9)
# PREREG_R9 9 (records only, never a condition): files the two check reports found not to be their run's final
# checkpoint (G2: ZX_G2_30_CHECK_20261103.md; K3fix: ZX_K3FIX_18_AND_RULES_CHECK_20261103.md)
CHECK_REPORT_NOT_FINAL = {"G2": {10: "still training (5090 redo, 180k max)", 14: "still training (5090 redo)",
                                 19: "series continues to 720k (his README)", 20: "run stopped 20:26:57; folder ends at 140500",
                                 22: "run stopped; folder ends at 136000", 27: "run stopped; folder ends at 117500",
                                 29: "run stopped; folder ends at 133000"},
                          "K3fix": {1: "our K3 stretch run stopped at 22500 of 90000 (10-21)", 11: "5090 round-2 run paused (59500 of 180000)",
                                    14: "a5 continuation stopped (276000 of 360000)", 17: "5090 round-2 run paused (74500 of 180000)",
                                    21: "5090 round-2 run paused (80500 of 180000)", 25: "a5 continuation stopped (297000 of 360000)",
                                    27: "5090 round-2 run paused (62500 of 180000)"}}  # fmt: skip
NEW_SKIP_TOP9 = {  # Zuxing's runs/ top folders that are not walked (PREREG_R9 2.2)
    "2026-11-03-tier2-g2-final-ckpts": "Zuxing's final-checkpoint bundle (copies, candidate lists, records): its files enter "
                                       "only through reports/tier2-G2 (PREREG_R9 D2); never entered",
    "2026-11-03-tier2-k3fix-final-ckpts": "Zuxing's final-checkpoint bundle (copies, candidate lists, evaluation scripts): its "
                                          "files enter only through reports/tier2-K3fix (PREREG_R9 D2); never entered",
    "2026-11-03-tier2-k3fix-ownsel": "selection / evaluation bundle with checkpoint copies; never entered",
    "2026-11-03-tier2-g2-reports-eval30": "report / evaluation bundle (not a training-run directory); never entered",
}  # fmt: skip
NEVER_ENTER9 = tuple(NEW_SKIP_TOP9) + ("2026-11-02-tier2-g2-ownsel",)  # round 8 read only its watcher files there


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


P8 = _load("r8_pool_pinned_r9", P8_FILE, P8_SHA)  # round 8's pool builder (its import re-binds round 7's resolver)
PT = P8.PT
PT.ZX_SKIP_TOP = {**PT.ZX_SKIP_TOP, **NEW_SKIP_TOP9}  # rebinding only: the pinned scan reads this table at call time
P8.NEVER_ENTER = tuple(P8.NEVER_ENTER) + NEVER_ENTER9  # rebinding only: round 8's guard reads this name at call time
K, MIN_LAST, SETTLE_S, GRID = PT.K, PT.MIN_LAST, P8.SETTLE_S, P8.GRID
HOST_PREFIX = {"pro6000": str(ZU) + "/", "ours": str(ME) + "/", **P8.HOST_PREFIX}


# ================================================================================================ guards and helpers
_ZGUARD8 = P8.zguard


def zguard(p) -> Path:
    """Round 8's guard (round 7's token guard first, then its allowlist for Zuxing's folder), plus: our round-9 folder
    (token guard only) and the report checkpoints reports/tier2-{G2,K3fix}/epNN-tier2-*.safetensors."""
    p = Path(p)
    s = str(p)
    if s.startswith(str(ROOT) + "/"):
        return P8._PT_GUARD(p)
    if s == str(REPORTS) or s.startswith(str(REPORTS) + "/"):
        P8._PT_GUARD(p)
        if p == REPORTS or (p.parent == REPORTS and p.name in REPORT_DIRS):
            return p
        if p.parent.parent == REPORTS and p.parent.name in REPORT_DIRS and REPORT_NAME.fullmatch(p.name):
            return p
        raise PermissionError(f"r9_pool may not open {p} (allowlist)")
    return _ZGUARD8(p)


P8.zguard = zguard  # rebinding only: round 8's helpers open files through this name at call time
PT.guard = zguard


def sha256_file(p) -> str:
    return P8.sha256_file(p)


def data_block_sha(p) -> str:
    return P8.data_block_sha(p)


def now_text(t: float | None = None) -> str:
    return P8.now_text(t)


def write_new(p: Path, text: str) -> None:
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp_r9p")
    tmp.write_text(text)
    os.replace(tmp, p)
    os.chmod(p, 0o444)


def csv_text(rows: list[dict], cols: list[str], delim: str = ",") -> str:
    return P8.csv_text(rows, cols, delim)


def full_header(p: Path) -> tuple[dict, dict]:
    """(safetensors metadata, resolved config) of a checkpoint header (never the tensors)."""
    with zguard(p).open("rb") as fp:
        n = struct.unpack("<Q", fp.read(8))[0]
        h = json.loads(fp.read(n))
    md = h.get("__metadata__") or {}
    return md, json.loads(md.get("user.resolved_config_json") or "{}")


def host_of(output_dir: str | None) -> str | None:
    """Training host class from the embedded output_dir."""
    od = output_dir or ""
    if od.startswith(str(ME) + "/"):
        return "ours"
    if od.startswith(str(ZU) + "/"):
        return "pro6000"
    for k in ("5090", "robolab"):
        if od.startswith(P8.HOST_PREFIX[k]):
            return k
    return None


def kind_of(parquet: str | None, host: str | None) -> tuple[str | None, str | None]:
    """(input kind, sequence id) of an as-trained parquet path."""
    if host in ("ours", "pro6000"):
        k, _, seq = PT.root_kind_of(parquet)
        return k, seq
    if host in ("5090", "robolab"):
        k, _, seq = P8.as_trained_kind(parquet, P8.HOST_PREFIX[host])
        return k, seq
    return None, None


# ================================================================================================ round-8 records
def load_r8records() -> dict:
    return json.loads(zguard(ROOT / "inputs_r9/R8_RECORDS_R9.json").read_text())


def r8_universe() -> dict[tuple, dict]:
    """Round 8's eligible runs keyed by its key (host class, embedded output_dir / realpath), with L at round 8's T_c."""
    data = json.loads((R8ROOT / "inputs_r8/runs_r8.json").read_text())
    out = {}
    for r in data["runs"]:
        if r.get("eligible"):
            out[tuple(r["key"])] = {"dir": r["dir"], "last": int(r["last"]), "label": r.get("label")}
    return out


# ================================================================================================ identity (D1)
class Identity:
    def __init__(self, rec: dict):
        self.rec = rec
        self.db_cache: dict[str, str | None] = {}

    def db(self, path: str | None) -> str | None:
        if not path:
            return None
        if path not in self.db_cache:
            try:
                self.db_cache[path] = data_block_sha(path) if Path(path).is_file() else None
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
            return "incumbent", inc, f"== the round-8 pick {inc['job']} (the round-9 incumbent; not screened)"
        for x in self.rec["barred"].get(k, []):
            if self.same(sha, path, x):
                return "barred", x, f"== {x['job']}: {x['why']}; never a finalist (PREREG_R9 2.7)"
        cands = [x for x in self.rec["phase_s_candidates"].values() if int(x["episode"]) == e]
        for x in sorted(cands, key=lambda x: (0 if x["round"] == "r8" else 1, x["job"])):
            if self.same(sha, path, x):
                same_input = x["parquet_sha256"] == parquet_sha and (kind == "raw" or x.get("input_reference_sha256") == frame_sha)
                rnd = x["round"][1]
                if same_input:
                    return f"reuse{rnd}", x, f"== round-{rnd} phase-S candidate {x['job']}: its round-{rnd} phase-S results and export are reused"
                return "screen", x, f"same file as round-{rnd} candidate {x['job']} but its input changed: fresh runs"
        return "screen", None, "new file"


# ================================================================================================ report files (D2 a)
def evidence_train_line(host: str, rel: str) -> tuple[str | None, str | None, str | None]:
    """The hand-over evidence file's 'training-log first lines' row of a run (5090: evidence/5090.txt): (line, commit,
    source)."""
    p = P8.HANDOVER / "evidence" / f"{host}.txt"
    if not p.is_file():
        return None, None, None
    for ln in zguard(p).read_text(errors="replace").splitlines():
        f = [x.strip() for x in ln.split("|", 2)]
        if len(f) == 3 and f[0] == rel and f[2].startswith("train_t2:"):
            m = re.search(r"commit=([0-9a-f]{7,40})", f[2])
            return f[2], (m.group(1) if m else None), f"host evidence file {p} (training-log first line of {rel})"
    return None, None, None


def run_lookup(output_dir: str, host: str, runs: list[dict]) -> list[dict]:
    """The run records on this server (pro6000 run directories, copy runs) whose embedded output_dir is output_dir."""
    out = []
    for r in runs:
        if host in ("ours", "pro6000"):
            if r.get("host_class") == "pro6000" and os.path.realpath(r["dir"]) == os.path.realpath(output_dir):
                out.append(r)
        elif r.get("host_class") == host and r.get("output_dir") == output_dir:
            out.append(r)
    return out


def final_status(it: int, max_it, runs_here: list[dict], t_c: float) -> tuple[str, str]:
    """PREREG_R9 2.1 (record only): 'yes' iff no later checkpoint of the same run is present on this server at T_c9 and
    the file is the run's planned end (iteration == its max_iterations, or a final.json of the run names it)."""
    later, fj = [], []
    for r in runs_here:
        its, settled = P8.settled_models8(Path(r["dir"]), t_c)
        later += [i for i in its if i > it]
        if r.get("final_json_checkpoint"):
            fj.append(r["final_json_checkpoint"])
    if later:
        return "no", f"later checkpoint(s) of the same run present on this server at T_c9 (largest {max(later)})"
    named = f"model_{it}.safetensors" in fj
    if max_it is not None and int(max_it) == it:
        return "yes", f"iteration == the run's max_iterations {max_it}; no later checkpoint on this server"
    if named:
        return "yes", "the run's final.json names this checkpoint; no later checkpoint on this server"
    return "no", (f"iteration {it} < the run's max_iterations {max_it} and no final.json names it (run still training or "
                  f"stopped; {len(runs_here)} run record(s) on this server)")  # fmt: skip


def report_records(t_c: float, runs: list[dict], log) -> list[dict]:
    """One record per file of reports/tier2-{G2,K3fix} (PREREG_R9 2.1): present / settled at T_c9 and eligibility."""
    out = []
    for sub, tag in REPORT_DIRS.items():
        d = REPORTS / sub
        names = sorted(os.listdir(zguard(d))) if d.is_dir() else []
        for n in names:
            f = d / n
            m = REPORT_NAME.fullmatch(n)
            rec = {"file": str(f), "folder": sub, "tag": tag, "name": n, "eligible": False, "reason": ""}
            if not m or m.group(2) != tag:
                rec["reason"] = "not an epNN-tier2-<folder>.safetensors file name (ignored)"
                out.append(rec)
                continue
            e = int(m.group(1))
            st = os.stat(f)
            rec.update(episode=e, mtime=now_text(st.st_mtime), ctime=now_text(st.st_ctime), size=st.st_size)
            if not (st.st_mtime <= t_c - SETTLE_S and st.st_ctime <= t_c):
                rec["reason"] = "not settled at T_c9 (mtime > T_c9 - 300 s or ctime > T_c9): not present (D3)"
                out.append(rec)
                continue
            md, cfg = full_header(f)
            tr = cfg.get("training") or {}
            it = int(md.get("user.iteration")) if str(md.get("user.iteration") or "").isdigit() else None
            od = cfg.get("output_dir")
            host = host_of(od)
            pq = (cfg.get("task") or {}).get("parquet")
            ref = (cfg.get("reference") or {}).get("parquet") if isinstance(cfg.get("reference"), dict) else None
            kind, seq = kind_of(pq, host)
            rec.update(iteration=it, format=md.get("format"), output_dir_as_trained=od, host_class=host, parquet_as_trained=pq,
                       reference_parquet_as_trained=ref, input=kind, sequence_id=seq, max_iterations=tr.get("max_iterations"),
                       resumed_from=tr.get("checkpoint"), sha256=sha256_file(f))  # fmt: skip
            here = run_lookup(od, host, runs) if od and host else []
            rec["run_records_here"] = [r["dir"] for r in here]
            if it is not None:
                rec["final_at_t_c"], rec["final_evidence"] = final_status(it, tr.get("max_iterations"), here, t_c)
            rec["check_report_final"] = ("no: " + CHECK_REPORT_NOT_FINAL[tag][e]) if e in CHECK_REPORT_NOT_FINAL[tag] else "yes"
            if md.get("format") != "learner_state":
                rec["reason"] = f"format {md.get('format')}"
            elif it is None or it < MIN_LAST:
                rec["reason"] = f"iteration {it} < {MIN_LAST} (R3)"
            elif host is None:
                rec["reason"] = f"embedded output_dir {od} names no known training host"
            elif kind is None:
                rec["reason"] = f"as-trained parquet {pq} not in a Tier-2-derived data root (R2)"
            elif ref not in (None, pq):
                rec["reason"] = f"reference.parquet {ref} != task.parquet"
            elif not PT.server_parquet(kind, seq).is_file():
                rec["reason"] = f"{PT.server_parquet(kind, seq)} missing on this server (R2)"
            elif PT.manifest_episode(PT.ROOTS[kind], seq) != e:
                rec["reason"] = f"sequence {seq} -> episode {PT.manifest_episode(PT.ROOTS[kind], seq)} != the file's ep{e:02d}"
            if rec["reason"]:
                out.append(rec)
                continue
            rec["parquet"] = str(PT.server_parquet(kind, seq))
            # provenance and training-commit evidence (PREREG_R9 2.1 (5); round 8's C2 sources)
            line = commit = src = None
            if host == "ours":
                own = Path(od) / f"model_{it}.safetensors"
                rec["our_original"] = str(own)
                rec["our_original_sha256"] = sha256_file(own) if own.is_file() else None
                if rec["our_original_sha256"] != rec["sha256"]:
                    rec["reason"] = f"our run's original {own} missing or != the report file (sha256)"
                    out.append(rec)
                    continue
                r6 = {r["dir"] for r in json.loads(PT.R6_RUNS.read_text())["runs"] if r.get("eligible") and r.get("owner") == "haoran"}
                if od not in r6:
                    rec["reason"] = "our run is not in round 6's eligible run list (round 7's R1)"
                    out.append(rec)
                    continue
                commit, src = "ours", "our run in round 6's eligible run list (round 7's R1); export tree fc_r2 61fa5d3"
            elif host == "pro6000":
                rel = Path(od).relative_to(ZR)
                lg = ZU / "logs" / rel.parts[0] / f"{rel.parts[-1]}.log"
                line, commit = PT.first_train_t2_line(lg)
                src = f"first train_t2: line of logs/{rel.parts[0]}/{rel.parts[-1]}.log" if commit else None
                own = Path(od) / f"model_{it}.safetensors"
                rec["run_file"] = str(own) if own.is_file() else None
                rec["run_file_sha256"] = sha256_file(own) if own.is_file() else None
            else:
                for r in here:
                    if r.get("commit"):
                        line, commit, src = r.get("train_t2_line"), r["commit"], r.get("commit_source")
                        break
                rel = od[len(P8.HOST_PREFIX[host] + "runs/"):] if od.startswith(P8.HOST_PREFIX[host] + "runs/") else None
                if not commit and host == "5090" and rel:
                    line, commit, src = evidence_train_line("5090", rel)
                if not commit and host == "5090":
                    c, s = P8.commit_from_evidence("5090")
                    commit, src = c, (f"host evidence file {s} (worktree HEAD from refs)" if c else None)
                if not commit and host == "robolab" and rel:  # a host without git whose evidence file lists the run
                    for ev in sorted((P8.HANDOVER / "evidence").glob("*.txt")):
                        if ev.stem == "5090" or not re.fullmatch(r"[A-Za-z0-9]+", ev.stem) or not any(ln.startswith(rel + " | ") for ln in zguard(ev).read_text(errors="replace").splitlines()):
                            continue
                        c, s = P8.commit_from_evidence(ev.stem)
                        if c:
                            commit, src = c, (f"host evidence file {s} (lists the run {rel}; worktree HEAD from refs; the host "
                                              "has no git binary)")  # fmt: skip
                            break
                for r in here:  # the same file in a copy run (byte identity with its checksum line, recorded only)
                    sums = next((x["sums"] for x in COPY_INDEX if str(x["dir"]) == r["dir"]), {})
                    if sums.get(f"model_{it}.safetensors"):
                        rec["copy_run_file"] = str(Path(r["dir"]) / f"model_{it}.safetensors")
                        rec["copy_run_checksum_line"] = sums[f"model_{it}.safetensors"]
            if line and not P8.tokens_ok(str(line)):
                line = f"train_t2: commit={commit} (the other fields of this line are not recorded)"
            rec.update(train_t2_line=line, commit=commit, commit_source=src)
            if not commit:
                rec["reason"] = "missing training-commit evidence (no train_t2: commit, MANIFEST commit or host evidence)"
                out.append(rec)
                continue
            rec.update(eligible=True, reason=f"eligible (PREREG_R9 D2: reports/{sub})")
            out.append(rec)
    log(f"report files at T_c9: {len(out)} files, {sum(1 for r in out if r['eligible'])} eligible; final at T_c9: "
        f"{sum(1 for r in out if r.get('final_at_t_c') == 'yes')} yes, {sum(1 for r in out if r.get('final_at_t_c') == 'no')} no")
    return out


# ================================================================================================ resolve
COPY_INDEX: list[dict] = []


def scan_copies_index() -> None:
    COPY_INDEX.clear()
    COPY_INDEX.extend(P8.copy_runs())


def all_runs(t_c: float) -> list[dict]:
    """Round 8's universe re-scanned at T_c9 (round 7's R1 pro6000 directories, round 8's R1c copy runs; one run copied
    twice counts once)."""
    r6_ours = {r["dir"] for r in json.loads(PT.R6_RUNS.read_text())["runs"] if r.get("eligible") and r.get("owner") == "haoran"}
    runs = []
    for d in PT.scan_pro6000():
        r = PT.run_record(d, t_c, r6_ours)
        r["host_class"] = "pro6000"
        r["key"] = ["pro6000", os.path.realpath(str(d))]
        runs.append(r)
    for c in COPY_INDEX:
        runs.append(P8.copy_run_record(c, t_c))
    by_key: dict[tuple, list[dict]] = {}
    for r in runs:
        if r["eligible"]:
            by_key.setdefault(tuple(r["key"]), []).append(r)
    for key, rs in by_key.items():
        rs.sort(key=lambda r: (-r["last"], -r.get("n_listed", r.get("n_settled", 0)), r["dir"]))
        for r in rs[1:]:
            r["eligible"] = False
            r["reason"] = f"duplicate copy of the same run (kept: {rs[0]['dir']}, L {rs[0]['last']})"
    for r in runs:
        if r["eligible"] and r["owner"] == "zuxing" and r["host_class"] == "pro6000":
            rel = Path(r["dir"]).relative_to(ZR)
            line, commit = PT.first_train_t2_line(ZU / "logs" / rel.parts[0] / f"{rel.parts[-1]}.log")
            r.update(train_t2_line=line, commit=commit, commit_source=f"first train_t2: line of logs/{rel.parts[0]}/{rel.parts[-1]}.log")
            if not commit:
                r["eligible"], r["reason"] = False, "missing commit evidence (no commit in the first train_t2: line)"
    for r in runs:
        ln = r.get("train_t2_line")
        if ln and not P8.tokens_ok(str(ln)):
            r["train_t2_line"] = f"train_t2: commit={r.get('commit')} (the other fields of this line are not recorded)"
    return runs


def resolve(t_c: float, log) -> dict:
    t0 = time.time()
    rec8 = load_r8records()
    ident = Identity(rec8)
    univ8 = r8_universe()
    names_old = set(rec8["phase_s_candidates"]) | {x["job"] for x in rec8["incumbents"].values()} | \
        {x["job"] for v in rec8["barred"].values() for x in v}  # fmt: skip
    runs = all_runs(t_c)
    elig = [r for r in runs if r["eligible"]]
    for r in elig:  # status against round 8's snapshot (PREREG_R9 D2 b)
        r8 = univ8.get(tuple(r["key"]))
        if r8 is None:
            r["r9_status"], r["r8_last"] = "new", None
        else:
            r["r8_last"] = r8["last"]
            r["r9_status"] = "trained further" if r["last"] > r8["last"] else "unchanged" if r["last"] == r8["last"] else \
                "fewer iterations than at round 8 (treated as unchanged)"  # fmt: skip
            if r8.get("label") and r.get("label") != r8["label"]:
                r["label"] = r8["label"]  # the same run keeps its earlier job label
    log(f"scan at T_c9 {now_text(t_c)} CST: {len(runs)} directories, {len(elig)} eligible runs; new "
        f"{sum(1 for r in elig if r['r9_status'] == 'new')}, trained further {sum(1 for r in elig if r['r9_status'] == 'trained further')}, "
        f"unchanged {sum(1 for r in elig if r['r9_status'].startswith(('unchanged', 'fewer')))} (vs round 8's {R8_T_C_TEXT} CST)")  # fmt: skip
    reports = report_records(t_c, runs, log)
    # candidates: (a) report files, (b) R4 picks of new / further-trained runs (keyed by file path)
    cand: dict[str, dict] = {}
    by_dir = {r["dir"]: r for r in elig}
    for rp in reports:
        if not rp["eligible"]:
            continue
        if REPORT_MODE[rp["folder"]] == "files":
            cand[rp["file"]] = {"kind": "report", "rec": rp, "routes": [f"report {rp['tag']} (D2 a)"]}
            continue
        # PREREG_R9 2.1 fallback (REPORT_MODE 'latest'): the latest checkpoint present at T_c9 of the file's run, verified
        # from the run's records on this server (an eligible run of round 8's rule), instead of the file itself
        here = [by_dir[d] for d in rp.get("run_records_here") or [] if d in by_dir]
        if not here:
            rp.update(eligible=False, reason="fallback rule: the file's run has no eligible run record on this server")
            continue
        r = sorted(here, key=lambda r: (-r["last"], r["dir"]))[0]
        f = str(Path(r["dir"]) / f"model_{r['last']}.safetensors")
        rp["fallback_candidate"] = f
        cand.setdefault(f, {"kind": "run", "run": r, "iteration": r["last"], "routes": []})["routes"].append(
            f"latest checkpoint at T_c9 of the run of reports/{rp['folder']}/{rp['name']} (D2 fallback)")
    for r in sorted(elig, key=lambda r: (int(r["episode"]), r["owner"], r["dir"])):
        if r["r9_status"] not in ("new", "trained further"):
            continue
        for it, slot in zip(r["picks"], r["slots"]):
            f = str(Path(r["dir"]) / f"model_{it}.safetensors")
            cand.setdefault(f, {"kind": "run", "run": r, "iteration": it, "routes": []})["routes"].append(f"R4 {slot} (D2 b)")
    log(f"candidates before roles: {len(cand)} files ({sum(1 for c in cand.values() if c['kind'] == 'report')} report files, "
        f"{sum(1 for c in cand.values() if c['kind'] == 'run')} mechanical picks)")
    rows = []
    for f, c in cand.items():
        src = Path(f)
        if c["kind"] == "report":
            rp = c["rec"]
            e, it, host, kind, seq = rp["episode"], rp["iteration"], rp["host_class"], rp["input"], rp["sequence_id"]
            owner = "haoran" if host == "ours" else "zuxing"
            label = f"zxrep{rp['tag'].lower()}"
            row = {"episode": e, "input": kind, "owner": owner, "machine": host if host != "ours" else "pro6000",
                   "host_class": host if host != "ours" else "pro6000", "copy_folder": f"reports/{rp['folder']}",
                   "tree": "fc_r2" if owner == "haoran" else "b39b69b", "run_dir": rp["output_dir_as_trained"],
                   "run_label": label, "iteration": it, "run_last": None, "run_status": "report file",
                   "routes": c["routes"], "slot": "; ".join(c["routes"]), "source_checkpoint": f, "sha256": rp["sha256"],
                   "parquet": rp["parquet"], "sequence_id": seq, "data_root": str(PT.ROOTS[kind]), "commit": rp["commit"],
                   "commit_source": rp["commit_source"], "final_at_t_c": rp.get("final_at_t_c"),
                   "final_evidence": rp.get("final_evidence"), "check_report_final": rp.get("check_report_final"),
                   "report_file": f, "max_iterations": rp.get("max_iterations")}  # fmt: skip
            if host in ("5090", "robolab"):
                row.update(parquet_as_trained=rp["parquet_as_trained"], output_dir_as_trained=rp["output_dir_as_trained"])
            if owner == "haoran":
                row["source_checkpoint"] = rp["our_original"]  # our original in our run folder (== the report file)
            sort_key = (e, owner, f"0 {f}", it)
        else:
            r, it = c["run"], c["iteration"]
            e = int(r["episode"])
            h = PT.header(src)
            sha = sha256_file(src)
            if r["host_class"] == "pro6000":
                kind, _, seq = PT.root_kind_of(h["parquet"])
                pq = h["parquet"]
            else:
                kind, _, seq = P8.as_trained_kind(h["parquet"], P8.HOST_PREFIX[r["host_class"]])
                pq = str(PT.server_parquet(kind, seq)) if kind else None
            row = {"episode": e, "input": kind, "owner": r["owner"], "machine": r["machine"], "host_class": r["host_class"],
                   "copy_folder": r.get("copy_folder"), "tree": r["tree"], "run_dir": r["dir"], "run_label": r["label"],
                   "iteration": it, "run_last": r["last"], "run_status": r["r9_status"], "routes": c["routes"],
                   "slot": "; ".join(c["routes"]), "source_checkpoint": f, "sha256": sha, "parquet": pq, "sequence_id": seq,
                   "data_root": str(PT.ROOTS[kind]) if kind else None, "commit": r.get("commit"),
                   "commit_source": r.get("commit_source"), "final_at_t_c": "yes" if it == r["last"] else "no",
                   "final_evidence": f"mechanical pick of a run with L {r['last']} at T_c9"}  # fmt: skip
            if r["host_class"] != "pro6000":
                row.update(parquet_as_trained=h["parquet"], output_dir_as_trained=h["output_dir"])
            if kind != r["input"] or seq != r["sequence_id"]:
                row.update(role="refused", reason=f"checkpoint parquet {h['parquet']} differs from the run's",
                           job=f"ep{e:02d}_{r['label']}_{it}")  # fmt: skip
                rows.append((e, r["owner"], f"1 {f}", it, row))
                continue
            if r["host_class"] != "pro6000":
                want = next((x["sums"] for x in COPY_INDEX if str(x["dir"]) == r["dir"]), {}).get(f"model_{it}.safetensors")
                row["checksum_line"] = want
                if want != sha:
                    row.update(role="refused", reason=f"sha256 {sha[:16]} != checksum line {str(want)[:16]}",
                               job=f"ep{e:02d}_{r['label']}_{it}")  # fmt: skip
                    rows.append((e, r["owner"], f"1 {f}", it, row))
                    continue
            sort_key = (e, r["owner"], f"1 {f}", it)
        rows.append((*sort_key, row))
    rows.sort(key=lambda x: x[:4])
    pool, seen = [], {}
    for *_, row in rows:
        e, it = row["episode"], row["iteration"]
        job = f"ep{e:02d}_{row['run_label']}_{it}"
        row.setdefault("job", job)
        if row.get("role") == "refused":
            pool.append(row)
            continue
        row["parquet_sha256"] = sha256_file(Path(row["parquet"]))
        kind = row["input"]
        fr = PT.ep_frame_ref(kind, e)
        row["frame_refs"] = str(PT.FRAME_REFS[kind]) if PT.FRAME_REFS[kind] else None
        row["frame_ref_file"] = str(fr) if fr else None
        row["frame_ref_sha256"] = sha256_file(fr) if fr and fr.is_file() else None
        role, x, why = ident.role(e, row["sha256"], row["source_checkpoint"], row["parquet_sha256"], row["frame_ref_sha256"], kind)
        if role == "screen" and row["sha256"] in seen:
            role, why = "duplicate", f"same file as {seen[row['sha256']]}"
        if role == "screen":  # a byte-different copy with the tensors of a file selected earlier in round 9
            db = ident.db(row["source_checkpoint"])
            for p2 in pool:
                if p2.get("role") == "screen" and p2["episode"] == e and db is not None and ident.db(p2["source_checkpoint"]) == db:
                    role, why = "duplicate", f"same tensors as {p2['job']} (relocated / re-written copy)"
                    p2["routes"] = list(p2["routes"]) + [f"{r} [also {row['source_checkpoint']}]" for r in row["routes"]]
                    break
        row.update(role=role, reason=why)
        if role in ("reuse8", "reuse7", "barred", "incumbent") and x is not None:
            row["job"] = x["job"]
            row.update(earlier_job=x["job"], earlier_training_sha256=x.get("training_sha256"),
                       earlier_training_checkpoint=x.get("training_checkpoint"))  # fmt: skip
            if role == "reuse8":
                row.update(r8_job=x["job"], r8_training_sha256=x["training_sha256"], r8_training_checkpoint=x["training_checkpoint"],
                           r8_export_record=x.get("export_record"), r8_export_record_sha256=x.get("export_record_sha256"),
                           r8_export_sha256=x.get("export_sha256"), r8_export_status=x.get("export_status"),
                           r8_s_eligible=x.get("s_eligible"), training_checkpoint=x["training_checkpoint"],
                           training_sha256=x["training_sha256"])  # fmt: skip
            if role == "reuse7":
                rp = x.get("export_record")
                row.update(r7_job=x["job"], r7_training_sha256=x["training_sha256"], r7_training_checkpoint=x["training_checkpoint"],
                           r7_export_record=rp, r7_export_record_sha256=sha256_file(Path(rp)) if rp else None,
                           r7_export_sha256=x.get("export_sha256"), r7_export_status=x.get("export_status"),
                           r7_s_eligible=x.get("s_eligible"), training_checkpoint=x["training_checkpoint"],
                           training_sha256=x["training_sha256"])  # fmt: skip
        if role == "screen":
            if row["job"] in names_old:
                row["job"] = row["job"] + "_r9"
            seen[row["sha256"]] = row["job"]
            src = Path(row["source_checkpoint"])
            if row["owner"] == "zuxing":
                row["copy"] = str(ROOT / "zuxing_copies_r9" / src.relative_to(ZU))
                row["copy_of"] = str(src)
                if row["host_class"] != "pro6000":
                    row["relocated_file"] = str(ROOT / "relocated_r9" / src.relative_to(ZU))
                    row["training_checkpoint"] = row["relocated_file"]
                    row["training_sha256"] = None  # filled from RELOCATION_R9.tsv
                else:
                    row["training_checkpoint"] = row["copy"]
                    row["training_sha256"] = row["sha256"]
            else:
                row["training_checkpoint"] = str(src)
                row["training_sha256"] = row["sha256"]
        pool.append(row)
    names = [p["job"] for p in pool if p["role"] in ("screen", "reuse8", "reuse7")]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise SystemExit(f"job names not unique: {dup}")
    for p in pool:
        for tok in ("report_r",):
            if tok in p["job"].lower():
                raise SystemExit(f"job name {p['job']} holds a guarded token")
    log(f"pool: {len(pool)} files: " + ", ".join(f"{k} {sum(1 for p in pool if p['role'] == k)}" for k in
                                                ("screen", "reuse8", "reuse7", "barred", "incumbent", "duplicate", "refused")))  # fmt: skip
    log(f"resolved in {time.time() - t0:.0f} s")
    return {"runs": runs, "reports": reports, "pool": pool, "t_c": t_c}


# ================================================================================================ reports
RUN_COLS = ["dir", "machine", "host_class", "copy_folder", "owner", "label", "input", "episode", "eligible", "reason", "r9_status",
            "r8_last", "first", "last", "n_files", "n_settled", "n_listed", "picks", "slots", "pick_set", "commit",
            "commit_source", "output_dir", "parquet", "parquet_as_trained", "server_parquet_sha256", "host_parquet_sha256",
            "host_parquet_sha256_source", "host_parquet_matches_server"]  # fmt: skip
REPORT_COLS = ["file", "folder", "episode", "eligible", "reason", "iteration", "max_iterations", "final_at_t_c", "final_evidence",
               "check_report_final", "host_class", "output_dir_as_trained", "input", "sequence_id", "sha256", "commit",
               "commit_source", "run_records_here", "run_file", "run_file_sha256", "copy_run_file", "copy_run_checksum_line",
               "our_original", "mtime", "ctime"]  # fmt: skip
POOL_COLS = ["job", "episode", "role", "reason", "routes", "input", "owner", "machine", "host_class", "copy_folder", "tree",
             "run_label", "run_dir", "run_status", "iteration", "run_last", "final_at_t_c", "check_report_final", "sha256",
             "checksum_line", "source_checkpoint", "training_checkpoint", "training_sha256", "copy", "relocated_file", "parquet",
             "parquet_sha256", "parquet_as_trained", "sequence_id", "data_root", "frame_ref_file", "frame_ref_sha256", "commit",
             "earlier_job", "r8_export_status", "r7_export_status"]  # fmt: skip


def short(s: str | None) -> str:
    return str(s or "").replace(str(ZU) + "/", "zx:").replace(str(ME / "fc_runs") + "/", "")


def md_report(res: dict, rec8: dict, title: str) -> str:
    runs, pool, reports = res["runs"], res["pool"], res["reports"]
    incs = rec8["incumbents"]
    L = [f"## {title}", "", f"Snapshot {now_text(res['t_c'])} CST (T_c9 of this run).", ""]
    L += ["### Candidates per episode", "",
          "| ep | round-9 incumbent (round-8 pick) | new screen | reuse round-8 S | reuse round-7 S | barred | == incumbent | duplicate | refused | evaluated |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]  # fmt: skip
    roles = ("screen", "reuse8", "reuse7", "barred", "incumbent", "duplicate", "refused")
    for e in range(30):
        ps = [p for p in pool if p["episode"] == e]
        n = {k: sum(1 for p in ps if p["role"] == k) for k in roles}
        ev = "yes" if n["screen"] + n["reuse8"] + n["reuse7"] else "no (keeps its round-8 pick)"
        L.append(f"| {e:02d} | `{incs[f'{e:02d}']['job']}` | {n['screen']} | {n['reuse8']} | {n['reuse7']} | {n['barred']} | "
                 f"{n['incumbent']} | {n['duplicate']} | {n['refused']} | {ev} |")
    L += ["", f"Totals: {sum(1 for p in pool if p['role'] == 'screen')} new screen candidates (x 3 runs = "
          f"{3 * sum(1 for p in pool if p['role'] == 'screen')} phase-S evaluations), "
          f"{sum(1 for p in pool if p['role'] == 'reuse8')} reusing round-8 phase-S results, "
          f"{sum(1 for p in pool if p['role'] == 'reuse7')} reusing round-7 phase-S results, "
          f"{sum(1 for p in pool if p['role'] == 'barred')} barred, {sum(1 for p in pool if p['role'] == 'incumbent')} equal to "
          f"the incumbent, {sum(1 for p in pool if p['role'] == 'duplicate')} duplicates, "
          f"{sum(1 for p in pool if p['role'] == 'refused')} refused.", ""]  # fmt: skip
    L += ["### Candidate list", "", "| ep | job | role | route(s) | input | source (status, L) | final at T_c9 | note |",
          "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for p in sorted(pool, key=lambda p: (p["episode"], p["role"], p["job"])):
        L.append(f"| {p['episode']:02d} | `{p['job']}` | {p['role']} | {', '.join(p['routes'])} | {p['input']} | "
                 f"{short(p.get('report_file') or p['run_dir'])} ({p['run_status']}, L {p['run_last']}) | "
                 f"{p.get('final_at_t_c')}{'' if not p.get('check_report_final') else ' (check report: ' + p['check_report_final'] + ')'} | {p['reason']} |")
    L += ["", "### Report files (reports/tier2-G2, reports/tier2-K3fix)", "",
          "| file | ep | it | trained on | as-trained run | input | commit (source) | final at T_c9 | check report | eligible / reason |",
          "|---|---|---:|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in reports:
        L.append(f"| {r['folder']}/{r['name']} | {r.get('episode', '')} | {r.get('iteration', '')} | {r.get('host_class', '')} | "
                 f"{short(r.get('output_dir_as_trained'))} | {r.get('input', '')} | {r.get('commit') or ''} ({r.get('commit_source') or ''}) | "
                 f"{r.get('final_at_t_c', '')}: {r.get('final_evidence', '')} | {r.get('check_report_final', '')} | "
                 f"{'yes' if r['eligible'] else 'NO'}: {r['reason']} |")
    L += ["", "### Runs (eligible; status against round 8's snapshot)", "",
          "| run | host | ep | input | status vs round 8 | L (round 8) | commit | picks |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted([r for r in runs if r["eligible"]], key=lambda r: (r["episode"], r["dir"])):
        L.append(f"| {short(r['dir'])} | {r['machine']} | {r['episode']:02d} | {r['input']} | {r['r9_status']} | {r['last']} "
                 f"({r.get('r8_last')}) | {r.get('commit') or ''} | "
                 f"{r.get('picks') if r['r9_status'] in ('new', 'trained further') else '-'} |")
    L += ["", "### Excluded directories and copies (not candidates)", "", "| directory | reason |", "|---|---|"]
    for r in sorted([r for r in runs if not r["eligible"]], key=lambda r: r["dir"]):
        L.append(f"| {short(r['dir'])} | {r['reason']} |")
    for k, v in sorted(PT.ZX_SKIP_TOP.items()):
        L.append(f"| zx:runs/{k}/ (not walked) | {v} |")
    return "\n".join(L) + "\n"


# ================================================================================================ freeze: copies + relocation
def freeze_outputs(res: dict, base: Path, log, lines: list[str], limit: int | None = None) -> int:
    """Copies, relocations and the frozen records under base (ROOT for the real freeze, a dryrun/trial_* folder for a
    trial)."""
    pool = res["pool"]
    if limit is not None:  # trial: the first `limit` screen rows, one per (host class, route kind) first, for plumbing
        pick, seen_h = [], set()
        for p in pool:
            key = (p["host_class"], "report" if p.get("report_file") else "run")
            if p["role"] == "screen" and key not in seen_h:
                pick.append(p["job"])
                seen_h.add(key)
        for p in pool:
            if p["role"] == "screen" and p["job"] not in pick and len(pick) < limit:
                pick.append(p["job"])
        pool = [p for p in pool if p["role"] != "screen" or p["job"] in pick]
        res = {**res, "pool": pool}
    copies = []
    zc, rd = base / "zuxing_copies_r9", base / "relocated_r9"
    for p in pool:
        if p["role"] != "screen" or p["owner"] != "zuxing":
            continue
        src = Path(p["copy_of"])
        rel = src.relative_to(ZU)
        p["copy"] = str(zc / rel)
        c = PT.copy_verified(src, zc / rel, True)
        copies.append(c)
        if not c["equal"] or c["sha256_copy"] != p["sha256"]:
            raise SystemExit(f"copy of {src} not verified: {c}")
        rsi = src.with_name(src.name.replace("model_", "rsi_"))
        if src.name.startswith("model_") and rsi.is_file():
            copies.append(PT.copy_verified(rsi, zc / rel.with_name(rsi.name), True))
        if p["host_class"] != "pro6000":
            p["relocated_file"] = str(rd / rel)
            p["training_checkpoint"] = p["relocated_file"]
        else:
            p["training_checkpoint"] = p["copy"]
    meta_srcs: set[Path] = set()
    used_runs = {p["run_dir"] for p in pool if p["role"] == "screen" and not p.get("report_file")}
    used_here = {d for r in res["reports"] if r.get("eligible") for d in (r.get("run_records_here") or [])}
    for r in res["runs"]:
        if not r["eligible"] or r["owner"] != "zuxing" or (r["dir"] not in used_runs and r["dir"] not in used_here):
            continue
        d = Path(r["dir"])
        for sub in ("hydra/.hydra/config.yaml", "hydra/.hydra/hydra.yaml", "hydra/.hydra/overrides.yaml", "hydra/config.yaml",
                    "hydra/hydra.yaml", "hydra/overrides.yaml", "final.json", "MANIFEST.txt", "SHA256SUMS"):  # fmt: skip
            if (d / sub).is_file():
                meta_srcs.add(d / sub)
        if r.get("sums_file") and Path(r["sums_file"]).is_file():
            meta_srcs.add(Path(r["sums_file"]))
    for r in res["reports"]:
        m = re.search(r"host evidence file (\S+)", r.get("commit_source") or "")
        if r.get("eligible") and m:
            meta_srcs.add(Path(m.group(1)))
    for src in sorted(meta_srcs):
        dst = zc / src.relative_to(ZU)
        if dst.exists():
            continue
        c = P8.copy_meta(src, dst)
        copies.append(c)
        if not c["equal"]:
            raise SystemExit(f"copy of {src} not verified: {c}")
    logs_done = set()
    for r in res["reports"]:  # the first train_t2: line of a pro6000 training log (excerpt)
        if r.get("eligible") and r.get("host_class") == "pro6000" and r.get("train_t2_line"):
            rel = Path(r["output_dir_as_trained"]).relative_to(ZR)
            lg = ZU / "logs" / rel.parts[0] / f"{rel.parts[-1]}.log"
            if lg.is_file() and str(lg) not in logs_done:
                logs_done.add(str(lg))
                copies.append(PT.write_excerpt(lg, zc / lg.relative_to(ZU).with_suffix(".first_train_t2_line.txt"),
                                               r["train_t2_line"], True))  # fmt: skip
    for r in res["runs"]:
        if r["eligible"] and r.get("train_t2_line") and r["owner"] == "zuxing" and r["dir"] in used_runs:
            lg = Path(r["dir"]) / "train.log" if r["host_class"] != "pro6000" else None
            if r["host_class"] == "pro6000":
                rel = Path(r["dir"]).relative_to(ZR)
                lg = ZU / "logs" / rel.parts[0] / f"{rel.parts[-1]}.log"
            if lg is not None and lg.is_file() and str(lg) not in logs_done:
                logs_done.add(str(lg))
                copies.append(PT.write_excerpt(lg, zc / lg.relative_to(ZU).with_suffix(".first_train_t2_line.txt"),
                                               r["train_t2_line"], True))  # fmt: skip
    bad = [c for c in copies if c.get("equal") is not True]
    if bad:
        raise SystemExit(f"unverified copies: {bad[:3]}")
    RT = PT.load_rt()
    PT.COPIES, PT.RELOC_DIR = zc, rd  # rebinding only: round 7's verify_row checks these folders
    rrows, rjson = [], []
    for p in pool:
        if p["role"] != "screen" or not p.get("relocated_file"):
            continue
        orig, dst = Path(p["copy"]), Path(p["relocated_file"])
        osha = PT.sha256_file(orig)
        problems = [] if osha == p["sha256"] and p.get("checksum_line") in (None, p["sha256"]) else \
            ["original copy sha256 != pool / checksum line"]  # fmt: skip
        raw = orig.read_bytes()
        new, info = RT.relocate_bytes(raw, p["parquet"], p["parquet_as_trained"])
        if dst.exists():
            raise SystemExit(f"refusing to overwrite {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp_r9reloc")
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
    inp = base / "inputs_r9"
    write_new(base / "RELOCATION_R9.tsv", csv_text(rrows, PT.RTSV_COLS, "\t"))
    write_new(base / "RELOCATION_R9.json", json.dumps({"written_at": now_text(), "relocate_code": str(PT.R5_RELOC_TOOL),
                                                       "relocate_code_sha256": PT.R5_RELOC_SHA, "rows": rjson}, indent=1,
                                                      default=str) + "\n")  # fmt: skip
    rlist = [base / "RELOCATION_R9.tsv", base / "RELOCATION_R9.json"] + [Path(r["relocated_file"]) for r in rrows]
    write_new(base / "RELOCATION_R9.sha256", "".join(f"{PT.sha256_file(p)}  {p.relative_to(base)}\n" for p in rlist))
    ccols = ["source", "copy", "sha256_source_before", "sha256_copy", "sha256_source_after", "equal", "size", "source_mtime", "copied_at"]
    write_new(base / "COPIES_r9.tsv", csv_text(copies, ccols, "\t"))
    tracks: dict[str, str] = {}
    for p in pool:
        if p["role"] in ("screen", "reuse8", "reuse7", "incumbent", "barred") and p.get("parquet_sha256"):
            tracks[p["parquet"]] = p["parquet_sha256"]
            if p.get("frame_ref_file"):
                tracks[p["frame_ref_file"]] = p["frame_ref_sha256"]
    for k, v in PT.ROOTS.items():
        m = v / "whole_body/soma/track2_manifest.json"
        tracks[str(m)] = PT.sha256_file(m)
    write_new(inp / "TRACKS_R9.sha256", "".join(f"{h}  {p}\n" for p, h in sorted(tracks.items())))
    write_new(inp / "runs_r9.json", json.dumps({"t_c": now_text(res["t_c"]), "runs": res["runs"]}, indent=1, default=str) + "\n")
    write_new(inp / "runs_r9.csv", csv_text(res["runs"], RUN_COLS))
    write_new(inp / "reports_r9.json", json.dumps({"t_c": now_text(res["t_c"]), "files": res["reports"]}, indent=1, default=str) + "\n")
    write_new(inp / "reports_r9.csv", csv_text(res["reports"], REPORT_COLS))
    write_new(inp / "pool_r9.csv", csv_text(pool, POOL_COLS))
    zrows = [{"source": r["file"], "episode": r.get("episode"), "commit": r.get("commit"), "commit_source": r.get("commit_source"),
              "train_t2_line": r.get("train_t2_line")} for r in res["reports"] if r.get("eligible")]
    zrows += [{"source": r["dir"], "episode": r.get("episode"), "commit": r.get("commit"), "commit_source": r.get("commit_source"),
               "train_t2_line": r.get("train_t2_line")} for r in res["runs"] if r["eligible"] and r["owner"] == "zuxing"]  # fmt: skip
    write_new(inp / "ZUXING_CODE_R9.tsv", csv_text(zrows, ["source", "episode", "commit", "commit_source", "train_t2_line"], "\t"))
    write_new(inp / "resolve_r9.log", "\n".join(lines) + "\n")
    tell = ["Zuxing: round 9 of the Tier-2 noisy-only selection (t2_consensus_r9_20261103) made sha256-verified READ-ONLY copies",
            f"of these files at the snapshot T_c9 {now_text(res['t_c'])} CST (nothing in your folder was written; no git command;",
            "none of your scripts was run). Of each training log only the first 'train_t2:' line was read.", ""]  # fmt: skip
    tell += [f"{c['source']} -> {c['copy']} ({c['sha256_copy']})" for c in copies]
    write_new(base / "TELL_ZUXING_r9.txt", "\n".join(tell) + "\n")
    meta = {"t_c": now_text(res["t_c"]) + " CST", "t_c_epoch": res["t_c"], "script": str(Path(__file__).resolve()),
            "script_sha256": PT.sha256_file(Path(__file__).resolve()), "r8_pool_sha256": P8_SHA,
            "r7_pool_tool_sha256": P8.PT_SHA, "prereg_r9_sha256": PREREG_SHA, "K": K, "min_last_iteration": MIN_LAST,
            "settle_seconds": SETTLE_S, "grid": GRID, "pruned_hosts": list(P8.PRUNED_HOSTS), "report_mode": REPORT_MODE,
            "r8_t_c": R8_T_C_TEXT + " CST", "n_screen": sum(1 for p in pool if p["role"] == "screen"),
            "rule": __doc__.split("Round-9 deltas (PREREG_R9):", 1)[1].split("Only checkpoint headers", 1)[0]}  # fmt: skip
    write_new(base / "POOL_R9.json", json.dumps({**meta, "rows": pool}, indent=1, default=str) + "\n")
    lst = [base / "POOL_R9.json", inp / "pool_r9.csv", inp / "runs_r9.json", inp / "runs_r9.csv", inp / "reports_r9.json",
           inp / "reports_r9.csv", inp / "TRACKS_R9.sha256", inp / "ZUXING_CODE_R9.tsv", inp / "resolve_r9.log",
           inp / "R8_RECORDS_R9.json", base / "COPIES_r9.tsv", base / "TELL_ZUXING_r9.txt", base / "RELOCATION_R9.tsv",
           base / "RELOCATION_R9.json", base / "RELOCATION_R9.sha256"]  # fmt: skip
    lst += sorted(p for p in zc.rglob("*") if p.is_file()) + sorted(p for p in rd.rglob("*") if p.is_file())
    write_new(base / "POOL_R9.sha256", "".join(f"{PT.sha256_file(p)}  {p.relative_to(base)}\n" for p in lst if p.is_file()))
    bad_rel = [r for r in rrows if r["status"] != "ok"]
    log(f"frozen: {len(pool)} pool rows, {len(copies)} copies, {len(rrows)} relocations ({len(bad_rel)} failed); POOL_R9.sha256 "
        f"{PT.sha256_file(base / 'POOL_R9.sha256')}")  # fmt: skip
    return 0 if not bad_rel else 1


# ================================================================================================ commands
def docs_problems() -> list[str]:
    """PREREG_R9.md == PREREG_R9.sha256 == hard-coded; PREREG_R9_PINS.md == its sha file, pins this script and every pin
    holds; LOCK_R9.sha256 holds (PREREG_R9 10)."""
    out = []
    pre, pre_sha = ROOT / "PREREG_R9.md", ROOT / "PREREG_R9.sha256"
    got = PT.sha256_file(pre) if pre.is_file() else "missing"
    rec = (pre_sha.read_text().split() or [""])[0] if pre_sha.is_file() else ""
    if not got == rec == PREREG_SHA:
        out.append(f"PREREG_R9.md {got[:16]} != sha file {rec[:16]} / hard-coded {PREREG_SHA[:16]}")
    pins, pins_sha = ROOT / "PREREG_R9_PINS.md", ROOT / "PREREG_R9_PINS.sha256"
    if not pins.is_file():
        return out + ["PREREG_R9_PINS.md missing"]
    if PT.sha256_file(pins) != ((pins_sha.read_text().split() or [""])[0] if pins_sha.is_file() else ""):
        out.append("PREREG_R9_PINS.md != PREREG_R9_PINS.sha256")
    mine = PT.sha256_file(Path(__file__).resolve())
    plines = [ln.split() for ln in pins.read_text().splitlines()]
    plist = [(f[1], f[2]) for f in plines if len(f) == 3 and f[0] == "pin" and len(f[2]) == 64]
    if ("r9_pool.py", mine) not in plist:
        out.append(f"PREREG_R9_PINS.md does not pin this script {mine[:16]}")
    for rel, h0 in plist:
        q = Path(rel) if rel.startswith("/") else ROOT / rel
        if (PT.sha256_file(q) if q.is_file() else "missing") != h0:
            out.append(f"pin {rel} changed")
    lock = ROOT / "LOCK_R9.sha256"
    if not lock.is_file():
        out.append("LOCK_R9.sha256 missing")
    else:
        for ln in lock.read_text().splitlines():
            f = ln.split(None, 1)
            if len(f) == 2 and len(f[0]) == 64:
                q = Path(f[1].strip()) if f[1].strip().startswith("/") else ROOT / f[1].strip()
                if (PT.sha256_file(q) if q.is_file() else "missing") != f[0]:
                    out.append(f"LOCK_R9.sha256: {f[1].strip()} changed")
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
    log(f"r9_pool DRY RUN: snapshot NOW ({now_text(t_c)} CST) as T_c9; nothing is copied, relocated or frozen")
    scan_copies_index()
    res = resolve(t_c, log)
    rec8 = load_r8records()
    stamp = dt.datetime.fromtimestamp(t_c, TZ8).strftime("%Y%m%d_%H%M%S")
    out = ROOT / "dryrun"
    out.mkdir(parents=True, exist_ok=True)
    write_new(out / f"pool_dryrun_{stamp}.json", json.dumps({"snapshot": now_text(t_c), "script_sha256": PT.sha256_file(Path(__file__).resolve()),
                                                             "runs": res["runs"], "reports": res["reports"], "pool": res["pool"],
                                                             "log": lines}, indent=1, default=str) + "\n")  # fmt: skip
    write_new(out / f"pool_dryrun_{stamp}.md", md_report(res, rec8, f"Pool dry run ({stamp})"))
    print(f"written: dryrun/pool_dryrun_{stamp}.json / .md")
    return 0


def cmd_freeze(a) -> int:
    if (ROOT / "POOL_R9.json").exists() or (ROOT / "POOL_R9.sha256").exists():
        print("FATAL: POOL_R9.json / POOL_R9.sha256 exists: the round-9 pool is already frozen")
        return 2
    problems = docs_problems()
    if problems:
        print("FATAL: " + "; ".join(problems))
        return 2
    t_c = time.time()  # PREREG_R9 D3: T_c9 = the time the freeze starts
    lines: list[str] = []
    log = make_log(lines)
    log(f"r9_pool FREEZE: T_c9 = {now_text(t_c)} CST (the freeze start); script sha256 {PT.sha256_file(Path(__file__).resolve())}")
    scan_copies_index()
    res = resolve(t_c, log)
    return freeze_outputs(res, ROOT, log, lines)


def cmd_trial(a) -> int:
    lines: list[str] = []
    log = make_log(lines)
    t_c = time.time()
    stamp = dt.datetime.fromtimestamp(t_c, TZ8).strftime("%Y%m%d_%H%M%S")
    base = ROOT / "dryrun" / f"trial_{stamp}"
    log(f"r9_pool TRIAL (plumbing only, snapshot NOW {now_text(t_c)}): {a.limit} screen rows frozen into {base}")
    base.mkdir(parents=True, exist_ok=False)
    (base / "inputs_r9").mkdir()
    scan_copies_index()
    res = resolve(t_c, log)
    shutil.copyfile(ROOT / "inputs_r9/R8_RECORDS_R9.json", base / "inputs_r9/R8_RECORDS_R9.json")
    return freeze_outputs(res, base, log, lines, limit=a.limit)


def cmd_verify(a) -> int:
    base = Path(a.base) if a.base else ROOT
    RT = PT.load_rt()
    PT.COPIES, PT.RELOC_DIR = base / "zuxing_copies_r9", base / "relocated_r9"
    with (base / "RELOCATION_R9.tsv").open(newline="") as fp:
        rows = list(csv.DictReader(fp, delimiter="\t"))
    bad = 0
    for r in rows:
        pr = PT.verify_row(RT, r)
        bad += bool(pr)
        if pr:
            print(f"{r['job']}: {pr}")
    nl = 0
    for ln in (base / "POOL_R9.sha256").read_text().splitlines():
        f = ln.split(None, 1)
        if len(f) == 2 and PT.sha256_file(base / f[1].strip()) != f[0]:
            nl += 1
            print(f"POOL_R9.sha256: {f[1].strip()} changed")
    print(f"{len(rows)} relocation rows verified, {bad} with problems; POOL_R9.sha256 lines changed: {nl}")
    return 0 if not bad and not nl else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("dryrun", "freeze", "trial", "verify"))
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--base", default=None, help="verify: a trial folder instead of the round-9 folder")
    a = ap.parse_args()
    if not sys.flags.dont_write_bytecode or not sys.flags.isolated:
        print("run with python -I -B")
        return 2
    return {"dryrun": cmd_dryrun, "freeze": cmd_freeze, "trial": cmd_trial, "verify": cmd_verify}[a.mode](a)


if __name__ == "__main__":
    raise SystemExit(main())

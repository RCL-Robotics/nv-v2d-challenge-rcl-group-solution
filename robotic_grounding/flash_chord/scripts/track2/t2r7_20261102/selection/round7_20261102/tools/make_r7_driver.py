#!/usr/bin/env python3
"""Builds scripts/t2cons_r7_eval.py from round 6's driver (r6scripts/t2cons_r6_eval.py) by block replacements.
Every replacement asserts that its old text occurs exactly once. Run on the Mac scratchpad only (text editing)."""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "r6scripts/t2cons_r6_eval.py").read_text()
PREREG_SHA = sys.argv[1] if len(sys.argv) > 1 else "0" * 64
POOL_TOOL_SHA = sys.argv[2] if len(sys.argv) > 2 else "0" * 64
N_SCREEN = int(sys.argv[3]) if len(sys.argv) > 3 else 154
s = src


def rep(old: str, new: str, count: int = 1) -> None:
    global s
    n = s.count(old)
    if n != count:
        raise SystemExit(f"expected {count} occurrence(s), found {n}: {old[:120]!r}")
    s = s.replace(old, new)


# ------------------------------------------------------------------ 1. docstring
head_end = s.index('"""\n\nfrom __future__ import annotations')
s = '''#!/usr/bin/env python3
"""t2cons_r7_eval.py: round 7 (STRICT) of the Tier-2 consensus selection, steps R2/R3/R5 (PREREG_R7.md section 3).

This file is scripts/t2cons_r6_eval.py (ada6e0f9; round 6's driver) with ONLY the round-7 bookkeeping changed (full text
diff in R7_SCRIPT_DIFFS.txt):
  Q1. ROOT = fc_runs/submissions/t2_consensus_r7_20261102; outputs exports/ (one export per NEW screen job), runs_s/
      (phase S) and runs_f/ (phase F), manifest_{s7,f7}.{json,csv}, own_input_evals_{s7,f7}.csv,
      summary_{s7,f7}_own_input.{csv,json,md}, seeds_{s7,f7}.tsv, eval_r7_driver.log, eval_r7_heartbeat_{s7,f7}.json,
      eval_r7_driver_stdout_<mode>.log (launcher); ROOT/evallock is our lane-lock folder;
  Q2. two phases (PREREG_R7 3.2 / 3.3): --phase screen = the 'screen' rows of inputs_r7/pool_r7.json (NEW files only;
      'reuse' rows keep their round-6 export and 3 runs and are never evaluated in phase S), 3 runs each;
      --phase final = the 23 incumbents of inputs_r7/incumbents_r7.json + the finalists of final_r7.json (which must be
      listed with its sha256 in SELECTION_S7.sha256) + the 7 report-only round-6 references of inputs_r7/r6refs_r7.json,
      10 runs (trials) each, trial-major order, within a trial the decision jobs before the references;
  Q3. designation evidence: a screen job = its pool_r7.json row (rule R1-R5), the file re-hashed: ours = the original in
      our run folder; Zuxing's pro6000 = our copy (COPIES_r7.tsv: source before == copy == after == the pool sha256);
      Zuxing's 5090 = our copy (same) + its metadata-only relocated file (RELOCATION_R7.tsv, re-verified at every start
      with the pinned scripts/t2cons_r7_pool.py verify_row, i.e. round 5's verify_pair). A reused finalist = its round-6
      pool row and round-6 export record (re-hashed; never re-exported; Zuxing's: COPIES_r6.tsv row). An incumbent / a
      reference = its earlier round's export record and exported file (re-hashed; NEVER re-exported);
  Q4. inputs raw, K3, K3fix, K3CA and G2 as round 6 (K3CA rebinding at load time); export tree fc_r2 ours, b39b69b his;
  Q5. pin chain: PREREG_R7.md == PREREG_R7.sha256 == hard-coded; its section-11 pins and commits; SNAPSHOT_R7,
      TRACKS_R7 and RELOCATION_R7 lists; round 1's own chain (pinned t2cons_eval check_pins); PREREG_R7_PINS.md (== its
      sha file, every pin holds) must pin this script and its launcher;
  Q6. GPUs 6 and 7 only (PREREG_R7 3.4), plain (no MPS), at most one evaluation of ours per GPU (cap 2); the 35 s
      cooldown becomes 10 s; SHARED_LOCKDIRS gains round 6's own evallock and the two Tier-3 report / retry lock folders;
  Q7. runs per job: 3 (phase S) / 10 (phase F); dry-run jobs ep00_z5k3fix_89500 (phase S: the first relocated 5090 job,
      new export) and ep01_zxg2_120000 (phase F path: the first reused round-6 candidate of Zuxing's, its round-6 export);
  Q8. docstring and log text; the path guard also refuses tier2_eval.csv, tier2_runs.csv, summary.txt,
      tier2_best_scan, eval_gen_local.
Exports (1e-4 then one 2e-4 fallback), the evaluation command, every ok check, retries (3 tries, 5 min apart), the lane
locks once per lock file (all or nothing, never blocking, never created), never MPS, the seed records, OVERDUE flagging
and never signalling are round 6's, unchanged. No clean diagnostic and no clean path (guard on every record read).

  python -I -B scripts/t2cons_r7_eval.py run --phase screen|final --dry-run       plan only (dryrun/r7_eval_plan_*.txt)
  python -I -B scripts/t2cons_r7_eval.py run --phase screen --dry-run-eval        ONE evaluation of ep00_z5k3fix_89500
  python -I -B scripts/t2cons_r7_eval.py run --phase final --dry-run-eval         ONE evaluation of ep01_zxg2_120000
  python -I -B scripts/t2cons_r7_eval.py run --phase screen --exports-only        every new screen job's single export
  python -I -B scripts/t2cons_r7_eval.py run --phase screen|final                 the evaluations of a phase (resumable)
  python -I -B scripts/t2cons_r7_eval.py summary|status --phase screen|final
Background start: scripts/t2cons_r7_eval_launch.sh --phase screen|final [--dry-run | --dry-run-eval | --exports-only]
''' + s[head_end:]

# ------------------------------------------------------------------ 2. constants
c_start = s.index('ME = Path("/root/autodl-tmp/haoran.zhang")\nROOT = ')
c_end = s.index('SEED_COLS = [')
consts = f'''ME = Path("/root/autodl-tmp/haoran.zhang")
ROOT = ME / "fc_runs/submissions/t2_consensus_r7_20261102"
R1 = ME / "fc_runs/submissions/t2_consensus_20261024"
R2 = ME / "fc_runs/submissions/t2_consensus_r2_20261027"
R3 = ME / "fc_runs/submissions/t2_consensus_r3_20261028"
R4 = ME / "fc_runs/submissions/t2_consensus_r4_20261028"
R5 = ME / "fc_runs/submissions/t2_consensus_r5_20261029"
R6 = ME / "fc_runs/submissions/t2_consensus_r6_20261031"
ZU = Path("/root/autodl-tmp/zuxing.lu")  # READ ONLY (round 7 evaluates our sha256-checked copies of his files)
FC = ME / "fc_runs"
TE_FILE = ME / "scripts/t2cons_eval.py"
TE_SHA = "2cf427866a9432a3bd1b8fbb7864149d4258a5d19100c20da92d32bfe4fbfa9b"  # round-1 PREREG_PINS.md, PREREG_R7 11
PREREG, PREREG_SHA_FILE = ROOT / "PREREG_R7.md", ROOT / "PREREG_R7.sha256"
PREREG_SHA = "{PREREG_SHA}"  # PREREG_R7.md (frozen)
PINS, PINS_SHA_FILE = ROOT / "PREREG_R7_PINS.md", ROOT / "PREREG_R7_PINS.sha256"
EXPORTS = ROOT / "exports"  # every round-7 export (one per NEW screen job)
RUNS_S, RUNS_F = ROOT / "runs_s", ROOT / "runs_f"
POOL = ROOT / "inputs_r7/pool_r7.json"
INCS = ROOT / "inputs_r7/incumbents_r7.json"
REFS = ROOT / "inputs_r7/r6refs_r7.json"
COPIES = ROOT / "COPIES_r7.tsv"
R6_COPIES = R6 / "COPIES_r6.tsv"
SNAP = ROOT / "inputs_r7/SNAPSHOT_R7.sha256"
TRACKS = ROOT / "inputs_r7/TRACKS_R7.sha256"
RELOC_TSV, RELOC_LIST = ROOT / "RELOCATION_R7.tsv", ROOT / "RELOCATION_R7.sha256"
POOL_TOOL = ME / "scripts/t2cons_r7_pool.py"
POOL_TOOL_SHA = "{POOL_TOOL_SHA}"  # PREREG_R7 11 (its verify_row re-checks every relocation)
FINAL_LIST = ROOT / "final_r7.json"  # written by the phase-S selection (PREREG_R7 5.4), locked by SELECTION_S7.sha256
SEL_S7 = ROOT / "SELECTION_S7.sha256"
TZ8 = dt.timezone(dt.timedelta(hours=8))
T_C = dt.datetime(2026, 11, 2, 6, 16, 8, tzinfo=TZ8)
NRUNS_S, NRUNS_F = 3, 10  # PREREG_R7 3.2 / 3.3
N_SCREEN = {N_SCREEN}
DRY_JOB = {{"screen": "ep00_z5k3fix_89500", "final": "ep01_zxg2_120000"}}  # PREREG_R7 3.6
OUR_RUN_DIRS = (ME / "fc_runs/C3r2_tier2", ME / "fc_runs/C3r2_tier2_k3fix", ME / "fc_runs/C3r2_tier2_k3easy",
                ME / "fc_runs/C3r2_tier2_k3_stretch90k", ME / "fc_runs/C3r2_tier2_k3ca", ME / "fc_runs/T2fix/T2B",
                ME / "fc_runs/T2fix/T2L")  # fmt: skip
ZX_COPY_DIR = ROOT / "zuxing_copies_r7"
R6_ZX_COPY_DIR = R6 / "zuxing_copies_r6"
RELOC_DIR = ROOT / "relocated_r7"
Z5 = ZU / "runs/2026-11-01-round7-5090-ckpts"
K3CA_ROOT = ME / "data/human_motion_data_tier2_k3ca"
K3CA_REFS = ME / "tmp_analysis/tier2/k3ca/frame_refs_k3ca"  # round 1's t2cons_consensus INPUT_REF["k3ca"]
ALLOWED_GPUS = (6, 7)  # PREREG_R7 3.4: GPUs 6 and 7 only, one evaluation of ours per GPU
OWN_LOCKDIR = ROOT / "evallock"
SHARED_LOCKDIRS = (FC / "queue_runin/evallock", FC / "queue_tier2v2/evallock", FC / "queue_t1fix/evallock",
                   FC / "queue_t1fix2/evallock", FC / "queue_t1fix3/evallock", FC / "queue_t1fix4/evallock",
                   FC / "queue_t1fix5/evallock", FC / "queue_t2fix/evallock",
                   FC / "submissions/t1_best20_20261026/evallock", FC / "submissions/t3_f0_final/evallock",
                   ME / "tmp_analysis/tier3/f0/evallock", ME / "tmp_analysis/tier3/opus_easy/evallock",
                   ME / "tmp_analysis/tier2/k3ca/train/evallock", ME / "tmp_analysis/tier2/k3fix_pilot/evallock",
                   ME / "tmp_analysis/tier2/k3full/evallock", ME / "tmp_analysis/tier2/k3easy/evallock",
                   FC / "queue_t1improve/evallock",  # AMENDMENTS_R2_c C2
                   R2 / "evallock",  # round 2's own lane locks (PREREG_R3 3.3)
                   R3 / "evallock", FC / "T1improve/confirm/evallock",  # PREREG_R4 3.3
                   R4 / "evallock",  # PREREG_R5 3.3
                   R5 / "evallock",  # PREREG_R6 3.4
                   R6 / "evallock", FC / "submissions/t3_f0_final/report_5trials_20261101/evallock",
                   ME / "tmp_analysis/tier3/f0_retry/evallock")  # fmt: skip  (PREREG_R7 3.4)
EXT_MARKERS = ("T2BEST5_JOB", "T3FINAL_JOB", "T1BEST20_JOB", "T1CONFIRM_JOB")  # other drivers' live evals (our caps)
HOST = socket.gethostname()
OUT_LOG = ROOT / "eval_r7_driver.log"
PHASE = {{  # per phase: runs base, runs per job, output names
    "screen": {{"tag": "s7", "runs": RUNS_S, "nruns": NRUNS_S, "manifest": ROOT / "manifest_s7.json",
               "manifest_csv": ROOT / "manifest_s7.csv", "evals": ROOT / "own_input_evals_s7.csv",
               "summary": ROOT / "summary_s7_own_input", "hb": ROOT / "eval_r7_heartbeat_s7.json",
               "seeds": ROOT / "seeds_s7.tsv"}},
    "final": {{"tag": "f7", "runs": RUNS_F, "nruns": NRUNS_F, "manifest": ROOT / "manifest_f7.json",
              "manifest_csv": ROOT / "manifest_f7.csv", "evals": ROOT / "own_input_evals_f7.csv",
              "summary": ROOT / "summary_f7_own_input", "hb": ROOT / "eval_r7_heartbeat_f7.json",
              "seeds": ROOT / "seeds_f7.tsv"}},
}}  # fmt: skip
'''
s = s[:c_start] + consts + s[c_end:]

# ------------------------------------------------------------------ 3. guard + pool tool
rep('''_GUARD_EXTRA = ("readme", "train_rl.log", "t2_zx07_20261028", "best_weights", "seed_best", "remote_reps", "report_r",
                "composites", "per_episode_zx")  # fmt: skip''',
    '''_GUARD_EXTRA = ("readme", "train_rl.log", "t2_zx07_20261028", "best_weights", "seed_best", "remote_reps", "report_r",
                "composites", "per_episode_zx", "tier2_eval.csv", "tier2_runs.csv", "summary.txt", "tier2_best_scan",
                "eval_gen_local")  # fmt: skip''')
rep('raise PermissionError(f"t2cons_r6_eval may not open {p}")', 'raise PermissionError(f"t2cons_r7_eval may not open {p}")')
rep('''INC_EXPORTS: dict[str, Path] = {}  # incumbent job -> its earlier round's export.json (filled from incumbents_r6.json)
''', '''INC_EXPORTS: dict[str, Path] = {}  # job -> its earlier export.json (incumbents, references, reused round-6 candidates)


def load_pool_tool():
    """scripts/t2cons_r7_pool.py compiled from the exact bytes pinned here and in PREREG_R7.md section 11; only its
    read-only relocation check verify_row (with round 5's pinned verify_pair) is called. Importing has no side effect."""
    src = POOL_TOOL.read_bytes()
    got = hashlib.sha256(src).hexdigest()
    if got != POOL_TOOL_SHA:
        raise SystemExit(f"{POOL_TOOL} sha256 {got} != pinned {POOL_TOOL_SHA}")
    mod = types.ModuleType("t2cons_r7_pool_readonly")
    mod.__file__ = str(POOL_TOOL)
    sys.modules[mod.__name__] = mod
    exec(compile(src, str(POOL_TOOL), "exec"), mod.__dict__)  # noqa: S102 (pinned bytes)
    return mod


PT = load_pool_tool()
RT = PT.load_rt()
''')
rep('''    D.ALLOWED_GPUS = ALLOWED_GPUS  # PREREG_R6 3.4 (the pinned Driver checks --gpus against this module name)''',
    '''    D.ALLOWED_GPUS = ALLOWED_GPUS  # PREREG_R7 3.4 (the pinned Driver checks --gpus against this module name)''')
rep('''    # PREREG_R6 3.1 (Q4): the K3CA input, rebinding only''', '''    # PREREG_R7 3.1 (Q4): the K3CA input, rebinding only''')
rep('''    def export_rec_r6(base, job_id):  # a screen job's export lives in exports/; an incumbent's is the earlier round's''',
    '''    def export_rec_r7(base, job_id):  # a new screen job's export lives in exports/; the others' are earlier records''')
rep('''    D.export_rec = export_rec_r6''', '''    D.export_rec = export_rec_r7''')

# ------------------------------------------------------------------ 4. pins
s = s.replace("PREREG_R6.md sha256", "PREREG_R7.md sha256").replace("PREREG_R6.sha256 == pinned", "PREREG_R7.sha256 == pinned")
rep('''            fatal.append(f"PREREG_R6 pin {rel}: {h[:16]} != {want[:16]}")''', '''            fatal.append(f"PREREG_R7 pin {rel}: {h[:16]} != {want[:16]}")''')
rep('''    info.append(f"PREREG_R6.md section 11: {len(pins)} pins checked, {len(commits)} commits")
    for listing, base in ((SNAP, ROOT), (TRACKS, None)):''',
    '''    info.append(f"PREREG_R7.md section 11: {len(pins)} pins checked, {len(commits)} commits")
    for listing, base in ((SNAP, ROOT), (TRACKS, None), (RELOC_LIST, ROOT)):''')
rep('''    # PREREG_R6_PINS.md (step R2): == its sha file, every pin holds, and it pins this script and its launcher''',
    '''    # PREREG_R7_PINS.md (step R2): == its sha file, every pin holds, and it pins this script and its launcher''')
rep('''            fatal.append(f"PREREG_R6_PINS.md sha256 {got[:16]} != PREREG_R6_PINS.sha256 {rec[:16]}")''',
    '''            fatal.append(f"PREREG_R7_PINS.md sha256 {got[:16]} != PREREG_R7_PINS.sha256 {rec[:16]}")''')
rep('''            need = ["scripts/t2cons_r6_eval.py", "scripts/t2cons_r6_eval_launch.sh"]
            fatal += [f"PREREG_R6_PINS.md does not pin {rel}" for rel in need if rel not in names]
            if ("scripts/t2cons_r6_eval.py", mine) not in ppins:
                fatal.append(f"PREREG_R6_PINS.md does not pin this script {mine[:16]}")''',
    '''            need = ["scripts/t2cons_r7_eval.py", "scripts/t2cons_r7_eval_launch.sh"]
            fatal += [f"PREREG_R7_PINS.md does not pin {rel}" for rel in need if rel not in names]
            if ("scripts/t2cons_r7_eval.py", mine) not in ppins:
                fatal.append(f"PREREG_R7_PINS.md does not pin this script {mine[:16]}")''')
rep('''                    fatal.append(f"PREREG_R6_PINS pin {rel}: {h[:16]} != {want[:16]}")''',
    '''                    fatal.append(f"PREREG_R7_PINS pin {rel}: {h[:16]} != {want[:16]}")''')
rep('''                info.append(f"PREREG_R6_PINS.md sha256 {got[:16]} == its sha file; its {len(ppins)} pins hold; pins this "''',
    '''                info.append(f"PREREG_R7_PINS.md sha256 {got[:16]} == its sha file; its {len(ppins)} pins hold; pins this "''')
rep('''        fatal.append("PREREG_R6_PINS.md missing: this script must be pinned before it exports or evaluates (PREREG_R6 10)")
    else:
        info.append(f"PREREG_R6_PINS.md not written yet (plan-only dry run); this script sha256 {mine}")''',
    '''        fatal.append("PREREG_R7_PINS.md missing: this script must be pinned before it exports or evaluates (PREREG_R7 10)")
    else:
        info.append(f"PREREG_R7_PINS.md not written yet (plan-only dry run); this script sha256 {mine}")''')

# ------------------------------------------------------------------ 5. final list
fl_start = s.index('def final_list_check()')
fl_end = s.index('# ================================================================================================ manifest')
s = s[:fl_start] + '''def final_list_check() -> tuple[dict | None, list[str]]:
    """PREREG_R7 3.3 / 5.4: final_r7.json must be listed, with its current sha256, in SELECTION_S7.sha256 (written last
    by the pinned phase-S selection), and every line of that lock must hold."""
    if not SEL_S7.is_file():
        return None, ["SELECTION_S7.sha256 missing: the phase-S selection is not frozen (finalists unknown)"]
    lines = [ln.split(None, 1) for ln in guard(SEL_S7).read_text().splitlines() if ln.strip()]
    problems, listed = [], False
    for parts in lines:
        if len(parts) != 2 or len(parts[0]) != 64:
            continue
        p = _pin_path(parts[1].strip())
        if _pin_hash(p) != parts[0]:
            problems.append(f"SELECTION_S7.sha256: {parts[1].strip()} changed")
        if p == FINAL_LIST:
            listed = True
    if not listed:
        problems.append("SELECTION_S7.sha256 does not list final_r7.json")
    data = guarded_json(FINAL_LIST)
    if data is None:
        problems.append("final_r7.json unreadable")
    return data, problems


''' + s[fl_end:]

# ------------------------------------------------------------------ 6. manifest section
m_start = s.index('def load_pool() -> list[dict]:')
m_end = s.index('MANIFEST_COLS = [')
s = s[:m_start] + '''def load_pool() -> list[dict]:
    data = guarded_json(POOL)
    if data is None:
        raise SystemExit(f"cannot read {POOL}")
    rows = list(data["rows"])
    for p in rows:  # reused round-6 candidates: their round-6 export records (PREREG_R7 2.1 R5)
        if p.get("role") == "reuse":
            INC_EXPORTS[p["job"]] = Path(p["r6_export_record"])
    return rows


def load_incumbents() -> dict[int, dict]:
    data = guarded_json(INCS)
    if data is None:
        raise SystemExit(f"cannot read {INCS}")
    out = {int(k): v for k, v in data["incumbents"].items()}
    if len(out) != 23 or any(e in out for e in PT.STRICT):
        raise SystemExit("incumbents_r7.json does not hold the 23 non-STRICT episodes")
    for x in out.values():
        INC_EXPORTS[x["job"]] = Path(x["export_record"])
    return out


def load_refs() -> dict[int, dict]:
    data = guarded_json(REFS)
    if data is None:
        raise SystemExit(f"cannot read {REFS}")
    out = {int(k): v for k, v in data["r6refs"].items()}
    if sorted(out) != list(PT.STRICT):
        raise SystemExit("r6refs_r7.json does not hold the 7 STRICT episodes")
    for x in out.values():
        INC_EXPORTS[x["job"]] = Path(x["export_record"])
    return out


def track_records() -> dict[str, str]:
    out = {}
    for line in guard(TRACKS).read_text().splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            out[os.path.normpath(parts[1].strip())] = parts[0]
    return out


def copy_rows(path: Path = COPIES) -> dict[str, dict]:
    with guard(path).open(newline="") as fp:
        return {os.path.normpath(r["copy"]): r for r in csv.DictReader(fp, delimiter="\\t")}


def relocation_rows() -> dict[str, dict]:
    with guard(RELOC_TSV).open(newline="") as fp:
        return {r["job"]: r for r in csv.DictReader(fp, delimiter="\\t")}


def _input_checks(D, job: dict, problems: list[str], notes: list[str], trk: dict[str, str], ep: int, kind_want: str,
                  parquet: Path, psha_want: str, fr_want: str | None) -> None:
    """Training parquet, data root, manifest and frame reference of a job (round 6's checks, re-keyed on TRACKS_R7)."""
    lay = D.parquet_layout(parquet)
    if lay is None or not parquet.is_file():
        problems.append(f"no / unexpected training parquet {parquet}")
        return
    data_root, seq = lay
    kind = D.root_kind(data_root)
    psha = sha256_file(parquet)
    if kind != kind_want:
        problems.append(f"data root kind {kind} != {kind_want}")
    if psha != psha_want or trk.get(os.path.normpath(str(parquet))) != psha:
        problems.append(f"parquet sha256 {psha[:16]} != record / TRACKS_R7")
    else:
        notes.append(f"training parquet sha256 {psha[:12]} == record == TRACKS_R7.sha256")
    manifest = D.MANIFEST_FOR.get(kind) if kind else None
    if manifest is None or D.manifest_episode_seq(manifest, ep) != seq:
        problems.append(f"manifest {manifest} does not map ep{ep:02d} to {seq}")
    fr = D.FRAME_REFS.get(kind) if kind else None
    if kind == "raw":
        if fr is not None:
            problems.append(f"raw job with frame refs {fr}")
        else:
            notes.append("raw input: no frame-reference npz (the scorer's own-input reference is the released Tier-2 parquet)")
    else:
        npz = Path(fr) / f"episode_{ep:06d}.npz" if fr else None
        if npz is None or not npz.is_file() or sha256_file(npz) != fr_want or trk.get(os.path.normpath(str(npz))) != fr_want:
            problems.append(f"{npz} missing or != record / TRACKS_R7 sha256")
    job.update(sequence_id=seq, parquet=str(parquet), parquet_sha256=psha, data_root=str(data_root), root_kind=kind,
               manifest=str(manifest), frame_refs=str(fr) if fr else None)  # fmt: skip


def _copy_ok(row: dict | None, sha: str, source: str) -> bool:
    return bool(row) and row.get("equal") == "True" and row["sha256_copy"] == row["sha256_source_before"] == \\
        row["sha256_source_after"] == sha and os.path.normpath(row["source"]) == os.path.normpath(source)  # fmt: skip


def build_screen(D, p: dict, trk: dict[str, str], cop: dict[str, dict], rel: dict[str, dict]) -> dict:
    """A phase-S job = one NEW 'screen' row of pool_r7.json (PREREG_R7 2.1 / 3.2)."""
    jid, ep, it = p["job"], int(p["episode"]), int(p["iteration"])
    problems: list[str] = []
    notes: list[str] = []
    ck = Path(p["training_checkpoint"])
    job = {"job": jid, "episode": ep, "object": None, "family": p["owner"], "owner": p["owner"],
           "arm": f"{p['run_label']} ({p['input']}) {p['slot']}", "stage": "round 7 phase S (PREREG_R7 3.2)",
           "source": p["run_dir"], "variants": "B", "priority": 1, "provisional": False, "role": "screen",
           "runs": NRUNS_S, "iteration": it, "tree": p["tree"], "run_label": p["run_label"], "slot": p["slot"],
           "machine": p.get("machine"), "strict_episode": bool(p.get("strict_episode")), "pool_sha256": p["sha256"],
           "expected_sha256_from": "inputs_r7/pool_r7.json (PREREG_R7 2)", "training_checkpoint": str(ck),
           "source_checkpoint": p["source_checkpoint"]}  # fmt: skip
    if p.get("role") != "screen":
        problems.append(f"pool role {p.get('role')} is not screen")
    if p["tree"] not in D.TREES or p["tree"] != ("fc_r2" if p["owner"] == "haoran" else "b39b69b"):
        problems.append(f"tree {p['tree']} is not the owner's export tree (PREREG_R7 3.1)")
    want_sha = p["sha256"]
    if p["owner"] == "haoran":
        if ck != Path(p["source_checkpoint"]) or not any(str(ck).startswith(str(d) + "/") for d in OUR_RUN_DIRS):
            problems.append(f"checkpoint {ck} is not our original in {[str(d) for d in OUR_RUN_DIRS]}")
        des = {"rule": "pool rule R1-R5 (PREREG_R7 2.1): our training run", "run_dir": p["run_dir"], "slot": p["slot"]}
    else:
        crow = cop.get(os.path.normpath(str(p.get("copy"))))
        if not str(p.get("copy")).startswith(str(ZX_COPY_DIR) + "/") or not _copy_ok(crow, p["sha256"], p["source_checkpoint"]):
            problems.append("COPIES_r7.tsv row does not show source before == copy == after == the pool sha256")
        des = {"rule": f"pool rule R1-R5 (PREREG_R7 2.1): Zuxing's {p.get('machine')} training run, sha256-verified copy",
               "run_dir": p["run_dir"], "slot": p["slot"], "copy_row": crow}  # fmt: skip
        if p.get("machine") == "5090":
            rr = rel.get(jid)
            if rr is None or rr.get("status") != "ok":
                problems.append("no ok RELOCATION_R7.tsv row")
            else:
                if os.path.normpath(rr["original_copy"]) != os.path.normpath(p["copy"]) or rr["original_sha256"] != p["sha256"] \\
                        or os.path.normpath(rr["relocated_file"]) != os.path.normpath(str(ck)) \\
                        or rr["new_parquet_path"] != p["parquet"] or rr["old_parquet_path"] != p.get("parquet_as_trained"):  # fmt: skip
                    problems.append("RELOCATION_R7.tsv row != pool_r7.json row (copy, sha256, relocated file, parquet paths)")
                vp = PT.verify_row(RT, rr)
                problems += [f"relocation: {x}" for x in vp]
                want_sha = rr["relocated_sha256"]
                des.update(relocated_file=rr["relocated_file"], relocated_sha256=rr["relocated_sha256"],
                           old_parquet_path=rr["old_parquet_path"], new_parquet_path=rr["new_parquet_path"],
                           relocation_verified=not vp)  # fmt: skip
        elif ck != Path(str(p.get("copy"))):
            problems.append(f"checkpoint {ck} is not the recorded copy {p.get('copy')}")
    tsha = sha256_file(ck) if ck.is_file() else None
    if tsha != want_sha:
        problems.append(f"checkpoint sha256 {str(tsha)[:16]} != expected {str(want_sha)[:16]}")
    job["expected_sha256"] = want_sha
    job["training_sha256"] = tsha
    job["designation_evidence"] = des
    _input_checks(D, job, problems, notes, trk, ep, p["input"], Path(p["parquet"]), p["parquet_sha256"], p.get("frame_ref_sha256"))
    job["problems"], job["notes"] = problems, notes
    job["status"] = "ok" if not problems else "refused: " + "; ".join(problems)
    return job


def build_reused(D, p: dict, trk: dict[str, str], cop6: dict[str, dict], role: str) -> dict:
    """A reused round-6 candidate in phase F (finalist; or the phase-F dry run): its round-6 training file and export
    record, re-hashed, never re-exported (PREREG_R7 2.1 R5 / 3.3)."""
    jid, ep, it = p["job"], int(p["episode"]), int(p["iteration"])
    problems: list[str] = []
    notes: list[str] = []
    ck = Path(p["training_checkpoint"])
    job = {"job": jid, "episode": ep, "object": None, "family": p["owner"], "owner": p["owner"],
           "arm": f"{p['run_label']} ({p['input']}) {p['slot']} [reused round-6 candidate]",
           "stage": "round 7 phase F (PREREG_R7 3.3)", "source": p["run_dir"], "variants": "B", "priority": 1,
           "provisional": False, "role": role, "runs": NRUNS_F, "iteration": it, "tree": p["tree"],
           "run_label": p["run_label"], "slot": p["slot"], "machine": p.get("machine"),
           "strict_episode": bool(p.get("strict_episode")), "pool_sha256": p["sha256"], "expected_sha256": p["sha256"],
           "expected_sha256_from": "inputs_r7/pool_r7.json role reuse (== round-6 pool_r6.json screen row)",
           "training_checkpoint": str(ck), "source_checkpoint": p["source_checkpoint"],
           "export_record": p["r6_export_record"]}  # fmt: skip
    if p.get("role") != "reuse":
        problems.append(f"pool role {p.get('role')} is not reuse")
    if p["tree"] not in D.TREES or p["tree"] != ("fc_r2" if p["owner"] == "haoran" else "b39b69b"):
        problems.append(f"tree {p['tree']} is not the owner's export tree (PREREG_R7 3.1)")
    if p["owner"] == "haoran":
        if ck != Path(p["source_checkpoint"]) or not any(str(ck).startswith(str(d) + "/") for d in OUR_RUN_DIRS):
            problems.append(f"checkpoint {ck} is not our original in {[str(d) for d in OUR_RUN_DIRS]}")
        crow = None
    else:
        crow = cop6.get(os.path.normpath(str(ck)))
        if not str(ck).startswith(str(R6_ZX_COPY_DIR) + "/") or not _copy_ok(crow, p["sha256"], p["source_checkpoint"]):
            problems.append("COPIES_r6.tsv row does not show source before == copy == after == the pool sha256")
    tsha = sha256_file(ck) if ck.is_file() else None
    if tsha != p["sha256"]:
        problems.append(f"checkpoint sha256 {str(tsha)[:16]} != pool {p['sha256'][:16]}")
    rp = Path(p["r6_export_record"])
    rec = guarded_json(rp) or {}
    f = Path(str(rec.get("file")))
    fsha = sha256_file(f) if f.is_file() else None
    if not str(rp).startswith(str(R6 / "exports") + "/") or sha256_file(rp) != p.get("r6_export_record_sha256"):
        problems.append("round-6 export record changed since pool_r7.json")
    if rec.get("status") != "ok" or rec.get("stock_verified") is not True or fsha != rec.get("sha256") \\
            or fsha != p.get("r6_export_sha256") or rec.get("source_sha256") != tsha:  # fmt: skip
        problems.append("round-6 export record / file is not an ok, stock-verified export of this file")
    job["training_sha256"] = tsha
    job["export_sha256"] = fsha
    job["designation_evidence"] = {"rule": "pool rule R1-R5 (PREREG_R7 2.1 R5): reused round-6 phase-S candidate",
                                   "r6_job": p.get("r6_job"), "run_dir": p["run_dir"], "slot": p["slot"], "copy_row": crow,
                                   "r6_export_record": str(rp)}  # fmt: skip
    _input_checks(D, job, problems, notes, trk, ep, p["input"], Path(p["parquet"]), p["parquet_sha256"], p.get("frame_ref_sha256"))
    job["problems"], job["notes"] = problems, notes
    job["status"] = "ok" if not problems else "refused: " + "; ".join(problems)
    return job


def build_incumbent(D, x: dict, trk: dict[str, str], role: str = "incumbent") -> dict:
    """A phase-F incumbent / report-only reference (PREREG_R7 3.3 / 4): its earlier export record and file, re-hashed;
    never re-exported."""
    jid, ep = x["job"], int(x["episode"])
    problems: list[str] = []
    notes: list[str] = []
    rec = guarded_json(Path(x["export_record"])) or {}
    f = Path(str(rec.get("file")))
    fsha = sha256_file(f) if f.is_file() else None
    if sha256_file(guard(Path(x["export_record"]))) != x["export_record_sha256"]:
        problems.append("export record changed since incumbents_r7.json / r6refs_r7.json")
    if rec.get("status") != "ok" or rec.get("stock_verified") is not True or fsha != x["export_sha256"] \\
            or rec.get("sha256") != fsha or Path(str(rec.get("file"))) != Path(x["export_file"]):  # fmt: skip
        problems.append(f"export record / file is not the recorded ok export {x['export_sha256'][:16]}")
    tck = Path(x["training_checkpoint"])
    tsha = sha256_file(tck) if tck.is_file() else None
    if tsha != x["training_sha256"] or rec.get("source_sha256") != tsha:
        problems.append("training file sha256 != record")
    what = "incumbent" if role == "incumbent" else "report-only round-6 reference (no decision role)"
    job = {"job": jid, "episode": ep, "object": x.get("object"), "family": x.get("owner"), "owner": x.get("owner"),
           "arm": f"{what} ({x['source_round']}, {x['input']})", "stage": "round 7 phase F (PREREG_R7 3.3)",
           "source": x["source_round"], "variants": "B", "priority": 1 if role == "incumbent" else 2, "provisional": False,
           "role": role, "runs": NRUNS_F, "iteration": int(x["iteration"]), "tree": x["tree"],
           "expected_sha256": x["training_sha256"],
           "expected_sha256_from": f"inputs_r7/{'incumbents' if role == 'incumbent' else 'r6refs'}_r7.json (PREREG_R7 4)",
           "training_checkpoint": str(tck), "training_sha256": tsha, "export_record": x["export_record"], "export_sha256": fsha,
           "designation_evidence": {"rule": f"{what}; earlier export reused", "m_P_frozen": x.get("m_P_frozen"),
                                    "selected_in": x.get("selected_in"), "r6_closest_run": x.get("r6_closest_run")}}  # fmt: skip
    if x["tree"] not in D.TREES:
        problems.append(f"tree {x['tree']} unknown")
    _input_checks(D, job, problems, notes, trk, ep, x["input"], Path(x["parquet"]), x["parquet_sha256"], x.get("frame_ref_sha256"))
    job["problems"], job["notes"] = problems, notes
    job["status"] = "ok" if not problems else "refused: " + "; ".join(problems)
    return job


def build_finalist(D, fl: dict, pool_by: dict[str, dict], trk: dict[str, str], cop: dict[str, dict], cop6: dict[str, dict],
                   rel: dict[str, dict]) -> dict:
    """A phase-F finalist (PREREG_R7 5.4): a new screen job with its phase-S export (exports/<job>/), or a reused round-6
    candidate with its round-6 export; 10 fresh runs."""
    p = pool_by[fl["job"]]
    if p.get("role") == "reuse":
        job = build_reused(D, p, trk, cop6, "finalist")
        if job.get("export_sha256") != fl.get("export_sha256"):
            job["problems"].append("round-6 export != final_r7.json")
    else:
        job = build_screen(D, p, trk, cop, rel)
        job.update(role="finalist", runs=NRUNS_F, stage="round 7 phase F (PREREG_R7 3.3)")
        exp = guarded_json(EXPORTS / fl["job"] / "export.json") or {}
        if exp.get("status") != "ok" or exp.get("sha256") != fl.get("export_sha256"):
            job["problems"].append("phase-S export record != final_r7.json")
    job["status"] = "ok" if not job["problems"] else "refused: " + "; ".join(job["problems"])
    return job


def build_manifest(D, log, phase: str, only: set | None = None) -> list[dict]:
    trk = track_records()
    inc = load_incumbents()
    refs = load_refs()
    pool = load_pool()
    cop = copy_rows(COPIES)
    cop6 = copy_rows(R6_COPIES)
    rel = relocation_rows()
    pool_by = {p["job"]: p for p in pool if p.get("role") in ("screen", "reuse")}
    if phase == "screen":
        rows = [p for p in pool if p.get("role") == "screen"]
        if len(rows) != N_SCREEN:
            raise SystemExit(f"pool_r7.json has {len(rows)} screen rows, PREREG_R7 2.2 says {N_SCREEN}")
        rows.sort(key=lambda p: (int(p["episode"]), p["job"]))
        jobs = [build_screen(D, p, trk, cop, rel) for p in rows if only is None or p["job"] in only]
        log(f"phase S job list from pool_r7.json: {len(jobs)} of {len(rows)} new screen jobs (PREREG_R7 3.2)")
    else:
        jobs = [build_incumbent(D, inc[e], trk) for e in sorted(inc) if only is None or inc[e]["job"] in only]
        jobs += [build_incumbent(D, refs[e], trk, "r6ref") for e in sorted(refs) if only is None or refs[e]["job"] in only]
        if only is None:
            data, problems = final_list_check()
            if problems:
                raise SystemExit("phase F needs the frozen phase-S selection: " + "; ".join(problems))
            for fl in data.get("finalists", []):
                jobs.append(build_finalist(D, fl, pool_by, trk, cop, cop6, rel))
        else:  # the phase-F dry run (PREREG_R7 3.6): a reused round-6 candidate on the finalist path
            for jid in sorted(only):
                p = pool_by.get(jid)
                if p is not None and p.get("role") == "reuse":
                    jobs.append(build_reused(D, p, trk, cop6, "finalist"))
        jobs.sort(key=lambda j: (int(j.get("priority", 1)), int(j["episode"]), j["job"]))
        log(f"phase F job list: {sum(1 for j in jobs if j['role'] == 'incumbent')} incumbents + "
            f"{sum(1 for j in jobs if j['role'] == 'finalist')} finalists + "
            f"{sum(1 for j in jobs if j['role'] == 'r6ref')} report-only references (PREREG_R7 3.3)")
    ok = [r for r in jobs if r["status"] == "ok"]
    hulls = D.hull_check(ok, log)
    for r in ok:
        h = hulls.get(r["parquet"], {"ok": False, "missing": ["not checked"]})
        r["hulls"] = h
        if not h["ok"]:
            r["status"] = f"hull_missing: {h['missing']}"
    return jobs


''' + s[m_end:]
rep('''MANIFEST_COLS = ["job", "episode", "role", "runs", "owner", "tree", "source", "arm", "iteration", "training_sha256",
                 "training_checkpoint", "export_record", "export_sha256", "parquet", "parquet_sha256", "root_kind",
                 "data_root", "manifest", "frame_refs", "status"]  # fmt: skip''',
    '''MANIFEST_COLS = ["job", "episode", "role", "runs", "owner", "machine", "strict_episode", "tree", "source", "arm", "iteration",
                 "pool_sha256", "training_sha256", "training_checkpoint", "export_record", "export_sha256", "parquet",
                 "parquet_sha256", "root_kind", "data_root", "manifest", "frame_refs", "status"]  # fmt: skip''')

# ------------------------------------------------------------------ 7. summary text
rep('''                "frame 0, explicit reset, world count 1). Comparable only within one training parquet sha256. floor = runs "
                "with SP-SR = MP-SR = 1. The selection uses consensus-B scores (PREREG_R6 5), not these values. No clean "
                "number anywhere in this file."}))''',
    '''                "frame 0, explicit reset, world count 1). Comparable only within one training parquet sha256. floor = runs "
                "with SP-SR = MP-SR = 1. The selection uses consensus-B scores (PREREG_R7 5), not these values. No clean "
                "number anywhere in this file."}))''')
rep('''    md = [f"# t2_consensus round 6, phase {phase}: official own-input evaluations ({D.now()})", "",''',
    '''    md = [f"# t2_consensus round 7, phase {phase}: official own-input evaluations ({D.now()})", "",''')
rep('''          "or read. The selection uses consensus-B scores (PREREG_R6 5), not these values.", "",''',
    '''          "or read. The selection uses consensus-B scores (PREREG_R7 5), not these values.", "",''')

# ------------------------------------------------------------------ 8. driver class
rep('''    class R6Driver(Base):''', '''    class R7Driver(Base):''')
rep('''    return R6Driver''', '''    return R7Driver''')
rep('''            self.exports_only = exports_only  # PREREG_R6 3.2: every screen job's single export, no evaluation''',
    '''            self.exports_only = exports_only  # PREREG_R7 3.2: every new screen job's single export, no evaluation''')
rep('''        def nruns(self, job: dict) -> int:  # PREREG_R6 3.2 / 3.3: 3 (screen) or 10 (final), capped by the mode''',
    '''        def nruns(self, job: dict) -> int:  # PREREG_R7 3.2 / 3.3: 3 (screen) or 10 (final), capped by the mode''')
rep('''                self.log(f"building {mf.name} (PREREG_R6 3.{2 if self.phase == 'screen' else 3}; phase {self.phase})")''',
    '''                self.log(f"building {mf.name} (PREREG_R7 3.{2 if self.phase == 'screen' else 3}; phase {self.phase})")''')
rep('''                    "built_at": D.now(), "phase": self.phase, "prereg_r6_sha256": sha256_file(PREREG),
                    "prereg_r6_pins_sha256": sha256_file(PINS) if PINS.is_file() else None,
                    "pool_r6_sha256": sha256_file(POOL), "incumbents_r6_sha256": sha256_file(INCS),
                    "final_r6_sha256": sha256_file(FINAL_LIST) if (self.phase == "final" and FINAL_LIST.is_file()) else None,
                    "selection_s6_lock_sha256": sha256_file(SEL_S6) if (self.phase == "final" and SEL_S6.is_file()) else None,
                    "t2best5_driver_sha256": TE.DRIVER_SHA, "t2cons_eval_sha256": TE_SHA,
                    "t2cons_r6_eval_sha256": sha256_file(Path(__file__).resolve()), "t_c": "2026-10-31 01:09:17 CST",''',
    '''                    "built_at": D.now(), "phase": self.phase, "prereg_r7_sha256": sha256_file(PREREG),
                    "prereg_r7_pins_sha256": sha256_file(PINS) if PINS.is_file() else None,
                    "pool_r7_sha256": sha256_file(POOL), "incumbents_r7_sha256": sha256_file(INCS),
                    "r6refs_r7_sha256": sha256_file(REFS), "relocation_r7_sha256": sha256_file(RELOC_TSV),
                    "final_r7_sha256": sha256_file(FINAL_LIST) if (self.phase == "final" and FINAL_LIST.is_file()) else None,
                    "selection_s7_lock_sha256": sha256_file(SEL_S7) if (self.phase == "final" and SEL_S7.is_file()) else None,
                    "t2best5_driver_sha256": TE.DRIVER_SHA, "t2cons_eval_sha256": TE_SHA, "t2cons_r7_pool_sha256": POOL_TOOL_SHA,
                    "t2cons_r7_eval_sha256": sha256_file(Path(__file__).resolve()), "t_c": "2026-11-02 06:16:08 CST",''')
rep('''            load_incumbents()  # fills INC_EXPORTS (the incumbents' earlier export records)''',
    '''            load_incumbents()  # fills INC_EXPORTS (the incumbents' / references' / reused candidates' earlier records)
            load_refs()
            load_pool()''')
rep('''                if g not in ALLOWED_GPUS:
                    why = "never (PREREG_R6 3.4: GPUs 0, 1, 2, 7 only)"''',
    '''                if g not in ALLOWED_GPUS:
                    why = "never (PREREG_R7 3.4: GPUs 6 and 7 only)"''')
rep('''                if D.export_rec(ROOT, jid) is not None or self.busy(("export", jid)):
                    continue  # an incumbent's earlier export record always exists: never re-exported (PREREG_R6 3.1)
                if job.get("role") == "incumbent":
                    continue''',
    '''                if D.export_rec(ROOT, jid) is not None or self.busy(("export", jid)):
                    continue  # an earlier export record always exists for these: never re-exported (PREREG_R7 3.1)
                if job.get("role") in ("incumbent", "r6ref") or job["job"] in INC_EXPORTS:
                    continue''')
rep('''            if self.exports_only:  # PREREG_R6 3.2: exports only, never an evaluation''',
    '''            if self.exports_only:  # PREREG_R7 3.2: exports only, never an evaluation''')
rep('''                slots = [] if job.get("role") == "incumbent" else [("export", None, EXPORTS / jid)]''',
    '''                slots = [] if (job.get("role") in ("incumbent", "r6ref") or jid in INC_EXPORTS) else [("export", None, EXPORTS / jid)]''')
rep('''                   "note": "the official path has no selectable seed (PREREG_R6 3.5): every run uses these defaults"}  # fmt: skip''',
    '''                   "note": "the official path has no selectable seed (PREREG_R7 3.5): every run uses these defaults"}  # fmt: skip''')
rep('''                    "exports_only": self.exports_only, "clean": "not run (PREREG_R6 7)"}  # fmt: skip''',
    '''                    "exports_only": self.exports_only, "clean": "not run (PREREG_R7 7)"}  # fmt: skip''')

# ------------------------------------------------------------------ 9. baseline, plan, status, main
rep('''    """PREREG_R6 3.1: the round-6 baseline = a byte copy of round-1 integrity/baseline.json if today's trees equal it."""''',
    '''    """PREREG_R7 3.1: the round-7 baseline = a byte copy of round-1 integrity/baseline.json if today's trees equal it."""''')
rep('''    p(f"t2cons_r6_eval PLAN-ONLY DRY RUN, phase {a.phase}, {D.now()}: nothing is exported, evaluated, locked or signalled. "
      "Written: dryrun/r6_eval_plan_*.txt and the read-only hull check's preflight/hulls_*.")''',
    '''    p(f"t2cons_r7_eval PLAN-ONLY DRY RUN, phase {a.phase}, {D.now()}: nothing is exported, evaluated, locked or signalled. "
      "Written: dryrun/r7_eval_plan_*.txt and the read-only hull check's preflight/hulls_*.")''')
rep('''        p("phase F finalists: " + ("; ".join(problems) + " -> plan covers the 30 incumbents only" if problems else
                                   f"{len(data.get('finalists', []))} from final_r6.json"))  # fmt: skip
        rows = build_manifest(D, D.Logger(None), "final", None if not problems else {x["job"] for x in load_incumbents().values()})''',
    '''        p("phase F finalists: " + ("; ".join(problems) + " -> plan covers the 23 incumbents + 7 references + the "
                                   "phase-F dry-run job only" if problems else
                                   f"{len(data.get('finalists', []))} from final_r7.json"))  # fmt: skip
        only = None if not problems else ({x["job"] for x in load_incumbents().values()} |
                                          {x["job"] for x in load_refs().values()} | {DRY_JOB["final"]})  # fmt: skip
        rows = build_manifest(D, D.Logger(None), "final", only)''')
rep('''            f0 = Path(D.export_rec(ROOT, r0["job"])["file"]) if r0.get("role") == "incumbent" else Path(r0["training_checkpoint"])''',
    '''            f0 = Path(D.export_rec(ROOT, r0["job"])["file"]) if r0["job"] in INC_EXPORTS else Path(r0["training_checkpoint"])''')
rep('''    D.write_atomic(ROOT / "dryrun" / f"r6_eval_plan_{a.phase}_{D.stamp()}.txt", "\\n".join(lines) + "\\n")''',
    '''    D.write_atomic(ROOT / "dryrun" / f"r7_eval_plan_{a.phase}_{D.stamp()}.txt", "\\n".join(lines) + "\\n")''')
rep('''    P = PHASE[phase]
    load_incumbents()
    hb = D.read_json(P["hb"]) or {}''', '''    P = PHASE[phase]
    load_incumbents()
    load_refs()
    load_pool()
    hb = D.read_json(P["hb"]) or {}''')
rep('''    g.add_argument("--exports-only", action="store_true", help="every screen job's single export, no evaluation (PREREG_R6 3.2)")
    r.add_argument("--gpus", default="0,1,2,7")  # PREREG_R6 3.4
    r.add_argument("--max-parallel", type=int, default=4)
    r.add_argument("--lane-max", type=int, default=2)
    r.add_argument("--cooldown", type=float, default=35.0)''',
    '''    g.add_argument("--exports-only", action="store_true", help="every new screen job's single export, no evaluation (PREREG_R7 3.2)")
    r.add_argument("--gpus", default="6,7")  # PREREG_R7 3.4
    r.add_argument("--max-parallel", type=int, default=2)
    r.add_argument("--lane-max", type=int, default=2)
    r.add_argument("--cooldown", type=float, default=10.0)''')
rep('''        if not 1 <= a.max_parallel <= 4 or not 1 <= a.lane_max <= 2 or a.cooldown < 35.0:
            ap.error("PREREG_R6 3.4: --max-parallel <= 4, --lane-max <= 2, --cooldown >= 35")''',
    '''        if not 1 <= a.max_parallel <= 2 or not 1 <= a.lane_max <= 2 or a.cooldown < 10.0:
            ap.error("PREREG_R7 3.4: --max-parallel <= 2, --lane-max <= 2, --cooldown >= 10")
        if any(int(x) not in ALLOWED_GPUS for x in str(a.gpus).split(",") if x.strip()):
            ap.error("PREREG_R7 3.4: --gpus may name GPUs 6 and 7 only")''')
rep('''                "note": "PREREG_R6 3.6: plumbing test (phase S: the K3CA final, first K3CA job of these drivers, fc_r2 export "
                        "kept as the job's single export; phase F: an incumbent with its earlier export reused); a dry run "
                        "is NOT a round-6 run or try and decides nothing"}))  # fmt: skip''',
    '''                "note": "PREREG_R7 3.6: plumbing test (phase S: the first relocated 5090 job, b39b69b export kept as the "
                        "job's single export; phase F path: the first reused round-6 candidate of Zuxing's, its round-6 "
                        "export reused); a dry run is NOT a round-7 run or try and decides nothing"}))  # fmt: skip''')
rep('''        log(f"t2cons_r6_eval.py (PREREG_R6) start phase {a.phase}{' EXPORTS ONLY' if a.exports_only else ''}: --gpus {a.gpus}; "''',
    '''        log(f"t2cons_r7_eval.py (PREREG_R7) start phase {a.phase}{' EXPORTS ONLY' if a.exports_only else ''}: --gpus {a.gpus}; "''')
rep('''    if a.cmd == "summary":
        load_incumbents()''', '''    if a.cmd == "summary":
        load_incumbents()
        load_refs()
        load_pool()''')

rep("+ ' (reused)' if r.get('role') == 'incumbent' else ''}", "+ ' (reused)' if r['job'] in INC_EXPORTS else ''}")

# leftovers check: any remaining round-6 token outside comments explaining provenance
left = [ln for ln in s.splitlines() if re.search(r"PREREG_R6|_r6\\b|r6_eval|SEL_S6|s6\\b|f6\\b|R6Driver|incumbents_r6|pool_r6", ln)]
(HERE / "t2cons_r7_eval.py").write_text(s)
print(f"written t2cons_r7_eval.py ({len(s.splitlines())} lines); lines still naming round 6: {len(left)}")
for ln in left:
    print("   ", ln.strip()[:150])

import os, sys
from pathlib import Path
RA = Path("/root/autodl-tmp/haoran.zhang/tmp_analysis/tier3/f0_retry")
F = RA / "PREREG_retry.md"
src = F.read_text()
R = []
R.append(("""Written 2026-10-26 ~21:15 CST. **Built, NOT launched.** Nothing trains until the main session writes `RETRY_GO`
after the user approves (section 7). `PREREG_f0.md` (sha256 `82702d85…`) is unchanged and keeps governing the f0 runs.
This file adds one rule: a retry may replace an f0 final in the submission (section 6).""",
"""Written 2026-10-26 ~21:15 CST (v1, sha256 `8f7da98f…`, kept in `r3fix/v1/`).
**Re-frozen as v2 on 2026-10-26 ~22:10 CST.** This happened before any RETRY_GO, launch, run or result, to apply the
three must-fix items of review R3. v2 changes:
- section 4, gate 2;
- the new section 4a;
- section 6 (the route into the packed file);
- the files list;
- disclosure 6.

Unchanged: rules 1, 3 and 5, the acceptance rule, the candidate list and the four frozen recipes.

**Built, NOT launched.** Nothing trains until the main session writes an accepted `RETRY_GO` after the user approves
(section 4). `PREREG_f0.md` (sha256 `82702d85…`) is unchanged and keeps governing the f0 runs.
This file adds one rule: a retry may replace an f0 final in the submission (section 6), through
`t3_f0_final/AMENDMENT_1.md`."""))
R.append(("""   - `f0/VERDICT.json`: it carries report-only raw Tier-2 numbers.

## 1. Candidate rule""",
"""   - `f0/VERDICT.json`: it carries report-only raw Tier-2 numbers.
6. **When v2 was written.**
   - `fc_runs/submissions/t3_f0_final`, the 5-run own-input evaluation of the f0 finals, had started at 21:38.
   - Its log showed the first own-input runs of ep00, which is not a candidate.
   - No Tier-3 Kaggle submission of ours existed.
   - No clean file was opened while v2 was written.
   - The v2 changes are gates and records only. No candidate, recipe, threshold or acceptance rule changed. The four
     recipes were frozen by v1 of `t3retry.py` (sha256 `969ce9d1…`, recorded in each recipe). v2 leaves the recipe
     computation unchanged.

## 1. Candidate rule"""))
R.append(("""2. `f0_retry/RETRY_GO` exists. It must contain the line `prereg_sha256 <this file's sha256>` and a line
   `user_approval: "<verbatim>", <time>`. The main session writes it only after the user approves; the template is
   `RETRY_GO.template`. **No RETRY_GO = nothing starts.**""",
"""2. `f0_retry/RETRY_GO` exists and `go_status()` accepts it (v2). It must contain:
   - the line `prereg_sha256 <this file's sha256>`;
   - a line `user_approval: "<the user's message, verbatim>", YYYY-MM-DD HH:MM CST`. A line still holding template
     text (`<` or `>`) is refused;
   - the line `tier3_board_scores_seen: none`, i.e. the GO was given before any Tier-3 Kaggle score of ours existed.

   It is refused when:
   - `f0_retry/RETRY_NOGO` exists (the user declined; template `RETRY_NOGO.template`);
   - `fc_runs/submissions/TIER3_FIRST_UPLOAD` exists and is not newer than RETRY_GO. The main session writes that
     marker right after our first Tier-3 Kaggle upload, interim or full;
   - `fc_runs/submissions/t3_f0_final/AMENDMENT_1.md` is not a byte copy of `f0_retry/AMENDMENT_1_t3_f0_final.md`
     (sha256 pinned in `RETRY_FILES.sha256`), or `AMENDMENT_1.sha256` there does not name it. The amendment is the only
     route by which a REPLACE verdict enters a packed Tier-3 file (section 6).

   The main session writes RETRY_GO, or RETRY_NOGO, only after the user decides, using the templates
   `RETRY_GO.template` / `RETRY_NOGO.template`. **No accepted RETRY_GO = nothing starts.**"""))
R.append(("""Launch mechanics:
- The start happens under the shared GPU-choice locks (`ch_runs/*/.gpulock`, opened read-only).""",
"""### 4a. When the user must decide (v2, R3 must-fix 3)

- The user's GO or NO-GO is obtained and recorded (RETRY_GO or RETRY_NOGO) before our first Tier-3 Kaggle upload,
  interim or full. That way no Tier-3 board score exists when the decision is made: board scores are clean aggregates
  over the 30 episodes.
- The interim plan gates its upload on this record (`fc_runs/submissions/t3_interim/PLAN.md` section 2.5 (e)).
- The driver refuses a GO written after `TIER3_FIRST_UPLOAD`. After that point the retry queue can only be opened by a
  new PREREG that discloses every Tier-3 board score seen.

Launch mechanics:
- **Config check (v2).** Before a clip's first start, the hydra config check (section 3) runs on its live recipe and
  is recorded write-once in `cfgcheck/epNN.json`.
  - A recorded FAIL, or a recipe that changed after its check, holds the clip for good; a human decides.
  - Recipes frozen after 21:12 (clips that become candidates later) are written read-only, and their sha256 is
    appended to `recipes.sha256`.
- The start happens under the shared GPU-choice locks (`ch_runs/*/.gpulock`, opened read-only)."""))
R.append(("""The outputs are `VERDICT_retry.json` and `SUBMISSION_CHOICE.json` (per episode). Whoever packs the Tier-3 submission
must read `SUBMISSION_CHOICE.json`. The frame references stay `frame_references_f0`, because the input is the same.""",
"""The outputs are `VERDICT_retry.json` and `SUBMISSION_CHOICE.json` (per episode).
- **chosen_run (v2).** For a REPLACE episode, SUBMISSION_CHOICE also records `chosen_run`: among the retry arm's ok
  runs, the one whose own-input ADD-AUC is closest to their mean. The arithmetic is exact rational, and ties go to the
  lower run index; this is the t3_f0_final selection rule. The record lists the run's results JSON, its parquet and
  its evaluated (exported) checkpoint, with their sha256.
- **Frame references.** They stay `frame_references_f0`, because the input is the same.

**Route into the packed file (v2, R3 must-fix 1).**
- `t3_f0_final/PREREG.md` (sha256 `e4e3a5da…`) takes its items from PREREG_f0 only.
- A REPLACE verdict enters a submission only through `t3_f0_final/AMENDMENT_1.md`. It is staged here as
  `AMENDMENT_1_t3_f0_final.md` (pinned in `RETRY_FILES.sha256`) and must be installed byte-identically before any
  retry starts (section 4, gate 2).
- In short:
  - (a) items come from SUBMISSION_CHOICE.json where it says REPLACE; otherwise from t3_f0_final's frozen item and
    selection;
  - (b) a REPLACE episode packs its `chosen_run`, reused and not re-run, after the t3_f0_final section-2 validity
    checks pass on it;
  - (c) the retry-updated file FINAL_r is built once: when every candidate is POST_DONE, LATE, CRASHED or unstarted at
    the training cut-off, or at 2026-11-02 18:00 CST, whichever comes first;
  - (e) the Kaggle Final mark goes to the full file these rules designate (FINAL_r if built and valid, else FINAL_f0),
    whatever the public scores show. Full files are never chosen by board score."""))
R.append(("""- `RETRY_GO.template`; `recipes/epNN.json`; `diag/`, the Tier-3-only diagnosis with its scripts.""",
"""- `RETRY_GO.template`, `RETRY_NOGO.template` (v2); `recipes/epNN.json`; `diag/`, the Tier-3-only diagnosis with its scripts.
- `AMENDMENT_1_t3_f0_final.md` (v2): the staged amendment to `t3_f0_final/PREREG.md`.
- v2 records: `recipes.sha256`, the append-only log of later recipes; `cfgcheck/epNN.json`, the write-once pre-launch
  config checks; `r3fix/`, holding the v1 files and the patch scripts."""))
R.append(("""- Hashes: `RETRY_FILES.sha256` (t3retry.py, launch.sh, recipes) and `PREREG_retry.sha256`.""",
"""- Hashes: `RETRY_FILES.sha256` (t3retry.py, launch.sh, the 4 recipes, AMENDMENT_1_t3_f0_final.md) and `PREREG_retry.sha256`."""))
R.append(("""- Controls: `STOP`, `HOLD_epNN`, `RETRY_GO`.""",
"""- Controls: `STOP`, `HOLD_epNN`, `RETRY_GO`, `RETRY_NOGO`; the marker `fc_runs/submissions/TIER3_FIRST_UPLOAD`."""))
for old, new in R:
    n = src.count(old)
    if n != 1:
        sys.exit(f"anchor count {n} != 1 for: {old[:90]!r}")
    src = src.replace(old, new)
os.chmod(F, 0o644)
F.write_text(src)
os.chmod(F, 0o444)
go = """# TEMPLATE ONLY. The driver reads tmp_analysis/tier3/f0_retry/RETRY_GO, never this file.
# The main session writes RETRY_GO only after the user approves the Tier-3 retry queue, and only BEFORE our first Tier-3
# Kaggle upload (PREREG_retry v2 section 4 gate 2 + 4a). First install the amendment (byte copy; the driver checks it):
#   ME=/root/autodl-tmp/haoran.zhang; cp $ME/tmp_analysis/tier3/f0_retry/AMENDMENT_1_t3_f0_final.md $ME/fc_runs/submissions/t3_f0_final/AMENDMENT_1.md
#   cd $ME/fc_runs/submissions/t3_f0_final && sha256sum AMENDMENT_1.md > AMENDMENT_1.sha256 && chmod 444 AMENDMENT_1.md AMENDMENT_1.sha256
# Then fill the lines below (a user_approval line still containing < or > is refused) and save as RETRY_GO.
prereg_sha256 <sha256 of PREREG_retry.md, as in PREREG_retry.sha256>
user_approval: "<the user's approving message, verbatim>", <YYYY-MM-DD HH:MM> CST
tier3_board_scores_seen: none
written_by: main session
"""
nogo = """# TEMPLATE ONLY. The main session saves a filled copy as tmp_analysis/tier3/f0_retry/RETRY_NOGO when the user declines
# the Tier-3 retry queue (record it BEFORE our first Tier-3 Kaggle upload; PREREG_retry v2 section 4a). With RETRY_NOGO
# present nothing starts, AMENDMENT_1 is void, and t3_f0_final's FINAL (PREREG_f0 items) is the only full file.
user_decision: "<the user's message, verbatim>", <YYYY-MM-DD HH:MM> CST
tier3_board_scores_seen: none
written_by: main session
"""
for name, text in (("RETRY_GO.template", go), ("RETRY_NOGO.template", nogo)):
    p = RA / name
    if p.exists():
        os.chmod(p, 0o644)
    p.write_text(text)
    os.chmod(p, 0o444)
print("PREREG_retry v2 written;", len(R), "replacements; templates written")

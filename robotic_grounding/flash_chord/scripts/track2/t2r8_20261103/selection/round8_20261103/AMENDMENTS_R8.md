# t2_consensus_r8_20261103: AMENDMENTS_R8 (deviations from PREREG_R8.md, section 10 "Amendments")

## A1. Evaluation GPUs 0, 4, 6 and 7 instead of 6 and 7 (operational)

**When.** Written 2026-11-03 from about 13:40 to 13:50 CST. This is after T_c (12:00:00 CST), the pool freeze and the 89
phase-S exports (CPU, all ok), and **before any round-8 evaluation**. At 13:39:19 the phase-S heartbeat read evals
ok 0, failed 0, pending 267, running none; `runs_s/` did not exist and `eval_r8_driver.log` had no evaluation start. No
round-8 own-input or consensus-B value existed, so none was seen.

**Why.** The user, 2026-11-03: "you could use all gpu for round 8 since some of gpu empty right now". From T_c to
13:39 no evaluation could start, because GPUs 6 and 7 each listed Zuxing's trainings (the 3.5 wait). At 13:27-13:33
GPUs 0 and 4 listed no compute process, and no MPS server, client or control daemon named them.

**Change.** An evaluation may start on GPU 0, 4, 6 or 7 (before A1: 6 or 7). GPUs 1, 2, 3 and 5 are never used.
Every other rule of 3.5 holds on each of these GPUs:
- plain: no `CUDA_MPS_*` variables, and never a GPU with an MPS server, client or control daemon;
- at most one evaluation of ours per GPU, and at least 4096 + 1536 MiB free;
- an evaluation starts on a GPU only while `nvidia-smi --query-compute-apps` lists no compute process on it; nothing
  of anyone else's is touched or signalled;
- round 7's lane locks (our `evallock/gpu<g>.lock` plus every shared lane-lock file of that GPU), `nice -n 10`, a 10 s
  cooldown per GPU; a running evaluation is never signalled; `STOP` stops new starts;
- at most 2 evaluations of ours per NUMA eval lane (round 7's `--lane-max 2`, unchanged). GPU 0 is on NUMA0 and GPUs 4,
  6 and 7 are on NUMA1, so at most 3 evaluations run at once: GPU 0 plus two of GPUs 4, 6 and 7. `--max-parallel` is 4
  (one per GPU of the set); the lane rule binds first.
- `taskset`: round 7's unchanged code pins each evaluation to the eval-lane cores of its GPU's NUMA node. For GPU 0
  these are the NUMA0 eval-lane cores 48-51,152-155. For GPUs 4, 6 and 7 they are the NUMA1 cores 100-103,204-207, as
  before.

**Unchanged.** The selection rule (sections 4 and 5), T_c and the frozen pool, the protocol of 3.1 (official evaluator,
stock runtime, export records), 3 runs per screen candidate and 10 per finalist, the run order, retries, the
deadlines of 7.3 and the integrity baseline. No input, candidate, export, manifest or record is added, removed or
re-made: the relaunched evaluation driver reuses `manifest_s8.json` and the 89 export records as they are.

**`PREREG_R8.md` is not edited.** It stays bb912c57180dde8255776884de169e12467349f57d584782308b88ff935abf8c. Its sha256 is
hard-coded in the four round-8 scripts and recorded in the frozen `POOL_R8.json` and in `manifest_s8.json`, and
`r8_select.py` checks the manifest's value. A1 instead supersedes these statements of `PREREG_R8.md`:
- "Standing rules", GPUs: read "Evaluations run only on GPUs 0, 4, 6 and 7 ... GPUs 1, 2, 3 and 5 are never used".
- C5 and 3.5 (GPUs, Only while free, Waiting): "GPUs 6 and 7" and "GPUs 6/7" read "the GPUs of A1". The Processes line
  reads as the `taskset` item above.
- 7.3: "if GPUs 6/7 never became free" reads "if no GPU of A1 ever became free".
- 8, wall time: about 55 evaluations per hour per GPU (round 7's measurement), with at most 3 at once under the lane
  rule.
- 9.1: phase F compares a finalist's round-8 runs, now on any GPU of A1, with the incumbent's round-7 runs on GPUs 6
  and 7. All eight GPUs are the same model (NVIDIA RTX PRO 6000 Blackwell Server Edition, driver 580.82.09) and use the
  same stock runtime and integrity baseline. There is no selectable seed (3.6). Each run records its GPU index and
  UUID.
- 10, hard rules: "GPUs 6 and 7 only" reads "the GPUs of A1 only".

**Implementation** (old files kept read-only in `superseded_a1_20261103/` as `*.bak`):
- `r8_eval.py` 81bc9d1c36c34455 -> 5dcf6a8235c4317f. It adds `ALLOWED_GPUS_R8 = (0, 4, 6, 7)` and re-binds round 7's
  module name `ALLOWED_GPUS` to it in `configure()`. Round 7's pinned `gpu_rows` and `load_driver` read that name at
  call time. Its GPU-row check also marks every GPU outside the set unusable. Defaults are now `--gpus 0,4,6,7
  --max-parallel 4`, the argument checks allow `--max-parallel` up to 4 and `--gpus` from the set only, and
  `--lane-max <= 2` and `--cooldown >= 10` are unchanged. Docstring updated.
- `r8_driver.py` 1d626bed15b8e2fe -> 3072efe2d8c57947. It starts `r8_eval.py run` with `--gpus 0,4,6,7
  --max-parallel 4 --lane-max 2 --cooldown 10` (before: `--gpus 6,7 --max-parallel 2`). Docstring updated.
- `r8_launch.sh` 7dee83ada1d7eb29 -> 3177cd9b2b0d7fd4. Comment only.
- `PREREG_R8_CHANGES.md`: one dated pointer entry to this file.
- `PREREG_R8_PINS.md`: section 3 (addendum A1) pins the new files and this file. The four superseded pin lines in
  section 2 are marked `superseded-by-A1` and keep their old values. `PREREG_R8_PINS.sha256` and `LOCK_R8.sha256` are
  re-written for these files, and `LOCK_R8.sha256` adds this file and `AMENDMENTS_R8.sha256`.
- `manifest_s8.json` keeps its record `prereg_r8_pins_sha256` 321bdd7c... of the pins file it was built under. No
  check reads that field.

**Procedure** (nothing was signalled):
- 13:39:33: `STOP` was created.
- 13:39:35: `r8_eval.py` (pid 778886) logged "STOP file: no new job; waiting for 0 running" and ended. Its heartbeat
  reads "stopped (STOP file)".
- 13:40:00: `r8_driver.py` (pid 877263) logged "STOP file (not a deadline): driver ends after phase S evaluations;
  nothing selected" and ended. Both processes were gone from /proc by 13:40:03.
- Then the files were replaced (new file + mv), re-pinned and checked (`sha256sum -c LOCK_R8.sha256`, every `pin`
  line, a plan-only `r8_eval.py run --phase screen --dry-run`). `STOP` was removed and `r8_launch.sh` started a new
  orchestrator, which resumes from the frozen pool, the reused manifest and the finished exports.

**What the writer of A1 saw:** the round-8 heartbeats and logs (the export lines hold export sha256 prefixes and
max_rel_diff values, no evaluation value), the top-level keys of `POOL_R8.json` and `manifest_s8.json`, the round-8
scripts and documents, round 7's driver code, and `nvidia-smi` / `/proc` / `/proc/locks`. No round-8 score.

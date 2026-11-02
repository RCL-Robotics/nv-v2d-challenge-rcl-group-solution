# What of the R7-CLOSEST file is not in a pushed commit of its author

Status at 2026-11-02 (item 2 updated 2026-11-03). The items below are the ones whose AUTHOR has not pushed them to a
branch of his own. Read-only copies of those that are on the team server are committed here; item 2 names the scripts
that are not on the team server, so they are not in this commit either.

## Pushed (nothing to do)

- Training code of all 30 checkpoints: ours `code/fc_r2` 61fa5d3 (src/, scripts/ and configs identical to f2923d6 in this history; branch
  dev.haoran.c3r2 41403d3); Zuxing's bc3b1bd (dev.zuxing.speedup; ep04, ep05, ep26) and 82837c8
  (dev.zuxing.hard_start; ep00, 16, 17, 19, 20, 21, 22, 24, 25, 28). All three are ancestors of this commit. Each commit is named by the first `train_t2:` line
  of the run's training log (and, for the 5090 runs, by his MANIFEST.tsv).
- Our input builders (raw retarget, K3, K3fix, K3CA) and selection / pack scripts: in this history.

## Not pushed by Zuxing (only our read-only copies are committed) -> please ask him to push

1. **G2 input builder** (6 picks train on G2: ep04, ep16, ep19, ep21, ep24, ep28):
   - `zuxing.lu/tmp/t2/g2data/build_g2_data.py` (sha256 ca19a161e0e53c28..., unchanged since 2026-10-24) and its build
     logs `tmp/t2/g2data/build*.log`;
   - the G2 surface-only generator `zuxing.lu/tmp/tier2_noise/kfix/gen/G2_surface_only/` (README, common.py, prep.py,
     model.py, detect.py, proxies.py, run_episode.py, make_csv.py, run_all.sh, check_ep28.py, out/proxies.csv) and
     `tmp/tier2_noise/kfix/run_fix.py`;
   - the data-root build records `zuxing.lu/data/human_motion_data_tier2_g2-2026-10-24/build_*.json` (the ones these
     picks use: build_ep04_01_19_28.json, build_ep16.json, build_ep21_09.json, build_ep24.json; unchanged since the
     round-6 record). The 2026-11-01 rest build (build_ep00_02_03_07_12_17_18_20_22_26_27_29.json, which also
     overwrote build.json) is used by no checkpoint of this file; its build log is not in tmp/t2/g2data.
2. **His launchers**: `zuxing.lu/tmp/t2/train_t2.sh`, `run_g2_runin.sh` (ep04), `run_ep05_ep26_runin_k3fix.sh`
   (ep05, ep26) and the others in tmp/t2/ (copies in `train/zuxing_r7/`), and the evaluation watcher that these
   `run_*.sh` scripts start, `tmp/t2/eval/eval_watch.sh` with `eval_watch_helper.py` (copies in
   `../t2ready_20261031/train/zuxing/eval/`; its own-input rows designated ep05 88000 and ep26 92000).
   Not on the team server at all, so not in this commit: the version of `tmp/t2/train_t2.sh` that launched ep05 / ep26
   on 2026-10-24 at 04:35:10 (the file was rewritten at 05:19:09; the committed copy is that later version, which
   launched his runs from 05:20 on, e.g. ep04), the launchers of his RTX 5090 runs
   (2026-10-24-tier2-runin-k3fix, 2026-10-25-tier2-runin-hard-g2 + cont + cont2, 2026-10-28-tier2-runin-a5 + cont,
   2026-10-28-tier2-5090-fill + cont, 2026-10-29-tier2-5090-queue, 2026-10-31-tier2-5090-round2), and the generic run
   wrapper `launch_run.sh` of his own tool set, which started the `run_*.sh` scripts. The exact Hydra overrides of every
   stage are recorded (`records/zuxing_runs_r7/`), so the runs are specified; these scripts themselves are missing.
3. **Uncommitted edits**: his launcher prints `git log -1` of the worktree, so uncommitted edits at training time would
   not show. Please confirm that the dev.zuxing.speedup / dev.zuxing.hard_start worktrees had no uncommitted changes (or
   push the diff).
4. **RTX 5090 data copies**: the 5090 copies of the raw / K3fix / G2 data roots cannot be hashed from the team server.
   The evaluator always used the team server's parquet (same root and relative path after relocation); please confirm
   the 5090 copies equal them (e.g. send `sha256sum` of the 10 parquets the 5090 picks trained on: G2 ep16, 19, 21, 24,
   28; K3fix ep00, 17, 20, 22, 25).
5. **5090 folder completeness**: `runs/2026-11-01-round7-5090-ckpts` keeps only the last two checkpoints for 22 of its
   45 runs, and 11 runs were still training when copied. Nothing of this file depends on the missing files, but
   round 7's pool could only see what the folder holds.

Suggested: one branch of his (e.g. `dev.zuxing.g2data`) with items 1-2, then this record can point to his commit
instead of our copies.

# Zuxing Lu's G2 input and launchers: copies, with provenance

These files are copies from Zuxing Lu's workdir on the training server, `<ZX>` = `/root/autodl-tmp/zuxing.lu`. They are
not in any pushed commit of his. They were copied read-only on 2026-10-31. For every file, the sha256 of the source
before the copy, of the copy, and of the source after the copy are equal. `COPIES_t2ready.tsv` lists the source path,
size, mtime and the three hashes. The bytes are unchanged, except in the two bundle notes
`../../train/zuxing/bundles/5090_best_*/README.txt` (edited copies, see `../../../t2r7_20261102/REDACTIONS.tsv`); only
the folder layout differs:

| here | his path |
|---|---|
| `kfix/run_fix.py` | `tmp/tier2_noise/kfix/run_fix.py` |
| `G2_surface_only/*` | `tmp/tier2_noise/kfix/gen/G2_surface_only/{README.md, common.py, prep.py, model.py, detect.py, proxies.py, run_episode.py, make_csv.py, run_all.sh, check_ep28.py, out/proxies.csv}` |
| `g2data/*` | `tmp/t2/g2data/{build_g2_data.py, build*.log}` |
| `data_root_records/*` | `data/human_motion_data_tier2_g2-2026-10-24/build*.json` |
| `../../train/zuxing/*.sh` | `tmp/t2/{train_t2.sh, run_g2_runin.sh, run_g2_runin_hard.sh, run_a5_runin.sh, run_ep05_ep26_runin_k3fix.sh, run_ep04_runin_k3fix.sh}` |
| `../../train/zuxing/eval/*` (copied 2026-11-03; `../../CODE_COPIES.tsv`) | `tmp/t2/eval/{eval_watch.sh, eval_watch_helper.py}` |
| `../../train/zuxing/bundles/*` | `runs/2026-10-24-5090-best/{5090_best_2026-10-24, 5090_best_2026-10-27, retest_best_2026-10-29}/{README.txt, SHA256SUMS*}` |
| `../../records/hydra/<job>/s<N>/` (his 10 checkpoints) | the run's `hydra/.hydra/{config,overrides,hydra}.yaml` (+ `final.json`), see `set/set_items.json` `record_source` |

## What G2 is and how the G2 input was built

1. **Initial estimate.** `kfix/run_fix.py agile+hgate` runs his variant of our K3. It uses per-episode process noise of
   max(pooled, the episode's own ML value), and a height-consistency gate on the K2 anchors. The output is
   `kfix/out/agile+hgate__K3__<ep>.npy`.
   - It imports our `kfilter.py`, `kalman_core.py` and `kcommon.py`. His copies in `tmp/tier2_noise/kfix/scripts/` are
     byte-identical to `../ours/k3/kalman/scripts/`: sha256 ef747a39..., aa635fc9... and ad143851....
   - It also reads our design files `design/d3_ml.json` and `frozen_params.json`, and the kit's `v2dlb`.
2. **G2_surface_only.** `G2_surface_only/run_all.sh <out>` writes `<out>/pose_ep<NN>.npy`. It is a batch MAP smoother
   of the object track, with a hand-surface contact model, hand slip, floor, support boxes and human-root drift. Every
   constant is the same for all episodes. Its `README.md` documents the model.
3. **Data root.** `g2data/build_g2_data.py <root> <episodes>` writes the G1 data root
   `<ZX>/data/human_motion_data_tier2_g2-2026-10-24` and `frame_refs_g2/`. The root is the raw Tier-2 G1 parquet with only
   the four object columns replaced by the G2 track, mapped into the G1 world with the same rigid map as our K3fix
   builder. The frame references are the G2 pose on the visible frames. The script checks each parquet with the kit
   packer's own functions: it must be accepted against `frame_refs_g2` and rejected against the raw input. Its build
   records are in `g2data/build*.log` and `data_root_records/`.

The scripts use absolute paths into both workdirs and his kit copy (`<ZX>/ref_repos/v2d_submission_kit`). To rerun
them, set those paths.

## Not copied

These files are not needed to rebuild the inputs or the checkpoints, so they were not copied:
- `kfix/run_variants.py` and the G1 variant;
- the `dev/` ablations;
- `score_*`, `compare_*` and `contact_gate*`;
- his other evaluation scripts and his score files (not needed to rebuild the inputs).

# Tier 3 downstream steps: exact commands (agent inputs → opus_v2 → frame-0 → supports)

These commands were reconstructed from the Claude Code session transcripts (`~/.claude/projects/-home-hanzhou-Projects-v2d/55432ff5-….jsonl`, timestamps in UTC) and checked against the NAS files.

- **NAS clock:** NAS mtimes are local time (UTC+4). The NAS clock also runs about 5–6 min behind the transcript clock. Evidence: the `t3_frame0` command was sent at 10:09:10Z, and its output directory has the mtime 14:03:53 local.
- **Paths:**
  - `N=$V2D_ROOT` (data root, README section 0)
  - `REPO=<checkout>/hoi_estimation`
  - `A=${APPTAINER:-apptainer}`
  - (on the workstation: `N=/mnt/nas/Group/hanzhou/v2d`, `REPO=/home/hanzhou/Projects/v2d`, `V2D_RUN=/home/hanzhou/Projects/v2d/run`, `A=~/.local/apptainer/usr/bin/apptainer`)
  - `SIF=$N/images/v2d_cari4d.sif`

## A. Agent inputs, the opus_v2 agent runs, and the opus release

### A1. `agent/export_inputs.py`: release inputs (`T1_ds_ep<eee>`), camera JC2s1 passed by CLI flags

**Answer: the release camera came from explicit CLI flags.** No default was edited and no environment variable was used.

```bash
N=${V2D_ROOT:?}; A=${APPTAINER:-apptainer}
# 2026-10-20T11:54:26Z: episodes 14 3 28 13 19 25
# 2026-10-20T13:40:46Z: the other 24 episodes
for n in $(seq 0 29); do e=$(printf %03d $n)
  CUDA_VISIBLE_DEVICES=0 $A exec --nv -B $N:$N -B $REPO $N/images/v2d_cari4d.sif \
    python3 $REPO/agent/export_inputs.py $n --ws $N/agent_runs/T1_ds_ep$e \
      --variant DB_JC2s1_shared_f820_pp --align $N/logs/JC2s1_shared_f820_pp_align.json --camera JC2s1_shared_f820_pp
done
```

**Outputs:** `$N/agent_runs/T1_ds_ep<eee>/inputs/` containing:
- `video.mp4`, `object_mesh.glb`, `ground_plane.json`, `task.txt`, `episode.json`, `camera.json`;
- `reconstruction.npz`, `frame0_reference_human.npz`, `masks.h5`.

The `.preexported` marker was added at about 13:35Z.

**Evidence:**
- **Script version:** the last edit to `export_inputs.py` was at 2026-10-20T11:52:55Z (mtime 15:52 local). That is before both export batches. The local file is byte-identical to the git copy (`diff -rq` is clean).
- **Export times:** the `reconstruction.npz` mtimes are 15:49–15:56 local for 6 episodes and 17:35–18:02 local for 24 episodes. These match the two commands.
- **Inputs used by the runs:** compared the sha256 of every input file in all 30 release manifests (`agent_runs/CL2*/logs/run_manifest.txt`) with `T1_ds_ep<eee>/inputs`. All 30 episodes match exactly. The CL2* runs copied the T1_ds inputs (`cp -r`), as `run_batch_CL.sh` does.

### A2. Agent batches: `agent/run_batch_CL.sh` (P = 3 parallel, run from `$REPO/agent`)

```bash
cd $REPO/agent
PFX=CL2  TASK=AGENT_TASK_temporal_v2.md nohup bash run_batch_CL.sh 3 $(seq 0 29)  >> $N/agent_runs/CL2_batch.log  2>&1 &  # 2026-10-21T05:04:38Z
PFX=CL2b TASK=AGENT_TASK_temporal_v2.md nohup bash run_batch_CL.sh 3 $(seq 11 29) >> $N/agent_runs/CL2b_batch.log 2>&1 &  # 2026-10-21T09:57:13Z
PFX=CL2c TASK=AGENT_TASK_temporal_v2.md nohup bash run_batch_CL.sh 3 $(seq 22 29) >> $N/agent_runs/CL2c_batch.log 2>&1 &  # 2026-10-21T14:56:52Z
```

**Batch coverage:**
- CL2 delivered episodes 0–10 and CL2b delivered 11–21. Each batch stopped launching new runs when the subscription quota ran out (the `*_STOP` logic).
- CL2c delivered 22–29.

**Run settings:**
- **Time limit and effort:** 90 min, effort high. These are hard-coded in the git version. The local copy later gained the env vars `TMIN`/`EFFORT` (2026-10-29), whose defaults are 90/high, so the behaviour is the same.
- **Per-run call:** `run_claude.sh <ws> claude-opus-5-5 90 open <pipeline_dir> AGENT_TASK_temporal_v2.md high`. Manifest: `model: claude-opus-5-5 effort=high net=open`, `task_sha256 5d5091be…`, `claude 2.1.303`.
- **`<pipeline_dir>`:** chosen by `run_batch_CL.sh` from `logs/JC2s1_shared_f820_pp_align.json[ep].src`:
  - `src == M1` → `experiments/M1_prompt_v2/out/<ep>`. These are episodes 3, 6, 12, 14, 16 and 21, confirmed by the manifest binds.
  - Otherwise `outputs/cari4d/<ep>`.
  - Episode 25 uses `outputs/cari4d_failed/episode_000025_out`.

Start time (UTC) and wall time of each run, from the manifests:

| ep | run | date_utc | wall_s | ep | run | date_utc | wall_s |
|---|---|---|---|---|---|---|---|
| 0 | CL2_ep000 | 10-21 05:04 | 3578 | 15 | CL2b_ep015 | 10:58 | 3548 |
| 1 | CL2_ep001 | 05:05 | 3201 | 16 | CL2b_ep016 | 11:16 | 3237 |
| 2 | CL2_ep002 | 05:05 | 3767 | 17 | CL2b_ep017 | 11:58 | 2849 |
| 3 | CL2_ep003 | 05:58 | 3676 | 18 | CL2b_ep018 | 12:10 | 2547 |
| 4 | CL2_ep004 | 06:04 | 5262 | 19 | CL2b_ep019 | 12:19 | 3876 |
| 5 | CL2_ep005 | 06:08 | 2300 | 20 | CL2b_ep020 | 12:45 | 2574 |
| 6 | CL2_ep006 | 06:47 | 3721 | 21 | CL2b_ep021 | 12:52 | 2607 |
| 7 | CL2_ep007 | 07:00 | 3622 | 22 | CL2c_ep022 | 14:56 | 4033 |
| 8 | CL2_ep008 | 07:32 | 1933 | 23 | CL2c_ep023 | 14:57 | 2721 |
| 9 | CL2_ep009 | 07:49 | 4043 | 24 | CL2c_ep024 | 14:57 | 3117 |
| 10 | CL2_ep010 | 08:00 | 3860 | 25 | CL2c_ep025 | 15:42 | 3018 |
| 11 | CL2b_ep011 | 09:57 | 4715 | 26 | CL2c_ep026 | 15:49 | 3567 |
| 12 | CL2b_ep012 | 09:57 | 3041 | 27 | CL2c_ep027 | 16:04 | 4242 |
| 13 | CL2b_ep013 | 09:58 | 3627 | 28 | CL2c_ep028 | 16:33 | 5156 |
| 14 | CL2b_ep014 | 10:48 | 5387 | 29 | CL2c_ep029 | 16:49 | 2450 |

### A3. Opus release (`tier3_opus_20261021`)

```bash
python3 $REPO/agent/build_release_opus.py $N/release/tier3_opus_20261021   # 2026-10-21T18:44:11Z (host python)
```

The script mtime is 10-21 22:44 local, just before the call, and the script is git-identical. The episode → run mapping (CL2 0–10, CL2b 11–21, CL2c 22–29) is hard-coded in the script.

## B. Frame-0 correction and the combined release

### B1. `tier2_refine/t3_frame0.py opus … 3 2 shift`

The `flip=3` mode was added and run in the same command, at 2026-10-23T10:09:10Z. That command looped over `"opus 2 0 shift" "opus 3 2 shift" "s1 3 2 shift"`. The release line is:

```bash
cd $N && $A exec -B $N -B $REPO \
  images/v2d_cari4d.sif python $REPO/tier2_refine/t3_frame0.py opus experiments/t3_frame0/opus_f3_c2_shift 3 2 shift
```

- **Output:** `$N/experiments/t3_frame0/opus_f3_c2_shift/episode_<6d>.npz` (key `T_world_mesh`) and `diag_scores.json`. The directory mtime is 10-23 14:03 local.
- **Runs on CPU** in the CARI4D image.
- **Verified 2026-11-02:** re-ran the git copy (`gh_hoi_estimation/hoi_estimation/tier2_refine/t3_frame0.py`) with the same arguments into a scratch directory. The max |ΔT_world_mesh| over all 30 episodes is **0**, so the output is bit-identical. It printed `MEAN auc 0.552` (a GT diagnostic only).
- The local `t3_frame0.py` differs from git only in the `run:A,B` prefix-list source (2026-10-28), which does not affect the `opus` source.

### B2. Combined release `tier3_frame0_supports_20261024`

```bash
cd $REPO/support && python3 build_release_combined.py $N/release/tier3_frame0_supports_20261024   # 2026-10-24T06:49:55Z (host python)
```

- **Defaults applied:**
  - `TRAJ=$N/experiments/t3_frame0/opus_f3_c2_shift`
  - variant `opus_v2_frame0_f3_c2_shift`
  - `SRC=release/tier3_opus_20261021`
  - `SUP=release/tier3_supports_20261024`
- **Version:** the git copy has these defaults hard-coded. The local copy only made them overridable with `TRAJ`/`VARIANT`/`AGENT_PFX` (2026-10-27), which were not used for this release.
- **Release mtimes:** `episodes.json` 10-24 10:43 local. `reconstructed_data/` and `agent_run/` date from the 10-21 opus build.
- **Inputs:** `$TRAJ/diag_scores.json` is optional (it is written only by the GT half of `t3_frame0.py`
  and was only printed). From `SUP` the script copies `episode_*/{support.json, support_mesh.glb, support_world.obj,
  fit_overlay.jpg}`, `fit_overview.jpg`, `contact_report.json`, `index.json` and `README.md` (as
  `supports/README_supports.md`); see C for where they come from.
- `README.md` of this release and of `tier3_opus_20261021` were written by hand afterwards (heredoc, 10-24T06:50:25Z
  and 10-21T18:44:44Z); the texts are in `release_files/`. The build scripts first copy the README of their source release.
- **`frame_references/`** of both releases are copies of the s1 files (README section 5).

## C. Support fit (`support/`, 14 episodes)

The episodes are 1, 2, 3, 6, 7, 9, 11, 13, 16, 17, 20, 25, 27 and 28. The default list is `SUPPORT` in `common.py`, and every step accepts an optional comma-separated episode list. All steps run with `--pwd $REPO/support`, so the scripts are found relative to that directory.

The run order below follows the last command for each output that ended in the release.

```bash
cd $N
B="-B $N -B $REPO --pwd $REPO/support"
export APPTAINERENV_CUDA_VISIBLE_DEVICES=0
A=${APPTAINER:-apptainer}
$A exec $B images/v2d_cari4d.sif python background.py                    # 2026-10-23T19:02:49Z, all 14 (median background, occupancy)
$A exec $B images/v2d_cari4d.sif python anchor.py                        # 2026-10-23T19:18:29Z
$A exec --nv $B images/v2d_gdino.sif  python detect.py                   # 2026-10-23T19:18:29Z  (GroundingDINO image)
$A exec --nv $B images/v2d_cari4d.sif bash -c "python segment.py"        # 2026-10-23T19:18:29Z  (SAM2)
# fit: final support.json files come from TWO runs with DIFFERENT fit.py versions (see caveat 1)
$A exec --nv $B images/v2d_cari4d.sif bash -c "python fit.py && python overlay.py && python sheet.py"   # 2026-10-23T20:28:26Z: eps 2..28 final
$A exec --nv $B images/v2d_cari4d.sif bash -c "python fit.py 1 && python overlay.py 1"                  # 2026-10-23T21:04:50Z: ep 1 final
$A exec $B images/v2d_cari4d.sif bash -c "python contact.py --snap > /dev/null && python contact.py --gt > /dev/null && python score_snap.py"  # 21:09:47Z (--gt and score_snap are GT diagnostics only)
$A exec $B images/v2d_cari4d.sif python package.py $N/release/tier3_supports_20261024   # 2026-10-23T21:13:44Z
# 2026-10-23T21:14:32Z, host shell: the three files package.py does not write
R=$N/release/tier3_supports_20261024
cp $N/support/vis/sheet_fits.jpg $R/fit_overview.jpg
cp $N/support/contact_report.json $R/
cat > $R/README.md <<'EOF' ... EOF        # hand-written; text in reproducibility/release_files/tier3_supports_20261024_README.md
```

- **`fit_overview.jpg`** is `support/vis/sheet_fits.jpg` from `sheet.py` in the 20:28Z run (checked with `cmp`). That
  sheet was not redrawn after the ep 1 refit, so its ep 1 tile shows the 20:28Z fit; the per-episode
  `fit_overlay.jpg` of ep 1 (`vis/ep001_fit.jpg`, from `overlay.py 1`) shows the final fit.
- **`contact_report.json`** is `support/contact_report.json` from `contact.py --snap` (checked with `cmp`).
  `contact.py --gt` writes `contact_report_gt_diag.json` instead and does not touch it.
- **`index.json`** and the `episode_*` folders are written by `package.py`.

### Reproducing the support release (copy-paste, bash)

```bash
N=${V2D_ROOT:?set V2D_ROOT to the data root (README section 0)}
REPO=<checkout>/hoi_estimation; A=${APPTAINER:-apptainer}
B=(-B $N -B $REPO --pwd $REPO/support); export APPTAINERENV_CUDA_VISIBLE_DEVICES=0 APPTAINERENV_V2D_ROOT=$N
# optional: export APPTAINERENV_V2D_SUPPORT_OUT=<copy of $N/support> to leave $N/support untouched
$A exec "${B[@]}" $N/images/v2d_cari4d.sif python background.py
$A exec "${B[@]}" $N/images/v2d_cari4d.sif python anchor.py
$A exec --nv "${B[@]}" $N/images/v2d_gdino.sif python detect.py
$A exec --nv "${B[@]}" $N/images/v2d_cari4d.sif python segment.py
# all 14 with the 20:28Z fit (13 final), overlays and the sheet; then ep 1 with the current fit.py
$A exec --nv "${B[@]}" $N/images/v2d_cari4d.sif bash -c "python fit_20261023T2028Z.py && python overlay.py && python sheet.py"
$A exec --nv "${B[@]}" $N/images/v2d_cari4d.sif bash -c "python fit.py 1 && python overlay.py 1"
$A exec "${B[@]}" $N/images/v2d_cari4d.sif python contact.py --snap > /dev/null
R=<out>/tier3_supports_20261024
$A exec "${B[@]}" $N/images/v2d_cari4d.sif python package.py $R
cp $N/support/vis/sheet_fits.jpg $R/fit_overview.jpg; cp $N/support/contact_report.json $R/
cp $REPO/reproducibility/release_files/tier3_supports_20261024_README.md $R/README.md
```
To re-run only the fits on the released intermediates (no GroundingDINO / SAM2): the two `fit` lines, with an episode
list, e.g. `python fit_20261023T2028Z.py 2,28` and `python fit.py 1` (`check_steps.sh fit` does this on a copy).

### Per-episode fit settings

| ep | support | FREE_H | FREE_S | GRID_M | fit.py version | support.json mtime (local) | release IoU |
|---|---|---|---|---|---|---|---|
| 1 | tall_bar_stool | unset | unset | 0.4 (default) | after 21:04Z: blurred-IoU grid and refinement | 10-24 01:00:49 | 0.607 |
| 2 | round_table | unset | unset | 0.4 | 20:28Z: sharp IoU, `occ<0.9` | 00:26:37 | 0.610 |
| 3 | round_table | unset | unset | 0.4 | 20:28Z | 00:28:48 | 0.731 |
| 6 | round_table | unset | unset | 0.4 | 20:28Z | 00:31:00 | 0.634 |
| 7 | round_table | unset | unset | 0.4 | 20:28Z | 00:33:04 | 0.662 |
| 9 | skinny_wood_chair | unset | unset | 0.4 | 20:28Z | 00:35:00 | 0.818 |
| 11 | white_desk | unset | unset | 0.4 | 20:28Z | 00:37:06 | 0.762 |
| 13 | white_desk | unset | unset | 0.4 | 20:28Z | 00:39:30 | 0.765 |
| 16 | white_desk | unset | unset | 0.4 | 20:28Z | 00:41:58 | 0.761 |
| 17 | round_table | unset | unset | 0.4 | 20:28Z | 00:45:08 | 0.754 |
| 20 | round_table | unset | unset | 0.4 | 20:28Z | 00:47:12 | 0.661 |
| 25 | round_table | unset | unset | 0.4 | 20:28Z | 00:49:17 | 0.762 |
| 27 | white_desk | unset | unset | 0.4 | 20:28Z | 00:51:25 | 0.814 |
| 28 | tall_bar_stool | unset | unset | 0.4 | 20:28Z | 00:53:27 | 0.825 |

**Evidence for the settings:**
- **FREE_H and FREE_S** are diagnostic switches only. With either one set, `fit.py` prints `diagnostic: height offset …, scale …` and **returns before writing `support.json`**. They were used only for diagnostic prints:
  - `FREE_H=1` on episodes 1 and 28 at 20:09Z;
  - `FREE_S=1` on episodes 1 and 28 at 21:00Z;
  - both on episode 1 at 21:07Z.

  Consistent with this, every `support.json` has `min_height_m ≈ 0`, and none stores a scale.
- **GRID_M=1.0** was used only once, for `fit.py 1` at 20:14:58Z. That result was overwritten by the full run at 20:28Z and again by `fit.py 1` at 21:04Z. Neither later run set `GRID_M`.
- **Release check:** the `T_world_mesh` of every `release/tier3_frame0_supports_20261024/supports/<seq>/support.json` equals `$N/support/ep<eee>/support.json` (`np.allclose`, 14 of 14).

## Caveats and uncertainties

1. **Mixed `fit.py` versions; the current script will not exactly reproduce episodes 2–28.**
   - At 21:04:50Z, `fit.py` was edited:
     - the grid search uses a blurred-silhouette IoU (`blur=4`);
     - the refinement uses a blur schedule of 4 → 2 → 0;
     - the best result is kept only from iteration 200 on;
     - `iou_grid` is recomputed as the sharp IoU.
   - Only episode 1 was refit after that edit. Episodes 2–28 in the release come from the 20:28Z version, which is the current code without that edit and without the 21:00Z edit (keep the best iterate; diagnostic `FREE_S`).
   - The current `fit.py` (mtime 10-24 01:04 local, git-identical) has the edit. Re-running it on episodes 2–28 can give somewhat different poses.
   - To reproduce exactly: run `support/fit_20261023T2028Z.py` for episodes 2–28 (current `fit.py` with the 21:00Z and 21:04Z edits reverted) and the current `fit.py` for episode 1, or use the released `support.json` files.
   - The `sed` that changed `iou=iou_fit if … else iou_f` to `iou=iou_f` ran at 2026-10-23T19:14Z, before every fit run; it touches only the reported number.
2. **Re-run 2026-11-02 (README section 6):** the fits of episodes 2 and 28 (`fit_20261023T2028Z.py`) and 1 (`fit.py`),
   `contact.py --snap` + `package.py`, B1 and B2. `background`/`anchor`/`detect`/`segment` (GroundingDINO, SAM2) were
   not re-run.
3. **Earlier support steps:** `background.py`/`anchor.py`/`detect.py`/`segment.py` for episodes 2 and 3 were also run earlier, at 19:14–19:15Z. The final run at 19:18Z covered all 14 episodes with default arguments and overwrote them.
4. **Contact step:** the plain `contact.py 2,3,…,28` (20:25Z) was superseded by `contact.py --snap` at 21:09Z. That run wrote `support/contact_report.json` (mtime 01:04 local) and `object_poses_world_snapped.npz`. `package.py` reads the contact report (for `T_world_mesh_contact`); the release copy (01:08 local) was made by the `cp` at 21:14:32Z.
5. **`package.py`:** it was edited just before its last run, at 21:13Z (the `hm` fix with `sed`). The current file (mtime 01:13 local) is that version and is git-identical.
6. **Local copies that differ from git:**
   - `agent/run_batch_CL.sh`: `TMIN`/`EFFORT` env vars, same defaults;
   - `tier2_refine/t3_frame0.py`: the `run:` prefix list;
   - `support/build_release_combined.py`: `TRAJ`/`VARIANT`/`AGENT_PFX` env vars;
   - `agent/collect_records.py`.

   All of these edits came after the releases and do not change the default behaviour.

   Local-only files (later Kimi/pi and v3/v4 work, not used by the releases):
   - `agent/AGENT_TASK_temporal_v3.md`, `agent/AGENT_TASK_temporal_v4.md`;
   - `agent/export_frame0_object.py`;
   - `agent/pi_models.json`, `agent/pipeline_pi.sh`, `agent/run_batch_pi.sh`, `agent/run_pi.sh`.

   `export_inputs.py`, `run_claude.sh`, `build_release_opus.py` and the support scripts are git-identical.

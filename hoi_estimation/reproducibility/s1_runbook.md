# s1 runbook: how `tier3_s1_20261020` was produced

The release `tier3_s1_20261020` was built from the object variant `DB_JC2s1_shared_f820_pp` and the camera variant
`JC2s1_shared_f820_pp`. This runbook reconstructs, step by step, the command that produced the files the next step
actually consumed.

## Sources and how to read them
- **Commands** come from the Claude Code session transcripts that ran them:
  - `b193eefa…` (2026-10-04 to 10-16);
  - `55432ff5…` (2026-10-16 onwards).
- **Times:**
  - Transcript times are UTC (`Z`).
  - NAS file mtimes are given in UTC as well, already converted from local Asia/Dubai (UTC+4).
  - The NAS clock runs about 5 minutes behind the workstation. A file can therefore appear a few minutes before the
    command that wrote it.
- **Script versions:** every script named here is byte-identical between `~/Projects/v2d/hoi_extraction/` and
  `gh_hoi_estimation/hoi_estimation/hoi_extraction/` (checked with `cmp`, 2026-11-02). Edits made *after* a step ran
  are listed under that step.
- **Scoring is not part of the chain.** Commands also ran `eval_official.py` / `compare_experiments.py`, which score
  against GT. Those scoring calls are omitted below: they only wrote CSVs under `logs/`, and no step reads them.

## Common definitions
```bash
N=/mnt/nas/Group/hanzhou/v2d
H=$HOME/Projects/v2d/hoi_extraction          # = hoi_estimation/hoi_extraction in the repo
S=$H/scripts
A=$HOME/.local/apptainer/usr/bin/apptainer
# "light" container: CPU-only scripts
RUN()  { $A exec -B $N:$N -B $HOME/Projects/v2d $N/images/v2d_cari4d.sif python "$@"; }
# "GPU" container with the SOMA / MHR / SAM-3D-Body libraries on PYTHONPATH
RUNC() { CUDA_VISIBLE_DEVICES=0 $A exec --nv -B $N:$N -B $HOME/Projects/v2d $N/images/v2d_cari4d.sif bash -c \
  "export PYTHONPATH=/workspace/v2d_cari4d/lib/cari4d:/workspace/v2d_sam3d_body/lib:\$PYTHONPATH LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH; $*"; }
PY=$HOME/Projects/KalmanHOI/.venv/bin/python   # host venv: python 3.12.13, numpy 2.5.2, torch 2.10.0+cu128, scipy 1.17.0
```

**GPU numbering.** The original runs used either GPU 0 or GPU 1 of the workstation (RTX 6000 Ada). The GPU index does
not change the results.

## Pipeline file copies on NAS
The per-episode pipeline reads its files from `$N/code/pipeline/`, not from the repo:
- `$N/code/pipeline/run_episode.sh`
- `$N/code/pipeline/gen_prompt_one.py`
- `$N/code/pipeline/gen_prompt_v2.py`
- `$N/code/pipeline/episodes.tsv`
- `$N/code/pipeline/object_info.json`

It also bind-mounts `$N/code/reconstruction/modules` (the CARI4D module copy) and the FoundationPose export patch
`$N/build/patches/run_foundationpose_mhr_export.py`.

The current copies are identical to the repo files `cluster/run_episode.sh`, `cluster/gen_prompt_v2.py` and
`patches/run_foundationpose_mhr_export.py` (checked with `cmp`).

---

## 1. CARI4D baseline, per episode → `$N/outputs/cari4d/<ep>/inference/coconet.pth`

The baseline was run in several batches, plus a separate run for episode 0.

**Episode 0.** It ran on the CIAI cluster:
- Job: `sbatch /l/users/han.zhou/v2d/build/slurm/smoke_ep0.sh` (repo: `slurm/smoke_ep0.sh`), last submitted at
  10-04T19:35Z.
- Output: `coconet.pth`, mtime 10-04T19:28Z (NAS clock).
- Copy to NAS: an rsync at 10-15T10:10Z (`logs/ep0_copy.log` = `EP0_COPIED`):
  ```bash
  mkdir -p $N/outputs/cari4d/episode_000000/{foundationpose,depth/moge2,work}
  rsync -a ciai-via-local:/l/users/han.zhou/v2d/outputs/cari4d/episode_000000/inference/{coconet.pth,refined.pth} $N/outputs/cari4d/episode_000000/inference/
  rsync -a ciai-via-local:/l/users/han.zhou/v2d/outputs/cari4d/episode_000000/foundationpose/ $N/outputs/cari4d/episode_000000/foundationpose/
  rsync -a ciai-via-local:/l/users/han.zhou/v2d/outputs/cari4d/episode_000000/depth/moge2/ $N/outputs/cari4d/episode_000000/depth/moge2/
  ```
  The transcript line is truncated at this point. The copy also wrote `work/` items such as masks.

**Episodes 1–29.** They ran locally with `scripts/run_local.sh "<gpus>" <eps…>`. It calls
`$N/code/pipeline/run_episode.sh <i>` with `WORK_ROOT=~/Projects/v2d/run/work` and `OUT_ROOT=~/Projects/v2d/run/out`,
and moves each PASS episode to `$N/outputs/cari4d/`. Stage markers make a rerun skip finished stages. The batches were:

| UTC start | command | episodes that finished OK (from the logs) |
|---|---|---|
| 10-05T11:21Z | `setsid nohup bash $S/run_local.sh "0 1" $(seq 1 29) > $N/logs/run_local_1005_1521.log` | 1 3 6 8 10 12 16 18 20 22 24 26 |
| 10-05T17:09Z (after kill/requeue) | `bash $S/run_local.sh '1' $(seq 1 29) > $N/logs/run_local_retry.log` | 4 15 17 21 27 29 |
| 10-06T10:06Z | `setsid nohup bash $S/run_local.sh "1" 2 5 7 9 11 13 14 19 23 25 28 > $N/logs/run_local_fix_1006_1406.log` | 2 5 7 9 11 13 14 19 23 28 (**25 FAILED**) |

- The first batch was killed partway (10-05T17:09Z), and the GPU-1-only retry pass was queued behind it.
- **Episode 25 has no baseline output.** Every later step uses its E2 bundle (step 2) instead.

**Caveats: the batches ran different versions of `run_episode.sh`.**
- **10-05T12:10Z edit:** added the bind of the NaN-centroid FoundationPose patch.
  - `run_local.sh` re-reads `run_episode.sh` for each episode, so episodes started before 12:10Z ran without this
    bind: episode 1 (OK) and the early tries of 2 and 4 (both failed, later redone).
  - The patch only changes a diagnostic value from `inf` to `NaN`. Without it, the validator would fail the episode.
- **10-06T10:06Z edit:** "never re-copy existing input video", plus a retry with `--postopt_batch_size 384` and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  - Only the third batch (2 5 7 9 11 13 14 19 23 28) had it.
  - The logs show which episodes took the windowed retry branch (`retrying with --postopt_batch_size`): episodes 2, 5,
    7, 11, 13, 19, 23 and 28 (and 25, which failed anyway). Episodes 9 and 14 finished without it.
- **Episode 0** used the CIAI copy of `run_episode.sh` from 10-04T19:35Z, which is an earlier version again.

## 2. E2_reregister: FoundationPose re-registration → `$N/experiments/E2_reregister/<ep>/coconet.pth`

`scripts/exp_fp_mode.sh` reruns CARI4D stage 5 (FoundationPose) and stage 6 (CoCoNet) on an episode's existing outputs.

The patched `run_foundationpose_mhr_export.py` adds the re-registration:
- It is enabled by `V2D_FP_REREGISTER=1`.
- After 3 consecutive frames with IoU below 0.3, it re-registers from the last reliable pose.
- It is bind-mounted from `$N/build/patches/`, which was updated at about 10-07T08:35Z, before E2 ran.

```bash
# 10-07T08:39Z  (KEEP_LOCAL=1 because E1 shared the inputs at the time)
CUDA_VISIBLE_DEVICES=1 V2D_FP_REREGISTER=1 KEEP_LOCAL=1 setsid nohup bash $S/exp_fp_mode.sh E2_reregister "--register-first-then-track" 18 8 27 17 15 13 > $N/logs/exp_E2_reregister.log 2>&1 < /dev/null &
# 10-07T09:24:52Z  (relaunch of a run killed at 09:24:41)
CUDA_VISIBLE_DEVICES=1 V2D_FP_REREGISTER=1 setsid nohup bash $S/exp_fp_mode.sh E2_reregister "--register-first-then-track" 1 2 4 6 7 9 11 12 14 16 19 28 25 > $N/logs/exp_E2_reregister_rest.log 2>&1 < /dev/null &
# 10-08T08:09:50Z  episode 3 rerun (after rm -rf $N/experiments/E2_reregister/episode_000003)
CUDA_VISIBLE_DEVICES=1 V2D_FP_REREGISTER=1 setsid nohup bash $S/exp_fp_mode.sh E2_reregister "--register-first-then-track" 3 > $N/logs/exp_E2_reregister_ep3.log 2>&1 < /dev/null &
```

- **Result:** 20 episodes, 1 2 3 4 6 7 8 9 11 12 13 14 15 16 17 18 19 25 27 28.
  - The `coconet.pth` mtimes are 10-07T08:41Z to 15:47Z, and episode 3 at 10-08T08:45Z. These match the batches above.
- **Later edits to `exp_fp_mode.sh`, plumbing only:**
  - 10-08T08:09Z: temp dir moved to `$N/experiments/.tmp`; symlink restore instead of copy. This was applied before the
    episode-3 rerun.
  - 10-15T18:45Z: optional `SRC_ROOT`, after all E2 runs.
  - Neither edit changes the FoundationPose or CoCoNet arguments.
- **Episode 25:** its baseline never produced `inference/`, but the stage inputs were available, so E2 could run it.
  Uncertain: which stage inputs these were. Probably the kept local files.

## 3. P1_temporal → `$N/experiments/P1_temporal/<ep>/coconet.pth`

The run at 10-15T08:03:38Z, in bash:
```bash
bash -c '
N=/mnt/nas/Group/hanzhou/v2d; EPS="1 2 3 4 6 7 8 9 11 12 13 14 15 16 17 18 19 25 27 28"
run() { $HOME/.local/apptainer/usr/bin/apptainer exec -B $N:$N -B /home/hanzhou/Projects/v2d $N/images/v2d_cari4d.sif python /home/hanzhou/Projects/v2d/hoi_extraction/scripts/"$@"; }
run postproc_temporal.py $EPS --out-exp P1_temporal'
```
- Defaults: `--src-exp E2_reregister`, static hold on.
- Outputs: mtimes 10-15T07:58Z to 08:04Z (NAS clock), so this run.
  - An identical zsh attempt at 08:03:04Z failed, because zsh does not word-split `$EPS`.
  - A single-episode run, `postproc_temporal.py 8`, at 08:03:08Z was overwritten by this run.
- `postproc_temporal.py` was last written at 10-15T08:02:55Z, before the run. Its input is only the E2 bundles, and it
  reads no reference data.

## 4. M1: re-prompted CARI4D for episodes 3, 6, 12, 14, 16, 21 (28 ran but is not used)

### 4a. `object_info.json`
Inline Python run at 10-15T08:28:54Z:
```bash
$A exec -B $N:$N $N/images/v2d_cari4d.sif python -c "
import json,numpy as np,trimesh
ds='$N/dataset/track_2/tier_1_multiview_caption'; info=json.load(open(ds+'/meta/info.json'))
out={}
for o in info['objects']:
    g=trimesh.load(ds+'/'+o['mesh'],force='scene',process=False).dump(concatenate=True)
    out[o['name']]={'diag_m':float(np.linalg.norm(g.extents)),'prompt':o['prompt']}
json.dump(out,open('$N/code/pipeline/object_info.json','w'),indent=1); print(len(out))
"
```

### 4b. Prompts with `gen_prompt_v2.py`
The final version of `gen_prompt_v2.py` (with `--near_person_m`) was written at 10-15T08:37:14Z and copied to
`$N/code/pipeline/`.

The prompts for episodes 3, 6 and 12 came from that same 08:37:14Z command, written to `/tmp/pv2`. The command also
did episodes 8, 13, 17 and 27; those were never used.
```bash
for i in 6 3 12 8 13 17 27; do e=$(printf %06d $i)
  obj=$(python3 -c "import json;[print(d['object']) for d in map(json.loads,open('$N/dataset/track_2/tier_1_multiview_caption/meta/episodes_metadata.jsonl')) if d['episode_index']==$i]")
  diag=$(python3 -c "import json;print(json.load(open('$N/code/pipeline/object_info.json'))['$obj']['diag_m'])")
  desc=$(python3 -c "import json;print(json.load(open('$N/code/pipeline/object_info.json'))['$obj']['prompt'])")
  D=$N/outputs/cari4d/episode_$e/depth/moge2
  CUDA_VISIBLE_DEVICES=1 $A exec --nv -B $N:$N -B /tmp $N/images/v2d_gdino.sif bash -c 'export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH; exec "$@"' _ \
    python $N/code/pipeline/gen_prompt_v2.py --video_path $N/dataset/track_2/tier_1_multiview_caption/videos/chunk-000/observation.images.exo_camera/episode_$e.mp4 \
      --object_prompt "${obj//_/ }" --object_description "$desc" --object_diag_m $diag \
      --raw_depth $D/raw.npy --intrinsics $D/intrinsics.pkl --model_dir $N/weights/grounding_dino --output_path /tmp/pv2/ep$i.json
done
# 10-15T08:42:27Z: place the prompts
X=$N/experiments/M1_prompt_v2; mkdir -p $X/work $X/out
for i in 3 6 12; do e=$(printf %06d $i); mkdir -p $X/work/episode_$e; cp /tmp/pv2/ep$i.json $X/work/episode_$e/prompts.json; cp /tmp/pv2/ep${i}_candidates.json $X/work/episode_$e/; done
```

For episodes 14, 16, 21 and 28 (10-15T10:43:58Z), the same loop wrote straight to
`--output_path $X/work/episode_$e/prompts.json` on `CUDA_VISIBLE_DEVICES=0`.

Evidence: the `prompts.json` mtimes are 08:36Z for episodes 3, 6 and 12, and 10:39–10:40Z for 14, 16, 21 and 28.
No later write to these files was found.

### 4c. The M1 CARI4D run
The run script is `$X/run.sh`, written at 10-15T08:42Z. Its content, verbatim:
```bash
#!/usr/bin/env bash
# Re-run episodes with prompts from gen_prompt_v2.py (prompts.json pre-placed in WORK) + E2 re-registration.
export V2D_ROOT=/mnt/nas/Group/hanzhou/v2d SIF_EXEC=$HOME/.local/apptainer/usr/bin/apptainer
export WORK_ROOT=$V2D_ROOT/experiments/M1_prompt_v2/work OUT_ROOT=$V2D_ROOT/experiments/M1_prompt_v2/out
export V2D_FP_REREGISTER=1
for i in "$@"; do
  EP=$(printf "%06d" "$i"); echo "[M1] episode $EP start $(date '+%F %T')"
  bash $V2D_ROOT/code/pipeline/run_episode.sh "$i" > $V2D_ROOT/logs/episode_${EP}.M1.log 2>&1 && echo "[M1] episode $EP OK $(date '+%F %T')" || echo "[M1] episode $EP FAILED $(date '+%F %T')"
done
echo M1_DONE
```
It was launched twice:
```bash
CUDA_VISIBLE_DEVICES=1 setsid nohup bash $X/run.sh 3 6 12 > $N/logs/exp_M1.log 2>&1 < /dev/null &          # 10-15T08:42Z
CUDA_VISIBLE_DEVICES=0 setsid nohup bash $X/run.sh 14 16 21 28 > $N/logs/exp_M1b.log 2>&1 < /dev/null &    # ~10-15T10:46Z
```
All seven episodes finished OK (see `logs/exp_M1.log` and `exp_M1b.log`).

### 4d. The M1 symlink directory (manual)
Episodes 3 6 12 14 16 21 were linked by hand into the `M1` directory, at 10-15T13:10Z / 14:22Z and 15:52Z (episode 21):
```bash
for i in 3 6 12 14 16 21; do e=episode_$(printf %06d $i); mkdir -p $N/experiments/M1/$e
  ln -sf $N/experiments/M1_prompt_v2/out/$e/inference/coconet.pth $N/experiments/M1/$e/coconet.pth; done
```
- **Episode 28 was never linked.** Its M1 run finished at 16:09Z, after the linking loops had run. Every later step
  uses P1_temporal for episode 28.

## 5. `ground_scale.py`: **not used by the s1 chain**
- Run at 10-15T10:10Z: `RUN $S/ground_scale.py $(seq 1 29) > $N/logs/ground_scale.log`, then episode 0 alone at
  11:30Z. It wrote `logs/ground_scales.json`.
- Only the `--ground-scales` options of `firstframe_align.py` and `eval_official.py` read this file. The ICP runs that
  feed the chain (step 6) do not pass that option. So `ground_scales.json` does not affect the release.
  - `ff_align_baseline_gs.json` is an unused experiment.

## 6. First-frame ICP: `firstframe_align.py --wide` → `logs/ffw_baseline.json`, `logs/ffw_M1.json`

**The chain uses the `ffw_*` files, not `ff_align_*.json`.** The latter are earlier experiments and are not consumed.
- `ff_align_merged.json` and `ff_align_M1_partial.json` are not used downstream.
- `ff_align_M1_partial.json` was even overwritten with episode 21 only at 15:52Z, then merged into `ff_align_merged.json`.

The final version of `firstframe_align.py` (patched at 10-15T15:55:19Z) ran at 10-15T15:57:45Z:
```bash
B=""; for i in $(seq 0 29); do f=$N/outputs/cari4d/episode_$(printf %06d $i)/inference/coconet.pth; [ -f $f ] && B="$B $i=$f"; done
B="$B 25=$N/experiments/E2_reregister/episode_000025/coconet.pth"           # episode 25: E2 bundle (no baseline)
RUNC python $S/firstframe_align.py --wide --bundles $B --out $N/logs/ffw_baseline.json
B=""; for i in 3 6 12 14 16 21; do B="$B $i=$N/experiments/M1/episode_$(printf %06d $i)/coconet.pth"; done
RUNC python $S/firstframe_align.py --wide --bundles $B --out $N/logs/ffw_M1.json
```
- Original quoting: in the original command, `$B` sat inside the double-quoted `bash -c` string, so the `--bundles`
  list was word-split by the inner bash.
- Evidence: `ffw_baseline.json` has mtime 10-15T16:01Z, and `ffw_M1.json` 16:03Z.
- Input: `ref_human_f0/<ep>.npz`, the frame-0 reference human only.
- **The human comes from the baseline (or M1) bundle, not from E2/P1.** The ICP aligns the frame-0 human. E2 and P1
  change only the object, so the baseline human is the same.

## 7. PnP: `firstframe_pnp.py` → `pnp_baseline.json`, `pnp_M1.json` → `pnp_merged.json` = `pnp_forE1.json`

```bash
# 10-15T16:31:49Z
B=$(python3 -c "import json;d=json.load(open('$N/logs/ffw_baseline.json'));print(' '.join(f'{k}={v[\"bundle\"]}' for k,v in d.items()))")
RUNC python $S/firstframe_pnp.py --bundles $B --icp $N/logs/ffw_baseline.json --out $N/logs/pnp_baseline.json
B=$(python3 -c "import json;d=json.load(open('$N/logs/ffw_M1.json'));print(' '.join(f'{k}={v[\"bundle\"]}' for k,v in d.items()))")
RUNC python $S/firstframe_pnp.py --bundles $B --icp $N/logs/ffw_M1.json --out $N/logs/pnp_M1.json
# 10-15T16:44:53Z  merge (inline python, verbatim)
python3 -c "
import json; b=json.load(open('$N/logs/pnp_baseline.json')); b.update(json.load(open('$N/logs/pnp_M1.json'))); json.dump(b,open('$N/logs/pnp_merged.json','w'))"
# 10-16T08:44:03Z  copy under the name the joint-camera scripts use by default (inline python, verbatim)
python3 -c "
import json; b=json.load(open('$N/logs/pnp_merged.json'))
# E1 on baseline inputs shares the baseline human; E1M1 shares the M1 human (pnp_merged already holds M1 entries for 3,6,12,14,16,21)
json.dump(b,open('$N/logs/pnp_forE1.json','w'))"
```
- **Verified by loading the files:**
  - `pnp_baseline.json` has 30 episodes. Its episode 25 entry points to the E2 bundle.
  - `pnp_M1.json` has episodes 3 6 12 14 16 21.
  - `pnp_merged.json` equals baseline updated by M1.
  - `pnp_forE1.json` has the same content as `pnp_merged.json`.
- **mtimes:**

  | file | mtime (UTC) |
  |---|---|
  | `pnp_baseline.json` | 10-15T16:29Z |
  | `pnp_M1.json` | 16:30Z |
  | `pnp_merged.json` | 16:38Z |
  | `pnp_forE1.json` | 10-16T08:38Z |

- `firstframe_pnp.py` was written at 10-15T16:31:35Z and has not changed since.
- Input: `ref_human_f0/<ep>.npz`.

## 8. FF2_feet_ref → `$N/experiments/FF2_feet_ref/<ep>/coconet.pth` + `feet_floor.json`

`postproc_feet_floor.py` was patched at 10-15T18:18:00Z and ran at 10-15T18:18:11Z:
```bash
RUNC python $S/postproc_feet_floor.py \$(seq 0 29) --use-sim --ref-foot-height --align default=$N/logs/pnp_merged.json --out-exp FF2_feet_ref
```
- Default `--src` order: `M1 P1_temporal E2_reregister baseline`. The first variant with a bundle is used.
- In the original command, `\$(seq 0 29)` was expanded inside the container's bash.
- Evidence: the outputs have mtimes 10-15T18:12–18:18Z, and `feet_floor.json` 18:18Z (NAS clock).
- **Source per episode, from `feet_floor.json`:**

  | source | episodes |
  |---|---|
  | `M1` | 3 6 12 14 16 21 |
  | `P1_temporal` | 1 2 4 7 8 9 11 13 15 17 18 19 25 27 28 |
  | `baseline` | 0 5 10 20 22 23 24 26 29 |

- **Alignment file:** it used `pnp_merged.json`. Step 9 uses `pnp_forE1.json`, which has the same content.
- **Input:** `ref_human_f0` (`--ref-foot-height`). It also reads the dataset folder for the meshes and metadata.

## 9. JC2s1_shared_f820_pp and DB_JC2s1_shared_f820_pp

`postproc_joint_camera_K.py` was patched at 10-19T16:58:50Z. The run started at 10-19T16:59:00Z:
- the camera step ran in the GPU container, without the `PYTHONPATH` exports;
- the depth blend ran on the host in the KalmanHOI venv;
- the working directory was `$S`.
```bash
cd $S
CUDA_VISIBLE_DEVICES=0 $A exec --nv -B $N:$N -B $HOME/Projects/v2d $N/images/v2d_cari4d.sif \
  python postproc_joint_camera_K.py $(seq 0 29) --out-exp JC2s1_shared_f820_pp --sig-logf 1e-4 --sig-pp 0.01 --K-prior 820 758.1 560.5
$PY postproc_depth_blend.py $(seq 0 29) --src JC2s1_shared_f820_pp --out-exp DB_JC2s1_shared_f820_pp
```
- **Defaults:**
  - `postproc_joint_camera_K.py`:
    - `--src-map $N/experiments/FF2_feet_ref/feet_floor.json`
    - `--align $N/logs/pnp_forE1.json`
    - `--sig-px 2.0 --sig-h 0.02 --window 15 --clip 0.3`
  - `postproc_depth_blend.py`: `--w 0.5 --max-dev 0.1`.
- **Original form:** a shell function `run JC2s1_shared_f820_pp --K-prior 820 758.1 560.5` that also ran JC2s2 and
  JC2s3 and scored each variant. The scoring is omitted here.
- **Outputs:**

  | output | mtime (UTC) |
  |---|---|
  | `experiments/JC2s1_shared_f820_pp/<ep>/coconet.pth` | 16:54–17:04Z |
  | `logs/JC2s1_shared_f820_pp_align.json` | 17:04Z |
  | `experiments/DB_JC2s1_shared_f820_pp/<ep>/coconet.pth` | 17:05–17:12Z |
  | `depth_blend.json` | 17:12Z |

  - `depth_blend.json` covers 30 episodes, all with `src` JC2s1 and `w` 0.5. None was skipped by `--max-dev`.
- **Inputs:**
  - `postproc_joint_camera_K.py` reads `ref_human_f0` (frame-0 human only) and the dataset folder.
  - `postproc_depth_blend.py` reads the dataset folder.
- No run of either script for these variants was found after 10-19T17:00Z.

## 10. Release → `$N/release/tier3_s1_20261020`

`build_release.py` was last edited at 10-20T07:30:25Z. The release was built at 10-20T07:32:32Z:
```bash
cd $S
CUDA_VISIBLE_DEVICES=0 $A exec --nv -B $N:$N -B $HOME/Projects/v2d $N/images/v2d_cari4d.sif python build_release.py $N/release/tier3_s1_20261020
# defaults: --variant DB_JC2s1_shared_f820_pp --camera JC2s1_shared_f820_pp --episodes 0..29
```
- Evidence: the `reconstructed_data/*/poses.npy` files have mtimes 10-20T07:29–07:39Z.
- An earlier test build, `$N/release/tier3_s1_test --episodes 0`, was deleted.
- The `README.md` (mtime 07:27Z) was written by hand.

**`code/` and `frame_references/`.** These were written at 10-20T07:45:49Z, with the host venv, from `$S`:
```bash
R=$N/release/tier3_s1_20261020; mkdir -p $R/code $R/frame_references
cp $S/build_release.py $S/postproc_joint_camera_K.py $S/postproc_joint_camera.py $S/postproc_depth_blend.py $S/eval_official.py $R/code/
```
The loop that wrote `frame_references`, verbatim. The `ref_poses` and metrics lines only check the release against
the official CSV (GT). The saved files depend only on `object_pose_xyz_wxyz.npy`.
```python
import json,numpy as np,csv,trimesh,sys
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,".")
from eval_object_pose import metrics, quat_wxyz_to_R
N="/mnt/nas/Group/hanzhou/v2d"; R=f"{N}/release/tier3_s1_20261020"; DS=f"{N}/dataset/track_2/tier_1_multiview_caption"
off={int(r["ep"]):r for r in csv.DictReader(open(f"{N}/logs/officialDB_JC2s1_shared_f820_pp.csv"))}
idx=json.load(open(f"{R}/episodes.json")); bad=0
for d in idx:
    e=d["episode_index"]; od=f"{R}/reconstructed_data/{d['sequence_id']}"
    T=np.load(f"{od}/poses.npy"); P=np.load(f"{od}/object_pose_xyz_wxyz.npy"); z=np.load(f"{od}/soma_params.npz",allow_pickle=True)
    assert T.shape[0]==d["frames"]==P.shape[0]==z["poses"].shape[0]==z["transl"].shape[0], e
    assert np.isfinite(T).all() and np.isfinite(z["poses"]).all()
    q=P[:,3:]; Rq=quat_wxyz_to_R(q.astype(float)); assert np.abs(Rq-T[:,:3,:3]).max()<1e-4 and np.abs(P[:,:3]-T[:,:3,3]).max()<1e-5
    xyzw=np.c_[P[:,:3],P[:,4:],P[:,3:4]].astype(np.float32)[:,None]
    np.savez(f"{R}/frame_references/episode_{e:06d}.npz",episode_index=np.int64(e),coordinate_frame=np.array("released_track2_world"),
             frame_index=np.arange(len(P),dtype=np.int64),pose_xyzw=xyzw)
    ref=np.load(f"{N}/ref_poses/episode_{e:06d}.npz")["pose_xyz_wxyz"].astype(float); n=min(len(T),len(ref)); ok=np.linalg.norm(ref[:n,3:],axis=1)>0.5
    g=trimesh.load(f"{od}/reconstructed_mesh/output_aligned.glb",force="scene",process=False).to_geometry(); np.random.seed(0); v=np.asarray(g.sample(1000))
    m=metrics(T[:n][ok][:,:3,:3],T[:n][ok][:,:3,3],quat_wxyz_to_R(ref[:n][ok][:,3:]),ref[:n][ok][:,:3],v)
    dm=abs(m["MPPE_cm"]-float(off[e]["MPPE_cm"]))
    if dm>0.01: bad+=1; print("MISMATCH",e,m["MPPE_cm"],off[e]["MPPE_cm"])
print("checked",len(idx),"episodes; MPPE mismatches:",bad)
```

---

## Data flow, in short
```
baseline CARI4D (ep0 CIAI; 1-29 local; 25 failed)
  ├─ E2_reregister (20 eps) ─ P1_temporal (same 20 eps)
  ├─ M1_prompt_v2 → M1 symlinks (3 6 12 14 16 21)
  ├─ firstframe_align --wide: ffw_baseline (baseline bundles, ep25 = E2), ffw_M1 ── [ref_human_f0]
  │    └─ firstframe_pnp: pnp_baseline + pnp_M1 → pnp_merged (= pnp_forE1) ─────── [ref_human_f0]
  └─ FF2_feet_ref (src: M1 > P1 > E2 > baseline; align pnp_merged) ─────────────── [ref_human_f0]
       └─ JC2s1_shared_f820_pp (align pnp_forE1, K prior 820/758.1/560.5) ──────── [ref_human_f0]
            └─ DB_JC2s1_shared_f820_pp (w 0.5) → build_release.py → tier3_s1_20261020
```
`ref_human_f0` is used only at frame 0, as organiser-approved for Tier 3. `ref_poses` is read only by the scoring and
check code.

## Open points
1. **Episode 0** used an older `run_episode.sh` on CIAI. The exact CIAI container and module copy are not checked here.
2. **Retry branch:** to reproduce episodes 2, 5, 7, 11, 13, 19, 23 and 28 exactly, use the current `run_episode.sh`.
   Their contact refinement ran windowed, with batch size 384.
3. **Episode 25's E2 inputs:** the stage inputs E2 used for episode 25 (no baseline PASS) are not traced.
4. **CARI4D determinism:** CARI4D, FoundationPose and the GroundingDINO prompts run on GPU and are not guaranteed to be
   bit-reproducible. The steps from step 3 onwards are deterministic given the same bundles. The SOMA decoding in
   `firstframe_align` uses the GPU, so expect float-level differences.

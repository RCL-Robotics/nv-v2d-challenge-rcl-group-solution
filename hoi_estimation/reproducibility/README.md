# Reproducibility addendum (Tier 3, rule 2.9)

This folder answers the four open items of the Tier 3 record review (`dev.haoran.t3final_20261102`, `a7e7d3a`,
`han_zhou_reconstruction/README.md` → "Still missing"). Everything below was checked on 2026-11-02 against the files
that were actually consumed; how each claim was checked is stated next to it.

| file | content |
|---|---|
| `s1_runbook.md` | Step-by-step commands that produced `tier3_s1_20261020` (CARI4D → E2 → P1 → M1 → first-frame ICP/PnP → FF2 → JC2s1 → DB → release). |
| `tier3_runbook.md` | Agent inputs (`export_inputs.py`), the opus_v2 agent batches, the frame-0 correction, the support fit (per-episode settings) and the two later releases. |
| `environment/` | Build definitions embedded in the two images, `pip freeze` of both images, sha256 of the weights. |
| `../hoi_extraction/scripts/export_reference.py` | Item 1. |
| `../support/fit_20261023T2028Z.py` | The `fit.py` version that produced the released supports of 13 of the 14 episodes (see item 2). |

Sources for the commands: the Claude Code session transcripts that ran them (timestamps UTC) and the NAS file mtimes
(local UTC+4; the NAS clock is ~5 min behind). GT-scoring calls that were chained onto pipeline commands are omitted;
they only wrote CSVs that no step reads.

## 1. Export of `ref_human_f0/<ep>.npz` and `ref_poses/<ep>.npz`

`hoi_extraction/scripts/export_reference.py` (host python with pandas + pyarrow; the CARI4D image has no parquet
engine). It is the two original inline commands (2026-10-06T08:19Z for `ref_poses`, 2026-10-15T10:32Z for
`ref_human_f0`) turned into a script.

- **Check:** re-exported into a scratch directory and compared with the NAS files: **60/60 files byte-identical**
  (pandas 3.0.2, pyarrow 23.0.1, numpy 2.4.4).
- **What the pipeline reads:** `ref_human_f0` (frame 0 only) in `firstframe_align.py`, `firstframe_pnp.py`,
  `postproc_feet_floor.py --ref-foot-height`, `postproc_joint_camera*.py` and `agent/export_inputs.py`; from
  `ref_poses` only **frame 0** in `tier2_refine/t3_frame0.py`.
- `ref_poses` holds all frames because the GT diagnostics (`eval_*.py`, `score_runs.py`, the diagnostic half of
  `t3_frame0.py`, `support/score_snap.py`, `support/contact.py --gt`) also read it. No pipeline output depends on
  frames > 0.

## 2. Exact commands and parameters

See `s1_runbook.md` and `tier3_runbook.md`. The points the review asked about:

| step | command / parameters | note |
|---|---|---|
| E2_reregister | `V2D_FP_REREGISTER=1 exp_fp_mode.sh E2_reregister "--register-first-then-track" <eps>`; 20 episodes (1 2 3 4 6 7 8 9 11 12 13 14 15 16 17 18 19 25 27 28) in three batches, ep 3 rerun 10-08 | later edits to `exp_fp_mode.sh` only touch temp paths |
| P1_temporal | `postproc_temporal.py <same 20 eps> --out-exp P1_temporal` (defaults: `--src-exp E2_reregister`, static hold on) | deterministic, no reference data |
| M1 | `gen_prompt_v2.py` prompts + `experiments/M1_prompt_v2/run.sh` (CARI4D with `V2D_FP_REREGISTER=1`) for 3 6 12 14 16 21 28; **hand-made symlinks** `experiments/M1/<ep>/coconet.pth` for 3 6 12 14 16 21 only | ep 28 finished after the linking and is not used |
| ground_scale | `ground_scale.py $(seq 1 29)`, then `0` | **not consumed** by the chain |
| first-frame ICP | `firstframe_align.py --wide` → `logs/ffw_baseline.json` (baseline bundles, ep 25 = E2 bundle) and `logs/ffw_M1.json` | the `ff_align_*.json` files are earlier experiments, **not consumed** |
| PnP | `firstframe_pnp.py --icp ffw_*.json` → `pnp_baseline.json`, `pnp_M1.json` → merged (inline python) → `pnp_merged.json` = `pnp_forE1.json` (same content, checked) | |
| FF2_feet_ref | `postproc_feet_floor.py $(seq 0 29) --use-sim --ref-foot-height --align default=logs/pnp_merged.json --out-exp FF2_feet_ref` | source per episode: M1 (3 6 12 14 16 21), P1 (1 2 4 7 8 9 11 13 15 17 18 19 25 27 28), baseline (0 5 10 20 22 23 24 26 29) |
| JC2s1_shared_f820_pp | `postproc_joint_camera_K.py $(seq 0 29) --out-exp JC2s1_shared_f820_pp --sig-logf 1e-4 --sig-pp 0.01 --K-prior 820 758.1 560.5` (defaults: `--src-map FF2_feet_ref/feet_floor.json --align logs/pnp_forE1.json --sig-px 2 --sig-h 0.02 --window 15 --clip 0.3`) | |
| DB_JC2s1_shared_f820_pp | `postproc_depth_blend.py $(seq 0 29) --src JC2s1_shared_f820_pp --out-exp DB_JC2s1_shared_f820_pp` (`--w 0.5 --max-dev 0.1`; no episode skipped) | |
| s1 release | `build_release.py $N/release/tier3_s1_20261020` (defaults `--variant DB_JC2s1_shared_f820_pp --camera JC2s1_shared_f820_pp`) | |
| export_inputs | `export_inputs.py <n> --ws $N/agent_runs/T1_ds_ep<eee> --variant DB_JC2s1_shared_f820_pp --align $N/logs/JC2s1_shared_f820_pp_align.json --camera JC2s1_shared_f820_pp` | the JC2s1 camera came from **these CLI flags**; input sha256 of all 30 release run manifests match these exports |
| agent batches | `PFX=CL2 / CL2b / CL2c TASK=AGENT_TASK_temporal_v2.md bash run_batch_CL.sh 3 <eps>` (90 min, effort high, `net=open`, `claude-opus-5-5`) | |
| frame-0 | `t3_frame0.py opus experiments/t3_frame0/opus_f3_c2_shift 3 2 shift` | re-run on 2026-11-02 from the git copy: **bit-identical** for all 30 episodes |
| support fit | all 14 episodes: `GRID_M` 0.4 (default), `FREE_H` / `FREE_S` **unset** | `FREE_H`/`FREE_S` are diagnostic switches that return before `support.json` is written; `GRID_M=1.0` was tried once on ep 1 and overwritten |

**Support-fit version caveat.** The released `support.json` of episodes 2 3 6 7 9 11 13 16 17 20 25 27 28 come from
the `fit.py` of 2026-10-23T20:28Z. Two later edits (21:00Z: keep the best iterate; 21:04Z: blurred-silhouette IoU)
were followed only by a refit of episode 1. To reproduce exactly:
- episodes 2–28: `support/fit_20261023T2028Z.py` (the current `fit.py` with both edits reverted; the reversal is
  exact, each reverted string occurs once);
- episode 1: the current `support/fit.py`.

Verification of the legacy file: re-run on 2026-11-02 (GPU 0, CARI4D image, into a scratch copy of the episode inputs) for episodes 2 and 28: `T_world_mesh` **bit-identical** to the released `support.json` (max |ΔT| = 0). The other 11 episodes were not re-run.

## 3. Revision of `reconstruction/modules` in the CARI4D image

**`33129dd` (Release v0.4.0, 2026-09-23)** of this repository.
- **Check:** `git hash-object` of every file under the 8 module directories that `v2d_cari4d.def` copies into the
  image (`/workspace/v2d_*`), compared with `git ls-tree` of candidate commits:

  | commit | identical | differ | not in tree |
  |---|---|---|---|
  | `1509e6e` (initial release) | 193 | 30 | 190 |
  | `cbee067` (0.2.0) | 200 | 27 | 186 |
  | `dd395a7` (0.3.0) | 368 | 3 | 42 |
  | **`33129dd` (0.4.0)** | **371** | **0** | 42 |

  The 42 extra files are all FoundationPose build artefacts (`mycpp/build/`, `bundlesdf/mycuda/build/`).
  `reconstruction/modules` has no change between `33129dd` and `a7e7d3a`, so the current tree is the same code.
- **Run-time binds:** `run_episode.sh` / `exp_fp_mode.sh` bind from the NAS checkout only
  `v2d_cari4d/lib/cari4d/lib_mhr/assets` and `v2d_sam3d_body/assets` (5 files, identical to `33129dd`) and the patch
  `hoi_extraction/patches/run_foundationpose_mhr_export.py` (identical to the NAS copy that was bound).
- **Build:** both images were built on 2026-10-04 (`v2d_cari4d` 20:45 +04, `v2d_gdino` 21:01 +04), singularity 4.1.1.
  The definitions embedded in the images are in `environment/*.sif.def`. `v2d_gdino` equals
  `hoi_extraction/singularity/v2d_gdino.def`; `v2d_cari4d` differs from the repo file in one line (the build-time
  kaolin import check is non-fatal: `|| echo WARN ...`).
- **Unpinned parts, as built:** nvdiffrast `253ac4fcea7de5f396371124af597e6cc957bfae` (0.4.0); the full package list
  is in `environment/pip_freeze_*.txt`.

## 4. Weights and images

Location on the NAS (lab-internal, `/mnt/nas/Group/hanzhou/v2d/`):

| item | path | size | sha256 |
|---|---|---|---|
| CARI4D image | `images/v2d_cari4d.sif` | 16.9 GB | `7c495bc5140189312ae64dc8110fe3f9cab70a90eca800d128f5f5e50974ebf5` |
| GroundingDINO image | `images/v2d_gdino.sif` | 7.7 GB | `0b21b6f64d975c0b5deb9f021a6ff42d18626fd92ba1cce141a9fe9e26239c8c` |
| weights | `weights/{cari4d, foundationpose, grounding_dino, hf_home, sam2, sam3d_body}` | 9.0 GB | per file in `environment/weights.sha256` (493 files) |

- **Weights source:** fetched on 2026-10-04 from Hugging Face (`weights_download.log` on the NAS records the
  destinations of `cari4d`, `sam3d_body/checkpoints/sam-3d-body-dinov3`, MoGe-2 `Ruicheng/moge-2-vitl-normal`
  snapshot `b135031b…` under `hf_home/`, and `sam2`); `grounding_dino` and `foundationpose` were staged as the
  repo's downloaders lay them out. The exact download commands were not recorded. To fetch them anew use the repo's
  downloaders, `python -m v2d.<module>.docker.run_download_weights --output_dir <weights>/<module>`; CARI4D and
  SAM-3D-Body are gated (`HF_TOKEN`). `environment/weights.sha256` pins the content either way. CARI4D checkpoint:
  run `2026-08-25-09-35-57`, `step200000.pth`, sha256 `78ff5cb8…5ba3` (as in its `manifest.json`).
- **Rebuilding the images instead:** `apptainer build v2d_cari4d.sif environment/v2d_cari4d.sif.def` from
  `reconstruction/` at `33129dd` (the def copies `modules/...` relative to it). The pip versions resolved at build
  time may differ; compare with `environment/pip_freeze_*.txt`.
- **Intermediate data** (CARI4D outputs, experiment bundles, alignment JSONs, agent inputs/outputs) are also on the NAS:
  `outputs/cari4d/`, `experiments/<variant>/`, `logs/*.json`, `agent_runs/<run>/`; paths per step in the runbooks.
  The images and weights can be copied to the training server on request.

## What is still not bit-reproducible
- GPU stages (CARI4D, FoundationPose, GroundingDINO prompts, SAM2, the support silhouette fit) are not guaranteed to
  be bit-identical on re-run. Every step from P1 on is deterministic given the same bundles, and the frame-0 step was
  re-run bit-identically.
- Episode 0's baseline ran on the CIAI cluster with an earlier `run_episode.sh`; episodes 2 5 7 11 13 19 23 28 took
  the `--postopt_batch_size 384` retry branch (details in `s1_runbook.md`).
- Which stage inputs E2 used for episode 25 (no baseline output) is not traced.

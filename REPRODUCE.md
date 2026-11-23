# Reproducing our V2D Challenge Track 2 submission files

This commit holds the code behind our three Track-2 submission files (Tier 1, 2 and 3; Unitree G1 + Dex3, 30 episodes
each). The steps below re-pack each submitted file from the stored rollouts in our verification package, re-run the
official evaluation of the submitted policies and, optionally, retrain them.

## Requirements

- Linux x86_64 with an NVIDIA GPU and a driver for CUDA 13 (version 580 or newer). No CUDA toolkit is needed.
- conda, git and git-lfs; internet access to GitHub, conda-forge, PyPI and Hugging Face; about 25 GB of free disk.
- The official submission kit (`KIT_MANIFEST.json` sha256 `398947a7783490245a47d70ec68d6894ecef7636e712532710ec448dc0ca1c3e`).
- The public challenge data: Hugging Face dataset `nvidia/video_to_data_challenge`, revision `5f68335f`, `track_2/`.
- Our verification package, provided with our submission:

| File | Size (bytes) | sha256 |
|---|---|---|
| `v2d_track2_verify_common.zip` | 5,949,336 | `b3eb5a73aeb448dc67494c16fa1937fb05c2f27c1f406431d90f688750fcee7f` |
| `v2d_track2_verify_tier1.zip` | 1,801,841,768 | `42bffb1d215358d4c2a5fc4f796b3ba64da1f2906c61f9b5dd099aed30e66879` |
| `v2d_track2_verify_tier2.zip` | 1,817,802,825 | `93a0b71681d12c4eb07c44341b62144506be13f274a8d7af038bc2fa53c421e2` |
| `v2d_track2_verify_tier3.zip` | 1,805,608,051 | `18ffc39a216eaa88341b424a411adf34f1d6fc5b90534836cf9bd1174df460ac` |

## 1. Install

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/RCL-Robotics/nv-v2d-challenge-rcl-group-solution.git v2d && cd v2d
GIT_LFS_SKIP_SMUDGE=1 git checkout <commit>        # the commit in the code_commit_url column of the submitted files
export V2D_WORK=/data/v2d/work                      # everything the scripts write (no spaces in the path)
export V2D_KIT=/data/v2d/v2d_submission_kit         # the official submission kit
export V2D_DATA=/data/v2d/video_to_data_challenge   # the challenge data (the folder that contains track_2/)
export V2D_RUNTIME=$V2D_WORK/video_to_data_8c778f9/robotic_grounding/flash_chord   # official runtime, cloned below
bash scripts/final/setup_env.sh --steps conda,stock,flashchord,kit   # envs flashchord and v2dkit in $V2D_WORK/envs
bash scripts/final/get_data.sh                      # skip if $V2D_DATA already holds the data; needs the hf CLI
# compare both envs with their recorded package lists; each check ends with "OK" when they match
"$V2D_WORK/envs/flashchord/bin/python" -I -B scripts/final/lib/env_check.py environment/flashchord.pip-freeze.txt
"$V2D_WORK/envs/v2dkit/bin/python" -I -B scripts/final/lib/env_check.py environment/v2dkit.pip-freeze.txt
```

## 2. Prepare the verification package

Unzip the four zips into one folder, check every file and create the site file `$PKG/site.env`. In it, set the lines
marked `EDIT`: `PKG` (this folder); `V2D_WORK`, `V2D_KIT`, `V2D_DATA`, `V2D_RUNTIME` as in section 1;
`FC_PYTHON=${V2D_WORK}/envs/flashchord/bin/python`; `KIT_PYTHON=${V2D_WORK}/envs/v2dkit/bin/python`; `GPUS`, the GPU
to use (`REQUIRE_IDLE_GPU=1` refuses a busy GPU). Keep the other lines (`ROOT_ME`, `ROOT_ZX`, `ROOT_HAN`, `T1_*`,
`FLASH_CHORD_CACHE_DIR`): they point the scripts at the input folders and the hull cache inside the package. The two
links are for section 4: each policy file stores the absolute path of its input, and the evaluator opens that path.
They need root (in a container, bind-mount the two folders at these paths instead).

```bash
for z in /path/to/v2d_track2_verify_*.zip; do unzip -q "$z" -d /data/v2d; done   # all four into one folder
PKG=/data/v2d/v2d_track2_verification_20261104
(cd "$PKG" && sha256sum -c --quiet MANIFEST.sha256)      # no output and exit status 0: every file is intact
cp "$PKG/site.env.template" "$PKG/site.env"              # then set its EDIT lines as described above
set -a; . "$PKG/site.env"; set +a
"$FC_PYTHON" -I "$PKG/fill_public_files.py" --dataset-root "$V2D_DATA"   # 76 byte copies of public files; "0 problems"
mkdir -p /root/autodl-tmp
ln -s "$PKG/ME" /root/autodl-tmp/haoran.zhang
ln -s "$PKG/ZX" /root/autodl-tmp/zuxing.lu
```

## 3. Reproduce the submitted files

```bash
bash "$PKG/repack.sh" 1 /path/to/<submitted Tier-1 CSV>
bash "$PKG/repack.sh" 2 /path/to/<submitted Tier-2 CSV>
bash "$PKG/repack.sh" 3 /path/to/<submitted Tier-3 CSV>
```

`repack.sh` runs the kit packer on that tier's stored rollouts with the URL in the file's `code_commit_url` column
(this commit: `https://github.com/Beat-in-our-hearts/nv-v2d-challenge-codebase/commit/<commit>`), writes
`repack_out/tierN_repacked.csv` in the current folder and prints `tier N: IDENTICAL to <file>` when the two files are
byte-identical (as `cmp` or `sha256sum` show). Another commit URL would change only the `code_commit_url` column.
The submitted files cite our original development repository (`nv-v2d-challenge-codebase`); the same commits, with
the same hashes, are in this repository's history, so pass the URL exactly as it appears in the submitted file.

## 4. Re-run the evaluation

```bash
bash scripts/final/eval_tier1.sh --site "$PKG/site.env" --episode 28   # --episode all (the default): all 30 episodes
bash scripts/final/eval_tier2.sh --site "$PKG/site.env" --episode 17
bash scripts/final/eval_tier3.sh --site "$PKG/site.env" --episode 7
# with all 30 episodes of a tier evaluated and site.env loaded (section 2), pack them into a CSV as repack.sh does:
URL=https://github.com/Beat-in-our-hearts/nv-v2d-challenge-codebase/commit/<commit>
pack() { "$KIT_PYTHON" -I -B "$V2D_KIT/tools/pack_submission.py" --track "track_2_tier${1}" \
  --results "$V2D_WORK/eval/tier${1}/t01/results" ${2:+--frame-references "$2"} --dataset-root "$V2D_DATA" \
  --code-commit "$URL" --eval-script "$V2D_KIT/flash_chord/scripts/evaluate_policy.py" -o "tier${1}_rerun.csv"; }
pack 1
pack 2 "$PKG/ME/fc_runs/submissions/t2r9_pack_20261104/pack_r9_placeholder/r9_closest/frame_refs"
pack 3 "$PKG/ME/fc_runs/submissions/t3_f0_final/pack/FINAL_r/frame_refs"
```

Each `eval_tierN.sh` checks the sha256 of the policy file (the submitted `checkpoint_sha256`) and of the kit
evaluator, runs the evaluator on the official runtime and writes the rollouts to `$V2D_WORK/eval/tierN/t01/results/`
and per-episode AUC, SP-SR, MP-SR and MPPE (against the policy's input trajectory) to `$V2D_WORK/eval/tierN/summary.tsv`.

## 5. Retrain (optional)

```bash
bash scripts/final/setup_env.sh --site "$PKG/site.env" --steps lfs    # the model assets of this checkout
bash scripts/final/train_tier1.sh --site "$PKG/site.env" --episode 8   # all stages of episode 8 -> $V2D_WORK/runs/tier1
bash scripts/final/train_tier2.sh --site "$PKG/site.env" --episode 17
bash scripts/final/train_tier3.sh --site "$PKG/site.env" --episode 7
bash scripts/final/build_inputs_tier2.sh --site "$PKG/site.env" --dry-run   # Tier-2 inputs: filters on the noisy data
bash scripts/final/build_inputs_tier3.sh --site "$PKG/site.env" --dry-run   # Tier-3 inputs: pipeline in hoi_estimation/
```

`train_tierN.sh` uses each stage's recorded code commit, settings and seed. A real builder run takes `--steps` (see
`--help`) and writes into `ROOT_ME` and `ROOT_ZX`, so point those at new folders first.

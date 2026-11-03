# verification/han_followup: checks after Han Zhou's later commits (6c10a22 … c0dbf5b)

These checks were run for the record update that follows Han's `c0dbf5b`. They are CPU only. They read no clean or
GT trajectory beyond frame 0 of the reference object poses, and they compute no score.

## Run on this commit (2026-11-04)

| file | what it shows |
|---|---|
| `check_set.log`, `check_set_report.json` | `tools/check_set_t3.py --files --hash` on this commit's tree (written out from git objects, before the commit): **ALL OK**. The README and restore-tool edits change none of the files it reads. |
| `code_identity.txt` | `tools/code_identity_t3.sh` on the staged tree (this commit without this folder and `SHA256SUMS`). <br>• Item 1: `src` equals `41403d3`'s (tree `b8000f82`). <br>• Item 2: outside the bundle and `TIER3_SUBMISSION.md`, only `hoi_estimation/` changed. <br>• Items 3–6 are unchanged from the merge. |
| `frame0_rerun_c0dbf5b.json`, `.log`; `run_refine_c0dbf5b.py`, `prep_inputs.py` | Han's frame-0 step (`t3_frame0.py opus <out> 3 2 shift`), re-run from `c0dbf5b`'s code, reproduces `tier3_frame0_supports_20261024` byte for byte: 30/30 episodes, 22,990 frames, max abs ΔT = 0. <br>• `prep_inputs.py` (from the 6c10a22 check) builds the inputs. They are the opus_v2 tracks, and a `ref_poses` that holds only the first parquet row, asserted `frame_index == 0`. <br>• The run cuts `t3_frame0.py` before its GT diagnostic loop. |
| `han_edits_after_48c217a.tsv`; `classify_han_edits.py` | Every `hoi_estimation/` file that Han modified between `48c217a` and `c0dbf5b` (111 files), and the class of each changed line. <br>• 103 change paths only; 2 are docs. <br>• `t3_frame0.py` and `support/common.py` add the opt-in `V2D_EPISODES` subset. <br>• `build_release_combined.py` makes `diag_scores.json` optional; it is only printed. <br>• `overlay.py` and `sheet.py` change overview images only. <br>• `diagnostics/rot_split.py` has an f-string tidy. <br>• `diagnostics/meshdim.py` (a GT diagnostic) gained a typo, `fos.environ`. |
| `restore_test.log` | The fixed `tools/restore_layout_t3.py`, run from this commit's tree with `--me <scratch>/t_me --han <scratch>/t_han --rewrite`. <br>• rc 0; 727 files written. <br>• Nothing is written under `t_me/data/`; the 9 data-root records are in `t_me/t3ready_expected/`. <br>• 148 of Han's document copies are under `t_han`. <br>• No training-server root is left in the rewritten files. |
| `hygiene_scan.json` | The files this commit changes or adds: no forbidden file name, no Git-LFS or binary type, and no credential pattern. |

## From the check of `6c10a22` (2026-11-02), copied unchanged into `from_6c10a22/`

| file | what it shows |
|---|---|
| `check_set_6c10a22.log`, `check_set_report_6c10a22.json` | `check_set_t3.py --files --hash` on `6c10a22`: ALL OK |
| `export_reference_sha256.txt`, `parquet_sha_vs_hf.txt`, `f0check.log`, `zipmeta_arrays.log` | `hoi_extraction/scripts/export_reference.py` on our Tier-1 parquet. <br>• The parquet files equal HF revision `5f68335f` (30/30 LFS sha256). <br>• It writes 60 files. Three runs, one in the kit's environment, gave the same sha256 list. <br>• Frame 0 of the export equals the frame-0 input of the frame-0 step (30/30). |
| `agent_inputs_rebuild.json` + `.py` | `video.mp4`, `object_mesh.glb`, `ground_plane.json`, `task.txt` and `episode.json` of the 30 release agent runs were rebuilt from the dataset. Each equals its run manifest's sha256 (30/30 each). |
| `camera_json_from_events.json` + `.py` | `camera.json` of the 30 release runs, recovered from their event streams, matches the run manifests (30/30). <br>• `T_world_cam` equals the s1 camera exactly. <br>• `K` is within 3.1e-5 px of the s1 camera. |
| `kprior_origin.txt` + `.py` | The per-episode MoGe-2 focal lengths have median 820.03, and their principal point is the image centre (768, 576). The agents' K is f ≈ 820, (758.1, 560.5). |
| `human_camera_unchanged.txt` | The human, camera and ground plane are byte-identical from s1 through opus_v2 to the frame-0 release: 180/180 files. |
| `agent_records_secret_scan.txt` | No credential pattern in `agent_records/` (events, transcripts, manifests). |
| `fit_version_hunks.txt` | The legacy support fit `support/fit_20261023T2028Z.py` differs from the current `fit.py` in 10 hunks. |

The scratch paths in the logs are on the training server, under `<ME>/tmp_analysis/`.

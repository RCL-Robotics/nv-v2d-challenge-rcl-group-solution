# Track 2 Tier 3 submission (FINAL_r)

The record of our Track 2 Tier 3 submission is in
[`robotic_grounding/flash_chord/scripts/track2/t3ready_20261102/`](robotic_grounding/flash_chord/scripts/track2/t3ready_20261102/README.md).
It covers:
- the file and its hashes;
- how each of the 30 episodes was produced;
- the data preparation from Han Zhou's video reconstruction;
- where each part of Han's reconstruction code lives, and what is still missing;
- the training, evaluation, selection and pack records;
- the verification;
- the disclosures and the grey areas for the organisers.

**Han Zhou's reconstruction code is in [`hoi_estimation/`](hoi_estimation/README.md).** It comes from his branch
`dev.hanzhou.hoi_estimation`:
- the code as it ran is `48c217a`, merged unchanged in `ab0d24c`;
- his later commits `6c10a22` … `c0dbf5b` (12 commits, all under `hoi_estimation/`) add the reproducibility material and
  the records of the 30 release agent runs. They also make the paths of 111 scripts portable. `c0dbf5b` is the parent of
  the record update that follows it.

It holds:
- the base pipeline (CARI4D on the exo video, camera and floor, post-processing, the s1 release builder);
- the agent harness and the records of every agent run;
- the frame-0 refinement;
- the support reconstruction;
- the export of the frame-0 reference data (`hoi_extraction/scripts/export_reference.py`);
- **how to reproduce it:** `hoi_estimation/reproducibility/`:
  - `README.md`: data, images and weights, and how to run from another root;
  - `s1_runbook.md` and `tier3_runbook.md`: the exact commands of every step;
  - `run_all_tier3.sh`: the end-to-end driver;
  - `environment/`: the image build definitions, pip freezes and weight hashes.

  Han's reproduction report and agent workflow notes are listed in `hoi_estimation/README.md`.
- GT-only diagnostics.

The record was first committed as `232e6b2`, which added only the record folder and this file. What is still open on
the reconstruction side is listed in the record README, section 6.

The FlashCHORD trainer that trained every Tier-3 policy is `robotic_grounding/flash_chord/src` of `41403d3`
(tree `b8000f82`). In a commit that also merges the other tiers' records, `src` can hold another tier's version.
To get the exact Tier-3 trainer, check out `41403d3`, which is an ancestor: `git worktree add <dir> 41403d3`.
`f2923d6` has the same `src`.

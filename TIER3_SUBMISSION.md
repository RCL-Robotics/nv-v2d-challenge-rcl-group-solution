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
`dev.hanzhou.hoi_estimation` (`48c217a`), merged unchanged. It holds:
- the base pipeline (CARI4D on the exo video, camera and floor, post-processing, the s1 release builder);
- the agent harness and the records of every agent run;
- the frame-0 refinement;
- the support reconstruction;
- GT-only diagnostics.

The record was first committed as `232e6b2`, which added only that folder and this file. The merge adds only
`hoi_estimation/`. The FlashCHORD trainer (`robotic_grounding/flash_chord/src`) is unchanged from `41403d3`.

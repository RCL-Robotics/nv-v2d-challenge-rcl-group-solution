# scripts/final: run every step of the three counted Track-2 files

The runbook is `REPRODUCE.md` at the repository root. These scripts turn its steps into commands. Each one:
- takes `--dry-run` (print the plan, i.e. the exact commands, and execute nothing), `--site FILE` and `--help`;
- builds its plan from the committed records only (the three set files, the record READMEs' commands);
- in a real run writes the plan to `$V2D_WORK/logs/<script>_<time>.sh` and runs it with `bash -euo pipefail`;
- writes only under `$V2D_WORK` (plus, for the builders and `pack_tier3.sh`, the record layout they are documented to
  write into).

| script | what it runs |
|---|---|
| `setup_env.sh` | conda envs `flashchord` / `v2d` at the recorded packages, the stock runtime 8c778f9, the hash-locked flashchord install, the kit venv, the retarget env, Git LFS; `--steps check` compares installed envs with `environment/` |
| `get_data.sh` | the challenge data at HF revision 5f68335f (`--verify-only` counts the files) |
| `build_inputs_tier2.sh` | raw / K3 / K3fix / G2 input roots (`--steps verify`: the 30 inputs of the counted file) |
| `build_inputs_tier3.sh` | Han Zhou's reconstruction, layout restore, f0 data prep steps 0 and A-F (`--steps verify`) |
| `train_tier{1,2,3}.sh --episode N` | every training stage of the episode, in a worktree of its recorded training commit, with the recorded overrides; resume pairs copied and sha256-checked; `--iterations K` for a smoke test |
| `eval_tier{1,2,3}.sh` | `--trials K` official kit evaluations of the pinned evaluated files (or `--reexport`), own input, and a summary |
| `select_tier2.sh`, `select_tier3.sh` | verify the selection locks and print the decision; the drivers are pointers |
| `pack_tier{1,2,3}.sh --commit-url URL` | re-pack the counted trial with the Oct-11 kit packer for a commit URL (only `code_commit_url` changes) |
| `selftest.sh` | syntax check and `--dry-run` of all of the above |

`lib/`: `common.sh` (options, defaults, plan runner), `helpers.sh` (functions the plans call: worktrees, LFS fill,
hull-cache seeding, resume pairs, idle-GPU check), `plan.py` (train / eval plans from the set records), `summarize.py`,
`lfs_fill.py`, `env_check.py`, `verify_inputs.py`, `show_selection.py`, `make_final_episodes.py` (wrote
`FINAL_EPISODES.tsv`), `kitpy_withhold.sh` (keeps the kit packer's clean-reference line out of Tier-2 logs).

Shared machines: `REQUIRE_IDLE_GPU=1` makes every GPU launch refuse a GPU on which `nvidia-smi` lists any compute
process; nothing here signals a process.

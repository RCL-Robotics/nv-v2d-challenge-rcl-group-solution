# scripts/final

The scripts behind our three submitted Track-2 files. [`REPRODUCE.md`](../../REPRODUCE.md) shows how to use them with
our verification package. Every script takes `--site FILE` (this machine's paths: the package's `site.env`, or a copy
of `site.env.example`), `--dry-run` (print the commands, run nothing) and `--help`. A real run writes its commands to
`$V2D_WORK/logs/` and runs them; its outputs go under `$V2D_WORK`.

| script | what it does |
|---|---|
| `setup_env.sh` | installs the environments recorded in `environment/` and the official runtime (nvidia-isaac/video_to_data at 8c778f9); `--steps check` compares installed envs with the records |
| `get_data.sh` | downloads the challenge data (Hugging Face `nvidia/video_to_data_challenge`, revision 5f68335f) |
| `eval_tier{1,2,3}.sh` | evaluates the submitted policy files with the official kit evaluator; rollouts and `summary.tsv` go to `$V2D_WORK/eval/tierN` |
| `pack_tier{1,2,3}.sh --commit-url URL` | re-packs the stored rollouts with the kit packer and runs the kit uploader's offline check, in the training machine's folder layout; with the verification package, use its `repack.sh` |
| `train_tier{1,2,3}.sh --episode N` | retrains one episode, every stage with its recorded code commit, settings and seed |
| `build_inputs_tier2.sh` | builds the Tier-2 inputs from the released noisy data; `--steps verify` checks the 30 inputs used |
| `build_inputs_tier3.sh` | runs the Tier-3 reconstruction pipeline (`hoi_estimation/`) and builds the Tier-3 inputs; `--steps verify` checks the 30 inputs used |
| `select_tier2.sh`, `select_tier3.sh` | checks the hash locks of the selection records and prints the policy chosen per episode |
| `selftest.sh` | syntax check and `--dry-run` of every script |

`FINAL_EPISODES.tsv` lists the 90 evaluated policy files with their inputs and sha256 values. `lib/` holds the helpers
that the scripts call. With `REQUIRE_IDLE_GPU=1`, every GPU launch refuses a GPU that already runs a process.

# t2_consensus_r8_20261103: LAUNCH_R8 (record of the orchestrator launch; decides nothing)

**Launched** 2026-11-02 21:50:26 CST, after the lock of the revised build and before T_c = 2026-11-03 12:00:00 CST.
Nothing had been exported or evaluated.

| item | value |
|---|---|
| command (from any directory) | `bash /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/r8_launch.sh` |
| what the launcher runs | `setsid nohup conda_envs/flashchord/bin/python -I -B <R8>/r8_driver.py run >> <R8>/r8_driver_stdout.log 2>&1 < /dev/null &` (stdin only is read from the null device) |
| pid | 877263 |
| process group (pgid) | 877263 (its own) |
| session (sid) | 877263 (its own, no terminal; parent pid 1), so it survives the ssh session and the Mac sleeping |
| r8_driver.py sha256 | 1d626bed15b8e2feabd3f555165a76d179ee773cdd28fdf8284752c1a10408c9 |
| `PREREG_R8.md` | bb912c57180dde8255776884de169e12467349f57d584782308b88ff935abf8c |
| `PREREG_R8_PINS.md` | 321bdd7cf2b7596ccb0973e93101e456a1f613eaafd7beeb5e3d9dbc8d0d84f9 (65 pins, 3 commits) |
| `LOCK_R8.sha256` | b12f540e035cf0fd2b236bf63a78a5b1de834ef62ee8cf1f400855b241375c4c (14 lines, all OK at launch) |
| state at launch | heartbeat `waiting for T_c`, 14.16 h left |
| GPUs 6 / 7 at launch | 3 compute processes listed on each (Zuxing's trainings): round 8 will not use them until none is listed |

**What it does.**
1. Sleeps until T_c, writing `R8_DRIVER_HEARTBEAT.json` every 60 s.
2. At T_c: `r8_pool.py freeze` (`POOL_R8.sha256` last), then `r8_pool.py verify`.
3. Phase S: `r8_eval.py run --phase screen` (exports on CPU at once; evaluations only on GPU 6 / 7, plain, one per GPU,
   and only while `nvidia-smi` lists no other compute process there); then `r8_select.py check / gate / score / select
   --phase S` (`SELECTION_S8.sha256` last).
4. Phase F (finalists only), then `select --phase F` (`SELECTION_R8.sha256` last). Without a finalist, every episode
   keeps its round-7 pick.
5. Deadlines: phase S not done by 2026-11-04 08:00 CST -> every episode keeps its round-7 pick; phase F not done by
   2026-11-04 12:00 CST -> unfinished finalists keep the round-7 pick.

It never commits, pushes, packs, uploads or signals anything. Packing, a commit and any upload need the user's approval.

**Status.**
- `cat <R8>/R8_DRIVER_HEARTBEAT.json` or `conda_envs/flashchord/bin/python -I -B <R8>/r8_driver.py status`;
- `<R8>/r8_driver.log` (steps), `<R8>/eval_r8_heartbeat_{s8,f8}.json` and `<R8>/eval_r8_driver.log` (evaluations).

**Stop.** `touch <R8>/STOP`. No new evaluation starts; running evaluations finish and are recorded; the orchestrator
ends at the next step boundary and selects nothing. Never kill or signal the processes.

**Relaunch** (for example after a server restart): `bash <R8>/r8_launch.sh`. It reports a running orchestrator instead
of starting a second one, and refuses while an `r8_eval.py` evaluation driver is still alive (wait until that has
ended, then relaunch). Every step resumes from its outputs.

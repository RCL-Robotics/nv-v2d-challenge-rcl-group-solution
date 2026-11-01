#!/usr/bin/env bash
# status.sh: one-screen status of the Tier-3 f0 FINAL pipeline (read-only; own-input numbers only, no clean number).
R=/root/autodl-tmp/haoran.zhang/fc_runs/submissions/t3_f0_final; PYTHONDONTWRITEBYTECODE=1 /root/autodl-tmp/haoran.zhang/conda_envs/flashchord/bin/python -B $R/t3final_driver.py status "$@"; echo "--- driver.log (last 6)"; tail -n 6 $R/driver.log 2>/dev/null

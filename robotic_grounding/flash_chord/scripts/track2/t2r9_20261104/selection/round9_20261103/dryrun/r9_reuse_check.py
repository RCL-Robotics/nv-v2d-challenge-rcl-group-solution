#!/usr/bin/env python3
"""r9_reuse_check.py: plan-only check of the phase-F job builders of r9_eval.py for reuse8 / reuse7 finalists (build step
B3; PREREG_R9 3.4). For one reuse8 and one reuse7 row of a trial pool, the job is rebuilt by round 8's driver from round
8's / round 7's records exactly as phase F would, and its earlier export record is checked. Nothing is exported,
evaluated, locked or written (except the printed report).

  conda_envs/flashchord/bin/python -I -B dryrun/r9_reuse_check.py <r9_eval.py> <trial pool folder>
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

EV, POOLDIR = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
mod = types.ModuleType("r9_eval_check")
mod.__file__ = str(EV)
sys.modules[mod.__name__] = mod
exec(compile(EV.read_bytes(), str(EV), "exec"), mod.__dict__)  # noqa: S102
mod.configure(POOLDIR)
D = mod.load_driver9()
rows = json.loads((POOLDIR / "POOL_R9.json").read_text())["rows"]
bad = 0
for role, builder in (("reuse8", mod.build_reused8), ("reuse7", mod.build_reused7_9)):
    ps = [p for p in rows if p["role"] == role]
    if not ps:
        print(f"{role}: no row in the trial pool")
        continue
    for p in ps:
        fl = {"job": p["job"], "export_sha256": p.get(f"r{role[-1]}_export_sha256")}
        job = builder(D, p, fl)
        ok = job.get("status") == "ok"
        bad += not ok
        print(f"{role} {p['job']}: status {job.get('status')}; export record {job.get('export_record')} "
              f"export {str(job.get('export_sha256'))[:12]}; training {str(job.get('training_sha256'))[:12]}; "
              f"in INC_EXPORTS {p['job'] in mod.E.INC_EXPORTS}; runs {job.get('runs')}; stage {job.get('stage')}")
print(f"reuse builders: {'OK' if not bad else f'{bad} PROBLEMS'}")
sys.exit(0 if not bad else 1)

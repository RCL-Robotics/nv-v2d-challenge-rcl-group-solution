"""simtest.py: CPU-only test of the item / selection / summary / status / pack path of t3final_driver.py, on stand-in run
records. A copy of the driver whose ROOT is dryrun/simtest_<stamp>/ freezes the items that resolve now; every run k of
an item is a stand-in pointing at the post step's own results (run 0) with a SYNTHETIC own-input AUC offset, so the
selection rule has something to choose. These are NOT evaluations and are never read by the real run.
  <flashchord python> -I -B simtest.py"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path  # noqa: E402

REAL = Path("/root/autodl-tmp/haoran.zhang/fc_runs/submissions/t3_f0_final")
SIM = REAL / "dryrun" / f"simtest_{dt.datetime.now():%Y%m%d_%H%M%S}"
SIM.mkdir(parents=True)
text = (REAL / "t3final_driver.py").read_text()
old = 'ROOT = ME / "fc_runs/submissions/t3_f0_final"'
assert text.count(old) == 1
(SIM / "t3final_driver.py").write_text(text.replace(old, f'ROOT = Path("{SIM}")'))
shutil.copy2(REAL / "t3final_kit.py", SIM / "t3final_kit.py")
spec = importlib.util.spec_from_file_location("simdriver", SIM / "t3final_driver.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)
log = D.Logger(SIM / "simtest.log")
states = D.resolve_all(log)
OFFSETS = (0.0, 0.004, -0.003, 0.001, -0.006)  # synthetic: mean -0.0008 -> run 1; ep03 without run 2: mean -0.002 -> run 3
for e in D.EPISODES:
    it = D.load_item(e)
    if it is None or it["state"] not in ("f0", "opus_v2"):
        continue
    for k in range(1, 6):
        base = SIM / "runs" / f"ep{e:02d}" / f"r{k}"
        if e == 3 and k == 2:  # one failed run (after 3 tries) to exercise the n_ok < 5 path
            rec = {"episode": e, "k": k, "status": "failed", "problems": ["SIMTEST failure"], "try": 3}
        else:
            m = dict(it["post_own_input"])
            m["auc"] = m["auc"] + OFFSETS[k - 1]
            rec = {"episode": e, "k": k, "kind": it["kind"], "object": it["object"], "try": 1, "status": "ok", "problems": [],
                   "remark": "SIMTEST stand-in = the post step's run 0 rollout; synthetic AUC offset", "gpu": -1,
                   "metrics": m, "results_json": it["post_results_json"], "json_sha256": it["post_results_json_sha256"],
                   "results_parquet": it["post_results_parquet"], "parquet_sha256": it["post_results_parquet_sha256"],
                   "pinned_sha256": it["eval_sha256"], "finished_at": D.now(), "seconds": 0}  # fmt: skip
        D.write_atomic(base / "run.json", D.dump(rec))
a = argparse.Namespace(runs=5, gpus="0,1,2,3,5,6,7", max_parallel=3, lane_max=2, cooldown=35.0, poll=5.0, resolve_every=60.0,
                       no_pack=True)
drv = D.Driver(a, log)
drv.refresh_items()
drv.freeze_selections()
drv.states = states
drv.heartbeat("simtest")
D.write_summary(log)
bad = []
for e, it in sorted(drv.items.items()):
    s = D.sel_rec(e)
    want = 3 if e == 3 else 1
    got = s["chosen_run"] if s else None
    log(f"ep{e:02d}: chosen run {got} (want {want}), n_ok {s and s['n_ok']}, distances {s and s['distances']}")
    if got != want:
        bad.append(e)
print("--- status")
D.cmd_status(argparse.Namespace())
rc = D.cmd_pack(argparse.Namespace(partial=True, files="FINAL,T1,T2"))
print("pack --partial rc", rc)
rc2 = D.cmd_pack(argparse.Namespace(partial=False, files="FINAL"))
print("pack (full) rc (expected 1: not all 30 episodes settled)", rc2)
print("--- selection.csv")
print((SIM / "selection.csv").read_text())
print("selection mismatches:", bad)
print("->", SIM)

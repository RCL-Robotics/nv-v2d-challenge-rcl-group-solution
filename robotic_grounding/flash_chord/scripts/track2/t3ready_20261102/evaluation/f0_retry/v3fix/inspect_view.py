import importlib.util, json, sys
from pathlib import Path
ME = Path("/root/autodl-tmp/haoran.zhang")
spec = importlib.util.spec_from_file_location("t3f0h_ro", ME / "scripts/t3f0_helper.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print("helper ALLOWED", m.ALLOWED_GPUS, "TIERS", m.PLACE_TIERS, "CAP", m.MAX_TRAIN_PER_GPU, "JOB_MB", m.JOB_MB, "MIN_FREE_AFTER_MB", m.MIN_FREE_AFTER_MB, "CORES", m.CORES)
v = m.observe()
for g, x in sorted(v.items()):
    print(g, "trainings", x["trainings"], "evals", x["evals"], "others", x["others"], "free", x["free"], "mps", x["mps"], "debt", x["debt"])
for g in range(8):
    print("cores node", 0 if g <= 2 else 1, "for gpu", g, len(m.own_node_cores(g, "")))

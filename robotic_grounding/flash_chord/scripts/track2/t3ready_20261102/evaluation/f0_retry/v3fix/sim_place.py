"""sim_place.py: simulate the v3 driver's 14 sequential starts with the unchanged t3f0 fill rule (plan(..., assume=...)),
using t3retry_v3's module (allowlist applied). Read-only; starts nothing."""
import importlib.util, json, sys
from pathlib import Path
RA = Path("/root/autodl-tmp/haoran.zhang/tmp_analysis/tier3/f0_retry")
spec = importlib.util.spec_from_file_location("t3r3", RA / "t3retry_v3.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
T = m.T()
print("t3retry_v3 ALLOWED", m.ALLOWED_GPUS, "| helper module ALLOWED", T.ALLOWED_GPUS, "| cap", T.MAX_TRAIN_PER_GPU, "| tiers", T.PLACE_TIERS)
view = T.observe()
base = {g: len(v["trainings"]) for g, v in view.items()}
print("trainings now (anybody):", base)
cands, _ = m.candidates()
assume, seq = {}, []
for ep in cands:
    line = T.plan(ep, "", assume=dict(assume))
    if not line.startswith("OK"):
        print(f"ep{ep:02d}: {line[:300]}"); seq.append((ep, None)); continue
    g = int(line.split()[1])
    assume[g] = assume.get(g, 0) + 1
    seq.append((ep, g))
    print(f"ep{ep:02d} -> gpu{g} | {line.split('|',1)[1][:230]}")
tot = {g: base.get(g, 0) + assume.get(g, 0) for g in sorted(view)}
print("placed:", sum(1 for _, g in seq if g is not None), "/", len(cands), "| tier-3 per gpu:", dict(sorted(assume.items())), "| total trainings per gpu after:", tot)
bad = [g for g in assume if g in (3, 4, 7) and g not in m.ALLOWED_GPUS] + [g for g in assume if g in (3, 4)]
over = [g for g, n in tot.items() if g in assume and n > 3]
print("CHECK nothing on 3/4:", not bad, "| CHECK <=3 per used gpu:", not over)
try:
    m.launch(1, {}, 3, "0", 1)
except RuntimeError as e:
    print("launch on gpu3 refused:", e)
try:
    m.launch(1, {}, 4, "0", 1)
except RuntimeError as e:
    print("launch on gpu4 refused:", e)

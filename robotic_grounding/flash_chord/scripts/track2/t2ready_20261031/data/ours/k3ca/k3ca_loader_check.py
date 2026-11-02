#!/usr/bin/env python3
"""k3ca_loader_check.py (CPU only, no training): can the C3r2 trainer (code/fc_r2) consume data/human_motion_data_tier2_k3ca?
Same checks as scripts/k3fix_loader_check.py for every parquet of the new root and of the raw Tier-2 root (control):
flash_chord.data.loader.load_reference (the motion_v1 schema validation the trainer runs), num_frames, object URDFs
exist, support_usda_for_reference resolves inside the root's own whole_body/reconstructed_stage. Also: the object track
loaded from the K3CA episodes' parquets differs from the K3fix root's, and is identical for the other 23.
Writes only tmp_analysis/tier2/k3ca/build/loader_check.json.
Run: cd code/fc_r2 && CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
     nice -n 19 $ME/conda_envs/flashchord/bin/python -B $ME/scripts/k3ca_loader_check.py
"""
import json, sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
from flash_chord.data.loader import load_reference  # noqa: E402
from flash_chord.assets.registry import support_usda_for_reference  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
ROOTS = {"k3ca": ME / "data/human_motion_data_tier2_k3ca", "tier2_control": ME / "data/human_motion_data_tier2"}
FIX = ME / "data/human_motion_data_tier2_k3fix"
OUT = ME / "tmp_analysis/tier2/k3ca/build/loader_check.json"
CHANGED = set(json.loads((ME / "tmp_analysis/tier2/k3ca/build/data_build.json").read_text())["k3ca_episodes"])
man = json.loads((ROOTS["k3ca"] / "whole_body/soma/track2_manifest.json").read_text())
ep_to_seq = {int(v["episode_index"]): v["sequence_id"] for v in man.values()}
cv = lambda x: x() if callable(x) else x  # noqa: E731
out = {k: [] for k in ROOTS}
for ep in range(30):
    seq = ep_to_seq[ep]
    for k, root in ROOTS.items():
        p = root / f"whole_body/soma/sequence_id={seq}/robot_name=g1/data.parquet"
        r = {"ep": ep}
        try:
            ref = load_reference(str(p))
            sup = support_usda_for_reference(ref)
            assets = cv(ref.object_assets)
            r.update(num_frames=int(cv(ref.num_frames)), obj_pos_shape=list(np.asarray(cv(ref.object_body_pos_w)).shape),
                     urdfs=[a.urdf_path for a in assets], urdfs_exist=all(Path(a.urdf_path).is_file() for a in assets),
                     support=str(sup) if sup else None,
                     support_in_own_root=bool(sup and str(sup).startswith(str(root / "whole_body/reconstructed_stage") + "/")))
            r["ok"] = bool(r["urdfs_exist"] and r["support_in_own_root"])
            if k == "k3ca":
                fref = load_reference(str(FIX / f"whole_body/soma/sequence_id={seq}/robot_name=g1/data.parquet"))
                d = float(np.abs(np.asarray(cv(ref.object_body_pos_w)) - np.asarray(cv(fref.object_body_pos_w))).max())
                dq = float(np.abs(np.asarray(cv(ref.object_body_quat_w)) - np.asarray(cv(fref.object_body_quat_w))).max())
                r["obj_pos_max_abs_diff_vs_k3fix_root_m"], r["obj_quat_max_abs_diff_vs_k3fix_root"] = d, dq
                r["changed"] = ep in CHANGED
                r["ok"] &= (d > 0 or dq > 0) == (ep in CHANGED)
        except Exception as e:  # noqa: BLE001
            r.update(ok=False, error=f"{type(e).__name__}: {e}"[:300])
        out[k].append(r)
    print(f"ep{ep:02d} " + " ".join(f"{k}:{'ok' if out[k][-1]['ok'] else 'FAIL'}/T={out[k][-1].get('num_frames')}" for k in ROOTS)
          + f" diff vs K3fix pos {out['k3ca'][-1].get('obj_pos_max_abs_diff_vs_k3fix_root_m')} quat {out['k3ca'][-1].get('obj_quat_max_abs_diff_vs_k3fix_root')}", flush=True)
same_T = all(len({out[k][ep].get("num_frames") for k in ROOTS}) == 1 for ep in range(30))
summary = {k: dict(n_ok=sum(r["ok"] for r in v), total_frames=sum(r.get("num_frames") or 0 for r in v)) for k, v in out.items()}
summary["same_num_frames"] = same_T
summary["k3ca_episodes"] = sorted(CHANGED)
summary["all_ok"] = bool(same_T and all(summary[k]["n_ok"] == 30 for k in ROOTS))
OUT.write_text(json.dumps({"summary": summary, "per_root": out}, indent=1) + "\n")
print(json.dumps(summary, indent=1))

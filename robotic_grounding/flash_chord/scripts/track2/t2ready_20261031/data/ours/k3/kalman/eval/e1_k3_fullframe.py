"""EVAL INPUT step 1 (NOISY-ONLY): the frozen K3 smoother on every frame of each clip (visible and invisible), from the
frozen code and frozen_params.json. On visible frames it must equal outputs/kalman_poses.npz bit-for-bit (checked).
Invisible frames (<= 0.11 %) get the smoothed state of the frozen model (their measurement is the nearest-visible fill,
exactly as in the frozen run). Writes KOUT/eval/k3_fullframe.npz.  Run with the v2d python.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

KOUT = Path("/root/autodl-tmp/haoran.zhang/tmp_analysis/tier2/kalman")
sys.path.insert(0, str(KOUT / "scripts"))
from kalman_core import BT, P_  # noqa: E402
from kfilter import Episode, episodes, load_params  # noqa: E402

EV = KOUT / "eval"
EV.mkdir(exist_ok=True)
prm = load_params()
KP = np.load(KOUT / "outputs/kalman_poses.npz")
out, rep = {}, []
for m in episodes():
    ep = m["episode_index"]
    E = Episode(m, prm)
    pose, X, diag = E.variant("K3")
    frozen = KP[f"K3__{ep}"]
    full = np.concatenate([X[:, P_] / 100.0, (R.from_rotvec(-X[:, BT]) * E.rot).as_quat()], 1)
    d1 = float(np.abs(pose - frozen).max())
    d2 = float(np.abs(full[E.vis] - frozen).max())
    assert d1 < 1e-12 and d2 < 1e-12, (ep, d1, d2)
    out[f"K3full__{ep}"] = full
    out[f"noisyfull__{ep}"] = np.concatenate([E.p / 100.0, E.rot.as_quat()], 1)
    out[f"vis__{ep}"] = E.vis
    rep.append({"episode": ep, "T": int(E.T), "invisible": int((~E.vis).sum()), "max_diff_vs_frozen": max(d1, d2)})
    print(rep[-1], flush=True)
np.savez_compressed(EV / "k3_fullframe.npz", **out)
(EV / "k3_fullframe.json").write_text(json.dumps(rep, indent=1))
print("total invisible frames", sum(r["invisible"] for r in rep), "of", sum(r["T"] for r in rep))

"""NOISY-ONLY sanity check: angular/position difference between K-variants, the noisy input and ZUPT v1 (no clean data)."""
import numpy as np, json
from scipy.spatial.transform import Rotation as R
K = np.load("/root/autodl-tmp/haoran.zhang/tmp_analysis/tier2/kalman/outputs/kalman_poses.npz")
Z = np.load("/root/autodl-tmp/haoran.zhang/tmp_analysis/tier2/filter_study/noise/filtered_object_poses.npz")
eps = sorted({int(k.split("__")[1]) for k in K.files})
def d(a, b):
    return float(np.degrees((R.from_quat(a[:, 3:]).inv() * R.from_quat(b[:, 3:])).magnitude()).mean()), float(np.linalg.norm(a[:, :3] - b[:, :3], axis=1).mean() * 100)
rows = []
for ep in eps:
    z, n = Z[f"zupt__{ep}"], Z[f"none__{ep}"]
    r = [ep]
    for v in ("K0", "K1", "K2", "K3"):
        r += [d(K[f"{v}__{ep}"], z), d(K[f"{v}__{ep}"], n)]
    rows.append(r)
    print(f"ep{ep:02d} " + "  ".join(f"{v}: vsZUPT {r[1+2*i][0]:5.1f}deg {r[1+2*i][1]:4.2f}cm vsNone {r[2+2*i][0]:5.1f}deg {r[2+2*i][1]:4.2f}cm" for i, v in enumerate(("K0", "K1", "K2", "K3"))))
z_none = [d(Z[f"zupt__{ep}"], Z[f"none__{ep}"]) for ep in eps]
print("mean ZUPT vs none: %.2f deg %.2f cm" % (np.mean([x[0] for x in z_none]), np.mean([x[1] for x in z_none])))
for i, v in enumerate(("K0", "K1", "K2", "K3")):
    print(v, "mean vs ZUPT: %.2f deg %.2f cm ; vs none: %.2f deg %.2f cm" % (np.mean([r[1+2*i][0] for r in rows]), np.mean([r[1+2*i][1] for r in rows]), np.mean([r[2+2*i][0] for r in rows]), np.mean([r[2+2*i][1] for r in rows])))

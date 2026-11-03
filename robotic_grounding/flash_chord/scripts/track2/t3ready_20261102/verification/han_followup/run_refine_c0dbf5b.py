"""run_refine_c0dbf5b.py <tree> <V> <wd> <outname>: re-run Han's frame-0 step from a checkout of c0dbf5b (or later).
Adapted from tmp_analysis/tier3/han_6c10a22_check/verify-deterministic-reruns/scripts/run_refine.py (written for 6c10a22):
since 81a3f6b t3_frame0.py takes its root from V2D_ROOT and its helper path relative to itself, so no path is rewritten.
V is the NAS-like layout built by prep_inputs.py (ref_poses hold the FIRST ROW ONLY, frame_index 0). The source is cut
before the GT diagnostic loop ('\nres = {}\n'); no score is computed. build_release_combined.py's own save lines are
applied and the bytes compared with Han's frame-0 release."""
import sys, io, os, json
from pathlib import Path
TREE, V, WD, OUTN = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
os.environ["V2D_ROOT"] = str(V); os.environ.pop("V2D_EPISODES", None)
import numpy as np, scipy, trimesh
OUT = WD / OUTN; OUT.mkdir(parents=True, exist_ok=True)
HAN = Path("/root/autodl-tmp/han.zhou/tier3/release"); REL = HAN / "tier3_frame0_supports_20261024"
f = TREE / "hoi_estimation/tier2_refine/t3_frame0.py"
src = f.read_text()
assert "/mnt/nas" not in src and "/home/hanzhou" not in src
assert src.count("\nres = {}\n") == 1; head = src.split("\nres = {}\n")[0]
sys.argv = [str(f), "opus", str(OUT), "3", "2", "shift"]; g = {"__name__": "t3f0", "__file__": str(f)}
exec(compile(head, str(f), "exec"), g)
brc = (TREE / "hoi_estimation/support/build_release_combined.py").read_text().splitlines()
blk = [l.strip() for l in brc if l.strip().startswith(("np.save(", "q = Rot.from_matrix"))]
print("env numpy", np.__version__, "scipy", scipy.__version__, "trimesh", trimesh.__version__, "| save lines:", blk, flush=True)
from scipy.spatial.transform import Rotation as Rot
eps = {int(e["episode_index"]): e["sequence_id"] for e in json.load(open(REL / "episodes.json"))}
res = {}
for i in range(30):
    T, info = g["refine"](i); np.savez(f"{OUT}/episode_{i:06d}.npz", T_world_mesh=T)
    T = np.load(f"{OUT}/episode_{i:06d}.npz")["T_world_mesh"].astype(np.float64)
    b1, b2 = io.BytesIO(), io.BytesIO()
    np.save(b1, T); q = Rot.from_matrix(T[:, :3, :3]).as_quat(); np.save(b2, np.c_[T[:, :3, 3], q[:, 3:], q[:, :3]].astype(np.float32))
    rd = REL / "reconstructed_data" / eps[i]
    eq1 = b1.getvalue() == (rd / "poses.npy").read_bytes(); eq2 = b2.getvalue() == (rd / "object_pose_xyz_wxyz.npy").read_bytes()
    dmax = float(np.abs(T - np.load(rd / "poses.npy")).max())
    res[i] = dict(poses_bytes_equal=eq1, xyzw_bytes_equal=eq2, max_abs_dT=dmax, frames=len(T))
json.dump(res, open(WD / f"{OUTN}.json", "w"), indent=1, default=float)
n = sum(r["poses_bytes_equal"] and r["xyzw_bytes_equal"] for r in res.values())
print(f"{OUTN}: byte-identical {n}/30, frames {sum(r['frames'] for r in res.values())}, max|dT| {max(r['max_abs_dT'] for r in res.values())}", flush=True)
print("non-identical:", [i for i, r in res.items() if not (r["poses_bytes_equal"] and r["xyzw_bytes_equal"])])

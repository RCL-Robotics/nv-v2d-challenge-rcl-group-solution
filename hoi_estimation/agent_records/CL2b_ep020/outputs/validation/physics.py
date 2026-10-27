"""Physical plausibility: height of the object's lowest point above the floor plane and its world-position spread
per phase (object rests on a table: height should be constant; static phases: position constant)."""
import sys, json, numpy as np, trimesh
inp = "/work/inputs"
res = np.load(sys.argv[1]); valj = json.load(open(sys.argv[2]))
rec = np.load(f"{inp}/reconstruction.npz"); Tb = rec["T_world_mesh"].astype(np.float64); To = res["T_world_mesh"]
pl = json.load(open(f"{inp}/ground_plane.json"))["plane"]; n = np.array(pl[:3]); d = pl[3]
m = trimesh.load(f"{inp}/object_mesh.glb", force="scene", process=False).to_geometry(); v = np.asarray(m.vertices)[::30]
def heights(T):
    w = np.einsum("fij,nj->fni", T[:, :3, :3], v) + T[:, None, :3, 3]
    s = w @ n + d
    return np.abs(s).min(1)  # distance of the lowest point to the floor
hb, ho = heights(Tb), heights(To)
print("phase | bottom height above floor (cm) baseline mean±std | ours mean±std | world centre spread (cm, std) baseline | ours")
for a, b, mv in valj["phases"]:
    sl = slice(a, b + 1)
    sb = np.linalg.norm(Tb[sl, :3, 3].std(0)) * 100; so = np.linalg.norm(To[sl, :3, 3].std(0)) * 100
    print(f"{a}-{b} {'moving' if mv else 'static'} | {100*hb[sl].mean():.1f}±{100*hb[sl].std():.1f} | {100*ho[sl].mean():.1f}±{100*ho[sl].std():.1f} | {sb:.1f} | {so:.1f}")
print(f"all | {100*hb.mean():.1f}±{100*hb.std():.1f} | {100*ho.mean():.1f}±{100*ho.std():.1f}")

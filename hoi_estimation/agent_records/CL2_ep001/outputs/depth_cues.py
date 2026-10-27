"""Depth cues (reported only): suggested shift along the viewing ray (cm, negative = closer to camera)."""
import json, os, sys
import numpy as np, trimesh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import load_inputs
inp = sys.argv[1] if len(sys.argv) > 1 else "/work/inputs"
res = sys.argv[2] if len(sys.argv) > 2 else "/work/outputs/object_poses_world.npz"
cam, K, Twc, rec = load_inputs(inp)
Tcw = np.linalg.inv(Twc)
pl = np.array(json.load(open(os.path.join(inp, "ground_plane.json")))["plane"]); n, d = pl[:3], pl[3]
hv = np.load(os.path.join(inp, "frame0_reference_human.npz"))["vertices_world"]
h0 = np.percentile(hv @ n + d, 0.1)
V = np.asarray(trimesh.load(os.path.join(inp, "object_mesh.glb"), force="scene", process=False).to_geometry().vertices)
diag = np.load(os.path.splitext(res)[0] + "_diag.npz"); ph = diag["phases"]; gs = float(diag["gscale"])
To = np.load(res)["T_world_mesh"]; Tb = rec["T_world_mesh"].astype(np.float64)
print(f"true-human sole level (plane distance, 0.1 pct): {h0 * 100:.1f} cm")
for tag, T in [("baseline", Tb), ("ours", To)]:
    for name, (a, b) in zip(["static_start", "static_end"], [ph[0], ph[2]]):
        # scale camera-frame vertex positions by f (move along rays); find f s.t. lowest point = h0
        Tc = Tcw @ T[a]
        Xc = V @ Tc[:3, :3].T + Tc[:3, 3]
        fs = np.linspace(0.85, 1.15, 601)
        lows = [(((f * Xc) @ Twc[:3, :3].T + Twc[:3, 3]) @ n + d).min() for f in fs]
        f = fs[np.argmin(np.abs(np.array(lows) - h0))]
        print(f"floor cue {tag} {name}: lowest point {lows[300] * 100:.1f} cm -> depth shift {(f - 1) * np.median(Xc[:, 2]) * 100:.1f} cm")
z = np.median((Tcw @ To)[:, 2, 3])
print(f"silhouette-size cue: scale {gs:.3f} -> {(gs - 1) * z * 100:.1f} cm (uncorrected); "
      f"with 3-5 % mask bloat measured on the true frame-0 human: {(gs * 1.03 - 1) * z * 100:.1f} .. {(gs * 1.05 - 1) * z * 100:.1f} cm")
# hand cue: left/right hand keypoint depth vs nearest handle point, carried phase
kp = rec["human_keypoints_world"].astype(np.float64)
top = V[V[:, 2] > 0.60].mean(0)
a, b = ph[1]
for nm, rng in [("right", range(21, 42)), ("left", range(42, 63))]:
    h = kp[a:b + 1, list(rng)].mean(1)
    g = np.einsum("fij,j->fi", To[a:b + 1, :3, :3], top) + To[a:b + 1, :3, 3]
    hc = h @ Tcw[:3, :3].T + Tcw[:3, 3]; gc = g @ Tcw[:3, :3].T + Tcw[:3, 3]
    dist = np.linalg.norm(h - g, axis=1)
    close = dist < np.percentile(dist, 30)
    print(f"hand cue ({nm} hand, 30% frames closest to the handle top): hand-handle depth diff median {np.median(hc[close, 2] - gc[close, 2]) * 100:.1f} cm, "
          f"MAD {np.median(np.abs(hc[close, 2] - gc[close, 2] - np.median(hc[close, 2] - gc[close, 2]))) * 100:.1f} cm, 3D dist median {np.median(dist[close]) * 100:.1f} cm")

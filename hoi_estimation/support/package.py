#!/usr/bin/env python3
"""Package the reconstructed supports for simulation, per episode (world frame = our Tier 3 release frame, OpenCV
world, y down; the ground plane is inputs/ground_plane.json):
  support.json            class, T_world_mesh for the scanned mesh, up vector, top height, fit IoU, top-surface box
  support_mesh.glb        the scanned furniture mesh in its own frame (apply T_world_mesh)
  support_world.obj       decimated mesh already in world coordinates (preview / quick collision)
  top_box                 oriented box primitive covering the load-bearing surface (table top or seat, found as the
                          highest 2-cm level whose horizontal footprint is >= 60% of the largest one), 3 cm thick, in world coordinates: centre, axes (columns), half extents
  T_world_mesh_contact    the same mesh shifted along the up axis by the median rest gap of our trajectory, so that the
                          object rests exactly on it (|shift| <= a few cm; the legs then sink into / lift off the floor
                          by that amount, harmless for a fixed body). Snapping the trajectory instead was rejected: a GT
                          diagnostic shows the residual gap is mostly support-height error, and snapping lowered AUC
  object_poses_world.npz  our best Tier 3 object trajectory (Opus v2 + frame-0 refinement), unchanged
Usage: package.py <out_dir>"""
import json, os, shutil, sys, numpy as np, trimesh
from common import *
TRAJ = f"{V}/experiments/t3_frame0/opus_f3_c2_shift"; DST = sys.argv[1]; os.makedirs(DST, exist_ok=True)
contact = json.load(open(f"{OUT}/contact_report.json")); index = {}
for i in SUPPORT:
    d = f"{OUT}/ep{i:03d}"; o = f"{DST}/episode_{i:06d}"; os.makedirs(o, exist_ok=True); s = json.load(open(f"{d}/support.json"))
    g = np.array(s["ground_plane"]); n = g[:3] / np.linalg.norm(g[:3]); g0 = g[3] / np.linalg.norm(g[:3])
    mesh = trimesh.load(f"{MESH}/{s['support']}/{s['support']}.glb", force="scene", process=False).to_geometry(); T = np.array(s["T_world_mesh"])
    W = np.asarray(mesh.vertices) @ T[:3, :3].T + T[:3, 3]; h = W @ n + g0
    # load-bearing level (table top / seat, not a backrest): the highest 2-cm band whose horizontal convex-hull area is
    # at least 60% of the largest band area above 30 cm
    e1 = np.cross(n, [1.0, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    lv = np.arange(.3, h.max() + .005, .01); ar = []
    for L in lv:
        b = W[(h > L - .02) & (h <= L)]
        ar.append(trimesh.convex.convex_hull(np.c_[b @ e1, b @ e2, np.zeros(len(b))], qhull_options="QJ").area / 2 if len(b) > 20 else 0.0)
    ar = np.array(ar); hs_ = float(lv[np.where(ar >= .6 * ar.max())[0].max()])
    top = W[(h > hs_ - .02) & (h <= hs_ + .005)]
    P = np.stack([top @ e1, top @ e2], 1); c2 = P.mean(0); U, _, Vt = np.linalg.svd(P - c2, full_matrices=False)
    ax2 = Vt; Q = (P - c2) @ ax2.T; lo, hi = np.percentile(Q, 1, 0), np.percentile(Q, 99, 0)
    ctr2 = c2 + ((lo + hi) / 2) @ ax2; a1 = ax2[0, 0] * e1 + ax2[0, 1] * e2; a2 = np.cross(n, a1)
    hm = float((top @ n + g0).max()); centre = ctr2[0] * e1 + ctr2[1] * e2 + (hm - .015 - g0) * n  # point at height hm - 1.5 cm
    box = dict(centre=centre.tolist(), axes=np.stack([a1, a2, n], 1).tolist(), half_extents=[float((hi - lo)[0] / 2), float((hi - lo)[1] / 2), .015])
    gap = contact.get(str(i), {}).get("gap_median_cm"); sh = (gap or 0.0) / 100
    Tc = T.copy(); Tc[:3, 3] += sh * n
    s.update(T_world_mesh_contact=Tc.tolist(), contact_shift_m=sh, support_surface_height_m=hm, up_world=n.tolist(), top_box=box, mesh="support_mesh.glb", note="T_world_mesh maps support_mesh.glb into the Tier 3 world (OpenCV world, y down)")
    s.pop("detection_box", None)
    json.dump(s, open(f"{o}/support.json", "w"), indent=1)
    shutil.copy(f"{MESH}/{SUPPORT[i]}/{SUPPORT[i]}.glb", f"{o}/support_mesh.glb"); shutil.copy(f"{d}/support_world.obj", f"{o}/support_world.obj")
    shutil.copy(f"{TRAJ}/episode_{i:06d}.npz", f"{o}/object_poses_world.npz")
    shutil.copy(f"{OUT}/vis/ep{i:03d}_fit.jpg", f"{o}/fit_overlay.jpg")
    index[i] = dict(support=SUPPORT[i], iou=s["iou"], surface_height_m=hm, gap_cm=contact.get(str(i), {}).get("gap_median_cm"))
    print(i, index[i], "top box", np.round(2 * np.array(box["half_extents"][:2]), 2))
json.dump(index, open(f"{DST}/index.json", "w"), indent=1)

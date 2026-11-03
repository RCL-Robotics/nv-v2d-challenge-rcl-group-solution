"""Report-only analysis: independent physical cues for a depth (along-ray) correction of the final trajectory.
Positive shift = move the object AWAY from the camera (cm). Not applied unless cues agree (they do not)."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stages import Episode, BAT_CENTER, anchor_depth


def shift_along_ray(Tc, dz):
    """Move the bat centre along its viewing ray by dz (metres of camera depth)."""
    c = Tc[:, :3, :3] @ BAT_CENTER + Tc[:, :3, 3]
    return anchor_depth(Tc, c[:, 2] + dz)


def main(final_npz, decoded_npz, s1, s2, out_json):
    ep = Episode("/work/inputs")
    import trimesh
    V = np.asarray(ep.mesh.vertices)[::20]
    Tc = ep.Tcw[None] @ np.load(final_npz)["T_world_mesh"]
    Td = np.load(decoded_npz)["T"]  # silhouette-only depth (before re-anchoring)
    phases = {"static-1": (0, s1), "moving": (s1 + 1, s2 - 1), "static-2": (s2, ep.F - 1)}
    grid = np.arange(-0.30, 0.301, 0.005)
    kpc = np.einsum("ij,fkj->fki", ep.Tcw[:3, :3], ep.kp_world) + ep.Tcw[:3, 3]
    seg = np.stack([np.zeros(30), np.zeros(30), np.linspace(-0.2, 0.33, 30)], 1)
    res = {}
    for name, (a, b) in phases.items():
        sl = slice(a, b + 1); r = {}
        cf = (Tc[sl, :3, :3] @ BAT_CENTER + Tc[sl, :3, 3])[:, 2]
        cd = (Td[sl, :3, :3] @ BAT_CENTER + Td[sl, :3, 3])[:, 2]
        r["silhouette_size_cm"] = 100 * float(np.median(cd - cf))
        if name.startswith("static"):
            # floor contact: lowest mesh point on the released floor plane
            hs = []
            for dz in grid:
                T = shift_along_ray(Tc[sl][::10], dz)
                P = np.einsum("fij,vj->fvi", ep.Twc[None, :3, :3] @ T[:, :3, :3], V) + (ep.Twc[None] @ T)[:, None, :3, 3]
                hs.append(np.median((P @ ep.plane[:3] + ep.plane[3]).min(1)))
            hs = np.array(hs)
            r["floor_contact_cm"] = 100 * float(grid[np.argmin(np.abs(hs))])
            r["floor_min_height_at_0_cm"] = 100 * float(hs[np.argmin(np.abs(grid))])
        else:
            # hand contact: shift minimising the median distance of the closest hand keypoint to the bat axis
            ds = []
            for dz in grid:
                T = shift_along_ray(Tc[sl], dz)
                P = np.einsum("fij,nj->fni", T[:, :3, :3], seg) + T[:, None, :3, 3]
                ds.append(np.median(np.linalg.norm(kpc[sl, 21:63, None] - P[:, None], axis=-1).min((1, 2))))
            r["hand_contact_cm"] = 100 * float(grid[np.argmin(ds)])
            r["hand_dist_at_0_cm"] = 100 * float(ds[np.argmin(np.abs(grid))])
        res[name] = r
    print(json.dumps(res, indent=1))
    json.dump(res, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])

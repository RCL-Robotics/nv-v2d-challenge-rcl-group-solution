#!/usr/bin/env python
"""Independent physical cues for a depth (along-ray) correction of a trajectory. Reports, per cue, the
depth shift (cm, + = farther from the camera) that the cue suggests; nothing is applied here.
usage: python depth_cues.py --inputs /work/inputs --traj result.npz --phases result_diag.json
 cue A  floor contact: in static phases on the floor, the lowest mesh point should touch the released floor plane
 cue B  hand contact: while carried, reconstructed hand keypoints (MHR 21-62) should touch the book surface
 cue C  metric size: silhouette chamfer distance as a function of the scale along the ray (static phases)
"""
import argparse, json, numpy as np, trimesh, h5py, cv2
from scipy.spatial import cKDTree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='/work/inputs'); ap.add_argument('--traj', required=True)
    ap.add_argument('--phases', required=True)
    a = ap.parse_args(); inp = a.inputs
    cam = json.load(open(f'{inp}/camera.json')); K = np.array(cam['K']); Twc = np.array(cam['T_world_cam'])
    Tw = np.load(a.traj)['T_world_mesh']; N = len(Tw); Tc = np.linalg.inv(Twc)[None] @ Tw
    rec = np.load(f'{inp}/reconstruction.npz')
    mesh = trimesh.load(f'{inp}/object_mesh.glb', force='scene', process=False).to_geometry()
    plane = np.array(json.load(open(f'{inp}/ground_plane.json'))['plane']); nrm, d0 = plane[:3], plane[3]
    phases = json.load(open(a.phases))['phases']
    V = np.asarray(mesh.vertices)[::20]
    ctr = mesh.bounds.mean(0)
    scales = np.linspace(0.94, 1.06, 121)

    def shifted_world(i, s):
        """pose of frame i with its camera-frame position scaled by s along the ray"""
        T = Tc[i].copy(); pc = T[:3, :3] @ ctr + T[:3, 3]; T[:3, 3] += (s - 1) * pc
        return Twc @ T

    print('## cue A: floor contact (static phases on the floor)')
    for s0, s1, kind in phases:
        if kind != 'static': continue
        P = np.einsum('ij,nj->ni', Tw[s0][:3, :3], V) + Tw[s0][:3, 3]
        h = (P @ nrm + d0).min()
        if h > 0.3: continue          # not on the floor
        hs = []
        for s in scales:
            T = shifted_world(s0, s); P = V @ T[:3, :3].T + T[:3, 3]; hs.append((P @ nrm + d0).min())
        hs = np.array(hs); k = np.argmin(np.abs(hs))
        z = Tc[s0, 2, 3]
        print(f'  frames {s0}-{s1 - 1}: lowest point {h * 100:+.1f} cm from plane -> contact at scale {scales[k]:.4f}'
              f' = depth shift {(scales[k] - 1) * z * 100:+.1f} cm  (plane tolerance: foot-plane distances '
              f'{json.load(open(f"{inp}/ground_plane.json"))["foot_plane_dist_stats"]["median"] * 100:.1f} cm median)')

    print('## cue B: hand contact while carried (moving phases, contact_prob > 0.9)')
    kp = rec['human_keypoints_world']; cp = rec['contact_prob_right_left']
    surf, _ = trimesh.sample.sample_surface_even(mesh, 4000, seed=0)
    res = []
    for s0, s1, kind in phases:
        if kind != 'moving': continue
        fr = [i for i in range(s0, s1) if cp[i].max() > 0.9]
        if len(fr) < 10: continue
        best = []
        for s in scales:
            dd = []
            for i in fr:
                T = shifted_world(i, s); P = surf @ T[:3, :3].T + T[:3, 3]; tree = cKDTree(P)
                hands = []
                for side, rng in ((0, range(21, 42)), (1, range(42, 63))):
                    if cp[i, side] > 0.9: hands.append(kp[i, list(rng)])
                H = np.concatenate(hands)
                dd.append(np.sort(tree.query(H)[0])[:5].mean())   # the 5 hand keypoints closest to the book
            best.append(np.median(dd))
        best = np.array(best); k = np.argmin(best)
        z = np.median(Tc[fr, 2, 3])
        print(f'  frames {s0}-{s1 - 1} ({len(fr)} contact frames): median hand-book gap {best[60] * 100:.1f} cm at the '
              f'current depth, minimum {best[k] * 100:.1f} cm at depth shift {(scales[k] - 1) * z * 100:+.1f} cm')
        res.append((scales[k] - 1) * z * 100)
    if res: print(f'  hand cue overall (median over phases): {np.median(res):+.1f} cm')

    print('## cue C: metric mesh size vs mask size (static phases, silhouette IoU vs scale along the ray)')
    H5 = h5py.File(f'{inp}/masks.h5', 'r')
    corners = np.array(trimesh.bounds.corners(mesh.bounds))
    for s0, s1, kind in phases:
        if kind != 'static': continue
        acc = np.zeros(len(scales[::10]))
        for i in range(s0, s1, 4):
            om = H5['object'][i].astype(np.uint8); pm = H5['person'][i]
            for j, s in enumerate(scales[::10]):
                T = np.linalg.inv(Twc) @ shifted_world(i, s)
                # silhouette by projecting a dense surface sample and closing it
                P = surf @ T[:3, :3].T + T[:3, 3]; q = (K @ P.T).T; q = (q[:, :2] / q[:, 2:]).astype(np.int32)
                m = np.zeros_like(om); cv2.fillConvexPoly(m, cv2.convexHull(q), 1)
                valid = ~(pm & ~om.astype(bool))
                inter = ((m & om) > 0)[valid].sum(); uni = ((m | om) > 0)[valid].sum()
                acc[j] += inter / max(uni, 1)
        k = np.argmax(acc); z = np.median(Tc[s0:s1, 2, 3])
        print(f'  frames {s0}-{s1 - 1}: best convex-hull IoU at scale {scales[::10][k]:.3f} -> depth shift '
              f'{(scales[::10][k] - 1) * z * 100:+.1f} cm  (IoU curve: {np.round(acc / acc.max(), 3).tolist()})')


if __name__ == '__main__':
    main()

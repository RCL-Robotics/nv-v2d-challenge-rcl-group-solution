#!/usr/bin/env python
"""Independent depth cues for the depth re-anchoring decision (report only; solve.py applies 0 shift).
For each static phase: (1) floor contact: depth shift along the viewing ray that puts the lowest mesh point
on the released floor plane (floor phases only); (2) MoGe-2 aligned depth vs rendered object depth on visible
object pixels (median); (3) for carried phases: camera-depth offset hand keypoints vs object centre."""
import argparse, os, sys, json
import numpy as np, h5py, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hoi_lib as H
from solve import STATIC
ap = argparse.ArgumentParser()
ap.add_argument('--inputs', default='/work/inputs'); ap.add_argument('--result', required=True)
ap.add_argument('--pipeline', default='/pipeline')
args = ap.parse_args()
K, Twc, rec = H.load_inputs(args.inputs)
Tb = H.mesh_to_cam(Twc, rec['T_world_mesh'].astype(np.float64))
Tr = H.mesh_to_cam(Twc, np.load(args.result)['T_world_mesh'])
pl = np.array(json.load(open(f'{args.inputs}/ground_plane.json'))['plane'])
Tcw = np.linalg.inv(Twc)
n_c = Tcw[:3, :3] @ pl[:3]; d_c = pl[:3] @ Twc[:3, 3] + pl[3]   # plane in camera frame: n_c.x + d_c = 0
R = H.MeshRenderer(f'{args.inputs}/object_mesh.glb')
V = R.verts_np[::20]
with h5py.File(f'{args.inputs}/masks.h5') as f:
    OM = f['object']; PM = f['person']
    fa = h5py.File(f'{args.pipeline}/depth/moge2/aligned.h5')
    print('| phase | frames | floor-contact shift (cm, + = farther) | MoGe-vs-render median shift (cm) |')
    print('|---|---|---|---|')
    for name, a, b in STATIC:
        T = Tr[a]
        P = V @ T[:3, :3].T + T[:3, 3]
        floor = ''
        if name in ('B', 'D1', 'D2'):
            # bottom point height as function of scale s along rays: h(s) = s * (n_c . P) + d_c, solve min_v h = 0
            hv = P @ n_c
            s = -d_c / hv.min() if d_c * hv.min() < 0 else np.nan
            # lowest point at scale s: choose vertex with min h(s): iterate
            for _ in range(3):
                j = np.argmin(s * hv + d_c); s = -d_c / hv[j]
            floor = f'{(s - 1) * T[2, 3] * 100:+.1f}'
        ds = []
        for i in range(a, b + 1, 4):
            om = OM[i] & ~PM[i]
            if om.sum() < 100: continue
            ys, xs = np.nonzero(om)
            crop = (int(xs.mean()) - 96, int(ys.mean()) - 72, 192, 144)
            _, m, dep = R.render(Tr[i:i + 1], K, crop, (192, 144), with_tex=False)
            m = m[0].cpu().numpy(); dep = dep[0].cpu().numpy()
            dm = cv2.imdecode(np.frombuffer(fa['aligned/front_stereo_camera_left'][i], np.uint8), cv2.IMREAD_UNCHANGED) / 1000.0
            x0, y0 = crop[0], crop[1]
            sub = dm[y0:y0 + 144, x0:x0 + 192]; omc = om[y0:y0 + 144, x0:x0 + 192]
            sel = m & omc
            if sel.sum() > 50:
                ds.append(np.median(sub[sel] - dep[sel]))
        print(f'| {name} | {a}-{b} | {floor or "n/a (on table)"} | {np.median(ds) * 100:+.1f} |')
kp = rec['human_keypoints_world'].astype(np.float64)
kc = np.einsum('ij,fkj->fki', Tcw[:3, :3], kp) + Tcw[:3, 3]
cp = rec['contact_prob_right_left']
print()
print('carried frames: median (hand keypoint z - object centre z) for the hand nearest to the object, cm')
mov = np.ones(len(Tr), bool)
for _, a, b in STATIC: mov[a:b + 1] = False
dz = []
for i in np.nonzero(mov)[0]:
    c = Tr[i, :3, 3]
    hands = [kc[i, 21:42], kc[i, 42:63]]
    dist = [np.linalg.norm(h - c, axis=1).min() for h in hands]
    h = hands[int(np.argmin(dist))]
    if min(dist) < 0.15:
        dz.append(np.median(h[:, 2]) - c[2])
print(f'n={len(dz)} median {np.median(dz) * 100:+.1f} cm, IQR {np.percentile(dz, 25) * 100:+.1f}..{np.percentile(dz, 75) * 100:+.1f} cm')

"""Ground-truth-free validation: how well a trajectory explains the video, per phase.
python validate.py result.npz [more.npz ...]   (baseline is always included)"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from hoi_lib import *
from solve import evidence, detect_phases
inp = '/work/inputs'
cam, K, Twc, rec = load_inputs(inp)
Rd = Renderer(load_mesh(inp)); fd = FrameData(inp)
Tcw = np.linalg.inv(Twc)
F = rec['T_world_mesh'].shape[0]
e1, s2, area, _ = detect_phases(fd.om)
phases = {'S1 static (table)': np.arange(0, e1 + 1), 'carried': np.arange(e1 + 1, s2), 'S2 static (floor)': np.arange(s2, F)}
plane = np.array(json.load(open(f'{inp}/ground_plane.json'))['plane'])
v = Rd.verts_np.astype(np.float64)
srcs = [('baseline', rec['T_world_mesh'].astype(np.float64))] + [(p, np.load(p)['T_world_mesh']) for p in sys.argv[1:]]
Tb = Tcw[None] @ srcs[0][1]
hw = (192, 192)
for name, Tw in srcs:
    Tc = Tcw[None] @ Tw
    tex = np.full(F, np.nan); iou = np.full(F, np.nan)
    for i in range(F):
        if area[i] < 400: continue
        box = frame_box(K, Tb[i], scale=1.3)
        img, om, pm = fd.crop(i, box, hw)
        t, u = evidence(Rd, K, box, hw, Tc[i:i + 1], img, om, pm)
        tex[i], iou[i] = t[0], u[0]
    dR = np.r_[0, rot_angle_deg(Tc[1:, :3, :3], Tc[:-1, :3, :3])]
    dt = np.r_[0, np.linalg.norm(np.diff(Tw[:, :3, 3], axis=0), axis=1)] * 100
    # lowest mesh point relative to the floor (cm, + above)
    sub = v[::50]
    low = np.array([((sub @ Tw[i, :3, :3].T + Tw[i, :3, 3]) @ plane[:3] + plane[3]).min() for i in range(F)])
    low = low * (1 if plane[1] < 0 else -1) * 100
    print(f'== {name}')
    for pn, ph in phases.items():
        dz = (Tc[ph, 2, 3] - Tb[ph, 2, 3]) * 100
        print(f'  {pn:18s} n={len(ph):3d} IoU {np.nanmean(iou[ph]):.3f} tex {np.nanmean(tex[ph]):.3f} '
              f'| rot step mean {dR[ph].mean():5.2f} p95 {np.percentile(dR[ph],95):5.1f} max {dR[ph].max():5.1f} deg '
              f'| trans step p95 {np.percentile(dt[ph],95):4.1f} cm | dz vs base mean {dz.mean():+5.1f} med {np.median(dz):+5.1f} cm '
              f'| lowest pt above floor med {np.median(low[ph]):+5.1f} cm')

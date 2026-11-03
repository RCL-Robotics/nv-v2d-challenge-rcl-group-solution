"""Validation table: baseline vs result per phase (no ground truth)."""
import sys, json; sys.path.insert(0, '/work/scratch/src')
import numpy as np, torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, '/work/outputs')
import solve as S

inp = '/work/inputs'
res_path, diag_path = sys.argv[1], sys.argv[2]
ho_diag = sys.argv[3] if len(sys.argv) > 3 else None
cam = json.load(open(f'{inp}/camera.json'))
K = np.array(cam['K']); Twc = np.array(cam['T_world_cam']); Tcw = np.linalg.inv(Twc)
rec = np.load(f'{inp}/reconstruction.npz')
Tb = Tcw[None] @ rec['T_world_mesh'].astype(float)
Tr = Tcw[None] @ np.load(res_path)['T_world_mesh']
D = np.load(diag_path)
S1, S2 = D['S1'], D['S2']
F = len(Tb)
PH = {'S1 static': (S1[0], S1[1]), 'lift (carried)': (S1[1], 230), 'pour (carried)': (230, 500),
      'put-down (carried)': (500, S2[0]), 'S2 static': (S2[0], S2[1]), 'ALL': (0, F)}

import h5py, trimesh
with h5py.File(f'{inp}/masks.h5') as f:
    om = f['object'][:]; pm = f['person'][:]
x0, y0, w, h = [int(v) for v in D['crop']]
om = om[:, y0:y0 + h, x0:x0 + w]; pm = pm[:, y0:y0 + h, x0:x0 + w]
R = S.SilRenderer(f'{inp}/object_mesh.glb', K, (x0, y0, w, h))
tg = torch.tensor(om, device='cuda').float(); va = torch.tensor(~(pm & ~om), device='cuda').float()


def ious(T):
    out = []
    with torch.no_grad():
        for c in range(0, F, 128):
            sil = R(torch.tensor(T[c:c + 128], device='cuda', dtype=torch.float32))
            out.append(S.soft_iou((sil > 0.5).float(), tg[c:c + 128], va[c:c + 128]).cpu().numpy())
    return np.concatenate(out)


I_b, I_r = ious(Tb), ious(Tr)
I_ho = None
if ho_diag:
    Dh = np.load(ho_diag)
    Th = Dh['T_final_c']; I_ho = ious(Th)
    I_hofit = ious(Dh['T_fit'])
    I_bfit = None


def rdeg(A, B):
    return np.degrees(np.linalg.norm(Rot.from_matrix(np.transpose(A[:, :3, :3], (0, 2, 1)) @ B[:, :3, :3]).as_rotvec(), axis=1))


drot_b = np.r_[np.nan, rdeg(Tb[:-1], Tb[1:])]; drot_r = np.r_[np.nan, rdeg(Tr[:-1], Tr[1:])]
dtr_b = np.r_[np.nan, np.linalg.norm(np.diff(Tb[:, :3, 3], axis=0), axis=1) * 100]
dtr_r = np.r_[np.nan, np.linalg.norm(np.diff(Tr[:, :3, 3], axis=0), axis=1) * 100]
rot_vs_b = rdeg(Tb, Tr)
dz = (Tr[:, 2, 3] - Tb[:, 2, 3]) * 100


def px(T):
    p = T[:, :3, 3]; return np.stack([K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2], K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]], 1)


dpx = px(Tr) - px(Tb)
# lateral (perpendicular to the ray) displacement in cm
lat = np.linalg.norm((Tr[:, :3, 3] / Tr[:, 2:3, 3] - Tb[:, :3, 3] / Tb[:, 2:3, 3])[:, :2], axis=1) * Tb[:, 2, 3] * 100
# ADD-like distance between result and baseline (mesh points), cm
V = trimesh.load(f'{inp}/object_mesh.glb', force='scene', process=False).to_geometry().vertices[::50]
def pts(T): return np.einsum('fij,nj->fni', T[:, :3, :3], V) + T[:, None, :3, 3]
add_rb = np.linalg.norm(pts(Tr) - pts(Tb), axis=2).mean(1) * 100

# floor
pl = json.load(open(f'{inp}/ground_plane.json'))['plane']; n_w = np.array(pl[:3]); d_w = pl[3]
n_c = Twc[:3, :3].T @ n_w; d_c = n_w @ Twc[:3, 3] + d_w
Vf = trimesh.load(f'{inp}/object_mesh.glb', force='scene', process=False).to_geometry().vertices
def floor(T):
    p = (T[:3, :3] @ Vf.T).T + T[:3, 3]; sd = p @ n_c + d_c
    return np.degrees(np.arccos(abs(T[:3, :3][:, 0] @ n_c))), np.percentile(sd, 0.2) * 100
# hand: right hand keypoints (21-41) closest-5 mean distance to mesh, in contact frames
kp = rec['human_keypoints_world'].astype(float); kpc = np.einsum('ij,fkj->fki', Tcw[:3, :3], kp) + Tcw[:3, 3]
tree = cKDTree(Vf[::5])
def hand(T, i):
    Ti = np.linalg.inv(T[i]); p = (Ti[:3, :3] @ kpc[i, 21:42].T).T + Ti[:3, 3]
    d, _ = tree.query(p); return np.sort(d)[:5].mean() * 100
cp = rec['contact_prob_right_left']
contact = cp.max(1) > 0.5
hb = np.array([hand(Tb, i) if contact[i] else np.nan for i in range(F)])
hr = np.array([hand(Tr, i) if contact[i] else np.nan for i in range(F)])

print('| phase | frames | IoU base | IoU ours | held-out IoU ours | rot step base/ours (deg, median) | trans step base/ours (cm, median) | rot change vs base (deg, median) | lateral shift (cm, median) | dz mean / median (cm) | ADD(ours,base) mean (cm) | hand-handle dist base/ours (cm, median) |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|')
for k, (a, b) in PH.items():
    sl = slice(a, b)
    ho = ('%.3f' % I_ho[a:b][np.arange(a, b) % 2 == 1].mean()) if I_ho is not None else '-'
    hbm = np.nanmedian(hb[sl]) if np.isfinite(hb[sl]).any() else np.nan
    hrm = np.nanmedian(hr[sl]) if np.isfinite(hr[sl]).any() else np.nan
    print(f'| {k} | {a}-{b - 1} | {I_b[sl].mean():.3f} | {I_r[sl].mean():.3f} | {ho} | {np.nanmedian(drot_b[sl]):.2f} / {np.nanmedian(drot_r[sl]):.2f} | '
          f'{np.nanmedian(dtr_b[sl]):.2f} / {np.nanmedian(dtr_r[sl]):.2f} | {np.median(rot_vs_b[sl]):.1f} | {np.median(lat[sl]):.2f} | '
          f'{dz[sl].mean():+.2f} / {np.median(dz[sl]):+.2f} | {add_rb[sl].mean():.2f} | {hbm:.1f} / {hrm:.1f} |')
print()
print('image offset ours-base (px): mean du %.2f dv %.2f' % tuple(dpx.mean(0)))
for name, (a, b) in [('S1', S1), ('S2', S2)]:
    tb = floor(S.mean_pose(Tb[a:b])); tr = floor(Tr[a])
    tbf = np.array([floor(T) for T in Tb[a:b]])
    print(f'{name}: floor tilt base-mean {tb[0]:.2f} deg / ours {tr[0]:.2f} deg; lowest point to plane base per-frame median {np.median(tbf[:,1]):.1f} cm (std {tbf[:,1].std():.1f}) / ours {tr[1]:.1f} cm')
    print(f'{name}: baseline camera-z std within phase {Tb[a:b,2,3].std()*100:.2f} cm, ours {Tr[a:b,2,3].std()*100:.2f} cm')
if I_ho is not None:
    print('held-out (odd frames) IoU: holdout-run final %.4f, holdout-run before depth re-anchoring %.4f, baseline %.4f' % (I_ho[1::2].mean(), I_hofit[1::2].mean(), I_b[1::2].mean()))
np.savez('/work/scratch/an/validate_out.npz', I_b=I_b, I_r=I_r, rot_vs_b=rot_vs_b, dz=dz, add_rb=add_rb)

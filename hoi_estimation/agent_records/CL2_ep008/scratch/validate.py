import sys, numpy as np, torch, h5py, cv2, json
sys.path.insert(0, '/work/outputs')
from solve import *

def main(res_path, out_json=None):
    K, Twc, Tw, rec, plane, mesh = load_inputs('/work/inputs')
    F = len(Tw); Tcw = np.linalg.inv(Twc)
    Tb = Tcw[None] @ Tw
    Tr = Tcw[None] @ np.load(res_path)['T_world_mesh']
    dbg = np.load('/work/scratch/run1/debug.npz'); a, b = int(dbg['a']), int(dbg['b'])
    phases = {'A_static': (0, a), 'moving': (a, b), 'B_static': (b, F), 'all': (0, F)}
    h5 = h5py.File('/work/inputs/masks.h5', 'r')
    R = CropRenderer(mesh); Kt = torch.tensor(K, dtype=torch.float32, device='cuda')
    idx = list(range(0, F, 2))
    ib = np.zeros(F) * np.nan; ir = np.zeros(F) * np.nan
    for k in range(0, len(idx), 96):
        sub = idx[k:k + 96]; d = MaskData(h5, sub)
        ib[sub] = eval_iou(R, Kt, d, Tb[sub]); ir[sub] = eval_iou(R, Kt, d, Tr[sub])
    leg_n, tips = leg_plane_normal(mesh)
    n, dpl = plane[:3], plane[3]
    hv = np.load('/work/inputs/frame0_reference_human.npz')['vertices_world']
    floor_ref = np.sort(hv @ n + dpl)[:50].mean()   # sole level of the true frame-0 human w.r.t. released plane
    def floor(T):
        Tw_ = Twc[None] @ T
        L = np.einsum('fij,nj->fni', Tw_[:, :3, :3], tips) + Tw_[:, None, :3, 3]
        h = L @ n + dpl
        up = np.einsum('fji,j->fi', Tw_[:, :3, :3], n)
        tilt = np.degrees(np.arccos(np.clip(np.abs(up @ leg_n), 0, 1)))
        return h, tilt
    hb, tb = floor(Tb); hr, tr = floor(Tr)
    # hand distance to chair surface (reconstructed human keypoints), frames with contact prob > .5
    kp = rec['human_keypoints_world']; cp = rec['contact_prob_right_left']
    V = np.asarray(mesh.vertices)[::20]
    def handdist(T):
        out = np.full((F, 2), np.nan)
        Tw_ = Twc[None] @ T
        for f in range(F):
            P = V @ Tw_[f, :3, :3].T + Tw_[f, :3, 3]
            for h, sl in enumerate([slice(21, 42), slice(42, 63)]):
                if cp[f, h] > 0.5:
                    hp = kp[f, sl].mean(0)
                    out[f, h] = np.min(np.linalg.norm(P - hp, axis=1))
        return out
    hdb, hdr = handdist(Tb), handdist(Tr)
    rows = []
    for name, (s0, s1) in phases.items():
        sl = slice(s0, s1)
        def stp(T):
            dr_ = rotdeg(T[s0:s1 - 1, :3, :3], T[s0 + 1:s1, :3, :3]); dt_ = np.linalg.norm(np.diff(T[sl, :3, 3], axis=0), axis=1)
            return dr_.mean(), dr_.max(), dt_.mean() * 100, dt_.max() * 100
        dz = (Tr[sl, 2, 3] - Tb[sl, 2, 3]) * 100
        row = dict(phase=name, frames=f'{s0}-{s1 - 1}',
                   iou_base=np.nanmean(ib[sl]), iou_ours=np.nanmean(ir[sl]),
                   step_rot_base=stp(Tb)[:2], step_rot_ours=stp(Tr)[:2], step_t_cm_base=stp(Tb)[2:], step_t_cm_ours=stp(Tr)[2:],
                   pos_std_cm_base=(Tb[sl, :3, 3].std(0) * 100).round(2).tolist(), pos_std_cm_ours=(Tr[sl, :3, 3].std(0) * 100).round(2).tolist(),
                   legs_above_plane_cm_base=hb[sl].mean() * 100, legs_above_plane_cm_ours=hr[sl].mean() * 100,
                   leg_spread_cm_base=(hb[sl].max(1) - hb[sl].min(1)).mean() * 100, leg_spread_cm_ours=(hr[sl].max(1) - hr[sl].min(1)).mean() * 100,
                   tilt_base=tb[sl].mean(), tilt_ours=tr[sl].mean(),
                   dz_mean_cm=dz.mean(), dz_median_cm=np.median(dz),
                   rot_diff_mean=rotdeg(Tb[sl, :3, :3], Tr[sl, :3, :3]).mean(), rot_diff_max=rotdeg(Tb[sl, :3, :3], Tr[sl, :3, :3]).max(),
                   hand_dist_cm_base=np.nanmean(hdb[sl]) * 100 if np.isfinite(hdb[sl]).any() else None,
                   hand_dist_cm_ours=np.nanmean(hdr[sl]) * 100 if np.isfinite(hdr[sl]).any() else None)
        rows.append(row)
    print('floor_ref (true human soles vs plane, cm): %.2f' % (floor_ref * 100))
    for r in rows:
        print(json.dumps({k: (np.round(v, 3).tolist() if isinstance(v, (float, np.floating, tuple)) else v) for k, v in r.items()}))
    np.savez('/work/scratch/val_arrays.npz', ib=ib, ir=ir, hb=hb, hr=hr, hdb=hdb, hdr=hdr)

if __name__ == '__main__':
    main(sys.argv[1])

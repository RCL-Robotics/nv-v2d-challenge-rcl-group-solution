"""Validation without ground truth: compares two pose files (baseline vs result) per phase.
usage: python validate.py base.npz result.npz  (base may be the literal 'base' = reconstruction.npz)"""
import sys, json, numpy as np, torch, h5py, trimesh, av
import torch.nn.functional as Fn
sys.path.insert(0, '/work/outputs')
import solve as S
from scipy.spatial.transform import Rotation as Rot
I = '/work/inputs'
C = S.Ctx()
cam = json.load(open(f'{I}/camera.json')); C.K = np.array(cam['K']); C.T_wc = np.array(cam['T_world_cam']); C.T_cw = np.linalg.inv(C.T_wc)
pl = json.load(open(f'{I}/ground_plane.json'))['plane']; C.PN = np.array(pl[:3]); C.PD = pl[3]
C.E1 = np.cross(C.PN, [0, 0, 1.]); C.E1 /= np.linalg.norm(C.E1); C.E2 = np.cross(C.PN, C.E1)
rec = np.load(f'{I}/reconstruction.npz'); kp = rec['human_keypoints_world'].astype(float)
C.mesh = trimesh.load(f'{I}/object_mesh.glb', force='scene', process=False).to_geometry()
with h5py.File(f'{I}/masks.h5') as f: C.obj = f['object'][:]; C.per = f['person'][:]
F = len(C.obj); area = C.obj.reshape(F, -1).sum(1); C.bb = np.zeros((F, 4))
for i in range(F):
    if area[i] > 0:
        ys = np.nonzero(C.obj[i].any(1))[0]; xs = np.nonzero(C.obj[i].any(0))[0]; C.bb[i] = [xs[0], ys[0], xs[-1], ys[-1]]
C.table = np.zeros((1152, 1536), bool)
A = rec['T_world_mesh'].astype(float) if sys.argv[1] == 'base' else np.load(sys.argv[1])['T_world_mesh']
B = np.load(sys.argv[2])['T_world_mesh']
rd = S.Renderer(C.mesh)
vis = area > S.MIN_MASK_AREA
# texture frames: decode crops for visible frames with area>400
need = set(i for i in range(F) if area[i] > S.TEX_MIN_AREA)
C.crops, C.crop_box = {}, {}
cont = av.open(f'{I}/video.mp4')
for i, frm in enumerate(cont.decode(video=0)):
    if i in need:
        img = frm.to_ndarray(format='rgb24'); b = C.bb[i]
        s_ = int(max(b[2] - b[0], b[3] - b[1]) * 1.25); x0, y0 = S.crop_box((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, s_)
        C.crops[i] = img[y0:y0 + s_, x0:x0 + s_].copy(); C.crop_box[i] = (x0, y0, s_)

def sil_metrics(P, frames):
    cs = [S.bbox_center_size(C, i) for i in frames]
    out = []
    for k in range(0, len(frames), 64):
        fr = frames[k:k + 64]
        d = S.crop_data(C, fr, [c for c, _ in cs[k:k + 64]], [s for _, s in cs[k:k + 64]])
        T = torch.tensor(np.stack([C.T_cw @ P[i] for i in fr]), dtype=torch.float32, device='cuda')
        out.append(S.iou_of(rd, d, T))
    return np.concatenate(out) if out else np.array([])

def tex_score(P, frames):
    """body-texture NCC at the given pose (single yaw = the pose's own)"""
    sc = []
    for i in frames:
        img = C.crops[i]; x0, y0, s_ = C.crop_box[i]; out = 192; s = out / s_
        Kt = torch.tensor(S.crop_K(C, x0, y0, s)[None], dtype=torch.float32, device='cuda')
        im = Fn.interpolate(torch.tensor(img, device='cuda').float().permute(2, 0, 1)[None] / 255, size=(out, out), mode='bicubic')[0].permute(1, 2, 0)
        om = Fn.interpolate(torch.tensor(C.obj[i, y0:y0 + s_, x0:x0 + s_], device='cuda').float()[None, None], size=(out, out))[0, 0]
        pm = Fn.interpolate(torch.tensor(C.per[i, y0:y0 + s_, x0:x0 + s_], device='cuda').float()[None, None], size=(out, out))[0, 0]
        with torch.no_grad():
            rgb, m, bl = rd.textured(torch.tensor((C.T_cw @ P[i])[None], dtype=torch.float32, device='cuda'), Kt, out, out)
            w = m.float() * om * (1 - pm); w = -Fn.max_pool2d(-w[:, None], 7, 1, 3)[:, 0]
            blk = Fn.max_pool2d(bl.float()[:, None], 9, 1, 4)[:, 0]
            sc.append(S.ncc_w(S.gray_gmag(rgb), S.gray_gmag(im[None]), w * (1 - blk)).item())
    return np.array(sc)

def hidden_violation(P, frames, center):
    """fraction of rendered object pixels NOT covered by the person mask, in frames without an object mask"""
    vals = []
    for i in frames:
        Tc = C.T_cw @ P[i]; u = C.K @ Tc[:3, 3]; cx, cy = u[0] / u[2], u[1] / u[2]
        x0, y0 = S.crop_box(cx, cy, 200); Kc = S.crop_K(C, x0, y0, 1.0)
        with torch.no_grad():
            m = rd.soft_mask(torch.tensor(Tc[None], dtype=torch.float32, device='cuda'), torch.tensor(Kc[None], dtype=torch.float32, device='cuda'), 200, 200)[0].cpu().numpy() > 0.5
        p = C.per[i, y0:y0 + 200, x0:x0 + 200]
        vals.append((m & ~p).sum() / max(m.sum(), 1))
    return np.array(vals)

def jumps(P, lo, hi):
    dr = np.array([np.degrees(np.linalg.norm(Rot.from_matrix(P[i][:3, :3].T @ P[i - 1][:3, :3]).as_rotvec())) for i in range(lo + 1, hi + 1)])
    dt = np.array([np.linalg.norm(P[i][:3, 3] - P[i - 1][:3, 3]) for i in range(lo + 1, hi + 1)]) * 100
    return dr, dt

def bottom_h(P, frames):
    return np.array([S.plane_coords(C, P[i])[2] for i in frames])

def tilt(P, frames):
    return np.array([np.degrees(np.arccos(np.clip(P[i][:3, 1] @ C.PN, -1, 1))) for i in frames])

def hand_dist(P, frames, base):
    pts = C.mesh.vertices[::40]
    return np.array([np.min(np.linalg.norm(kp[i, base:base + 21][:, None] - (pts @ P[i][:3, :3].T + P[i][:3, 3])[None], axis=2)) for i in frames])

phases = [('P1 table static', S.P1), ('P2 carry-1', S.P2), ('P3 floor static', S.P3), ('P4a carry-2 visible', (436, 745)),
          ('P4b carry-2 hidden', (746, 851)), ('P5 floor static (hidden)', S.P5)]
rows = []
for name, (lo, hi) in phases:
    fr = [i for i in range(lo, hi + 1) if vis[i]]
    frt = [i for i in fr if area[i] > S.TEX_MIN_AREA]
    r = {'phase': name, 'frames': f'{lo}-{hi}', 'n_vis': len(fr)}
    for tag, P in [('base', A), ('ours', B)]:
        if fr:
            iou = sil_metrics(P, fr); r[f'iou_{tag}'] = float(iou.mean())
            r[f'iou_odd_{tag}'] = float(sil_metrics(P, [i for i in fr if i % 2 == 1]).mean())
        if frt: r[f'tex_{tag}'] = float(tex_score(P, frt).mean())
        dr, dt = jumps(P, lo, hi)
        r[f'rot_step_med_{tag}'] = float(np.median(dr)); r[f'n_rot_jumps>10deg_{tag}'] = int((dr > 10).sum())
        r[f'trans_step_med_cm_{tag}'] = float(np.median(dt))
        r[f'tilt_med_{tag}'] = float(np.median(tilt(P, range(lo, hi + 1))))
        if 'floor' in name: r[f'bottom_h_med_{tag}'] = float(np.median(bottom_h(P, range(lo, hi + 1))))
        hidden = [i for i in range(lo, hi + 1) if area[i] == 0]
        if hidden: r[f'hidden_violation_{tag}'] = float(hidden_violation(P, hidden[::3], None).mean())
        if 'carry' in name:
            hb = 21 if 'carry-1' in name else 42
            r[f'hand_dist_med_cm_{tag}'] = float(np.median(hand_dist(P, list(range(lo, hi + 1, 3)), hb)) * 100)
    rows.append(r); print(json.dumps(r))
json.dump(rows, open(sys.argv[3] if len(sys.argv) > 3 else '/work/outputs/validation/validation.json', 'w'), indent=1)

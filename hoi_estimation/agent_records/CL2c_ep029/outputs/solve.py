#!/usr/bin/env python
"""Trajectory-level re-estimation of the rolling-luggage pose (episode 29).

Method (see REPORT.md):
  1. phases from the SAM2 object mask motion (static / moving);
  2. physical model: the luggage stands on its 4 wheels on the released floor plane the whole episode
     (verified by eye: it is only rolled by a foot push, never lifted or tilted). Its pose is therefore
     yaw about the floor normal + 2D floor position; static phases share ONE pose;
  3. global render-and-compare (nvdiffrast silhouettes vs. the object mask, person pixels ignored) of all
     frames jointly, with an acceleration penalty on yaw/position and a global image-scale nuisance
     factor (the object appears ~4% larger than the metric mesh at the floor-consistent distance);
  4. the 180-deg yaw flip hypothesis is tested per static phase and the better one kept;
  5. depth re-anchoring: camera-frame positions are scaled along the viewing rays so that the median
     camera depth of each static phase equals the baseline's (moving frames: linear interpolation of
     the factor). No extra depth shift is applied (cues disagree, see REPORT.md).
Run:  python solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz [--validate]
"""
import argparse, json, time, os, sys
import numpy as np, h5py, trimesh, torch, roma, cv2
import nvdiffrast.torch as dr
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot
from scipy.ndimage import median_filter, binary_closing, binary_opening, binary_dilation, label

np.random.seed(0); torch.manual_seed(0)
DEV = 'cuda'

# ---------------------------------------------------------------- constants (with reasons)
BODY_TOP_Z = 0.235   # mesh-file z above which only the telescopic handle exists (vertex cross-section
                     # shrinks from 0.33x0.23 m to two 2 cm rods at z=0.235); the SAM2 mask excludes the rods
N_DEC_FACES = 20000  # decimated body mesh for silhouettes
MOVE_THR_PX = 2.5    # mask bbox/centroid change over +-3 frames (median-filtered) that counts as motion
STATIC_MARGIN = 3    # frames next to motion that are NOT forced into the static pose
W_ACC = 1e2          # weight of the acceleration penalty (yaw in rad x 0.5 m lever, position in m);
                     # chosen by held-out IoU (fit even frames, test odd): 1e4 underfits pushes, 2e1..1e2 plateau, 1e3 worse on push 1
ITERS = 600          # Adam with cosine-decayed learning rate -> converged, run-to-run stable result
LR = 2e-3


def tt(x):
    return torch.tensor(np.asarray(x), dtype=torch.float32, device=DEV)


def rot_mean(Rs):
    U, S, Vt = np.linalg.svd(Rs.sum(0)); D = np.diag([1, 1, np.linalg.det(U @ Vt)]); return U @ D @ Vt


def rot_a2b(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b); v = np.cross(a, b); c = a @ b
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]); return np.eye(3) + vx + vx @ vx / (1 + c)


def rotdeg(Ra, Rb):
    tr = np.trace(np.swapaxes(Ra, -1, -2) @ Rb, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


class SilRenderer:
    def __init__(self, V, Fc):
        self.v = tt(V); self.f = torch.tensor(np.asarray(Fc), dtype=torch.int32, device=DEV)
        self.glctx = dr.RasterizeCudaContext(device=DEV)

    @staticmethod
    def proj(K, x0, y0, W, H, near=0.05, far=20.):
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2] - x0, K[1, 2] - y0
        return tt([[2 * fx / W, 0, 2 * cx / W - 1, 0], [0, 2 * fy / H, 2 * cy / H - 1, 0],
                   [0, 0, (far + near) / (far - near), -2 * far * near / (far - near)], [0, 0, 1, 0]])

    def render(self, Rc, tc, P, H, W):
        vc = torch.einsum('bij,nj->bni', Rc, self.v) + tc[:, None, :]
        vh = torch.cat([vc, torch.ones_like(vc[..., :1])], -1)
        clip = torch.einsum('bij,bnj->bni', P, vh).contiguous()
        rast, _ = dr.rasterize(self.glctx, clip, self.f, (H, W))
        m = dr.antialias((rast[..., 3:4] > 0).float(), rast, clip, self.f)
        return m[..., 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='/work/inputs')
    ap.add_argument('--out', default='/work/outputs/object_poses_world.npz')
    ap.add_argument('--validate', action='store_true', help='also run a held-out fit (even frames) and print tables')
    ap.add_argument('--w_acc', type=float, default=W_ACC)
    args = ap.parse_args()
    t_start = time.time()
    INP = args.inputs
    rec = np.load(f'{INP}/reconstruction.npz')
    cam = json.load(open(f'{INP}/camera.json'))
    K = np.array(cam['K']); Twc = np.array(cam['T_world_cam']); Tcw = np.linalg.inv(Twc)
    pl = json.load(open(f'{INP}/ground_plane.json'))['plane']; NRM = np.array(pl[:3]); DPL = pl[3]
    if NRM @ np.array([0, -1., 0]) < 0: NRM, DPL = -NRM, -DPL      # normal pointing up (world is y-down)
    Tw_base = rec['T_world_mesh'].astype(np.float64); F = len(Tw_base)
    Tc_base = Tcw @ Tw_base
    hf = h5py.File(f'{INP}/masks.h5', 'r'); OMall = hf['object'][:]; PMall = hf['person'][:]

    # ------------------------------------------------------------ mesh: body silhouette, wheel geometry
    mesh = trimesh.load(f'{INP}/object_mesh.glb', force='scene', process=False).to_geometry()
    mm = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.faces), process=True)   # merge seams
    fz = mm.vertices[mm.faces][:, :, 2].max(1)
    om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mm.vertices), o3d.utility.Vector3iVector(mm.faces[fz < BODY_TOP_Z]))
    om.remove_unreferenced_vertices(); om = om.simplify_quadric_decimation(N_DEC_FACES)
    body = SilRenderer(np.asarray(om.vertices), np.asarray(om.triangles))
    V = np.asarray(mm.vertices)
    low = V[V[:, 2] < V[:, 2].min() + 0.03]; WP = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            s = low[(np.sign(low[:, 0] - np.median(low[:, 0])) == sx) & (np.sign(low[:, 1] - np.median(low[:, 1])) == sy)]
            WP.append(s[np.argmin(s[:, 2])])
    WP = np.array(WP); C_M = WP.mean(0)                   # wheel-contact centre (mesh frame)
    u = np.linalg.svd(WP - C_M)[2][2]; U_M = u if u[2] > 0 else -u   # wheel-plane normal (mesh "up")
    E1 = np.cross(NRM, [0, 0, 1.]); E1 /= np.linalg.norm(E1); E2 = np.cross(NRM, E1)

    # ------------------------------------------------------------ phases
    feat = []
    for i in range(F):
        ys, xs = np.nonzero(OMall[i]); feat.append([xs.mean(), ys.mean(), xs.min(), xs.max(), ys.min(), ys.max()])
    fm = median_filter(np.array(feat), size=(5, 1), mode='nearest')
    mv = np.array([np.abs(fm[min(F - 1, t + 3)] - fm[max(0, t - 3)]).max() for t in range(F)])
    moving = binary_opening(binary_closing(mv > MOVE_THR_PX, np.ones(9)), np.ones(5))
    lab, nst = label(~binary_dilation(moving, np.ones(2 * STATIC_MARGIN + 1)))
    seg = np.zeros(F, int); k = 0; static_ids = []; segmap = {}
    for i in range(F):
        if lab[i] > 0:
            if lab[i] not in segmap: segmap[lab[i]] = k; static_ids.append(k); k += 1
            seg[i] = segmap[lab[i]]
        else:
            seg[i] = k; k += 1
    nseg = k
    # report phases: static runs and moving runs (unmargined)
    edges = np.flatnonzero(np.diff(moving.astype(int))) + 1
    bounds = [0] + list(edges) + [F]
    phases = [(bounds[j], bounds[j + 1], 'moving' if moving[bounds[j]] else 'static') for j in range(len(bounds) - 1)]
    print('phases:', phases)

    # ------------------------------------------------------------ upright parametrisation, init from baseline
    def upright(Tw):
        Rw = Tw[:3, :3]; Rb = rot_a2b(Rw @ U_M, NRM) @ Rw
        cw = Rw @ C_M + Tw[:3, 3]; return Rb, cw - (cw @ NRM + DPL) * NRM
    a0 = np.flatnonzero(seg == static_ids[0])
    Tm = np.eye(4); Tm[:3, :3] = rot_mean(Tw_base[a0, :3, :3]); Tm[:3, 3] = np.median(Tw_base[a0, :3, 3], 0)
    Rb, cw0 = upright(Tm)
    yaw_b = np.zeros(F); p_b = np.zeros((F, 2))
    for i in range(F):
        Rbi, cwi = upright(Tw_base[i])
        yaw_b[i] = Rot.from_matrix(Rbi @ Rb.T).as_rotvec() @ NRM
        dc = cwi - cw0; p_b[i] = [dc @ E1, dc @ E2]
    yaw_b = np.unwrap(yaw_b)
    yaw0 = np.array([np.median(yaw_b[seg == s]) for s in range(nseg)])
    p0 = np.array([np.median(p_b[seg == s], 0) for s in range(nseg)])

    # crop around all object masks
    un = OMall.any(0); ys, xs = np.nonzero(un); mg = 40
    x0, y0 = max(0, xs.min() - mg), max(0, ys.min() - mg)
    W, H = min(OMall.shape[2], xs.max() + mg) - x0, min(OMall.shape[1], ys.max() + mg) - y0
    P = SilRenderer.proj(K, x0, y0, W, H)
    Og = torch.tensor(OMall[:, y0:y0 + H, x0:x0 + W], device=DEV).float()
    Vg = 1 - torch.tensor(PMall[:, y0:y0 + H, x0:x0 + W], device=DEV).float()
    del OMall, PMall
    Rb_t, cw_t, n_t, e1_t, e2_t, cm_t = tt(Rb), tt(cw0), tt(NRM), tt(E1), tt(E2), tt(C_M)
    Rcw_t, tcw_t = tt(Tcw[:3, :3]), tt(Tcw[:3, 3])

    def cam_from(yaw, p, ls):
        Ry = roma.rotvec_to_rotmat(n_t[None] * yaw[:, None]); Rw = Ry @ Rb_t[None]
        cw = cw_t[None] + p[:, 0:1] * e1_t + p[:, 1:2] * e2_t
        s = torch.exp(ls)
        Rc = Rcw_t @ Rw * s
        tc = torch.einsum('ij,bj->bi', Rcw_t, cw - (Rw @ cm_t) * s) + tcw_t
        return Rc, tc, Rw, cw - Rw @ cm_t

    def iou_loss(idx, yaw, p, ls, hard=False):
        Rc, tc, _, _ = cam_from(yaw, p, ls)
        m = body.render(Rc, tc, P[None].expand(len(idx), -1, -1), H, W)
        if hard: m = (m > 0.5).float()
        o = Og[idx]; v = Vg[idx]
        return (m * o * v).sum((1, 2)) / ((m + o - m * o) * v).sum((1, 2))

    # ------------------------------------------------------------ flip hypotheses per static phase
    # The body silhouette is nearly symmetric under a 180-deg yaw (after re-fitting position), so the
    # flip test also uses the telescopic handle (not in the SAM2 mask): chamfer distance between the
    # rendered handle outline and image edges. A flip is adopted only if BOTH cues favour it clearly.
    hm = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mm.vertices), o3d.utility.Vector3iVector(mm.faces[fz >= BODY_TOP_Z]))
    hm.remove_unreferenced_vertices(); hm = hm.simplify_quadric_decimation(3000)
    handle = SilRenderer(np.asarray(hm.vertices), np.asarray(hm.triangles))
    y0h = max(0, y0 - 220); Hh = y0 + H - y0h
    Ph = SilRenderer.proj(K, x0, y0h, W, Hh)
    sub_all = {s: np.flatnonzero(seg == s)[::max(1, (seg == s).sum() // 15)] for s in static_ids}
    need = set(np.concatenate(list(sub_all.values())).tolist())
    cap = cv2.VideoCapture(f'{INP}/video.mp4'); edt = {}
    for i in range(F):
        ok, fr_img = cap.read()
        if not ok: break
        if i in need:
            g = cv2.cvtColor(fr_img, cv2.COLOR_BGR2GRAY)[y0h:y0h + Hh, x0:x0 + W]
            edt[i] = cv2.distanceTransform((~(cv2.Canny(g, 30, 90) > 0)).astype(np.uint8), cv2.DIST_L2, 3)
    cap.release()
    hfile = h5py.File(f'{INP}/masks.h5', 'r')

    def handle_chamfer(idx, yaw, p, ls):
        with torch.no_grad():
            Rc, tc, _, _ = cam_from(yaw, p, ls)
            mh = handle.render(Rc, tc, Ph[None].expand(len(idx), -1, -1), Hh, W).cpu().numpy() > 0.5
            mb = body.render(Rc, tc, Ph[None].expand(len(idx), -1, -1), Hh, W).cpu().numpy() > 0.5
        out = []
        for j, i in enumerate(idx):
            om_ = hfile['object'][i][y0h:y0h + Hh, x0:x0 + W]; pm_ = hfile['person'][i][y0h:y0h + Hh, x0:x0 + W]
            e = (cv2.Canny(((mh[j] & ~mb[j]) * 255).astype(np.uint8), 50, 150) > 0) & ~om_ & ~pm_
            out.append(np.minimum(edt[i][e], 10).mean() if e.sum() > 20 else 10.0)
        return float(np.mean(out))

    flip_log = []
    for s in static_ids:
        sub = sub_all[s]; res = []
        for dyaw in [0.0, np.pi]:
            y = torch.tensor([float(yaw0[s] + dyaw)], dtype=torch.float32, device=DEV, requires_grad=True)
            pp = torch.tensor(p0[s:s + 1], dtype=torch.float32, device=DEV, requires_grad=True)
            ls = torch.zeros((), device=DEV, requires_grad=True)
            opt = torch.optim.Adam([y, pp, ls], lr=3e-3)
            for it in range(150):
                l = 1 - iou_loss(sub, y.expand(len(sub)), pp.expand(len(sub), -1), ls).mean()
                opt.zero_grad(); l.backward(); opt.step()
            with torch.no_grad():
                sc = iou_loss(sub, y.expand(len(sub)), pp.expand(len(sub), -1), ls, hard=True).mean().item()
            ch = handle_chamfer(sub, y.detach().expand(len(sub)), pp.detach().expand(len(sub), -1), ls.detach())
            res.append((sc, ch, y.item(), pp.detach().cpu().numpy()[0]))
        flip = (res[1][0] > res[0][0] + 0.02) and (res[1][1] < res[0][1] - 0.5)
        flip_log.append((s, res[0][0], res[0][1], res[1][0], res[1][1], flip))
        if flip:
            yaw0[s] = res[1][2]; p0[s] = res[1][3]
    print('flip test per static phase (seg, IoU/handle-chamfer px kept, IoU/chamfer yaw+180, flipped):')
    for f_ in flip_log: print('   seg %d: keep %.3f / %.2f px   flip %.3f / %.2f px   -> %s' % f_)
    if any(f_[5] for f_ in flip_log):
        # unwrap: shift the yaw of all frames after a flipped static phase consistently is ambiguous;
        # re-initialise moving frames by linear interpolation between the static phases
        sf = np.array([np.flatnonzero(seg == s).mean() for s in static_ids])
        for s in range(nseg):
            if s in static_ids: continue
            t = np.flatnonzero(seg == s)[0]
            yaw0[s] = np.interp(t, sf, yaw0[static_ids]); p0[s] = [np.interp(t, sf, p0[static_ids, j]) for j in range(2)]

    # ------------------------------------------------------------ joint optimisation
    seg_t = torch.tensor(seg, device=DEV)

    def joint_fit(data_frames, iters=ITERS, w_acc=args.w_acc, chunk=120):
        yaw = torch.tensor(yaw0, dtype=torch.float32, device=DEV, requires_grad=True)
        p = torch.tensor(p0, dtype=torch.float32, device=DEV, requires_grad=True)
        ls = torch.zeros((), device=DEV, requires_grad=True)
        opt = torch.optim.Adam([yaw, p, ls], lr=LR)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: 0.02 + 0.98 * 0.5 * (1 + np.cos(np.pi * min(it, iters) / iters)))
        for it in range(iters):
            opt.zero_grad(); tot = 0.
            for c in range(0, len(data_frames), chunk):
                idx = data_frames[c:c + chunk]; it_ = torch.as_tensor(idx, device=DEV)
                l = (1 - iou_loss(idx, yaw[seg_t[it_]], p[seg_t[it_]], ls)).sum() / len(data_frames)
                l.backward(); tot += l.item()
            yf = yaw[seg_t]; pf = p[seg_t]
            acc = (yf[2:] - 2 * yf[1:-1] + yf[:-2]) ** 2 * 0.25 + ((pf[2:] - 2 * pf[1:-1] + pf[:-2]) ** 2).sum(1)
            (w_acc * acc.mean()).backward(); opt.step(); sched.step()
        print('  fit: data %.4f  image-scale %.3f' % (tot, torch.exp(ls).item()))
        return yaw.detach(), p.detach(), ls.detach()

    def eval_iou(yaw, p, ls=None, Tcams=None, chunk=120):
        out = []
        with torch.no_grad():
            for c in range(0, F, chunk):
                idx = np.arange(c, min(F, c + chunk)); it_ = torch.as_tensor(idx, device=DEV)
                if Tcams is None:
                    Rc, tc, _, _ = cam_from(yaw[seg_t[it_]], p[seg_t[it_]], ls)
                else:
                    T = tt(Tcams[idx]); Rc, tc = T[:, :3, :3], T[:, :3, 3]
                m = (body.render(Rc, tc, P[None].expand(len(idx), -1, -1), H, W) > 0.5).float()
                o = Og[idx]; v = Vg[idx]
                out.append(((m * o * v).sum((1, 2)) / (((m + o) > 0).float() * v).sum((1, 2))).cpu().numpy())
        return np.concatenate(out)

    def to_world(yaw, p):
        with torch.no_grad():
            _, _, Rw, tw = cam_from(yaw[seg_t], p[seg_t], torch.zeros((), device=DEV))
        T = np.tile(np.eye(4), (F, 1, 1)); T[:, :3, :3] = Rw.cpu().numpy(); T[:, :3, 3] = tw.cpu().numpy(); return T

    def reanchor(Tw_new):
        """scale camera-frame positions along the rays: per static phase median z = baseline median z;
        moving frames get a linearly interpolated factor."""
        Tcn = Tcw @ Tw_new; ks = []; cs = []
        for s in static_ids:
            fr = np.flatnonzero(seg == s)
            ks.append(np.median(Tc_base[fr, 2, 3]) / np.median(Tcn[fr, 2, 3])); cs.append(fr.mean())
        kf = np.interp(np.arange(F), cs, ks)
        Tcn[:, :3, 3] *= kf[:, None]
        return Twc @ Tcn, kf

    Ib = eval_iou(None, None, Tcams=Tc_base)
    if args.validate:
        print('held-out validation: fit on even frames, evaluate on odd frames')
        yv, pv, lv = joint_fit(np.arange(0, F, 2))
        Iv = eval_iou(yv, pv, lv)
        Tv, _ = reanchor(to_world(yv, pv)); Iv1 = eval_iou(None, None, Tcams=Tcw @ Tv)
        print('| phase | frames | IoU base (odd) | IoU ours+scale (odd) | IoU ours final pose (odd) |')
        for a, b, kind in phases:
            sel = np.zeros(F, bool); sel[a:b] = True; sel &= (np.arange(F) % 2 == 1)
            print('| %s | %d-%d | %.3f | %.3f | %.3f |' % (kind, a, b - 1, Ib[sel].mean(), Iv[sel].mean(), Iv1[sel].mean()))

    yaw, p, ls = joint_fit(np.arange(F))
    Is = eval_iou(yaw, p, ls)
    Tw_new, kf = reanchor(to_world(yaw, p))
    Tc_new = Tcw @ Tw_new
    I1 = eval_iou(None, None, Tcams=Tc_new)
    np.savez(args.out, T_world_mesh=Tw_new.astype(np.float64))

    # ------------------------------------------------------------ diagnostics table
    Vb = V[::7]
    def wheel_h(T): return (((Vb @ T[:3, :3].T + T[:3, 3]) @ NRM) + DPL).min()
    print('re-anchor factors per frame: min %.4f max %.4f' % (kf.min(), kf.max()))
    print('| phase | frames | IoU base | IoU ours (+scale nuisance) | IoU ours final | rot diff ours-base mean/max (deg) | '
          'rot jitter base / ours (deg/frame) | z diff ours-base mean / median (cm) | wheel height above plane base / ours (cm) |')
    for a, b, kind in phases:
        fr = np.arange(a, b)
        rd = rotdeg(Tw_base[fr, :3, :3], Tw_new[fr, :3, :3])
        jb = rotdeg(Tw_base[fr[:-1], :3, :3], Tw_base[fr[1:], :3, :3]).mean()
        jn = rotdeg(Tw_new[fr[:-1], :3, :3], Tw_new[fr[1:], :3, :3]).mean()
        dz = (Tc_new[fr, 2, 3] - Tc_base[fr, 2, 3]) * 100
        hb = np.median([wheel_h(Tw_base[i]) for i in fr[::3]]) * 100; hn = np.median([wheel_h(Tw_new[i]) for i in fr[::3]]) * 100
        print('| %s | %d-%d | %.3f | %.3f | %.3f | %.1f / %.1f | %.2f / %.2f | %.1f / %.1f | %.1f / %.1f |' % (
            kind, a, b - 1, Ib[fr].mean(), Is[fr].mean(), I1[fr].mean(), rd.mean(), rd.max(), jb, jn, dz.mean(), np.median(dz), hb, hn))
    print('total |z diff| episode: mean %.2f cm' % (np.abs(Tc_new[:, 2, 3] - Tc_base[:, 2, 3]).mean() * 100))
    print('max GPU mem %.2f GB, run time %.1f s' % (torch.cuda.max_memory_allocated() / 1e9, time.time() - t_start))


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""Trajectory-level re-estimation of the 6-DoF pose of the blue easter basket (episode 2).

Run:  python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz

Method (details, reasoning and validation in REPORT.md):
  A. Phases (constants below, verified by eye + automatic mask-stability check printed at run time):
     table-static P1 / carry-1 P2 / floor-static P3 / carry-2 P4 (visible, then hidden behind the body) /
     floor-static P5 (hidden behind the legs).
  B. Geometry from silhouettes (nvdiffrast, SAM2 masks, person pixels and - while carried - the table top ignored):
     static phases: ONE upright pose per phase fitted to the pooled silhouettes (floor phases: bottom on the
     released floor plane; table phase: baseline depth along the viewing ray kept);
     carried frames: per-frame refinement of rotation + image-plane translation at the temporally smoothed baseline
     depth, then SE(3) smoothing.
  C. Yaw about the basket's symmetry axis (the hard part; a silhouette of a near-rotationally symmetric bucket does
     not constrain it): render-and-compare of the mesh texture (gradient-magnitude NCC, wire handle excluded) for
     every yaw hypothesis in every visible frame ->
       - static phases: pooled curve -> 45-deg periodic bunny-pattern phase -> snap a coarse yaw to it;
       - carry-2: Viterbi tracking of the bunny/hole stripe pattern through time (relative yaw), absolute offset
         from two anchors (the floor-static yaw at pick-up and the CoCoNet window that agrees best with the track);
       - carry-1 (small, blurred, occluded): smooth yaw interpolation between the two static phases.
  D. Hidden carry: basket follows the holding hand (MHR wrist position + hand rotation about the vertical; tilt kept
     from the last reliable visible frames), then a minimal image-plane shift keeps it behind the person silhouette
     (no object pixel is visible in those frames); blended into the placement pose. P5: upright on the floor, fitted
     to the sliver silhouettes + "hidden behind the person" in later frames; yaw = mean of hand propagation and
     CoCoNet (sliver frames).
"""
import argparse, json, time
import numpy as np
import torch, torch.nn.functional as Fn
import h5py, trimesh, cv2, av
import nvdiffrast.torch as dr
import roma
from scipy.spatial.transform import Rotation as Rot, Slerp
from scipy.ndimage import gaussian_filter1d, median_filter

torch.manual_seed(0); np.random.seed(0)
torch.backends.cudnn.deterministic = True

# ----------------------------------------------------------------------------------------------------------------
# Phase constants (frame indices, inclusive). Verified by eye on the video and by mask IoU w.r.t. a reference frame
# of the phase (printed at run time): basket leaves the table at 123, is set on the floor at ~223 (hand still on it
# until ~265), lifted at 436, carried (handover ~520-530), hidden behind the body from ~746, placed on the floor
# behind the legs at ~847-851 (hand releases), visible as a sliver left of the legs until 867, then hidden.
P1 = (0, 122)       # static on the round table
P2 = (123, 222)     # carried (1st carry)
P3 = (223, 435)     # static on the floor
P4 = (436, 851)     # carried (2nd carry, handover, hidden behind the body ~746-846, placement)
P5 = (852, 979)     # static on the floor behind the person's legs (hidden after 867)
P5_SLIVER = (847, 867)   # frames around/after placement where a sliver of the basket is visible next to the legs
# Coarse yaw for the table phase: CoCoNet yaw while the basket is being lifted off the table (frames 123-153; it
# cannot have rotated much yet). The CoCoNet estimate during P1 itself (-115 deg) places the scanned wooden grip in
# front of the basket, but the averaged P1 image has no tan pixels on the basket at all (checked: 2 of 2337 px),
# and it contradicts CoCoNet's own estimates at 123-153 (120-137 deg) and FoundationPose at 105-123 (118-147 deg).
P1_COARSE_FRAMES = (123, 153)
FLOOR_H = 0.01      # bottom of basket above the released floor plane [m]; the true frame-0 human's lowest vertex is
                    # 2.3 cm above that plane (shoe soles), i.e. the plane is 0-2 cm below the true floor.
MIN_MASK_AREA = 150 # px; frames with smaller object masks are treated as not visible
TEX_MIN_AREA = 400  # px; frames used for texture (yaw) scoring
HAND_BLOCK = 42     # MHR70 keypoint block (42-62) of the hand holding the basket in carry-2 (closest hand 540-745)
AX_C = np.array([0.0043, 0.0, 0.0032])   # bucket symmetry axis (mesh x,z) from circle fits to rim / bottom vertices
BOT_Y = -0.0666                           # mesh y of the bucket bottom (mesh +y = opening)
YAWS = np.arange(-180, 180, 2.0)


def log(*a):
    print('[solve %6.1fs]' % (time.time() - T0), *a, flush=True)


def wrapd(a):
    return (np.asarray(a) + 180.) % 360. - 180.


class Ctx:
    pass


# ----------------------------------------------------------------------------------------------------------------
class Renderer:
    def __init__(self, mesh):
        self.v = torch.tensor(mesh.vertices, dtype=torch.float32, device='cuda')
        self.f = torch.tensor(mesh.faces, dtype=torch.int32, device='cuda')
        uv = np.array(mesh.visual.uv, dtype=np.float32).copy(); uv[:, 1] = 1 - uv[:, 1]
        self.uv = torch.tensor(uv, device='cuda')
        img = mesh.visual.material.baseColorTexture.convert('RGB')
        img = img.resize((1024, int(1024 * img.size[1] / img.size[0])))
        self.tex = torch.tensor(np.array(img, dtype=np.float32) / 255., device='cuda')[None].contiguous()
        # wire handle / grip / lugs: faces sticking out of the bucket body profile (rim lip excluded by height).
        # The handle is a movable part, so it is excluded from texture scoring.
        fc = mesh.vertices[mesh.faces].mean(1); r = np.hypot(fc[:, 0] - AX_C[0], fc[:, 2] - AX_C[2])
        rb = 0.088 + (fc[:, 1] - BOT_Y) / 0.1556 * (0.1135 - 0.088)
        self.bail = torch.tensor(np.r_[False, (r - rb > 0.005) & (fc[:, 1] < 0.080)], device='cuda')
        self.ctx = dr.RasterizeCudaContext()

    def _clip(self, Tcm, Kc, H, W, near=0.05, far=20.):
        p = torch.einsum('bij,nj->bni', Tcm[:, :3, :3], self.v) + Tcm[:, None, :3, 3]
        z = p[..., 2]
        x = Kc[:, None, 0, 0] * p[..., 0] / z + Kc[:, None, 0, 2]
        y = Kc[:, None, 1, 1] * p[..., 1] / z + Kc[:, None, 1, 2]
        return torch.stack([(2 * x / W - 1) * z, (2 * y / H - 1) * z, ((z - near) / (far - near) * 2 - 1) * z, z], -1).contiguous()

    def soft_mask(self, Tcm, Kc, H, W):
        clip = self._clip(Tcm, Kc, H, W)
        rast, _ = dr.rasterize(self.ctx, clip, self.f, (H, W))
        m = (rast[..., 3:] > 0).float()
        return dr.antialias(m, rast, clip, self.f)[..., 0]

    def textured(self, Tcm, Kc, H, W):
        clip = self._clip(Tcm, Kc, H, W)
        rast, _ = dr.rasterize(self.ctx, clip, self.f, (H, W))
        uvi, _ = dr.interpolate(self.uv[None], rast, self.f)
        rgb = dr.texture(self.tex, uvi, filter_mode='linear')
        tid = rast[..., 3].long()
        return rgb, tid > 0, self.bail[tid]


def crop_K(C, x0, y0, s):
    Kc = C.K.copy(); Kc[0, 2] -= x0; Kc[1, 2] -= y0; Kc[:2] *= s; Kc[0, 2] += 0.5 * s; Kc[1, 2] += 0.5 * s
    return Kc


def crop_box(cx, cy, S):
    return int(np.clip(cx - S / 2, 0, 1536 - S)), int(np.clip(cy - S / 2, 0, 1152 - S))


def crop_data(C, frames, centers, sizes, R=128, use_table=False):
    Ks, om, vis = [], [], []
    for k, i in enumerate(frames):
        S = int(sizes[k]); x0, y0 = crop_box(*centers[k], S); s = R / S
        Ks.append(crop_K(C, x0, y0, s))
        o = torch.tensor(C.obj[i, y0:y0 + S, x0:x0 + S]).float()[None, None]
        p = C.per[i, y0:y0 + S, x0:x0 + S].copy()
        if use_table:
            p |= C.table[y0:y0 + S, x0:x0 + S]
        p = torch.tensor(p).float()[None, None]
        om.append(Fn.interpolate(o, size=(R, R), mode='area')[0, 0])
        vis.append((Fn.interpolate(p, size=(R, R), mode='area')[0, 0] < 0.5).float())
    d = Ctx()
    d.R = R; d.K = torch.tensor(np.stack(Ks), dtype=torch.float32, device='cuda')
    d.om = torch.stack(om).cuda(); d.vis = torch.stack(vis).cuda()
    return d


def bbox_center_size(C, i, scale=2.0):
    b = C.bb[i]
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), max(b[2] - b[0], b[3] - b[1], 20) * scale


def sil_loss(rd, d, Tc):
    m = rd.soft_mask(Tc.contiguous(), d.K, d.R, d.R)
    return (((m - d.om) ** 2) * d.vis).sum((1, 2)) / d.R


def iou_of(rd, d, Tc):
    with torch.no_grad():
        m = (rd.soft_mask(Tc.contiguous(), d.K, d.R, d.R) > 0.5).float(); o = (d.om > 0.5).float()
        return ((m * o * d.vis).sum((1, 2)) / (((m + o) > 0).float() * d.vis).sum((1, 2)).clamp(min=1)).cpu().numpy()


# ----------------------------------------------------------------------------------------------------------------
def yaw_deg(C, Rw):
    x = Rw[:, 0]
    return np.degrees(np.arctan2(x @ C.E2, x @ C.E1))


def circ_median(a_deg):
    a = np.radians(np.asarray(a_deg))
    d = np.abs(np.angle(np.exp(1j * (a[:, None] - a[None]))))
    return np.degrees(a[np.argmin(d.sum(1))])


def circ_mean(a_deg, w=None):
    a = np.radians(np.asarray(a_deg)); w = np.ones_like(a) if w is None else np.asarray(w)
    return np.degrees(np.angle(np.sum(w * np.exp(1j * a))))


def rot_about_axis(T, ang_deg):
    """rotate the object about its own symmetry axis (mesh +y through AX_C)"""
    Ry = np.eye(4); Ry[:3, :3] = Rot.from_euler('y', ang_deg, degrees=True).as_matrix()
    Tc = np.eye(4); Tc[:3, 3] = AX_C; Ti = np.eye(4); Ti[:3, 3] = -AX_C
    return T @ Tc @ Ry @ Ti


def set_yaw(C, Tw, target):
    T = Tw
    for _ in range(3):   # (yaw of a tilted basket is not exactly linear in the axis rotation)
        T = rot_about_axis(T, wrapd(target - yaw_deg(C, T[:3, :3])))
    return T


def upright_pose_t(C, yaw, a, b, h):
    """torch: world pose of an upright basket (mesh +y = up), yaw in rad, axis foot at plane coords (a,b), bottom at h"""
    up = torch.tensor(C.PN, dtype=torch.float32, device='cuda')
    e1 = torch.tensor(C.E1, dtype=torch.float32, device='cuda'); e2 = torch.tensor(C.E2, dtype=torch.float32, device='cuda')
    n = yaw.shape[0]
    x = torch.cos(yaw)[:, None] * e1 + torch.sin(yaw)[:, None] * e2
    z = torch.cross(x, up.expand_as(x), dim=1)
    R = torch.stack([x, up.expand_as(x), z], 2)
    ax = torch.tensor([AX_C[0], BOT_Y, AX_C[2]], dtype=torch.float32, device='cuda')
    P = torch.tensor(-C.PD * C.PN, dtype=torch.float32, device='cuda') + a[:, None] * e1 + b[:, None] * e2 + h[:, None] * up
    T = torch.eye(4, device='cuda').repeat(n, 1, 1); T[:, :3, :3] = R; T[:, :3, 3] = P - R @ ax
    return T


def plane_coords(C, Tw):
    P = Tw[:3, :3] @ np.array([AX_C[0], BOT_Y, AX_C[2]]) + Tw[:3, 3]
    q = P + C.PD * C.PN
    return q @ C.E1, q @ C.E2, P @ C.PN + C.PD


def fit_static(C, rd, frames, yaw, init_Tw, mode, h=FLOOR_H, hidden_frames=(), iters=250):
    """One upright pose for all frames (yaw given, deg). mode='floor': bottom at height h on the floor plane, fit the
    plane position (a,b). mode='ray': keep the camera depth of init_Tw's axis foot, fit the image-plane shift."""
    cs = [bbox_center_size(C, i) for i in frames]
    d = crop_data(C, frames, [c for c, _ in cs], [s for _, s in cs])
    B = len(frames)
    Tcw = torch.tensor(C.T_cw, dtype=torch.float32, device='cuda')
    a0, b0, h0 = plane_coords(C, init_Tw)
    yaw_t = torch.tensor([np.radians(yaw)], dtype=torch.float32, device='cuda')
    if mode == 'floor':
        p = torch.tensor([a0, b0], dtype=torch.float32, device='cuda', requires_grad=True)
        hh = torch.tensor([h], dtype=torch.float32, device='cuda')
        def pose(): return upright_pose_t(C, yaw_t, p[0:1], p[1:2], hh)
    else:
        T_init = upright_pose_t(C, yaw_t, torch.tensor([a0], device='cuda').float(), torch.tensor([b0], device='cuda').float(),
                                torch.tensor([h0], device='cuda').float())
        p = torch.zeros(2, device='cuda', requires_grad=True)
        Rwc = Tcw[:3, :3].T
        def pose():
            T = T_init.clone()
            T[0, :3, 3] = T[0, :3, 3] + Rwc @ torch.stack([p[0], p[1], torch.zeros_like(p[0])])
            return T
    hd = None
    if len(hidden_frames):
        # frames where the object is invisible: rendered object pixels must be covered by the person mask
        cm = (np.mean([c[0] for c, _ in cs]), np.mean([c[1] for c, _ in cs]))
        hd = crop_data(C, hidden_frames, [cm] * len(hidden_frames), [160] * len(hidden_frames))
    opt = torch.optim.Adam([p], lr=2e-3 if mode == 'floor' else 1e-3)
    for it in range(iters):
        Tw = pose()
        loss = sil_loss(rd, d, (Tcw @ Tw).expand(B, -1, -1)).mean()
        if hd is not None:
            m = rd.soft_mask((Tcw @ Tw).expand(len(hidden_frames), -1, -1).contiguous(), hd.K, hd.R, hd.R)
            loss = loss + (m * hd.vis).sum((1, 2)).mean() / hd.R
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Tw = pose()
        iou = iou_of(rd, d, (Tcw @ Tw).expand(B, -1, -1))
    return Tw[0].cpu().numpy().astype(np.float64), iou


def tex_targets(C, frames, out):
    """per-frame image gradient-magnitude, object mask and visibility at the texture-crop geometry"""
    Ks, gis, oms, vis = [], [], [], []
    for i in frames:
        x0, y0, S = C.crop_box[i]; s = out / S
        Ks.append(crop_K(C, x0, y0, s))
        im = torch.tensor(C.crops[i], device='cuda').float().permute(2, 0, 1)[None] / 255
        im = Fn.interpolate(im, size=(out, out), mode='bicubic', align_corners=False)[0].permute(1, 2, 0)
        gis.append(gray_gmag(im[None])[0])
        oms.append(Fn.interpolate(torch.tensor(C.obj[i, y0:y0 + S, x0:x0 + S], device='cuda').float()[None, None], size=(out, out), mode='area')[0, 0])
        vis.append((Fn.interpolate(torch.tensor(C.per[i, y0:y0 + S, x0:x0 + S], device='cuda').float()[None, None], size=(out, out), mode='area')[0, 0] < 0.5).float())
    d = Ctx(); d.R = out; d.K = torch.tensor(np.stack(Ks), dtype=torch.float32, device='cuda')
    d.gi = torch.stack(gis); d.om = torch.stack(oms); d.vis = torch.stack(vis)
    return d


def tex_ncc(rd, d, Tc):
    """differentiable body-texture NCC (gradient magnitude) + soft silhouette loss in the texture crops"""
    clip = rd._clip(Tc.contiguous(), d.K, d.R, d.R)
    rast, _ = dr.rasterize(rd.ctx, clip, rd.f, (d.R, d.R))
    uvi, _ = dr.interpolate(rd.uv[None], rast, rd.f)
    rgb = dr.texture(rd.tex, uvi, filter_mode='linear')
    m = (rast[..., 3:] > 0).float(); ms = dr.antialias(m, rast, clip, rd.f)[..., 0]
    bl = rd.bail[rast[..., 3].long()].float()
    w = (m[..., 0] * d.om * d.vis).detach(); w = -Fn.max_pool2d(-w[:, None], 7, 1, 3)[:, 0]
    blk = Fn.max_pool2d(bl[:, None], 9, 1, 4)[:, 0]
    ncc = ncc_w(gray_gmag(rgb), d.gi, w * (1 - blk))
    sil = (((ms - d.om) ** 2) * d.vis).sum((1, 2)) / d.R
    return ncc, sil


def refine_frames(C, rd, frames, Tc_init, w_rot=5.0, iters=150, R=128, chunk=64, use_table=True, w_tex=0.0, tex_ok=None):
    """per-frame refinement: rotation about the object centre (regularised to the init) + image-plane shift (camera
    depth fixed); objective = silhouette loss (+ w_tex * (-body-texture NCC) for frames with a texture crop)."""
    out, ious0, ious1 = [], [], []
    c = torch.tensor(C.mesh.bounds.mean(0), dtype=torch.float32, device='cuda')
    for s in range(0, len(frames), chunk):
        fr = frames[s:s + chunk]; B = len(fr)
        cs = [bbox_center_size(C, i) for i in fr]
        d = crop_data(C, fr, [c_ for c_, _ in cs], [s_ for _, s_ in cs], R=R, use_table=use_table)
        tf = [k for k, i in enumerate(fr) if w_tex > 0 and i in C.crops and (tex_ok is None or tex_ok(i))]
        td = tex_targets(C, [fr[k] for k in tf], 160) if tf else None
        T0 = torch.tensor(Tc_init[s:s + chunk], dtype=torch.float32, device='cuda')
        rv = torch.zeros(B, 3, device='cuda', requires_grad=True); dxy = torch.zeros(B, 2, device='cuda', requires_grad=True)
        def pose():
            Rd = roma.rotvec_to_rotmat(rv)
            ctr = torch.einsum('bij,j->bi', T0[:, :3, :3], c) + T0[:, :3, 3]
            Rn = Rd @ T0[:, :3, :3]
            z = ctr[:, 2:3]
            ctr2 = ctr + torch.cat([dxy * z / 3.0, torch.zeros_like(z)], 1)
            ctr2 = ctr2 * (z / ctr2[:, 2:3])     # keep camera depth
            T = torch.eye(4, device='cuda').repeat(B, 1, 1); T[:, :3, :3] = Rn; T[:, :3, 3] = ctr2 - torch.einsum('bij,j->bi', Rn, c)
            return T
        ious0.append(iou_of(rd, d, T0))
        opt = torch.optim.Adam([rv, dxy], lr=2e-3)
        for it in range(iters):
            T = pose()
            loss = sil_loss(rd, d, T).sum() + w_rot * (rv ** 2).sum() * R
            if td is not None:
                ncc, _ = tex_ncc(rd, td, T[tf])
                loss = loss - w_tex * ncc.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            T = pose()
        ious1.append(iou_of(rd, d, T)); out.append(T.cpu().numpy().astype(np.float64))
    return np.concatenate(out), np.concatenate(ious0), np.concatenate(ious1)


# ----------------------------------------------------------------------------------------------------------------
def gray_gmag(img):  # (B,H,W,3) -> gradient magnitude (B,H,W)
    g = Fn.avg_pool2d(img.mean(-1, keepdim=True).permute(0, 3, 1, 2), 3, 1, 1)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=g.dtype, device=g.device)[None, None] / 8
    gx = Fn.conv2d(g, kx, padding=1); gy = Fn.conv2d(g, kx.transpose(2, 3), padding=1)
    return torch.sqrt(gx[:, 0] ** 2 + gy[:, 0] ** 2 + 1e-8)


def ncc_w(a, b, w):
    n = w.sum((1, 2)).clamp(min=1); am = (a * w).sum((1, 2)) / n; bm = (b * w).sum((1, 2)) / n
    a_ = (a - am[:, None, None]) * w; b_ = (b - bm[:, None, None]) * w
    return (a_ * b_).sum((1, 2)) / torch.sqrt((a_ ** 2).sum((1, 2)) * (b_ ** 2).sum((1, 2)) + 1e-9)


def yaw_curve(C, rd, i, Tw, out=192, chunk=60):
    """texture agreement (gradient-magnitude NCC over bucket-body pixels, handle excluded) for every yaw in YAWS,
    other pose parameters fixed to Tw"""
    img = C.crops[i]; x0, y0, S = C.crop_box[i]; s = out / S
    Kt = torch.tensor(crop_K(C, x0, y0, s)[None], dtype=torch.float32, device='cuda')
    im = torch.tensor(img, device='cuda').float().permute(2, 0, 1)[None] / 255
    im = Fn.interpolate(im, size=(out, out), mode='bicubic', align_corners=False)[0].permute(1, 2, 0)
    om = Fn.interpolate(torch.tensor(C.obj[i, y0:y0 + S, x0:x0 + S], device='cuda').float()[None, None], size=(out, out))[0, 0]
    pm = Fn.interpolate(torch.tensor(C.per[i, y0:y0 + S, x0:x0 + S], device='cuda').float()[None, None], size=(out, out))[0, 0]
    gi = gray_gmag(im[None])
    res = []
    with torch.no_grad():
        for c in range(0, len(YAWS), chunk):
            Ts = np.stack([C.T_cw @ set_yaw(C, Tw, y) for y in YAWS[c:c + chunk]]); B = len(Ts)
            rgb, m, bl = rd.textured(torch.tensor(Ts, dtype=torch.float32, device='cuda'), Kt.expand(B, -1, -1), out, out)
            w = m.float() * om * (1 - pm); w = -Fn.max_pool2d(-w[:, None], 7, 1, 3)[:, 0]
            blk = Fn.max_pool2d(bl.float()[:, None], 9, 1, 4)[:, 0]
            res.append(ncc_w(gray_gmag(rgb), gi.expand(B, -1, -1), w * (1 - blk)).cpu().numpy())
    return np.concatenate(res)


def stripe_phase(curve, period=45.):
    """phase (deg, in [0,period)) of the period-periodic component of a yaw curve: its maxima are phase + k*period"""
    c = np.sum(curve * np.exp(-2j * np.pi * YAWS / period))
    return (np.degrees(np.angle(c)) / 360. * period) % period, np.abs(c) / len(YAWS)


def snap(yaw, phase, period=45.):
    """nearest angle to yaw of the form phase + k*period"""
    return yaw + ((phase - yaw + period / 2) % period - period / 2)


def track_stripes(frames, curves, alpha=0.02, maxstep=4.):
    """Viterbi tracking of the texture-stripe pattern (envelope removed) through time -> relative yaw path"""
    H = curves - gaussian_filter1d(curves, 10, axis=1, mode='wrap')
    S = len(YAWS); D = np.abs(wrapd(YAWS[:, None] - YAWS[None, :]))
    acc = H[0].copy(); bp = []
    for j in range(1, len(frames)):
        tr = alpha * D; tr[D > maxstep * (frames[j] - frames[j - 1])] = 1e9
        tot = acc[:, None] - tr; b = np.argmax(tot, 0); acc = tot[b, np.arange(S)] + H[j]; bp.append(b)
    p = [int(np.argmax(acc))]
    for b in bp[::-1]: p.append(int(b[p[-1]]))
    return YAWS[np.array(p[::-1])]


# ----------------------------------------------------------------------------------------------------------------
def smooth_se3(Ts, sigma):
    """Gaussian smoothing of a pose sequence: translation directly, rotation via sign-aligned quaternions."""
    Ts = np.array(Ts)
    t = gaussian_filter1d(Ts[:, :3, 3], sigma, axis=0, mode='nearest')
    q = Rot.from_matrix(Ts[:, :3, :3]).as_quat()
    for k in range(1, len(q)):
        if q[k] @ q[k - 1] < 0: q[k] = -q[k]
    q = gaussian_filter1d(q, sigma, axis=0, mode='nearest'); q /= np.linalg.norm(q, axis=1, keepdims=True)
    out = Ts.copy(); out[:, :3, :3] = Rot.from_quat(q).as_matrix(); out[:, :3, 3] = t
    return out


def robust_rot_track(Rs, win=9, thr_deg=25.):
    """replace rotations deviating > thr from the medoid of their temporal window (isolated network spikes)"""
    Rs = np.array(Rs); n = len(Rs); out = Rs.copy(); nrep = 0
    for k in range(n):
        lo, hi = max(0, k - win // 2), min(n, k + win // 2 + 1)
        W = Rs[lo:hi]
        D = np.array([[np.linalg.norm(Rot.from_matrix(A.T @ B).as_rotvec()) for B in W] for A in W])
        med = W[np.argmin(D.sum(1))]
        if np.degrees(np.linalg.norm(Rot.from_matrix(med.T @ Rs[k]).as_rotvec())) > thr_deg:
            out[k] = med; nrep += 1
    return out, nrep


def hand_frame(kp, i, base):
    """hand frame from MHR70 hand keypoints: wrist base+20, index MCP base+7, middle MCP base+11, pinky MCP base+19"""
    w = kp[i, base + 20]; idx = kp[i, base + 7]; mid = kp[i, base + 11]; pk = kp[i, base + 19]
    x = mid - w; x /= np.linalg.norm(x)
    n = np.cross(idx - w, pk - w); n -= x * (n @ x); n /= np.linalg.norm(n)
    T = np.eye(4); T[:3, :3] = np.stack([x, n, np.cross(x, n)], 1); T[:3, 3] = w
    return T


def avg_pose(Ts):
    T = np.eye(4); T[:3, :3] = Rot.from_matrix(np.array([T_[:3, :3] for T_ in Ts])).mean().as_matrix()
    T[:3, 3] = np.median([T_[:3, 3] for T_ in Ts], 0)
    return T


def interp_pose(Ta, Tb, w):
    r = Slerp([0, 1], Rot.from_matrix(np.stack([Ta[:3, :3], Tb[:3, :3]])))([w]).as_matrix()[0]
    T = np.eye(4); T[:3, :3] = r; T[:3, 3] = (1 - w) * Ta[:3, 3] + w * Tb[:3, 3]
    return T


def hidden_fix(C, rd, out, frames, iters=100, R=128, S=200):
    """frames without any visible object pixel: the basket must be occluded by the person. Minimal image-plane
    shift (camera depth kept) that moves the rendered basket behind the person mask; temporal smoothness."""
    if not frames: return
    Tc0 = torch.tensor(np.stack([C.T_cw @ out[i] for i in frames]), dtype=torch.float32, device='cuda')
    cen = []
    for k, i in enumerate(frames):
        u = C.K @ Tc0[k, :3, 3].cpu().numpy(); cen.append((u[0] / u[2], u[1] / u[2]))
    d = crop_data(C, frames, cen, [S] * len(frames), R=R)
    notp = 1 - d.vis   # vis = not-person; we want rendered pixels inside the person mask
    dxy = torch.zeros(len(frames), 2, device='cuda', requires_grad=True)
    opt = torch.optim.Adam([dxy], lr=2e-3)
    for it in range(iters):
        T = Tc0.clone(); z = Tc0[:, 2, 3:4]
        T[:, :3, 3] = Tc0[:, :3, 3] + torch.cat([dxy * z / 3.0, torch.zeros_like(z)], 1)
        m = rd.soft_mask(T, d.K, R, R)
        viol = (m * (1 - notp)).sum((1, 2)) / R
        smooth = ((dxy[1:] - dxy[:-1]) ** 2).sum() * 1e4
        loss = viol.sum() + smooth + 10 * (dxy ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        T = Tc0.clone(); z = Tc0[:, 2, 3:4]
        T[:, :3, 3] = Tc0[:, :3, 3] + torch.cat([dxy * z / 3.0, torch.zeros_like(z)], 1)
    for k, i in enumerate(frames):
        out[i] = C.T_wc @ T[k].cpu().numpy().astype(np.float64)


# ----------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='/work/inputs')
    ap.add_argument('--out', default='/work/outputs/object_poses_world.npz')
    ap.add_argument('--diag', default=None, help='optional path for a diagnostics npz')
    args = ap.parse_args()
    I = args.inputs
    C = Ctx()
    cam = json.load(open(f'{I}/camera.json'))
    C.K = np.array(cam['K']); C.T_wc = np.array(cam['T_world_cam']); C.T_cw = np.linalg.inv(C.T_wc)
    pl = json.load(open(f'{I}/ground_plane.json'))['plane']; C.PN = np.array(pl[:3]); C.PD = pl[3]
    if C.PN @ C.T_wc[:3, 3] + C.PD < 0: C.PN, C.PD = -C.PN, -C.PD   # normal points to the camera side (up)
    C.E1 = np.cross(C.PN, [0, 0, 1.]); C.E1 /= np.linalg.norm(C.E1); C.E2 = np.cross(C.PN, C.E1)
    rec = np.load(f'{I}/reconstruction.npz')
    Tw_base = rec['T_world_mesh'].astype(np.float64); F = len(Tw_base)
    kp = rec['human_keypoints_world'].astype(np.float64)
    C.mesh = trimesh.load(f'{I}/object_mesh.glb', force='scene', process=False).to_geometry()
    base_yaw = np.array([yaw_deg(C, T[:3, :3]) for T in Tw_base])

    log('loading masks')
    with h5py.File(f'{I}/masks.h5') as f:
        C.obj = f['object'][:]; C.per = f['person'][:]
    area = C.obj.reshape(F, -1).sum(1)
    C.bb = np.zeros((F, 4))
    for i in range(F):
        if area[i] > 0:
            ys = np.nonzero(C.obj[i].any(1))[0]; xs = np.nonzero(C.obj[i].any(0))[0]
            C.bb[i] = [xs[0], ys[0], xs[-1], ys[-1]]
    visible = area > MIN_MASK_AREA

    log('decoding video')
    tex_frames = [i for i in range(F) if area[i] > TEX_MIN_AREA and (i <= P1[1] or P3[0] <= i <= P3[1] or P4[0] <= i <= 745)]
    tex_frames = [i for i in tex_frames if not (P1[0] <= i <= P1[1] or P3[0] <= i <= P3[1]) or i % 2 == 0]
    need = set(tex_frames)
    C.crops, C.crop_box = {}, {}
    cont = av.open(f'{I}/video.mp4')
    for i, frm in enumerate(cont.decode(video=0)):
        if i == 0:
            fr0 = frm.to_ndarray(format='rgb24')
        if i in need:
            img = frm.to_ndarray(format='rgb24'); b = C.bb[i]
            S = int(max(b[2] - b[0], b[3] - b[1]) * 1.25); x0, y0 = crop_box((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, S)
            C.crops[i] = img[y0:y0 + S, x0:x0 + S].copy(); C.crop_box[i] = (x0, y0, S)
    cont.close()

    # table occluder mask (static scene): beige table top segmented in frame 0
    hsv = cv2.cvtColor(fr0, cv2.COLOR_RGB2HSV)
    tm = ((hsv[..., 0] >= 15) & (hsv[..., 0] <= 26) & (hsv[..., 1] >= 40) & (hsv[..., 1] <= 80) & (hsv[..., 2] >= 170)).astype(np.uint8)
    roi = np.zeros_like(tm); roi[560:720, 640:1000] = 1; tm *= roi
    tm = cv2.morphologyEx(tm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(tm); k = 1 + np.argmax(st[1:, 4]); tm = (lab == k).astype(np.uint8)
    tm = cv2.morphologyEx(tm, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    C.table = cv2.dilate(tm, np.ones((9, 9), np.uint8)).astype(bool)   # incl. the darker table edge

    # automatic check of the phase constants: mask IoU with the phase's reference frame (unoccluded pixels)
    def ref_iou(ref, i):
        v = ~C.per[i]; a = C.obj[ref]; b = C.obj[i]
        return (a & b & v).sum() / max(((a | b) & v).sum(), 1)
    for (lo, hi), ref in [(P1, 30), (P3, 330)]:
        ious = [ref_iou(ref, i) for i in range(lo, hi + 1)]
        log('phase %d-%d: mask IoU to frame %d: min %.3f median %.3f; frame before/after: %.3f %.3f' %
            (lo, hi, ref, np.min(ious), np.median(ious), ref_iou(ref, lo - 1) if lo > 0 else 1, ref_iou(ref, hi + 1)))

    rd = Renderer(C.mesh)
    out = Tw_base.copy()
    src = np.array(['baseline'] * F, dtype=object)
    diag = {}

    # ================= B. geometry: carried frames (per-frame silhouette refinement) =================
    for (lo, hi), lab_ in [(P2, 'carry1'), (P4, 'carry2')]:
        frs = [i for i in range(lo, hi + 1) if visible[i]]
        Tc_b = np.stack([C.T_cw @ Tw_base[i] for i in frs])
        z = Tc_b[:, 2, 3].copy()
        zs = gaussian_filter1d(median_filter(z, size=9, mode='nearest'), 2.0, mode='nearest')
        Tc0 = Tc_b.copy(); Tc0[:, :3, 3] *= (zs / z)[:, None]
        Tc0[:, :3, :3], nrep = robust_rot_track(Tc0[:, :3, :3])
        Tc, i0, i1 = refine_frames(C, rd, frs, Tc0)
        log('%s: %d visible frames, %d rotation spikes replaced, silhouette IoU %.3f -> %.3f' % (lab_, len(frs), nrep, i0.mean(), i1.mean()))
        for k, i in enumerate(frs):
            if i1[k] > 0.6:
                out[i] = C.T_wc @ Tc[k]; src[i] = lab_

    # ================= C. yaw =================
    log('texture yaw curves for %d frames' % len(tex_frames))
    # static phases: geometry first with the baseline yaw (silhouette barely depends on yaw), then texture
    T_P1g, _ = fit_static(C, rd, [i for i in range(P1[0], P1[1] + 1, 2) if visible[i]], circ_median(base_yaw[P1[0]:P1[1] + 1]), Tw_base[60], 'ray')
    T_P3g, _ = fit_static(C, rd, [i for i in range(P3[0], P3[1] + 1, 2) if visible[i]], circ_median(base_yaw[P3[0]:P3[1] + 1]), Tw_base[330], 'floor')
    curves = {}
    for i in tex_frames:
        Tref = T_P1g if i <= P1[1] else (T_P3g if P3[0] <= i <= P3[1] else out[i])
        curves[i] = yaw_curve(C, rd, i, Tref)
    log('texture curves done')

    def pooled(P):
        return np.mean([curves[i] for i in tex_frames if P[0] <= i <= P[1]], 0)
    ph1, amp1 = stripe_phase(pooled(P1)); ph3, amp3 = stripe_phase(pooled(P3))
    coarse3 = circ_median(base_yaw[P3[0]:P3[1] + 1])
    yaw3 = snap(coarse3, ph3)
    coarse1 = circ_median(base_yaw[P1_COARSE_FRAMES[0]:P1_COARSE_FRAMES[1] + 1])
    yaw1 = snap(coarse1, ph1)
    log('P1 yaw: coarse %.1f (baseline P1 %.1f) stripe phase %.1f (amp %.4f) -> %.1f' % (coarse1, circ_median(base_yaw[P1[0]:P1[1] + 1]), ph1, amp1, yaw1))
    log('P3 yaw: coarse %.1f stripe phase %.1f (amp %.4f) -> %.1f' % (coarse3, ph3, amp3, yaw3))

    # carry-2: stripe tracking
    trk = [i for i in tex_frames if P4[0] <= i <= 745]
    path = track_stripes(np.array(trk), np.stack([curves[i] for i in trk]))
    c_a = wrapd(yaw3 - path[0])                       # anchor 1: floor-static yaw at pick-up
    offs = wrapd(base_yaw[trk] - path)                # anchor 2: CoCoNet window most consistent with the track
    best = None
    for s in range(0, len(trk) - 40):
        win = [k for k in range(s, len(trk)) if trk[k] - trk[s] < 90]
        if len(win) < 40: continue
        dd = wrapd(offs[win] - circ_median(offs[win])); iqr = np.subtract(*np.percentile(dd, [75, 25]))
        if best is None or iqr < best[0]: best = (iqr, trk[win[0]], trk[win[-1]], circ_median(offs[win]))
    c_b = best[3]
    c = circ_mean([c_a, c_b])
    c_snap = 22.5 * np.round(c / 22.5)   # the track follows a stripe maximum; stripes repeat every 22.5 deg
    log('carry-2 yaw track: anchor(P3 pick-up) %.1f, anchor(CoCoNet %d-%d, IQR %.1f) %.1f -> offset %.1f, snapped %.1f' %
        (c_a, best[1], best[2], best[0], c_b, c, c_snap))
    trk_yaw = dict(zip(trk, path + c_snap))
    diag.update(path=path, trk=np.array(trk), c_a=c_a, c_b=c_b, c=c_snap, yaw1=yaw1, yaw3=yaw3)

    # static poses with final yaw
    def static_phase(P, mode, yaw, init, hidden=()):
        frs = [i for i in range(P[0], P[1] + 1) if visible[i]]
        Tw, _ = fit_static(C, rd, frs[::2], yaw, init, mode, hidden_frames=hidden)
        out[P[0]:P[1] + 1] = Tw; src[P[0]:P[1] + 1] = 'static'
        a, b, h = plane_coords(C, Tw)
        log('static %s: yaw %.1f, bottom height above floor plane %.3f m' % (P, yaw, h))
        return Tw
    T_P1 = static_phase(P1, 'ray', yaw1, T_P1g)
    T_P3 = static_phase(P3, 'floor', yaw3, T_P3g)

    # apply yaw to carried frames
    vis2 = [i for i in range(P4[0], P4[1] + 1) if src[i] == 'carry2']
    ky = np.array(sorted(trk_yaw)); vy = np.unwrap(np.radians([trk_yaw[k] for k in ky]))
    for i in vis2:
        if i > 745: continue
        out[i] = set_yaw(C, out[i], np.degrees(np.interp(i, ky, vy)))
    vis1 = [i for i in range(P2[0], P2[1] + 1) if src[i] == 'carry1']
    dy = wrapd(yaw3 - yaw1)
    for i in vis1:   # carry-1: smooth yaw ramp between the two static phases (no reliable texture: small, blurred)
        w = (i - P1[1]) / (P3[0] - P1[1])
        out[i] = set_yaw(C, out[i], yaw1 + w * dy)

    # temporal smoothing of the carried segments; interpolate small gaps inside visible runs
    for lab_ in ['carry1', 'carry2']:
        idx = np.nonzero(src == lab_)[0]
        idx = idx[idx <= 745] if lab_ == 'carry2' else idx
        runs = np.split(idx, np.nonzero(np.diff(idx) > 3)[0] + 1)
        for r in runs:
            if len(r) < 3: continue
            full = np.arange(r[0], r[-1] + 1)
            for i in full:
                if src[i] != lab_:
                    a_ = r[r < i].max(); b_ = r[r > i].min()
                    out[i] = interp_pose(out[a_], out[b_], (i - a_) / (b_ - a_)); src[i] = lab_ + '_interp'
            out[full] = smooth_se3(out[full], sigma=1.5)
    # junctions static <-> carry: short slerp blend (3 frames) for continuity
    for edge, Ts, side in [(P2[0], T_P1, +1), (P2[1], T_P3, -1), (P4[0], T_P3, +1)]:
        for k in range(3):
            i = edge + side * k; w = (k + 1) / 4.
            out[i] = interp_pose(Ts, out[i], w)

    # ================= D. hidden carry + P5 =================
    vis4 = [i for i in range(P4[0], 746) if src[i] in ('carry2', 'carry2_interp')]
    last_vis = max(vis4)
    kps = gaussian_filter1d(kp, 3.0, axis=0, mode='nearest')
    hand_yaw = np.unwrap(np.radians([yaw_deg(C, hand_frame(kps, i, HAND_BLOCK)[:3, :3]) for i in range(F)]))
    wrist = kps[:, HAND_BLOCK + 20]
    good = [i for i in vis4 if area[i] > 1500]          # reliable (large, unoccluded) frames only
    anchor = [i for i in good if i >= good[-1] - 20]
    T_anchor = avg_pose([out[i] for i in anchor]); i_a = int(np.median(anchor))
    for i in range(anchor[-1] + 1, last_vis + 1):     # the last, barely visible frames: follow the hand as well
        src[i] = 'hand_attached'
    last_vis = anchor[-1]
    def hand_follow(i):
        """basket follows the wrist position and the hand's rotation about the vertical; tilt kept from the last
        visible frames (the full hand orientation from occluded keypoints is too noisy)"""
        dpsi = np.degrees(hand_yaw[i] - hand_yaw[i_a])
        Rz = Rot.from_rotvec(np.radians(dpsi) * C.PN).as_matrix()
        T = np.eye(4); T[:3, :3] = Rz @ T_anchor[:3, :3]
        T[:3, 3] = wrist[i] + Rz @ (T_anchor[:3, 3] - wrist[i_a])
        return T
    release = P5_SLIVER[0]
    yaw_hand = yaw_deg(C, hand_follow(release)[:3, :3])
    yaw_cc = circ_median(base_yaw[P5_SLIVER[0]:P5_SLIVER[1] + 1])
    yaw5 = circ_mean([yaw_hand, yaw_cc])
    log('P5 yaw: hand-propagated %.1f, CoCoNet (sliver frames) %.1f -> %.1f' % (yaw_hand, yaw_cc, yaw5))
    sl = [i for i in range(P5[0], P5_SLIVER[1] + 1) if visible[i]]
    hid = [i for i in range(P5_SLIVER[1] + 1, P5[1] + 1) if not visible[i]][::4]
    T_P5, iou5 = fit_static(C, rd, sl, yaw5, Tw_base[sl[len(sl) // 2]], 'floor', hidden_frames=hid)
    out[P5[0]:P5[1] + 1] = T_P5; src[P5[0]:P5[1] + 1] = 'static'
    log('static P5: sliver IoU %.3f, bottom height %.3f' % (iou5.mean(), plane_coords(C, T_P5)[2]))
    for i in range(last_vis + 1, P5[0]):
        w = np.clip((i - (P5[0] - 30)) / 30.0, 0, 1)   # blend into the placement pose over the last 30 frames
        out[i] = interp_pose(hand_follow(i), T_P5, w); src[i] = 'hand_attached'
    hidden_fix(C, rd, out, [i for i in range(last_vis + 1, P5[0]) if not visible[i]])
    seg = np.arange(last_vis - 5, P5[0])
    out[seg] = smooth_se3(out[seg], sigma=3.0)
    out[P5[0]:] = T_P5
    diag.update(yaw5=yaw5, yaw_hand=yaw_hand, yaw_cc=yaw_cc)

    np.savez(args.out, T_world_mesh=out.astype(np.float64))
    log('wrote', args.out, out.shape)
    if args.diag:
        np.savez(args.diag, src=src.astype(str), T_P1=T_P1, T_P3=T_P3, T_P5=T_P5, **diag)


if __name__ == '__main__':
    T0 = time.time()
    main()

#!/usr/bin/env python3
"""Recover a monocular iron trajectory with re-segmentation and render-and-compare.

The supplied baseline mask segmented a tripod standing behind the iron.  This
solver therefore re-prompts SAM2 on the iron itself, then exploits the task's
physical structure: a fixed lying pose while sliding, a smooth slerp to the
final upright pose, and constant poses in the two untouched intervals.
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.signal import medfilt, savgol_filter
from scipy.spatial.transform import Rotation as Rot

# The package is installed in the container but is not always on sys.path when
# this file is invoked from /work/outputs.
sys.path.insert(0, "/workspace/v2d_sam2")
from v2d.sam2.lib.video_to_masks import video_to_masks  # noqa: E402


def set_deterministic(seed=7):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cv2.setRNGSeed(seed)


def slerp(a, b, s):
    qa, qb = a.as_quat(), b.as_quat()
    if np.dot(qa, qb) < 0:
        qb = -qb
    th = np.arccos(np.clip(np.dot(qa, qb), -1.0, 1.0))
    if th < 1e-8:
        return Rot.from_quat(qa)
    q = (np.sin((1.0 - s) * th) * qa + np.sin(s * th) * qb) / np.sin(th)
    return Rot.from_quat(q)


def clean_component(mask):
    """Discard tiny disconnected SAM blobs but keep split visible parts."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.where(areas >= max(20, 0.015 * areas.max()))[0] + 1
    return np.isin(lab, keep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    set_deterministic()
    t_start = time.time()
    inp = Path(args.inputs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- inputs
    with open(inp / "episode.json") as f:
        n_frames = int(json.load(f)["frames"])
    with open(inp / "camera.json") as f:
        cam = json.load(f)
    K = np.asarray(cam["K"], dtype=np.float64)
    T_wc = np.asarray(cam["T_world_cam"], dtype=np.float64)
    T_cw = np.linalg.inv(T_wc)
    width, height = int(cam["width"]), int(cam["height"])

    mesh = trimesh.load(inp / "object_mesh.glb", force="scene", process=False).to_geometry()
    V = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device="cuda")
    faces_i = np.asarray(mesh.faces, dtype=np.int32)
    F = torch.tensor(faces_i, dtype=torch.int32, device="cuda")
    UV = torch.tensor(np.asarray(mesh.visual.uv), dtype=torch.float32, device="cuda")
    texture = torch.tensor(
        np.asarray(mesh.visual.material.baseColorTexture), dtype=torch.float32, device="cuda"
    )[None] / 255.0
    T_cw_t = torch.tensor(T_cw, dtype=torch.float32, device="cuda")
    glctx = dr.RasterizeCudaContext()

    # ------------------------------------------------------- re-segmentation
    # The frame-0 iron occupies x=685..754, y=613..657.  This tight box excludes
    # the tripod (the cause of the supplied object-mask failure).  It was chosen
    # from the frame-0 image and is intentionally explicit and episode-specific.
    with tempfile.TemporaryDirectory(prefix="iron_solve_") as td:
        td = Path(td)
        prompt = {
            "prompts": [{
                "frame_index": 0,
                "object_id": 0,
                "points": None,
                "point_labels": None,
                "box": {"x0": 665.0, "y0": 610.0, "x1": 780.0, "y1": 675.0},
                "mask_path": None,
            }]
        }
        prompt_path = td / "prompt.json"
        prompt_path.write_text(json.dumps(prompt))
        os.environ["CONFIG_FILE"] = "configs/sam2.1/sam2.1_hiera_l.yaml"
        os.environ["CHECKPOINT_FILE"] = "sam2.1_hiera_large.pt"
        masks_dir = td / "masks"
        video_to_masks(str(inp / "video.mp4"), str(prompt_path), str(masks_dir), "/weights/sam2")

        def obj_mask(frame):
            m = cv2.imread(str(masks_dir / "0" / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0
            return clean_component(m)

        cap = cv2.VideoCapture(str(inp / "video.mp4"))
        h5 = h5py.File(inp / "masks.h5", "r")

        # ------------------------------------------------------- pose helpers
        def load_frame(frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {frame}")
            return image

        def render(R, Ts, Kc, W, H):
            """Render object masks/colours for world mesh poses Ts (N,4,4)."""
            Ts = torch.tensor(np.asarray(Ts), dtype=torch.float32, device="cuda")
            Tcam = torch.eye(4, device="cuda")[None].repeat(len(Ts), 1, 1)
            Tcam = T_cw_t[None] @ Ts
            Vc = V[None] @ Tcam[:, :3, :3].transpose(1, 2) + Tcam[:, None, :3, 3]
            z = Vc[..., 2:3]
            uv = Vc[..., :2] / z * torch.tensor(
                [Kc[0, 0], Kc[1, 1]], dtype=torch.float32, device="cuda"
            ) + torch.tensor([Kc[0, 2], Kc[1, 2]], dtype=torch.float32, device="cuda")
            ndc = torch.stack([2 * uv[..., 0] / W - 1, 2 * uv[..., 1] / H - 1], dim=-1)
            # z/w is normalized into the rasterizer's clip range.  Using z as
            # both depth and w (instead of z=w) is essential for correct z-buffer
            # ordering and perspective interpolation.
            pos = torch.cat([ndc * z, ((z - 5.0) / 5.0) * z, z], dim=-1)
            rast, _ = dr.rasterize(glctx, pos, F, resolution=(H, W))
            icolor = dr.interpolate(UV, rast, F)[0]
            color = dr.texture(texture, icolor, filter_mode="linear")
            return rast[..., 3] > 0, color

        def fit_fixed(frame, R, z_values, offset=(0.0, 0.0)):
            """Fit translation/depth for a known rotation by render comparison.

            Translation is first placed on the target-mask centre ray, then the
            rendered bbox centre is aligned exactly.  Person-occluded pixels are
            excluded from false-positive costs.
            """
            image = load_frame(frame)
            obj = obj_mask(frame)
            person = h5["person"][frame]
            ys, xs = np.where(obj)
            pad = 36
            x0, y0 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
            x1 = min(width, int(xs.max()) + pad + 1)
            y1 = min(height, int(ys.max()) + pad + 1)
            W, H = x1 - x0, y1 - y0
            Kc = K.copy()
            Kc[0, 2] -= x0
            Kc[1, 2] -= y0
            tgt = torch.tensor(obj[y0:y1, x0:x1], dtype=torch.bool, device="cuda")
            pers = torch.tensor(person[y0:y1, x0:x1], dtype=torch.bool, device="cuda")
            roi = torch.tensor(
                cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2RGB),
                dtype=torch.float32, device="cuda",
            ) / 255.0
            tuc = (xs.min() + xs.max()) / 2.0 + offset[0]
            tvc = (ys.min() + ys.max()) / 2.0 + offset[1]
            zs = np.asarray(z_values, dtype=np.float64)

            def make_ts(centers_u, centers_v):
                ans = []
                for u, v, z in zip(centers_u, centers_v, zs):
                    X = (u - K[0, 2]) / K[0, 0] * z
                    Y = (v - K[1, 2]) / K[1, 1] * z
                    t = T_wc[:3, :3] @ np.array([X, Y, z]) + T_wc[:3, 3]
                    Tm = np.eye(4)
                    Tm[:3, :3] = R
                    Tm[:3, 3] = t
                    ans.append(Tm)
                return ans

            ts = make_ts([tuc] * len(zs), [tvc] * len(zs))
            pred, _ = render(R, ts, Kc, W, H)
            us2, vs2 = [], []
            for j in range(len(zs)):
                yy, xx = torch.where(pred[j])
                pu = (xx.min() + xx.max()).float().item() / 2.0 + x0
                pv = (yy.min() + yy.max()).float().item() / 2.0 + y0
                us2.append(tuc - (pu - tuc))
                vs2.append(tvc - (pv - tvc))
            ts = make_ts(us2, vs2)
            pred, color = render(R, ts, Kc, W, H)
            best = None
            for j in range(len(zs)):
                inter = (pred[j] & tgt).sum().item()
                fp = (pred[j] & ~tgt & ~pers).sum().item()
                fn = (tgt & ~pred[j]).sum().item()
                iou = inter / max(inter + fp + fn, 1)
                mi = (pred[j] & tgt)[..., None].float()
                den = mi.sum().item() * 3.0 + 1e-6
                l1 = ((color[j] - roi).abs() * mi).sum().item() / den
                ncc_den = ((color[j] ** 2 * mi).sum().sqrt() *
                           (roi ** 2 * mi).sum().sqrt()).item()
                ncc = (color[j] * roi * mi).sum().item() / (ncc_den + 1e-6)
                score = iou + 0.25 * ncc - 0.30 * l1
                if best is None or score > best[0]:
                    best = (score, iou, l1, ncc, float(zs[j]), ts[j])
            return best

        # Fixed orientations (verified by silhouette + texture/SIFT comparisons):
        # lying pose yaw=51deg,tilt=3deg,roll=0deg in the object's local frame;
        # upright orientation yaw=319deg,tilt=10deg,roll=-25deg.
        B_flat = np.column_stack([
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, -1.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        ])
        R_flat = (Rot.from_euler("y", 51, degrees=True).as_matrix()
                  @ Rot.from_euler("x", 3, degrees=True).as_matrix() @ B_flat)
        B_up = np.column_stack([
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, -1.0, 0.0]),
        ])
        R_upright_raw = (Rot.from_euler("y", 319, degrees=True).as_matrix()
                         @ Rot.from_euler("x", 10, degrees=True).as_matrix()
                         @ Rot.from_euler("z", -25, degrees=True).as_matrix() @ B_up)
        # The static upright pose is 95% along the observed lying->upright motion.
        R_final = slerp(Rot.from_matrix(R_flat),
                        Rot.from_matrix(R_upright_raw), 0.95).as_matrix()

        # Phase boundaries are explicit because the masks show exactly zero
        # motion before f159, a long hand-driven slide, stand-up during 660..794,
        # and a constant final pose from f795 onward.
        F_STATIC_FLAT_END = 159
        F_TRANSITION_START = 660
        F_TRANSITION_END = 794
        F_STATIC_UP_START = 795

        # ------------------------------------------------- lying/sliding fits
        z_flat = np.arange(2.00, 3.801, 0.05)
        flat_fits = {}
        # Static interval: one anchor pose serves all untouched frames.
        anchor = fit_fixed(0, R_flat, z_flat, offset=(0.0, 0.0))
        # Sliding: correct the small systematic SAM bbox-centre bias measured
        # against the rendered mesh centre (+2,+1 pixels).
        for f in range(F_STATIC_FLAT_END + 1, F_TRANSITION_START):
            flat_fits[f] = fit_fixed(f, R_flat, z_flat, offset=(2.0, 1.0))
            if f % 50 == 0:
                print(f"[flat] {f}: IoU={flat_fits[f][1]:.3f}", flush=True)

        # ------------------------------------------------ stand-up fits + DP
        ss = np.arange(0.0, 1.001, 0.025)
        zs = np.arange(2.20, 3.301, 0.10)
        rotations = np.array([
            slerp(Rot.from_matrix(R_flat), Rot.from_matrix(R_upright_raw), float(s)).as_matrix()
            for s in ss
        ], dtype=np.float32)
        trans_frames = list(range(F_TRANSITION_START, F_TRANSITION_END + 1))
        score_grid = np.zeros((len(trans_frames), len(ss), len(zs)), dtype=np.float32)
        t_grid = np.zeros((len(trans_frames), len(ss), len(zs), 4, 4), dtype=np.float32)
        for ii, f in enumerate(trans_frames):
            image = load_frame(f)
            obj = obj_mask(f)
            person = h5["person"][f]
            ys, xs = np.where(obj)
            pad = 38
            x0, y0 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
            x1 = min(width, int(xs.max()) + pad + 1)
            y1 = min(height, int(ys.max()) + pad + 1)
            W, H = x1 - x0, y1 - y0
            Kc = K.copy()
            Kc[0, 2] -= x0
            Kc[1, 2] -= y0
            tgt = torch.tensor(obj[y0:y1, x0:x1], dtype=torch.bool, device="cuda")
            pers = torch.tensor(person[y0:y1, x0:x1], dtype=torch.bool, device="cuda")
            roi = torch.tensor(
                cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2RGB),
                dtype=torch.float32, device="cuda",
            ) / 255.0
            tuc = (xs.min() + xs.max()) / 2.0 + 2.0
            tvc = (ys.min() + ys.max()) / 2.0 + 1.5
            cells = [(a, b) for a in range(len(ss)) for b in range(len(zs))]
            first_T = {}
            first_pred = {}
            for st in range(0, len(cells), 16):
                cc = cells[st:st + 16]
                ts = []
                for a, b in cc:
                    z = zs[b]
                    X, Y = (tuc - K[0, 2]) / K[0, 0] * z, (tvc - K[1, 2]) / K[1, 1] * z
                    t = T_wc[:3, :3] @ np.array([X, Y, z]) + T_wc[:3, 3]
                    Tm = np.eye(4); Tm[:3, :3] = rotations[a]; Tm[:3, 3] = t
                    ts.append(Tm)
                pr, _ = render(rotations[[a for a, b in cc]], ts, Kc, W, H)
                for j, cell in enumerate(cc):
                    first_T[cell] = ts[j]
                    first_pred[cell] = pr[j]
            second_T = []
            for a, b in cells:
                z = zs[b]
                yy, xx = torch.where(first_pred[(a, b)])
                pu = (xx.min() + xx.max()).float().item() / 2.0 + x0
                pv = (yy.min() + yy.max()).float().item() / 2.0 + y0
                u, v = tuc - (pu - tuc), tvc - (pv - tvc)
                X, Y = (u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z
                t = T_wc[:3, :3] @ np.array([X, Y, z]) + T_wc[:3, 3]
                Tm = np.eye(4); Tm[:3, :3] = rotations[a]; Tm[:3, 3] = t
                second_T.append(Tm)
            second_T = np.asarray(second_T)
            for st in range(0, len(cells), 16):
                cc = cells[st:st + 16]
                pr, co = render(rotations[[a for a, b in cc]], second_T[st:st + 16], Kc, W, H)
                for j, (a, b) in enumerate(cc):
                    inter = (pr[j] & tgt).sum().item()
                    fp = (pr[j] & ~tgt & ~pers).sum().item()
                    fn = (tgt & ~pr[j]).sum().item()
                    iou = inter / max(inter + fp + fn, 1)
                    mi = (pr[j] & tgt)[..., None].float()
                    den = mi.sum().item() * 3.0 + 1e-6
                    l1 = ((co[j] - roi).abs() * mi).sum().item() / den
                    ncc_den = ((co[j] ** 2 * mi).sum().sqrt() *
                               (roi ** 2 * mi).sum().sqrt()).item()
                    ncc = (co[j] * roi * mi).sum().item() / (ncc_den + 1e-6)
                    score_grid[ii, a, b] = iou + 0.25 * ncc - 0.30 * l1
                    t_grid[ii, a, b] = second_T[st + j]
            if f % 20 == 0:
                print(f"[stand] {f}", flush=True)

        # Global monotone decoding: stand-up progresses monotonically and depth
        # changes smoothly.  This suppresses one-frame texture/silhouette flips.
        unary = -8.0 * score_grid
        for ii in range(len(trans_frames)):
            for a in range(len(ss)):
                unary[ii, a, :] += 8.0 * ((ss[a] - 0.95) / 0.05) ** 2 if trans_frames[ii] >= 795 else 0.0
        INF = 1e30
        dp = np.full((len(ss), len(zs)), INF)
        back = np.zeros((len(trans_frames), len(ss), len(zs), 2), dtype=np.int16)
        dp[0, :] = unary[0, 0, :]
        for ii in range(1, len(trans_frames)):
            newdp = np.full_like(dp, INF)
            for a in range(len(ss)):
                for b in range(len(zs)):
                    best = INF; ba = bb = 0
                    for pa in range(a + 1):
                        for pb in range(len(zs)):
                            cost = dp[pa, pb] + 0.35 * (a - pa) ** 2 + 0.08 * (b - pb) ** 2
                            if cost < best:
                                best, ba, bb = cost, pa, pb
                    newdp[a, b] = best + unary[ii, a, b]
                    back[ii, a, b] = (ba, bb)
            dp = newdp
        end = np.unravel_index(np.argmin(dp), dp.shape)
        path = []
        for ii in range(len(trans_frames) - 1, -1, -1):
            path.append(end)
            end = tuple(int(v) for v in back[ii, end[0], end[1]])
        path = path[::-1]

        # ------------------------------------------------------ final anchor
        z_up = np.arange(2.20, 2.811, 0.05)
        final_fit = fit_fixed(895, R_final, z_up, offset=(2.0, 1.0))

        # --------------------------------------------------------- assemble
        Tall = np.zeros((n_frames, 4, 4), dtype=np.float64)
        Tall[:, 3, 3] = 1.0
        Tall[:F_STATIC_FLAT_END + 1, :3, :3] = R_flat
        Tall[:F_STATIC_FLAT_END + 1, :3, 3] = anchor[5][:3, 3]

        flat_trans = np.asarray([flat_fits[f][5][:3, 3]
                                 for f in range(F_STATIC_FLAT_END + 1, F_TRANSITION_START)])
        flat_trans = np.stack([
            savgol_filter(medfilt(flat_trans[:, j], 7), 15, 2) for j in range(3)
        ], axis=1)
        for k, f in enumerate(range(F_STATIC_FLAT_END + 1, F_TRANSITION_START)):
            Tall[f, :3, :3] = R_flat
            Tall[f, :3, 3] = flat_trans[k]

        trans_trans = np.asarray([t_grid[ii, a, b, :3, 3]
                                  for ii, (a, b) in enumerate(path)])
        trans_trans = np.stack([
            savgol_filter(medfilt(trans_trans[:, j], 7), 15, 2) for j in range(3)
        ], axis=1)
        for ii, (a, b) in enumerate(path):
            f = trans_frames[ii]
            Tall[f, :3, :3] = rotations[a]
            Tall[f, :3, 3] = trans_trans[ii]

        Tall[F_STATIC_UP_START:, :3, :3] = R_final
        Tall[F_STATIC_UP_START:, :3, 3] = final_fit[5][:3, 3]

        # Smoothstep blends avoid centimetre-scale seams at phase boundaries.
        def blend(f, fa, fb, Ta, Tb):
            u = (f - fa) / float(fb - fa)
            s = u * u * (3.0 - 2.0 * u)
            R = slerp(Rot.from_matrix(Ta[:3, :3]), Rot.from_matrix(Tb[:3, :3]), s).as_matrix()
            Tall[f, :3, :3] = R
            Tall[f, :3, 3] = (1.0 - s) * Ta[:3, 3] + s * Tb[:3, 3]

        Ta = Tall[155].copy(); Tb = Tall[170].copy()
        for f in range(155, 171):
            blend(f, 155, 170, Ta, Tb)
        Ta = Tall[785].copy(); Tb = Tall[800].copy()
        for f in range(785, 797):
            blend(f, 785, 797, Ta, Tb)

        # Numerical cleanup.
        for f in range(n_frames):
            Tall[f, :3, :3] = Rot.from_matrix(Tall[f, :3, :3]).as_matrix()
            Tall[:, 3, :] = np.array([0.0, 0.0, 0.0, 1.0])

        np.savez(out_path, T_world_mesh=Tall.astype(np.float64))
        h5.close()
        cap.release()
        print(f"Wrote {out_path} with shape {Tall.shape}; elapsed {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()

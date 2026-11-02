"""Camera-model hypothesis test.

Camera model = (K focal f, world_R_cam, camera centre C).  Given a camera, the scene is rebuilt as:
  rays from our 2D (our camera-frame points re-projected through our K, then back-projected through f'),
  per-frame depth factor a_t chosen so the human's lowest vertex sits at height h0 above the released plane
  (feet-on-floor, as FF2), the object gets the same a_t.
Cameras compared:  'ours'   = PnP first-frame camera (should reproduce FF2),
                   'oracle' = DIAGNOSTIC ONLY: fit to the GT human over all frames (upper bound),
                   'self'   = compliant self-calibration (see fit_self), uses only our reconstruction,
                              the released plane and the frame-0 reference human (height h0, size).
"""
import json, os, sys
import numpy as np, torch, trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, "/workspace/v2d_cari4d/lib/cari4d"); sys.path.insert(0, "/home/hanzhou/Projects/v2d/hoi_extraction/scripts")
from eval_object_pose import metrics, quat_wxyz_to_R
from eval_official import bundle_path
from lib_mhr.mhr_layer import MHRLayer
from tools.render_mhr_wild_inference import _params
import soma

V2D = "/mnt/nas/Group/hanzhou/v2d"; ds = f"{V2D}/dataset/track_2/tier_1_multiview_caption"
meta = {d["episode_index"]: d for d in map(json.loads, open(f"{ds}/meta/episodes_metadata.jsonl"))}
al = json.load(open(f"{V2D}/logs/pnp_forE1.json")); ff = json.load(open(f"{V2D}/experiments/FF2_feet_ref/feet_floor.json"))
dev = torch.device("cuda")
mhr = MHRLayer.from_mhr_assets(mhr_assets_root=f"{V2D}/weights/sam3d_body", device="cuda")
sl = soma.SOMALayer(identity_model_type="mhr", device="cuda")
FLIP = np.diag([1.0, -1.0, -1.0])
WIN = 15
FOCALS = lambda f, fo: [("pnp_same", f), ("f_oracle", f * fo), ("f855", 855.0)] + ([("f_ext", EXT[IDX])] if EXT else [])
EXT = json.load(open(os.environ["EXT"])) if os.environ.get("EXT") else None


def soma_verts(pose, ident, scale, transl):
    n = len(pose)
    with torch.no_grad():
        return sl(torch.tensor(pose).float().view(n, 77, 3).to(dev), torch.tensor(ident).float().to(dev), scale_params=torch.tensor(scale).float().to(dev),
                  transl=torch.tensor(transl).float().to(dev)).vertices.cpu().numpy().astype(np.float64) @ FLIP


def rebuild(cam, Hc, K, n, d, h0, Tobj):
    """Hc: (F,V,3) our human (source, before FF) in our camera; returns world object translations and per-frame a_t."""
    f2, Rwc, C = cam
    sx = K[0, 0] / f2
    def remap(X):  # same pixel, new focal: scale x,y about principal ray (principal point unchanged)
        Y = X.copy(); Y[..., 0] *= sx; Y[..., 1] *= sx; return Y
    Hn = remap(Hc)
    a_coef = np.einsum("j,tvj->tv", n @ Rwc, Hn).min(1)          # lowest vertex height slope
    c0 = n @ C + d
    a = np.where(a_coef < -1e-6, (h0 - c0) / a_coef, 1.0)
    a = np.clip(a, 0.7, 1.3)
    a = np.array([np.median(a[max(0, i - WIN):i + WIN + 1]) for i in range(len(a))])
    to = remap(Tobj[:, :3, 3]) * a[:, None]
    return to @ Rwc.T + C, Rwc[None] @ Tobj[:, :3, :3], a


def main():
    eps = [int(x) for x in sys.argv[1].split(",")]
    out = {}
    for idx in eps:
        global IDX; IDX = str(idx)
        ep = f"episode_{idx:06d}"
        src = ff[str(idx)]["src"]
        b = torch.load(bundle_path(src, ep), map_location="cpu", weights_only=False)
        Ps = np.asarray(b["pr"]["pose_abs"], np.float64); F = len(Ps)
        K = np.asarray(b["observations"]["K_full"], np.float64)[0]
        with torch.no_grad():
            Hc = np.concatenate([mhr.mhr_forward(_params(b["pr"], slice(a, min(a + 128, F)), dev)).vertices.cpu().numpy().astype(np.float64)[:, ::8] for a in range(0, F, 128)])
        W = np.asarray(al[str(idx)]["sim"]); s = np.cbrt(np.linalg.det(W[:3, :3])); Rw = W[:3, :3] / s; C = W[:3, 3]
        gp = np.asarray(json.load(open(f"{ds}/{meta[idx]['ground_plane']}"))["plane"]); n, d = gp[:3] / np.linalg.norm(gp[:3]), gp[3] / np.linalg.norm(gp[:3])
        if n @ C + d < 0: n, d = -n, -d
        r0 = np.load(f"{V2D}/ref_human_f0/{ep}.npz")
        g0 = soma_verts(r0["pose"][None], r0["identity"][None], r0["scale"][None], r0["transl"][None])[0]
        h0 = float((g0 @ n + d).min())
        glb = trimesh.load(f"{ds}/{meta[idx]['mesh']}", force="scene", process=False).dump(concatenate=True)
        S = np.asarray(trimesh.bounds.oriented_bounds(glb)[0]); verts = np.asarray(glb.sample(1000))
        Tobj = Ps @ S
        ref = np.load(f"{V2D}/ref_poses/{ep}.npz")["pose_xyz_wxyz"].astype(np.float64)
        m_ = min(F, len(ref)); ok = (np.linalg.norm(ref[:m_, 3:], axis=1) > 0.5) & np.isfinite(Tobj[:m_]).all(axis=(1, 2))
        Rr, tr = quat_wxyz_to_R(ref[:m_][ok][:, 3:]), ref[:m_][ok][:, :3]
        cams = {"ours": (K[0, 0], Rw, C)}
        # ---- oracle camera (DIAGNOSTIC): GT human every 5th frame vs our 2D (projections of our human through K)
        g = np.load(f"{V2D}/diag_only_ref_human/{ep}.npz"); fr = g["frame"]; fr = fr[fr < F][::2]
        G = soma_verts(g["pose"][::2][:len(fr)], g["identity"][::2][:len(fr)], g["scale"][::2][:len(fr)], g["transl"][::2][:len(fr)])
        # vertex map MHR(sub-sampled) -> SOMA at frame 0 using the first-frame alignment
        h_w0 = s * Hc[0] @ Rw.T + C
        _, jmap = cKDTree(G[0]).query(h_w0)
        uv = (Hc[fr] @ K.T); uv = uv[..., :2] / uv[..., 2:3]
        _, jmap0 = cKDTree(g0).query(h_w0)
        uvf0 = (Hc[0] @ K.T); uvf0 = uvf0[:, :2] / uvf0[:, 2:3]
        X = G[:, jmap]
        def resid(p):
            R = Rot.from_rotvec(p[:3]).as_matrix() @ Rw; Cc = C + p[3:6]; f2 = K[0, 0] * np.exp(p[6])
            Xc = (X - Cc) @ R
            pr = Xc[..., :2] / Xc[..., 2:3] * f2 + K[:2, 2]
            return (pr - uv).ravel()
        sol = least_squares(resid, np.zeros(7), loss="soft_l1", f_scale=5.0)
        p = sol.x
        cams["oracle"] = (K[0, 0] * np.exp(p[6]), Rot.from_rotvec(p[:3]).as_matrix() @ Rw, C + p[3:6])
        # focal-only variants: re-solve the frame-0 PnP (frame-0 reference human only) for a given focal
        def pnp0(f2):
            X0 = g0[jmap0]; uv0 = uvf0
            def r0_(q):
                R = Rot.from_rotvec(q[:3]).as_matrix() @ Rw; Cc = C + q[3:6]
                Xc = (X0 - Cc) @ R
                return (Xc[:, :2] / Xc[:, 2:3] * f2 + K[:2, 2] - uv0).ravel()
            q = least_squares(r0_, np.zeros(6), loss="soft_l1", f_scale=3.0).x
            return (f2, Rot.from_rotvec(q[:3]).as_matrix() @ Rw, C + q[3:6])
        for name, f2 in FOCALS(K[0, 0], np.exp(p[6])):
            cams[name] = pnp0(f2)
        out[idx] = {"oracle_df": float(np.exp(p[6])), "oracle_drot_deg": float(np.degrees(np.linalg.norm(p[:3]))), "oracle_dC_cm": float(np.linalg.norm(p[3:6]) * 100)}
        line = f"{idx:2d} {meta[idx]['object'][:10]:10s} oracle: f x{np.exp(p[6]):.3f} rot {np.degrees(np.linalg.norm(p[:3])):.2f}deg C {np.linalg.norm(p[3:6])*100:.1f}cm |"
        for name, cam in cams.items():
            tw, Rwo, a = rebuild(cam, Hc, K, n, d, h0, Tobj)
            m = metrics(Rwo[:m_][ok], tw[:m_][ok], Rr, tr, verts)
            line += f" {name} AUC {m['AUC']:.3f} pos {m['pos_cm']:.1f}"
            out[idx][name] = m["AUC"]
        print(line, flush=True)
    json.dump(out, open(os.environ.get("OUT", "/dev/null"), "w"), indent=1)


if __name__ == "__main__":
    main()

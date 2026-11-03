"""Recover the 6-DoF trajectory of the white desk (episode 10) in the released world frame.

Method (see REPORT.md):
  1. Phases (constants verified by eye + automatic checks, see below): static A (desk upright, untouched / only
     held), moving (desk turned, tilted and laid on its side), static B (desk lying on the floor).
  2. Static phases: ONE pose per phase, fitted by render-and-compare (nvdiffrast) against the SAM2 object mask
     pooled over all frames of the phase (pixels covered by the person are ignored).  Hypotheses: the baseline
     pose and its 180-degree flips about the three mesh axes; the best pooled IoU wins unless its margin is
     < 0.01 (near-symmetric silhouette), then the hypothesis continuous with the tracked moving phase wins.  Depth along the viewing
     ray is then fused (inverse variance) with two weak cues: floor contact and MoGe-2 depth.
  3. Moving phase: per-frame silhouette fit initialised from the baseline, followed by a joint optimisation of
     all frames with silhouette + constant-velocity (acceleration) prior on the mesh bounding-box corners, with the
     static poses as fixed boundary conditions.  (A KLT reprojection term is implemented in optimise(mu>0) but
     disabled: in tests it did not remove the occlusion bias and, fitted on alternate frame pairs, it made the
     trajectory less smooth and increased the held-out KLT residual; KLT tracks are used for validation only.)
Run:  python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoi_common import *

# ---------------------------------------------------------------- phase constants (frame indices, 30 fps)
# Static A: desk upright on its feet.  The person walks up, grasps the near edge at ~175 and starts to turn
# the desk at ~265.  Template matching of unoccluded desk patches (frame 160 vs 180..260) shows <= 1 px motion,
# while the per-frame silhouette fits drift by 2-3 cm only because the person occludes the near half: so the
# whole range is treated as one static pose, fitted on the unoccluded frames 0..170.
A_END = 265            # first frame of the moving phase
A_FIT = (0, 170)       # frames used to fit the static-A pose (person not touching / not occluding the desk)
# Static B: desk lying on its front edge on the floor; per-frame fits stay within 1 cm of the pooled pose from
# frame 600 on, and the object mask IoU with the last frame is >= 0.95 from 600 on.
B_START = 600
SCALE = 0.5            # silhouette resolution (768x576)
SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--out", default="/work/outputs/object_poses_world.npz")
    ap.add_argument("--pipeline", default="/pipeline")
    args = ap.parse_args()
    I = args.inputs
    torch.manual_seed(SEED); np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)

    K, Twc, Kp = load_camera(I)
    Tcw = np.linalg.inv(Twc)
    m, v, f, uv, tex = load_mesh(I)
    rec = np.load(f"{I}/reconstruction.npz")
    Tb_w = rec["T_world_mesh"].astype(np.float64)
    F = Tb_w.shape[0]
    Tb = Tcw @ Tb_w                                   # baseline in camera frame
    Ks = scale_K(K, SCALE)
    SD = SilhouetteData(I, SCALE)
    log("masks loaded")
    R = Renderer(v, f)
    h, w = SD.h, SD.w
    ctr = (v.min(0) + v.max(0)) / 2

    # ------------------------------------------------------------ static phases
    plane = json.load(open(f"{I}/ground_plane.json"))["plane"]
    n_pl, d_pl = np.array(plane[:3]), plane[3]
    # floor offset calibrated with the TRUE frame-0 human: its lowest vertex (shoe sole) lies h0 above the plane
    h0 = float((np.load(f"{I}/frame0_reference_human.npz")["vertices_world"] @ n_pl + d_pl).min())
    vv = v.astype(np.float64)
    moge = h5py.File(f"{args.pipeline}/depth/moge2/aligned.h5", "r")["aligned/front_stereo_camera_left"]

    def fit_hypotheses(name, fit_frames):
        """Pooled silhouette fit from the baseline pose and its 180-degree flips about the 3 mesh axes."""
        no, nb = pooled_counts(SD, fit_frames)
        Tinit = Tb[fit_frames[len(fit_frames) // 2]]
        hyps = []
        for ax in [None, 0, 1, 2]:
            Ti = Tinit if ax is None else flip_about_axis(Tinit, ax, ctr)
            T, iou = fit_static_pose(R, Ks, h, w, no, nb, Ti, iters=300)
            hyps.append((iou, ax, T))
            log(f"static {name}: hypothesis flip={ax} pooled IoU {iou:.4f}")
        return sorted(hyps, key=lambda x: -x[0])

    def choose(name, hyps, T_track):
        """Silhouette decides if the IoU margin is clear (>= 0.01); otherwise (near-symmetric silhouette) the
        hypothesis that is continuous with the tracked moving phase at the phase boundary wins."""
        rot = [rot_angle_deg(hh[2][:3, :3], T_track[:3, :3]) for hh in hyps]
        log(f"static {name}: rotation to tracked boundary pose: " + ", ".join(f"flip={hh[1]}: {r:.1f}deg" for hh, r in zip(hyps, rot)))
        margin = hyps[0][0] - hyps[1][0]
        if margin >= 0.01:
            k = 0
        else:
            cands = [j for j in range(len(hyps)) if hyps[0][0] - hyps[j][0] < 0.01]
            k = min(cands, key=lambda j: rot[j])
        log(f"static {name}: chose flip={hyps[k][1]} (IoU margin best-2nd {margin:.4f})")
        return hyps[k][2]

    def depth_fuse(name, T, all_frames):
        def minh(eps):
            T2 = T.copy(); T2[:3, 3] *= 1 + eps; Tw = Twc @ T2
            return ((vv @ Tw[:3, :3].T + Tw[:3, 3]) @ n_pl + d_pl).min()
        epss = np.linspace(-0.06, 0.06, 241)
        hs = np.array([minh(e) for e in epss])
        e_floor = float(epss[np.argmin(np.abs(hs - h0))])
        dh = abs(minh(0.01) - minh(-0.01)) / 0.02
        mk, dd, _ = R.render(torch.tensor(T[None], dtype=torch.float32, device="cuda"), K, W, H)
        dd = dd[0].cpu().numpy()
        rat = []
        for i in all_frames[::10]:
            D = cv2.imdecode(np.frombuffer(moge[i], np.uint8), cv2.IMREAD_UNCHANGED) / 1000.0
            o = cv2.resize(SD.obj[i].cpu().numpy().astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            vis = cv2.resize(SD.valid[i].cpu().numpy().astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            sel = (cv2.erode(o, np.ones((5, 5))) > 0) & (vis > 0) & (dd > 0)
            if sel.sum() > 500:
                rat.append(np.median(D[sel] / dd[sel]))
        e_moge = float(np.median(rat)) - 1
        # sigmas (relative depth): silhouette 1% (pure-scale scan of the pooled IoU), floor 2 cm contact
        # tolerance converted with dh/deps, MoGe 2.5% (frame-to-frame spread / its own scale alignment)
        s_sil, s_floor, s_moge = 0.01, 0.02 / max(dh, 1e-3), 0.025
        wts = np.array([1 / s_sil ** 2, 1 / s_floor ** 2, 1 / s_moge ** 2])
        e_fused = float((wts * np.array([0.0, e_floor, e_moge])).sum() / wts.sum())
        log(f"static {name}: depth cues eps silhouette 0, floor {e_floor:+.4f} (h(0)={minh(0):.3f}, h0={h0:.3f}),"
            f" moge {e_moge:+.4f} -> fused {e_fused:+.4f} ({e_fused * T[2, 3] * 100:+.1f} cm)")
        T = T.copy(); T[:3, 3] *= 1 + e_fused
        return T

    hypA = fit_hypotheses("A", list(range(*A_FIT)))
    hypB = fit_hypotheses("B", list(range(B_START, F)))

    # ------------------------------------------------------------ moving phase
    a, b = A_END, B_START
    fr = np.arange(a, b); N = len(fr)
    c = torch.tensor(v.mean(0), device="cuda")
    bb = torch.tensor(trimesh.bounds.corners(np.stack([v.min(0), v.max(0)])), dtype=torch.float32, device="cuda")
    corners = lambda T: torch.einsum("nij,kj->nki", T[:, :3, :3], bb) + T[:, None, :3, 3]
    CH = 40

    def optimise(T_init, iters, lam, mu, tracks=None, lr=2e-3):
        T0 = torch.tensor(T_init, dtype=torch.float32, device="cuda")
        wv = torch.zeros(N, 3, device="cuda", requires_grad=True); dt = torch.zeros(N, 3, device="cuda", requires_grad=True)
        opt = torch.optim.Adam([wv, dt], lr=lr)
        if lam > 0:
            tA = torch.tensor(TA[None], dtype=torch.float32, device="cuda"); tB = torch.tensor(TB[None], dtype=torch.float32, device="cuda")
        if mu > 0:
            ti = torch.tensor(tracks[:, 0].astype(int) - a, device="cuda")
            P0 = torch.tensor(tracks[:, 1:3], dtype=torch.float32, device="cuda"); P1 = torch.tensor(tracks[:, 3:5], dtype=torch.float32, device="cuda")
            Kt = torch.tensor(K, dtype=torch.float32, device="cuda")
            def proj(T, X):
                Xc = torch.einsum("nij,nj->ni", T[:, :3, :3], X) + T[:, :3, 3]; q = Xc @ Kt.T
                return q[:, :2] / q[:, 2:]
        for it in range(iters):
            opt.zero_grad()
            for k in range(0, N, CH):
                T = compose(T0[k:k + CH], wv[k:k + CH], dt[k:k + CH], c)
                mk, _, _ = R.render(T, Ks, w, h)
                SD.loss(mk, fr[k:k + CH]).sum().backward()
            T = compose(T0, wv, dt, c)
            L = 0
            if mu > 0:
                if it % 40 == 0:   # re-lift track starts to mesh points with the current poses
                    Tc = T.detach().cpu().numpy().astype(np.float64)
                    Xnp, val = klt_lift(R, K, {i: Tc[i - a] for i in fr}, tracks, SCALE)
                    vm = torch.tensor(val, device="cuda")
                    Xm = torch.tensor(Xnp, dtype=torch.float32, device="cuda"); Xm[~vm] = c
                e = (torch.nn.functional.huber_loss(proj(T[ti], Xm), P0, reduction="none", delta=1.0).sum(-1)
                     + torch.nn.functional.huber_loss(proj(T[ti + 1], Xm), P1, reduction="none", delta=1.0).sum(-1))
                L = L + mu * torch.where(vm, e, torch.zeros_like(e)).sum()
            if lam > 0:
                P = torch.cat([corners(tA).expand(2, -1, -1), corners(T), corners(tB).expand(2, -1, -1)])
                acc = P[2:] - 2 * P[1:-1] + P[:-2]
                L = L + lam * (acc ** 2).sum(-1).mean(-1).sum()
            if torch.is_tensor(L):
                L.backward()
            opt.step()
        with torch.no_grad():
            return compose(T0, wv, dt, c).cpu().numpy().astype(np.float64)

    T1 = optimise(Tb[a:b], iters=150, lam=0, mu=0, lr=3e-3)
    log("moving phase: per-frame silhouette fit done")
    TA = depth_fuse("A", choose("A", hypA, T1[0]), list(range(0, A_END)))
    TB = depth_fuse("B", choose("B", hypB, T1[-1]), list(range(B_START, F)))
    T2 = optimise(T1, iters=250, lam=1e4, mu=0)
    log(f"moving phase: joint fit done; peak GPU mem {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    Tcam = np.zeros((F, 4, 4))
    Tcam[:a] = TA; Tcam[a:b] = T2; Tcam[b:] = TB
    T_world = Twc @ Tcam
    np.savez(args.out, T_world_mesh=T_world.astype(np.float64))
    log(f"wrote {args.out}  shape {T_world.shape}")


if __name__ == "__main__":
    main()

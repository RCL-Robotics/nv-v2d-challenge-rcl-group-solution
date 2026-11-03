"""Ground-truth-free validation: baseline vs result, per phase.
Metrics: silhouette IoU (person pixels ignored), KLT reprojection residual on HELD-OUT odd-frame tracks,
patch template-matching residual (frame i -> i+5, 41x41 patches on visible desk texture/edges, independent of
both the silhouette and the KLT term), temporal smoothness (mean acceleration of the mesh bbox corners),
and lowest-point height above the released floor plane in the static phases.
python validate.py result.npz"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoi_common import *
from solve import A_END, B_START

I = "/work/inputs"
K, Twc, Kp = load_camera(I); Tcw = np.linalg.inv(Twc)
m, v, f, uv, tex = load_mesh(I); R = Renderer(v, f)
rec = np.load(f"{I}/reconstruction.npz")
cands = {"baseline": Tcw @ rec["T_world_mesh"].astype(float), "result": Tcw @ np.load(sys.argv[1])["T_world_mesh"]}
F = 794
phases = [("static A", 0, A_END), ("moving", A_END, B_START), ("static B", B_START, F)]
SD = SilhouetteData(I, 0.5, cache="/work/scratch/masks_half.npz")
Ks = scale_K(K, 0.5)
pl = json.load(open(f"{I}/ground_plane.json"))["plane"]; n_pl, d_pl = np.array(pl[:3]), pl[3]
bbc = trimesh.bounds.corners(np.stack([v.min(0), v.max(0)]))
ious = {}
for k, T in cands.items():
    out = []
    with torch.no_grad():
        for s in range(0, F, 40):
            idx = np.arange(s, min(s + 40, F))
            mk, _, _ = R.render(torch.tensor(T[idx], dtype=torch.float32, device="cuda"), Ks, SD.w, SD.h)
            out.append(SD.iou(mk, idx).cpu().numpy())
    ious[k] = np.concatenate(out)
# held-out KLT (odd frames) over the whole video
frs = list(range(0, F))
vid = read_video(f"{I}/video.mp4", frs)
om = load_masks(I, frs); pm = load_masks(I, frs, 1, "person")
tracks = klt_tracks(vid, om, pm, frs)
tracks = tracks[tracks[:, 0].astype(int) % 2 == 1]
# patch template matching i -> i+5
rng = np.random.default_rng(0)
patches = []
for i in range(0, F - 5, 5):
    g0 = cv2.cvtColor(vid[i], cv2.COLOR_RGB2GRAY).astype(np.float32); g1 = cv2.cvtColor(vid[i + 5], cv2.COLOR_RGB2GRAY).astype(np.float32)
    gm = np.hypot(cv2.Sobel(g0, cv2.CV_32F, 1, 0), cv2.Sobel(g0, cv2.CV_32F, 0, 1))
    farp = ~(cv2.dilate((pm[i] | pm[i + 5]).astype(np.uint8), np.ones((61, 61))) > 0)
    cand = np.argwhere(om[i] & farp & (gm > 50))
    if len(cand) == 0:
        continue
    cand = cand[rng.choice(len(cand), min(len(cand), 400), replace=False)]
    sel = []
    for y, x in cand:
        if 60 < y < H - 60 and 60 < x < W - 60 and all((y - a) ** 2 + (x - b) ** 2 > 50 ** 2 for a, b in sel):
            sel.append((y, x))
    for y, x in sel[:12]:
        P = 20; S = 25
        res = cv2.matchTemplate(g1[y - P - S:y + P + S + 1, x - P - S:x + P + S + 1], g0[y - P:y + P + 1, x - P:x + P + 1], cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(res)
        if mv > 0.85 and 0 < ml[0] < 2 * S and 0 < ml[1] < 2 * S:
            # sub-pixel parabola refinement
            dx = (res[ml[1], ml[0] - 1] - res[ml[1], ml[0] + 1]) / (2 * (res[ml[1], ml[0] - 1] - 2 * mv + res[ml[1], ml[0] + 1]) + 1e-9)
            dy = (res[ml[1] - 1, ml[0]] - res[ml[1] + 1, ml[0]]) / (2 * (res[ml[1] - 1, ml[0]] - 2 * mv + res[ml[1] + 1, ml[0]]) + 1e-9)
            patches.append((i, x, y, x + ml[0] - S + dx, y + ml[1] - S + dy))
patches = np.array(patches)
del vid
rows = {}
for k, T in cands.items():
    Xm, val = klt_lift(R, K, T, tracks); kres = klt_residual(K, T, tracks, Xm, val)
    # patch residual (i -> i+5)
    Xp, valp = klt_lift(R, K, T, patches)
    i5 = patches[:, 0].astype(int) + 5
    Tn = T[i5]; Xc = np.einsum("nij,nj->ni", Tn[:, :3, :3], Xp) + Tn[:, :3, 3]; q = Xc @ K.T; q = q[:, :2] / q[:, 2:]
    pres = np.linalg.norm(q - patches[:, 3:5], axis=1); pres[~valp] = np.nan
    P = np.einsum("nij,kj->nki", T[:, :3, :3], bbc) + T[:, None, :3, 3]
    acc = np.linalg.norm(P[2:] - 2 * P[1:-1] + P[:-2], axis=-1).mean(-1)
    Tw = Twc @ T
    hmin = np.array([((v[::20] @ Tw[i, :3, :3].T + Tw[i, :3, 3]) @ n_pl + d_pl).min() for i in range(F)])
    for name, a, b in phases:
        kf = (tracks[:, 0] >= a) & (tracks[:, 0] < b); pf = (patches[:, 0] >= a) & (patches[:, 0] < b)
        rows.setdefault(name, []).append((k, ious[k][a:b].mean(), np.nanmedian(kres[kf]), np.nanmean(np.minimum(kres[kf], 5)), kf.sum(),
                                          np.nanmedian(pres[pf]), np.nanmean(np.minimum(pres[pf], 10)), pf.sum(),
                                          acc[max(a - 1, 0):b - 1].mean() * 1000, np.median(hmin[a:b])))
print("| phase | frames | traj | mask IoU | held-out KLT px (median / mean<=5) | patch i->i+5 px (median / mean<=10) | accel mm/frame^2 | median lowest-point height above plane (m) |")
print("|---|---|---|---|---|---|---|---|")
for name, a, b in phases:
    for r in rows[name]:
        print(f"| {name} | {a}-{b - 1} | {r[0]} | {r[1]:.3f} | {r[2]:.2f} / {r[3]:.2f} (n={r[4]}) | {r[5]:.2f} / {r[6]:.2f} (n={r[7]}) | {r[8]:.2f} | {r[9]:.3f} |")
for k in cands:
    print(k, "all-frames IoU", ious[k].mean())

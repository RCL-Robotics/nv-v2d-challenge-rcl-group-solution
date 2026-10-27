# Visualisation/diagnostic only (uses GT poses): fit the exo camera (T_world_cam + focal) so that the GT object mesh
# projects onto the SAM2 object masks over many frames (pixels under the person mask ignored) and the frame-0 GT human
# vertices fall inside the person mask. The dataset ships no camera calibration; this camera is only for rendering GT.
import sys, json, numpy as np, torch, h5py, trimesh, cv2
import nvdiffrast.torch as dr
sys.path.insert(0, "/home/hanzhou/Projects/v2d/hoi_extraction/scripts")
from eval_object_pose import quat_wxyz_to_R
V = "/mnt/nas/Group/hanzhou/v2d"; idx = int(sys.argv[1]); out = sys.argv[2]
inp = f"{V}/agent_runs/T1_ds_ep{idx:03d}/inputs"
cam = json.load(open(f"{inp}/camera.json")); K0 = np.array(cam["K"]); W0 = np.array(cam["T_world_cam"]); Hh, Ww = cam["height"], cam["width"]
g = trimesh.load(f"{inp}/object_mesh.glb", force="scene", process=False).to_geometry()
ref = np.load(f"{V}/ref_poses/episode_{idx:06d}.npz")["pose_xyz_wxyz"].astype(float); okr = np.linalg.norm(ref[:, 3:], axis=1) > 0.5
hum = np.load(f"{inp}/frame0_reference_human.npz")["vertices_world"].astype(np.float64)
dev = "cuda"; s = 2  # half resolution
with h5py.File(f"{inp}/masks.h5") as f:
    F = min(len(f["object"]), len(ref))
    area = np.array([f["object"][t][::8, ::8].sum() for t in range(F)])
    cand = np.where(okr[:F] & (area > 15))[0]
    fr = cand[np.linspace(0, len(cand) - 1, min(60, len(cand))).astype(int)]
    rs = lambda m: cv2.resize(m.astype(np.float32), (Ww // s, Hh // s), interpolation=cv2.INTER_AREA)
    MO = torch.tensor(np.stack([rs(f["object"][t]) for t in fr]), device=dev)
    MP = torch.tensor(np.stack([rs(f["person"][t]) for t in fr]), device=dev)
    P0 = f["person"][0]
dt = cv2.distanceTransform((~P0).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)  # px outside person mask
DT = torch.tensor(dt, device=dev)[None, None]
R = quat_wxyz_to_R(ref[fr, 3:]); Tg = np.tile(np.eye(4), (len(fr), 1, 1)); Tg[:, :3, :3] = R; Tg[:, :3, 3] = ref[fr, :3]
Vw = torch.tensor(np.einsum("fij,vj->fvi", Tg[:, :3, :3], np.asarray(g.vertices)) + Tg[:, None, :3, 3], device=dev, dtype=torch.float32)
Fc = torch.tensor(np.asarray(g.faces), device=dev, dtype=torch.int32)
Hw = torch.tensor(hum, device=dev, dtype=torch.float32)
Wi = np.linalg.inv(W0); Rcw0 = torch.tensor(Wi[:3, :3], device=dev, dtype=torch.float32); tcw0 = torch.tensor(Wi[:3, 3], device=dev, dtype=torch.float32)
w = torch.zeros(3, device=dev, requires_grad=True); t = torch.zeros(3, device=dev, requires_grad=True); lf = torch.zeros(1, device=dev, requires_grad=True)
glc = dr.RasterizeCudaContext()
def so3(v):
    th = v.norm() + 1e-12; k = v / th; Kx = torch.zeros(3, 3, device=dev)
    Kx[0, 1], Kx[0, 2], Kx[1, 0], Kx[1, 2], Kx[2, 0], Kx[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    return torch.eye(3, device=dev) + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx
def to_cam(X):
    return X @ (so3(w) @ Rcw0).T + (so3(w) @ tcw0 + t)
def proj(Xc):
    f = torch.tensor(K0[0, 0], device=dev) * torch.exp(lf); u = f * Xc[..., 0] / Xc[..., 2] + K0[0, 2]; v = f * Xc[..., 1] / Xc[..., 2] + K0[1, 2]
    return u, v
def render(Xc):
    u, v = proj(Xc); z = Xc[..., 2]; n, fa = 0.1, 30.
    clip = torch.stack([(2 * u / Ww - 1) * z, (2 * v / Hh - 1) * z, z * (fa + n) / (fa - n) - 2 * fa * n / (fa - n), z], -1).contiguous()
    rast, _ = dr.rasterize(glc, clip, Fc, resolution=[Hh // s, Ww // s])
    m = (rast[..., 3:] > 0).float()
    return dr.antialias(m, rast, clip, Fc)[..., 0]
def loss_fn():
    R_ = render(to_cam(Vw)); keep = 1 - MP
    inter = (R_ * MO * keep).sum((1, 2)); uni = ((R_ + MO - R_ * MO) * keep).sum((1, 2)) + 1
    iou = inter / uni
    u, v = proj(to_cam(Hw)); gx = torch.stack([2 * u / (Ww - 1) - 1, 2 * v / (Hh - 1) - 1], -1)[None, None]
    dh = torch.nn.functional.grid_sample(DT, gx, align_corners=True).mean()
    return 1 - iou.mean() + 0.002 * dh, iou, dh
opt = torch.optim.Adam([{"params": [w], "lr": 2e-3}, {"params": [t], "lr": 5e-3}, {"params": [lf], "lr": 2e-3}])
with torch.no_grad(): l0, iou0, dh0 = loss_fn()
for it in range(400):
    opt.zero_grad(); l, iou, dh = loss_fn(); l.backward(); opt.step()
    if it == 250:
        for gpar in opt.param_groups: gpar["lr"] *= 0.2
with torch.no_grad():
    l, iou, dh = loss_fn(); Rn = (so3(w) @ Rcw0).cpu().numpy().astype(float); tn = (so3(w) @ tcw0 + t).cpu().numpy().astype(float)
Tcw = np.eye(4); Tcw[:3, :3], Tcw[:3, 3] = Rn, tn; Wn = np.linalg.inv(Tcw); K = K0.copy(); K[0, 0] *= float(torch.exp(lf)); K[1, 1] *= float(torch.exp(lf))
ang = np.degrees(np.arccos(np.clip((np.trace(Wn[:3, :3].T @ W0[:3, :3]) - 1) / 2, -1, 1)))
rep = dict(K=K.tolist(), T_world_cam=Wn.tolist(), width=Ww, height=Hh, frames=fr.tolist(),
           iou_before=float(iou0.mean()), iou_after=float(iou.mean()), iou_after_median=float(iou.median()), human_dt_px_before=float(dh0), human_dt_px_after=float(dh),
           change_vs_s1_deg=float(ang), change_vs_s1_cm=float(np.linalg.norm(Wn[:3, 3] - W0[:3, 3]) * 100), focal_scale=float(torch.exp(lf)))
json.dump(rep, open(out, "w"), indent=1)
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rep.items() if k not in ("K", "T_world_cam", "frames")})

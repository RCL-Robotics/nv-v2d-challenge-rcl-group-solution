#!/usr/bin/env python3
"""Refine the per-episode camera (rotation + centre) with silhouettes, using the frame-0 object pose (allowed input).
  object term : the mesh at its frame-0 pose, rendered over the static start (frames from 0 while the Tier 2 position
                stays within 1 cm of frame 0; at most 30 frames, stride 3), vs the SAM2 object masks (soft IoU,
                pixels under the person mask ignored);
  human term  : decoded Tier 2 SOMA vertices at NF frames, mean distance (px) to the SAM2 person mask.
Init and prior: the joint human/object camera (cam_join.py). Intrinsics: the shared K of inputs/camera.json. Projection
ignores the sim3 scale, so only R and the camera centre t are optimised. No GT beyond frame 0.
Output json has the same keys as the joint camera (s kept), plus iou/dt before and after.
mode "r" optimises the rotation only (camera centre kept from the joint fit), "rt" both.
Usage: cam_sil_f0.py <cam_joint.json> <out.json> [eps|all] [mode=r]"""
import os
import json, sys, numpy as np, torch, h5py, trimesh, cv2
import nvdiffrast.torch as dr
import soma
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hoi_extraction", "scripts"))
from eval_object_pose import quat_wxyz_to_R
V = os.environ.get("V2D_ROOT", "/mnt/nas/Group/hanzhou/v2d"); T2DS = f"{V}/dataset/track_2/tier_2_synthetic_noise"; FLIP = np.diag([1.0, -1.0, -1.0])
meta = {d["episode_index"]: d for d in map(json.loads, open(f"{T2DS}/meta/episodes_metadata.jsonl"))}
cams = json.load(open(sys.argv[1])); OUT = sys.argv[2]
eps = list(map(int, sys.argv[3].split(","))) if len(sys.argv) > 3 and sys.argv[3] != "all" else range(30)
MODE = sys.argv[4] if len(sys.argv) > 4 else "r"
dev = "cuda"; DS = 2; NF = 40; glc = dr.RasterizeCudaContext(); layer = soma.SOMALayer(identity_model_type="mhr", device=dev)

def so3(v):
    th = v.norm() + 1e-12; k = v / th; Kx = torch.zeros(3, 3, device=dev)
    Kx[0, 1], Kx[0, 2], Kx[1, 0], Kx[1, 2], Kx[2, 0], Kx[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    return torch.eye(3, device=dev) + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx

res = {}
for i in eps:
    inp = f"{V}/agent_runs/T1_ds_ep{i:03d}/inputs"; c = cams[str(i)]
    cj = json.load(open(f"{inp}/camera.json")); K = np.array(cj["K"]); Hh, Ww = cj["height"] // DS, cj["width"] // DS
    f0 = np.load(f"{V}/ref_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"][0].astype(float)  # frame 0 only
    t2 = np.load(f"{V}/tier2_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float)
    st = np.linalg.norm(t2[:, :3] - t2[0, :3], axis=1) < .01; Kst = int(np.argmin(st)) if not st.all() else len(st)
    ofr = np.arange(0, max(min(Kst, 30), 1), 3)
    g = trimesh.load(f"{T2DS}/{meta[i]['mesh']}", force="scene", process=False).to_geometry()
    Vo = torch.tensor(np.asarray(g.vertices) @ quat_wxyz_to_R(f0[None, 3:])[0].T + f0[:3], device=dev, dtype=torch.float32)
    Fc = torch.tensor(np.asarray(g.faces), device=dev, dtype=torch.int32)
    h = np.load(f"{V}/tier2_human/episode_{i:06d}.npz")
    with h5py.File(f"{inp}/masks.h5") as fm:
        F = min(len(fm["person"]), len(h["pose"])); hfr = np.linspace(0, F - 1, NF).astype(int)
        rs = lambda m: cv2.resize(m.astype(np.float32), (Ww, Hh), interpolation=cv2.INTER_AREA)
        MO = torch.tensor(np.stack([rs(fm["object"][t]) for t in ofr]), device=dev)
        MP = torch.tensor(np.stack([rs(fm["person"][t]) for t in ofr]), device=dev)
        DT = torch.tensor(np.stack([cv2.distanceTransform((~fm["person"][t][::DS, ::DS].astype(bool)).astype(np.uint8), cv2.DIST_L2, 5)
                                    for t in hfr]), device=dev)[:, None]
    with torch.no_grad():
        o = layer(torch.tensor(h["pose"][hfr]).float().view(-1, 77, 3).to(dev), torch.tensor(h["identity"][hfr]).float().to(dev),
                  scale_params=torch.tensor(h["scale"][hfr]).float().to(dev), transl=torch.tensor(h["transl"][hfr]).float().to(dev))
        Hv = (o.vertices.double() @ torch.tensor(FLIP.T, device=dev)).float()[:, ::10]
    R0 = torch.tensor(np.array(c["R"]), device=dev, dtype=torch.float32); t0 = torch.tensor(np.array(c["t"]), device=dev, dtype=torch.float32)
    w = torch.zeros(3, device=dev, requires_grad=True); dt = torch.zeros(3, device=dev, requires_grad=True)
    fx, cx, cy = float(K[0, 0]) / DS, float(K[0, 2]) / DS, float(K[1, 2]) / DS
    def to_cam(X):  # world -> camera (up to the sim3 scale, which projection ignores)
        R = so3(w) @ R0; return (X - (t0 + dt)) @ R
    def proj(Xc): return fx * Xc[..., 0] / Xc[..., 2] + cx, fx * Xc[..., 1] / Xc[..., 2] + cy
    def render(Xc):
        u, v = proj(Xc); z = Xc[..., 2]; n, fa = 0.05, 50.
        clip = torch.stack([(2 * u / Ww - 1) * z, (2 * v / Hh - 1) * z, z * (fa + n) / (fa - n) - 2 * fa * n / (fa - n), z], -1)
        clip = clip[None].expand(len(ofr), -1, -1).contiguous()
        rast, _ = dr.rasterize(glc, clip, Fc, resolution=[Hh, Ww]); return dr.antialias((rast[..., 3:] > 0).float(), rast, clip, Fc)[..., 0]
    def loss_fn():
        Rm = render(to_cam(Vo)); keep = 1 - MP
        iou = (Rm * MO * keep).sum((1, 2)) / (((Rm + MO - Rm * MO) * keep).sum((1, 2)) + 1)
        u, v = proj(to_cam(Hv)); gx = torch.stack([2 * u / (Ww - 1) - 1, 2 * v / (Hh - 1) - 1], -1)[:, None]
        dh = torch.nn.functional.grid_sample(DT, gx, align_corners=True, padding_mode="border").mean()
        return 1 - iou.mean() + 0.002 * dh, iou.mean(), dh
    with torch.no_grad(): _, iou0, dh0 = loss_fn()
    opt = torch.optim.Adam([{"params": [w], "lr": 2e-3}] + ([{"params": [dt], "lr": 5e-3}] if MODE == "rt" else []))
    for it in range(300):
        opt.zero_grad(); l, _, _ = loss_fn(); l.backward(); opt.step()
        if it == 200:
            for gp in opt.param_groups: gp["lr"] *= 0.2
    with torch.no_grad():
        _, iou1, dh1 = loss_fn(); Rn = (so3(w) @ R0).cpu().numpy().astype(float); tn = (t0 + dt).cpu().numpy().astype(float)
    dang = float(np.degrees(np.arccos(np.clip((np.trace(Rn.T @ np.array(c["R"])) - 1) / 2, -1, 1))))
    res[str(i)] = dict(s=c["s"], R=Rn.tolist(), t=tn.tolist(), iou0=float(iou0), iou1=float(iou1), dt0=float(dh0), dt1=float(dh1),
                       change_deg=dang, change_cm=float(torch.norm(dt) * 100), obj_frames=ofr.tolist())
    print(f"{i:2d} {meta[i]['object'][:16]:16s} iou {float(iou0):.3f}->{float(iou1):.3f} humdt {float(dh0):.1f}->{float(dh1):.1f}px  "
          f"dR {dang:.1f}d dt {float(torch.norm(dt))*100:.1f}cm  nobj {len(ofr)}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)

#!/usr/bin/env python3
"""Fit the support furniture mesh to the video: the mesh stands upright on the known ground plane (its lowest point on
the plane), leaving yaw and the 2-D position on the plane free. Objective: soft IoU between the rendered silhouette
(static camera of our Tier 3 reconstruction, inputs/camera.json) and the SAM2 support mask on the median background;
pixels covered by the person or the object in > 90% of the sampled frames (no clean background there) are ignored. Coarse grid (which of the 6 mesh axis directions is up, yaw, position
around the back-projected bottom centre of the detection box), then gradient refinement with nvdiffrast.
Writes OUT/epXXX/support.json (T_world_mesh of the support mesh, mesh path, IoU, top height) and support_world.obj
(decimated mesh in world coordinates)."""
import json, sys, numpy as np, torch, trimesh, cv2
import nvdiffrast.torch as dr
from common import *
import os
FREE_H = os.environ.get("FREE_H") == "1"; FREE_S = os.environ.get("FREE_S") == "1"; GR = float(os.environ.get("GRID_M", "0.4")); dev = "cuda"; DS = 4; UPS = [[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]; glc = dr.RasterizeCudaContext()

def axis_angle(k, a):  # torch rotation about unit axis k (3,) by angle a (scalar tensor)
    K = torch.zeros(3, 3, device=dev, dtype=k.dtype)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    return torch.eye(3, device=dev, dtype=k.dtype) + torch.sin(a) * K + (1 - torch.cos(a)) * K @ K

def fit(i):
    d = f"{OUT}/ep{i:03d}"; name = SUPPORT[i]
    cam = json.load(open(f"{inp(i)}/camera.json")); K = np.array(cam["K"]); Tw = np.array(cam["T_world_cam"]); Wi = np.linalg.inv(Tw)
    Hh, Ww = cam["height"] // DS, cam["width"] // DS
    g = np.array(json.load(open(f"{inp(i)}/ground_plane.json"))["plane"]); nrm = g[:3] / np.linalg.norm(g[:3]); g0 = g[3] / np.linalg.norm(g[:3])
    # nrm points up (height above floor = nrm.x + g0)
    mesh = trimesh.load(f"{MESH}/{name}/{name}.glb", force="scene", process=False).to_geometry()
    small = mesh.simplify_quadric_decimation(face_count=20000) if len(mesh.faces) > 20000 else mesh
    Vm = np.asarray(mesh.vertices); Fm = np.asarray(mesh.faces)  # render the full mesh (decimation opens holes)
    mask = cv2.resize((cv2.imread(f"{d}/support_mask.png", 0) > 127).astype(np.float32), (Ww, Hh), interpolation=cv2.INTER_AREA)
    occ = cv2.resize(np.load(f"{d}/occupancy.npy").astype(np.float32), (Ww, Hh), interpolation=cv2.INTER_AREA)
    keep = torch.tensor((occ < 0.9).astype(np.float32), device=dev); M = torch.tensor(mask, device=dev)
    box = json.load(open(f"{d}/detections.json"))["detections"][0]["box"]
    # back-project the bottom centre of the box onto the ground plane
    ray = np.linalg.inv(K) @ np.array([(box[0] + box[2]) / 2, box[3] * 0.97, 1.0]); rw = Tw[:3, :3] @ ray; c = Tw[:3, 3]
    q0 = c + rw * (-(nrm @ c + g0) / (nrm @ rw))
    e1 = np.cross(nrm, [1.0, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(nrm, e1)
    Kt = torch.tensor(K, device=dev, dtype=torch.float32); Wt = torch.tensor(Wi, device=dev, dtype=torch.float32)
    Fc = torch.tensor(Fm, device=dev, dtype=torch.int32); nt = torch.tensor(nrm, device=dev, dtype=torch.float32)
    E1 = torch.tensor(e1, device=dev, dtype=torch.float32); E2 = torch.tensor(e2, device=dev, dtype=torch.float32)

    def pose(sign, yaw, xy):  # mesh -> world
        A = trimesh.geometry.align_vectors(UPS[sign], nrm)[:3, :3]  # mesh up axis -> world up
        Vu = torch.tensor(Vm @ A.T, device=dev, dtype=torch.float32)
        Vu = Vu - (Vu @ nt).min() * nt                                   # lowest point on the plane through the origin
        Ry = axis_angle(nt, yaw); q = torch.tensor(q0 - (nrm @ q0 + g0) * nrm, device=dev, dtype=torch.float32) + xy[0] * E1 + xy[1] * E2
        return (Vu * SC) @ Ry.T + q + DH * nt, A, Ry, q

    def render(Vw):
        Xc = Vw @ Wt[:3, :3].T + Wt[:3, 3]; z = Xc[:, 2].clamp(min=1e-3)
        u = (Kt[0, 0] * Xc[:, 0] / z + Kt[0, 2]) / DS; v = (Kt[1, 1] * Xc[:, 1] / z + Kt[1, 2]) / DS; n_, f_ = 0.05, 50.
        clip = torch.stack([(2 * u / Ww - 1) * z, (2 * v / Hh - 1) * z, z * (f_ + n_) / (f_ - n_) - 2 * f_ * n_ / (f_ - n_), z], -1)[None].contiguous()
        rast, _ = dr.rasterize(glc, clip, Fc, resolution=[Hh, Ww]); return dr.antialias((rast[..., 3:] > 0).float(), rast, clip, Fc)[0, ..., 0]

    def iou(R_, blur=0):  # blur > 0: IoU of box-blurred silhouettes (smoother landscape for thin parts)
        Mb, R2 = M, R_
        if blur:
            pool = lambda x: torch.nn.functional.avg_pool2d(x[None, None], 2 * blur + 1, 1, blur)[0, 0]
            Mb, R2 = pool(M), pool(R_)
        return (R2 * Mb * keep).sum() / (((R2 + Mb - R2 * Mb) * keep).sum() + 1)

    DH = torch.zeros((), device=dev, requires_grad=FREE_H)  # optional free height offset (diagnostic only)
    SC = torch.ones((), device=dev, requires_grad=FREE_S)   # optional free mesh scale (diagnostic only)
    best = (-1, None)
    with torch.no_grad():
        for sign in range(6):
            for yaw in np.radians(np.arange(0, 360, 15)):
                for dx in np.arange(-GR, GR + .01, 0.1):
                    for dy in np.arange(-GR, GR + .01, 0.1):
                        s = float(iou(render(pose(sign, torch.tensor(yaw, device=dev), torch.tensor([dx, dy], device=dev))[0]), blur=4))
                        if s > best[0]: best = (s, (sign, yaw, dx, dy))
    iou_grid = best[0]; sign, yaw0, dx0, dy0 = best[1]
    yaw = torch.tensor(float(yaw0), device=dev, requires_grad=True); xy = torch.tensor([dx0, dy0], device=dev, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([{"params": [yaw], "lr": 0.01}, {"params": [xy], "lr": 0.005}] + ([{"params": [DH], "lr": 0.003}] if FREE_H else []) + ([{"params": [SC], "lr": 0.003}] if FREE_S else []))
    with torch.no_grad(): iou_grid = float(iou(render(pose(sign, yaw, xy)[0])))  # sharp IoU of the grid optimum
    keep_best = (iou_grid, yaw.detach().clone(), xy.detach().clone(), DH.detach().clone(), SC.detach().clone())
    for it in range(300):
        opt.zero_grad(); Rr = render(pose(sign, yaw, xy)[0]); l = 1 - iou(Rr, blur=4 if it < 100 else 2 if it < 200 else 0)
        if it >= 200 and 1 - float(l) > keep_best[0]: keep_best = (1 - float(l), yaw.detach().clone(), xy.detach().clone(), DH.detach().clone(), SC.detach().clone())
        l.backward(); opt.step()
        if it == 200:
            for gp in opt.param_groups: gp["lr"] *= 0.2
    with torch.no_grad():  # best parameters seen (the soft-IoU gradient of thin legs is noisy)
        yaw.copy_(keep_best[1]); xy.copy_(keep_best[2]); DH.copy_(keep_best[3]); SC.copy_(keep_best[4])
        Vw, A, Ry, q = pose(sign, yaw, xy); Rf = render(Vw); iou_f = float(iou(Rf))
        cv2.imwrite(f"{d}/fit_render_mask_keep.png", (np.concatenate([Rf.cpu().numpy(), mask, keep.cpu().numpy()], 1) * 255).astype(np.uint8))
    # full-resolution mesh transform: x_world = R x_mesh + t
    R = Ry.cpu().numpy().astype(float) @ A
    Vfull = np.asarray(mesh.vertices); hmin = (Vfull @ R.T @ nrm).min()
    t = q.cpu().numpy().astype(float) - hmin * nrm
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    hs = Vfull @ R.T @ nrm + t @ nrm + g0                              # vertex heights above the floor
    rep = dict(episode=i, support=name, mesh=f"{MESH}/{name}/{name}.glb", T_world_mesh=T.tolist(), up_axis_mesh=UPS[sign],
               yaw_deg=float(np.degrees(float(yaw)) % 360), iou_grid=iou_grid, iou=iou_f,
               top_height_m=float(hs.max()), min_height_m=float(hs.min()), detection_box=box, ground_plane=g.tolist())
    if FREE_H or FREE_S: print(i, "diagnostic: height offset %.3f m, scale %.3f, IoU %.3f" % (float(DH), float(SC), iou_f)); return
    json.dump(rep, open(f"{d}/support.json", "w"), indent=1)
    sw = small.copy(); sw.vertices = (np.asarray(small.vertices) @ R.T + t); sw.export(f"{d}/support_world.obj")
    print(f"{i:2d} {name:18s} up {str(UPS[sign]):12s} yaw {rep['yaw_deg']:5.1f} IoU grid {iou_grid:.3f} -> {iou_f:.3f}  top {rep['top_height_m']:.3f} m", flush=True)

for i in (map(int, sys.argv[1].split(",")) if len(sys.argv) > 1 else SUPPORT):
    fit(i)

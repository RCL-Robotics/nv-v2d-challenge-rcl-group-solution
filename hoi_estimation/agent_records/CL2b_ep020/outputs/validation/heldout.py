"""Held-out validation: refit theta,u,v with every 4th frame removed from the data term (weight 0) and compare
image agreement on those held-out frames with the baseline. Depth fixed to the final solve.py depths."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, cv2, h5py
from hoi_lib import Renderer, load_camera, load_mesh
from traj_opt import TrajectoryOptimizer
from solve import upright_basis, yaw_of
inp = "/work/inputs"
res = np.load(sys.argv[1]); valj = json.load(open(sys.argv[2]))
K, Twc, _, W, H = load_camera(inp); Tcw = np.linalg.inv(Twc); Rcw = Tcw[:3, :3]
plane = json.load(open(f"{inp}/ground_plane.json"))["plane"]; nup = np.array(plane[:3]); nup /= np.linalg.norm(nup)
if plane[3] < 0: nup = -nup
rec = np.load(f"{inp}/reconstruction.npz"); Twm_b = rec["T_world_mesh"].astype(np.float64); F = len(Twm_b)
Tcm_b = Tcw[None] @ Twm_b; Tcm_o = Tcw[None] @ res["T_world_mesh"]
v, f, uv, tex = load_mesh(f"{inp}/object_mesh.glb"); Rd = Renderer(v, f, uv, tex)
o = Tcm_b[:, :3, 3]; pu = o @ K.T; pu = pu[:, :2] / pu[:, 2:]
cu, cv_ = np.median(pu, 0); x0 = int(round(cu - 64)); y0 = int(round(cv_ - 56)); crop = (x0, y0, 128, 112)
cap = cv2.VideoCapture(f"{inp}/video.mp4"); imgs = []
while True:
    ok, fr = cap.read()
    if not ok: break
    full = np.zeros((H, W, 3), np.uint8); full[y0:y0+112, x0:x0+128] = fr[y0:y0+112, x0:x0+128]; imgs.append(full)
hf = h5py.File(f"{inp}/masks.h5", "r"); oms = list(hf["object"][:]); pms = list(hf["person"][:])
phases = [tuple(p) for p in valj["phases"]]
B = upright_basis(Twm_b[0, :3, :3], nup)
yaw_b = np.unwrap(yaw_of(B, Twm_b[:, :3, :3]))
init = np.concatenate([pu, np.array(valj["z_final"])[:, None]], 1)
held = np.zeros(F, bool); held[2::4] = True
opt = TrajectoryOptimizer(Rd, K, Rcw, B, crop, imgs, oms, pms, phases, yaw_b, init)
opt.optimize(iters=120, fix_z=True, frame_w=(~held).astype(np.float32), log=lambda *a: None)
iou_h, ncc_h = opt.eval_frames()
D = opt.D; dev = "cuda"
def score(T):
    I, N = [], []
    with torch.no_grad():
        for s in range(0, F, 96):
            idx = torch.arange(s, min(F, s + 96), device=dev)
            a, b = D.losses(torch.tensor(T[s:s+96, :3, :3], device=dev, dtype=torch.float32),
                            torch.tensor(T[s:s+96, :3, 3], device=dev, dtype=torch.float32), idx)
            I.append(a.cpu().numpy()); N.append(b.cpu().numpy())
    return np.concatenate(I), np.concatenate(N)
iou_b, ncc_b = score(Tcm_b)
print("phase | held-out frames | IoU baseline | IoU ours(held-out fit) | NCC baseline | NCC ours(held-out fit)")
out = []
for a, b, mv in phases:
    m = np.zeros(F, bool); m[a:b+1] = True; m &= held
    row = (f"{a}-{b}{' (moving)' if mv else ' (static)'}", int(m.sum()), iou_b[m].mean(), iou_h[m].mean(), ncc_b[m].mean(), ncc_h[m].mean())
    out.append(row); print("%s | %d | %.3f | %.3f | %.3f | %.3f" % row)
m = held
print("all | %d | %.3f | %.3f | %.3f | %.3f" % (m.sum(), iou_b[m].mean(), iou_h[m].mean(), ncc_b[m].mean(), ncc_h[m].mean()))
print("fraction of held-out frames where ours(held-out) IoU > baseline IoU: %.2f" % (iou_h[m] > iou_b[m]).mean())
print("fraction of held-out frames where ours(held-out) NCC > baseline NCC: %.2f" % (ncc_h[m] > ncc_b[m]).mean())

# Visualisation only (uses GT): render in the GT-fitted camera (gt_camera_fit.py). GT = green outline (should sit on the
# object); each estimate = its WORLD poses projected with the same camera, i.e. exactly the error the metric sees.
import sys, json, numpy as np, cv2, trimesh, subprocess, os
sys.path.insert(0, "/home/hanzhou/Projects/v2d/hoi_extraction/scripts"); sys.path.insert(0, "/home/hanzhou/Projects/v2d/agent")
from eval_object_pose import metrics, quat_wxyz_to_R
from render_compare import draw
V2 = "/mnt/nas/Group/hanzhou/v2d"; DS = f"{V2}/dataset/track_2/tier_1_multiview_caption"; REL = f"{V2}/release/tier3_s1_20261020/reconstructed_data"
idx, camf, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]; runs = [a.split("=", 1) for a in sys.argv[4:]]
m = {d["episode_index"]: d for d in map(json.loads, open(f"{DS}/meta/episodes_metadata.jsonl"))}[idx]
cam = json.load(open(camf)); K = np.array(cam["K"]); Wi = np.linalg.inv(np.array(cam["T_world_cam"]))
ws = f"{V2}/agent_runs/T1_ds_ep{idx:03d}"
mesh = trimesh.load(f"{ws}/inputs/object_mesh.glb", force="scene", process=False).to_geometry(); Vm, Fc = np.asarray(mesh.vertices), np.asarray(mesh.faces)
ref = np.load(f"{V2}/ref_poses/episode_{idx:06d}.npz")["pose_xyz_wxyz"].astype(float); okr = np.linalg.norm(ref[:, 3:], axis=1) > 0.5
G = np.tile(np.eye(4), (len(ref), 1, 1)); G[okr, :3, :3] = quat_wxyz_to_R(ref[okr, 3:]); G[:, :3, 3] = ref[:, :3]; Gc = Wi[None] @ G
np.random.seed(0); vs = np.asarray(trimesh.load(f"{DS}/{m['mesh']}", force="scene", process=False).to_geometry().sample(1000))
rows = []
for lab, p in runs:
    T = np.load(f"{REL}/{m['sequence_id']}/poses.npy") if p == "s1" else np.load(p)["T_world_mesh"]
    n = min(len(T), len(ref)); ok = okr[:n]
    a = metrics(T[:n][ok][:, :3, :3], T[:n][ok][:, :3, 3], G[:n][ok][:, :3, :3], G[:n][ok][:, :3, 3], vs)
    rows.append((lab, Wi[None] @ T[:n], a["AUC"], a["pos_cm"])); print(lab, "AUC %.3f pos %.1f cm" % (a["AUC"], a["pos_cm"]), flush=True)
def outline(img, Tc, col, th=3):
    Vc = Vm @ Tc[:3, :3].T + Tc[:3, 3]
    if (Vc[:, 2] < 0.05).any(): return img
    q = np.round((Vc[:, :2] / Vc[:, 2:3] * np.diag(K)[:2] + K[:2, 2]) * 4).astype(np.int32)
    mk = np.zeros(img.shape[:2], np.uint8); cv2.fillPoly(mk, list(q[Fc]), 255, shift=2)
    cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); img = img.copy(); cv2.drawContours(img, cs, -1, col, th, cv2.LINE_AA); return img
cap = cv2.VideoCapture(f"{ws}/inputs/video.mp4"); fps = cap.get(cv2.CAP_PROP_FPS) or 30; wr, fi, sc, cr = None, 0, 0.5, 320
while True:
    okf, fr = cap.read()
    if not okf: break
    if fi % 2 == 0:
        cols = []
        for lab, Tc, auc, pos in rows:
            img, uv = fr, None
            if fi < len(Tc): img, uv = draw(fr, K, Tc[fi], Vm, Fc, (255, 140, 40))
            if fi < len(Gc) and okr[fi]: img = outline(img, Gc[fi], (60, 255, 60))
            if fi < len(Gc) and okr[fi]:  # crop centred on the GT object
                Vg = Vm.mean(0) @ Gc[fi, :3, :3].T + Gc[fi, :3, 3]; c = Vg[:2] / Vg[2] * np.diag(K)[:2] + K[:2, 2]
            else: c = uv.mean(0) if uv is not None else np.array(img.shape[1::-1]) / 2
            cx, cy = np.clip(c, 0, np.array(img.shape[1::-1]) - 1).astype(int); h = cr // 2
            pad = cv2.copyMakeBorder(img, h, h, h, h, cv2.BORDER_CONSTANT); z = img.shape[0] * sc / cr
            crop = cv2.resize(pad[cy:cy + cr, cx:cx + cr], None, fx=z, fy=z); small = cv2.resize(img, None, fx=sc, fy=sc)
            cv2.rectangle(small, (0, 0), (small.shape[1], 84), (0, 0, 0), -1)
            cv2.putText(small, f"{lab}  frame {fi}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(small, f"AUC {auc:.3f} | mean pos err {pos:.1f} cm", (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
            cols.append(np.concatenate([small, crop], 1))
        cv = np.concatenate(cols, 0)
        cv2.putText(cv, "blue: estimate (world pose)   green outline: GT   camera fitted to GT + masks", (12, cv.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (60, 255, 60), 2, cv2.LINE_AA)
        if wr is None: wr = cv2.VideoWriter(out + ".tmp.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps / 2, cv.shape[1::-1])
        wr.write(cv)
    fi += 1
wr.release(); subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out + ".tmp.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", out], check=True); os.remove(out + ".tmp.mp4")

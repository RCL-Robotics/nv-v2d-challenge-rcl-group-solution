#!/usr/bin/env python3
"""Visual check: median background, SAM2 support mask (red tint), fitted support mesh projected (green outline),
chosen detection box (yellow). Writes OUT/vis/epXXX_fit.jpg."""
import json, sys, numpy as np, cv2, trimesh
from common import *
os.makedirs(f"{OUT}/vis", exist_ok=True)
for i in (map(int, sys.argv[1].split(",")) if len(sys.argv) > 1 else SUPPORT):
    d = f"{OUT}/ep{i:03d}"; im = cv2.imread(f"{d}/background.png"); s = json.load(open(f"{d}/support.json"))
    cam = json.load(open(f"{inp(i)}/camera.json")); K = np.array(cam["K"]); Wi = np.linalg.inv(np.array(cam["T_world_cam"]))
    m = cv2.imread(f"{d}/support_mask.png", 0) > 127; im[m] = (0.5 * im[m] + [0, 0, 127]).astype(np.uint8)
    w = trimesh.load(f"{d}/support_world.obj", process=False); V = np.asarray(w.vertices) @ Wi[:3, :3].T + Wi[:3, 3]
    q = np.round((V[:, :2] / V[:, 2:3] * np.diag(K)[:2] + K[:2, 2]) * 4).astype(np.int32)
    mk = np.zeros(im.shape[:2], np.uint8); cv2.fillPoly(mk, list(q[np.asarray(w.faces)]), 255, shift=2)
    cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(im, cs, -1, (60, 255, 60), 2)
    b = np.round(s["detection_box"]).astype(int); cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), (0, 255, 255), 2)
    cv2.putText(im, f"ep{i:02d} {s['support']} IoU {s['iou']:.2f} top {s['top_height_m']:.2f} m", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    x0, y0 = max(b[0] - 300, 0), max(b[1] - 250, 0); cv2.imwrite(f"{OUT}/vis/ep{i:03d}_fit.jpg", im[y0:b[3] + 200, x0:b[2] + 300])

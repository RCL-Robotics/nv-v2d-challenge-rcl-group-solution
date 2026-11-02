#!/usr/bin/env python3
"""GroundingDINO on the median background: all boxes for the support prompt; the chosen box is the one whose
bottom-centre region is closest to the anchor pixel (the resting object stands on / leans against it). Runs in the
v2d_gdino image. Writes OUT/epXXX/detections.json."""
import json, sys, os, numpy as np, torch, groundingdino
from PIL import Image
import groundingdino.datasets.transforms as Tr
from groundingdino.util.inference import load_model, predict
from torchvision.ops import box_convert
from common import *
TF = Tr.Compose([Tr.RandomResize([800], max_size=1333), Tr.ToTensor(), Tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
model = load_model(os.path.join(groundingdino.__path__[0], "config", "GroundingDINO_SwinT_OGC.py"), f"{V}/weights/grounding_dino/groundingdino_swint_ogc.pth")
for i in (map(int, sys.argv[1].split(",")) if len(sys.argv) > 1 else SUPPORT):
    d = f"{OUT}/ep{i:03d}"; im = np.array(Image.open(f"{d}/background.png").convert("RGB")); H, W = im.shape[:2]
    x, _ = TF(Image.fromarray(im), None)
    b, lg, ph = predict(model=model, image=x, caption=PROMPT[SUPPORT[i]], box_threshold=0.25, text_threshold=0.2)
    b = (box_convert(b, "cxcywh", "xyxy") * torch.tensor([W, H, W, H])).numpy()
    a = np.array(json.load(open(f"{d}/anchor.json"))["uv"])
    # distance from the anchor to the box (0 inside), ties broken by score
    dist = [float(np.hypot(max(bb[0] - a[0], 0, a[0] - bb[2]), max(bb[1] - a[1], 0, a[1] - bb[3]))) for bb in b]
    order = sorted(range(len(b)), key=lambda k: (dist[k], -float(lg[k])))
    det = [dict(box=b[k].tolist(), score=float(lg[k]), phrase=ph[k], anchor_dist_px=dist[k]) for k in order]
    json.dump(dict(prompt=PROMPT[SUPPORT[i]], detections=det), open(f"{d}/detections.json", "w"), indent=1)
    print(i, SUPPORT[i], "n", len(b), "chosen", [round(v) for v in det[0]["box"]] if det else None, "dist %.0f score %.2f" % (det[0]["anchor_dist_px"], det[0]["score"]) if det else "", flush=True)

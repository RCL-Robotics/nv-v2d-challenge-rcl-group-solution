#!/usr/bin/env python3
"""SAM2 (image predictor, box prompt = chosen GroundingDINO box) on the median background -> support mask.
Runs in the v2d_cari4d image. Writes OUT/epXXX/support_mask.png."""
import json, os, sys, numpy as np, torch
from PIL import Image
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sam2.sam2_image_predictor import SAM2ImagePredictor
from common import *
CK = f"{V}/weights/sam2"
from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()  # sam2 initialises hydra on import
with initialize_config_dir(config_dir=CK, job_name="sam2"):
    cfg = compose(config_name="sam2.1_hiera_l.yaml")
OmegaConf.resolve(cfg); model = instantiate(cfg.model, _recursive_=True)
model.load_state_dict(torch.load(f"{CK}/sam2.1_hiera_large.pt", map_location="cpu")["model"], strict=True)
pred = SAM2ImagePredictor(model.to("cuda").eval())
for i in (map(int, sys.argv[1].split(",")) if len(sys.argv) > 1 else SUPPORT):
    d = f"{OUT}/ep{i:03d}"; im = np.array(Image.open(f"{d}/background.png").convert("RGB"))
    det = json.load(open(f"{d}/detections.json"))["detections"]
    if not det: print(i, "no detection"); continue
    with torch.inference_mode():
        pred.set_image(im); m, s, _ = pred.predict(box=np.array(det[0]["box"]), multimask_output=False)
    Image.fromarray((m[0] > 0).astype(np.uint8) * 255).save(f"{d}/support_mask.png")
    print(i, SUPPORT[i], "score %.3f area %d px" % (float(s[0]), int((m[0] > 0).sum())), flush=True)

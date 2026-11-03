#!/usr/bin/env python3
"""Contact sheet of all fit overlays -> OUT/vis/sheet_fits.jpg"""
import cv2, numpy as np
from common import *
ims = []
for i in SUPPORT:
    im = cv2.imread(f"{OUT}/vis/ep{i:03d}_fit.jpg"); h, w = im.shape[:2]; s = 420 / h; ims.append(cv2.resize(im, (int(w * s), 420)))
W = max(x.shape[1] for x in ims); ims = [cv2.copyMakeBorder(x, 0, 0, 0, W - x.shape[1], cv2.BORDER_CONSTANT) for x in ims]
ims += [np.zeros((420, W, 3), np.uint8)] * (-len(ims) % 4)  # 4 per row (14 episodes -> 4x4 with 2 blank)
cv2.imwrite(f"{OUT}/vis/sheet_fits.jpg", np.concatenate([np.concatenate(ims[k:k + 4], 1) for k in range(0, len(ims), 4)], 0))

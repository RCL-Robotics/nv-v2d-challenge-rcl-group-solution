from pathlib import Path
p=Path('/work/outputs')
pieces=['''#!/usr/bin/env python3
"""Conservative deterministic image-evidence refinement for the supplied episode.

Run offline: python solve.py --inputs /work/inputs --out object_poses_world.npz
No trajectory cache, scratch files, downloaded data, or model weights are read.
All thresholds below are explicit reproducible algorithm constants.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import cv2, h5py, torch, trimesh
import nvdiffrast.torch as dr
from scipy.ndimage import distance_transform_edt, binary_dilation
from scipy.spatial.transform import Rotation
''']
s=(p/'render_utils.py').read_text();pieces.append(s[s.index('class Renderer:'):])
s=(p/'refine.py').read_text();pieces.append(s[s.index('def make_data('):s.index('def rotmat(')])
s=(p/'deterministic_search.py').read_text();pieces.append(s[s.index('@torch.no_grad()'):])
s=(p/'build_deterministic.py').read_text()
s=s[s.index('def generate('):s.index("if __name__=='__main__':")]
s=s.replace('def generate(inputs,save,steps=85):','def generate(inputs):').replace('optimize(r,','search(r,')
s=s.replace("np.savez_compressed(save,R=Rall,t=tall,scores=score,stats=stats,**data);print('done',time.time()-start,flush=True)","print('base search seconds',time.time()-start,flush=True);return dict(R=Rall,t=tall,scores=score,stats=stats,**data),r")
pieces.append(s)
s=(p/'blue_metrics.py').read_text();pieces.append(s[s.index('@torch.no_grad()'):])
s=(p/'roll_refine.py').read_text();pieces.append(s[s.index('def add_roll_candidates('):s.index("if __name__=='__main__':")])
s=(p/'conservative_select.py').read_text();pieces.append(s[s.index('def angle('):s.index("if __name__=='__main__':")])
pieces.append((p/'final_main.py').read_text())
(p/'solve.py').write_text(chr(10).join(pieces))
print('standalone solver lines',len((p/'solve.py').read_text().splitlines()))

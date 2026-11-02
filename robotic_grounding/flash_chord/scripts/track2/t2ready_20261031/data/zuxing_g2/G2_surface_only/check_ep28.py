"""Ep28 check: distance of this pipeline's pose_ep28.npy (each <out dir> given) to the original B_hands output and to
the initial estimate: max rotation difference (deg) and mean vertex distance over the kit's 500 points (cm)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import proxies as PX
from common import kc, mesh_path_for, sample_vertices, INIT_FMT
REF = '/root/autodl-tmp/zuxing.lu/tmp/tier2_noise/kfix/ep28/B_hands/pose_ep28.npy'
pts = sample_vertices(mesh_path_for(kc.T2 / 'mesh', 'squeegee')) * 100.0
ref = np.load(REF); init = np.load(INIT_FMT.format(ep=28))
print('initial vs B_hands', {k: round(v, 3) for k, v in PX.pose_diff(pts, init, ref).items()})
for d in sys.argv[1:]:
    a = np.load(os.path.join(d, 'pose_ep28.npy'))
    print(os.path.basename(os.path.normpath(d)), 'vs B_hands', {k: round(v, 3) for k, v in PX.pose_diff(pts, a, ref).items()})
    print(os.path.basename(os.path.normpath(d)), 'vs initial', {k: round(v, 3) for k, v in PX.pose_diff(pts, a, init).items()})

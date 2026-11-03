"""Independent physical depth cues per phase (report only; solve.py keeps the baseline depth by default).
python depth_cues.py result.npz
Each cue gives the camera-depth shift (cm, + = farther) of the result that the cue would suggest."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from hoi_lib import *
from solve import detect_phases, median_image, evidence

inp = '/work/inputs'
cam, K, Twc, rec = load_inputs(inp)
Tcw = np.linalg.inv(Twc)
Rd = Renderer(load_mesh(inp)); fd = FrameData(inp)
Tw = np.load(sys.argv[1])['T_world_mesh']
Tc = Tcw[None] @ Tw
F = len(Tw)
e1, s2, area, _ = detect_phases(fd.om)
plane = np.array(json.load(open(f'{inp}/ground_plane.json'))['plane'])
n, d = plane[:3], plane[3]
sgn = 1 if n[1] < 0 else -1  # positive = above the floor (world y-down)
v = Rd.verts_np.astype(np.float64)
kp = rec['human_keypoints_world'].astype(np.float64)
cp = rec['contact_prob_right_left']
LTIPS = [42, 46, 50, 54, 58]


def scale_ray(X, s):
    Y = X.copy(); Y[:3, 3] *= s; return Y


def size_cue(frames):
    X = Tc[frames[len(frames) // 2]]
    box = frame_box(K, X, scale=1.4); hw = (256, 256)
    med, om, pm = median_image(fd, frames, box, hw)
    zs = np.arange(-0.30, 0.301, 0.01)
    H = np.stack([scale_ray(X, (X[2, 3] + dz) / X[2, 3]) for dz in zs])
    _, iou = evidence(Rd, K, box, hw, H, med, om, pm)
    return zs[np.argmax(iou)] * 100, iou.max(), iou[np.argmin(np.abs(zs))]


def floor_cue(frames):
    X = Tc[frames[len(frames) // 2]]
    best = None
    for dz in np.arange(-0.40, 0.401, 0.002):
        Y = Twc @ scale_ray(X, (X[2, 3] + dz) / X[2, 3])
        low = (((v[::20] @ Y[:3, :3].T + Y[:3, 3]) @ n + d) * sgn).min()
        if best is None or abs(low) < abs(best[1]):
            best = (dz, low)
    return best[0] * 100


def hand_cue(frames):
    """Object head (grasped trigger) vs left-hand fingertip centroid, camera depth difference."""
    out = []
    for i in frames:
        tips = (Tcw[:3, :3] @ kp[i, LTIPS].mean(0) + Tcw[:3, 3])
        head = Tc[i, :3, :3] @ np.array([0, 0, 0.12]) + Tc[i, :3, 3]
        out.append(tips[2] - head[2])
    return np.median(out) * 100, np.std(out) * 100


S1 = np.arange(0, e1 + 1); S2 = np.arange(s2, F); C = np.arange(e1 + 1, s2)
grasp = [i for i in C if cp[i].max() > 0.5 and area[i] > 400]
print('phase S1 (table):')
print('  metric size vs image (median-image IoU over depth shifts): shift %+.0f cm (IoU %.3f vs %.3f at 0)' % size_cue(S1))
g1 = [i for i in grasp if i < e1 + 15]
print('  hand contact at lift-off (frames %d-%d): shift %+.1f cm (spread %.1f)' % (g1[0], g1[-1], *hand_cue(g1)))
print('phase carried:')
print('  hand contact (%d frames with contact>0.5 and visible object): shift %+.1f cm (spread %.1f)' % (len(grasp), *hand_cue(grasp)))
print('phase S2 (floor):')
print('  floor contact (lowest mesh point on the released plane): shift %+.1f cm' % floor_cue(S2))
print('  metric size vs image: shift %+.0f cm (IoU %.3f vs %.3f at 0)' % size_cue(S2))
g2 = [i for i in grasp if i > s2 - 25]
if g2:
    print('  hand contact at release (frames %d-%d): shift %+.1f cm (spread %.1f)' % (g2[0], g2[-1], *hand_cue(g2)))

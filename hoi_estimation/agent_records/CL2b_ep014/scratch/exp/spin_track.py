# per-frame best spin (mod 72) relative to given poses by silhouette IoU (full-res), to detect real spinning
import sys; sys.path.insert(0, '/work/outputs')
from common import *
import h5py
poses = sys.argv[1]; frames = eval(sys.argv[2])
D = load_inputs('/work/inputs'); mesh = load_mesh('/work/inputs')
Tw = D['rec']['T_world_mesh'].astype(np.float64) if poses == 'base' else np.load(poses)['T_world_mesh']
Tc = np.linalg.inv(D['Twc']) @ Tw
R = Renderer(mesh, D['K'], 1152, 1536)
h = h5py.File('/work/inputs/masks.h5')
ax_p = np.array([0.006, 0.004, 0.0])
angs = np.arange(-36, 36, 1.5)


def spinT(T, deg):
    S = np.eye(4); S[:3, :3] = axis_angle([0, 0, 1], np.radians(deg)); S[:3, 3] = ax_p - S[:3, :3] @ ax_p
    return T @ S


for f in frames:
    m = torch.tensor(h['object'][f], device='cuda'); p = torch.tensor(h['person'][f], device='cuda')
    valid = ~(p & ~m)
    o = R.render(np.stack([spinT(Tc[f], a) for a in angs]), rgb=False)['mask'] & valid
    iou = ((o & m).sum((1, 2)) / (o | m).sum((1, 2))).cpu().numpy()
    i = np.argmax(iou)
    print(f, 'best %.1f iou %.3f (min %.3f)' % (angs[i], iou[i], iou.min()), flush=True)

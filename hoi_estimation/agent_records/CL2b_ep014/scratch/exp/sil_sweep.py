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
angs = np.arange(-180, 180, 2)


def spinT(T, deg):
    S = np.eye(4); S[:3, :3] = axis_angle([0, 0, 1], np.radians(deg)); S[:3, 3] = ax_p - S[:3, :3] @ ax_p
    return T @ S


acc = np.zeros(len(angs))
for f in frames:
    m = torch.tensor(h['object'][f], device='cuda'); p = torch.tensor(h['person'][f], device='cuda')
    valid = ~(p & ~m)
    ious = []
    for s0 in range(0, len(angs), 30):
        o = R.render(np.stack([spinT(Tc[f], a) for a in angs[s0:s0 + 30]]), rgb=False)['mask'] & valid
        ious.append(((o & m).sum((1, 2)) / (o | m).sum((1, 2))).cpu().numpy())
    acc += np.concatenate(ious)
acc /= len(frames)
for g in [0, 72, 144, -144, -72]:
    i = np.argmin(np.abs(((angs - g) + 180) % 360 - 180) - 0) if False else None
    w = np.abs(((angs - g + 180) % 360) - 180) <= 8
    j = np.where(w)[0][np.argmax(acc[w])]
    print('grid %4d: best %4d iou %.4f' % (g, angs[j], acc[j]))

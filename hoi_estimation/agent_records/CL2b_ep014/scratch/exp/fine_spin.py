import sys; sys.path.insert(0, '/work/outputs')
from common import *
import h5py
poses = sys.argv[1]; frames = eval(sys.argv[2])
D = load_inputs('/work/inputs'); mesh = load_mesh('/work/inputs')
Tc = np.linalg.inv(D['Twc']) @ np.load(poses)['T_world_mesh']
R = Renderer(mesh, D['K'], 1152, 1536)
h = h5py.File('/work/inputs/masks.h5')
c = np.array([0.0007, 0.0137, 0]); ax = np.array([0.0073, -0.001, 1]); ax /= np.linalg.norm(ax)
angs = np.arange(-16, 17, 2)
def spinT(T, deg):
    S = np.eye(4); S[:3, :3] = axis_angle(ax, np.radians(deg)); S[:3, 3] = c - S[:3, :3] @ c
    return T @ S
acc = np.zeros(len(angs))
for f in frames:
    m = torch.tensor(h['object'][f], device='cuda'); p = torch.tensor(h['person'][f], device='cuda')
    valid = ~(p & ~m)
    o = R.render(np.stack([spinT(Tc[f], a) for a in angs]), rgb=False)['mask'] & valid
    acc += ((o & m).sum((1, 2)) / (o | m).sum((1, 2))).cpu().numpy()
print(' '.join('%d:%.4f' % (a, v) for a, v in zip(angs, acc / len(frames))))

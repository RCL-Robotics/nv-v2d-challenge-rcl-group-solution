# sweep spin about table axis for a few frames; score silhouette IoU and texture NCC
import sys; sys.path.insert(0, '/work/outputs')
from common import *
import torch.nn.functional as Fn
poses = sys.argv[1]; frames = [int(x) for x in sys.argv[2].split(',')]
D = load_inputs('/work/inputs'); mesh = load_mesh('/work/inputs')
Tw = D['rec']['T_world_mesh'].astype(np.float64) if poses == 'base' else np.load(poses)['T_world_mesh']
Tc = np.linalg.inv(D['Twc']) @ Tw
V = np.load('/work/scratch/vid_half.npy', mmap_mode='r'); M = np.load('/work/scratch/mobj_half.npy', mmap_mode='r'); P = np.load('/work/scratch/mper_half.npy', mmap_mode='r')
K = D['K'].copy(); K[:2] *= 0.5
R = Renderer(mesh, K, 576, 768)
ax_p = np.array([0.006, 0.004, 0.0])
angs = np.arange(-180, 180, 3)


def spinT(T, deg):
    S = np.eye(4); S[:3, :3] = axis_angle([0, 0, 1], np.radians(deg)); S[:3, 3] = ax_p - S[:3, :3] @ ax_p
    return T @ S


def hp(x, s=2.0):
    return x - Fn.avg_pool2d(x, 9, 1, 4)


for f in frames:
    Ts = np.stack([spinT(Tc[f], d) for d in angs])
    out = R.render(Ts)
    im = torch.tensor(V[f], device='cuda').float().mean(-1) / 255
    m = torch.tensor(M[f], device='cuda'); p = torch.tensor(P[f], device='cuda')
    valid = ~(p & ~m)
    rm = out['mask'] & valid
    iou = ((rm & m).sum((1, 2)) / (rm | m).sum((1, 2))).cpu().numpy()
    g = out['rgb'].mean(-1)
    reg = (out['mask'] & m & ~p).float()
    a = hp(g[:, None])[:, 0]; b = hp(im[None, None])[:, 0].expand_as(a)
    def ncc(a, b, w):
        n = w.sum((1, 2)) + 1
        am = (a * w).sum((1, 2)) / n; bm = (b * w).sum((1, 2)) / n
        ac = (a - am[:, None, None]) * w; bc = (b - bm[:, None, None]) * w
        return ((ac * bc).sum((1, 2)) / ((ac ** 2).sum((1, 2)).sqrt() * (bc ** 2).sum((1, 2)).sqrt() + 1e-6)).cpu().numpy()
    n1 = ncc(a, b, reg); n0 = ncc(g, im[None].expand_as(g), reg)
    print('frame', f)
    best = np.argsort(-iou)[:6]
    print('  top iou:', [(int(angs[i]), round(float(iou[i]), 3), round(float(n1[i]), 3)) for i in best])
    best = np.argsort(-n1)[:6]
    print('  top hpNCC:', [(int(angs[i]), round(float(iou[i]), 3), round(float(n1[i]), 3)) for i in best])
    best = np.argsort(-n0)[:4]
    print('  top NCC:', [(int(angs[i]), round(float(iou[i]), 3), round(float(n0[i]), 3)) for i in best])

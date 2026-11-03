import sys; sys.path.insert(0, '/work/outputs')
from common import *
from rectify import *
import h5py
poses = sys.argv[1]; which = sys.argv[2]; segs = eval(sys.argv[3]); tag = sys.argv[4]
D = load_inputs('/work/inputs'); mesh = load_mesh('/work/inputs')
Tw = D['rec']['T_world_mesh'].astype(np.float64) if poses == 'base' else np.load(poses)['T_world_mesh']
Tc = np.linalg.inv(D['Twc']) @ Tw
K = D['K']
R = Renderer(mesh, K, 1152, 1536)
if which == 'under':
    z0, sign = -0.092, +1
else:
    z0, sign = -0.115, -1
P = plane_grid(z0)
# mesh reference: frontal synthetic camera, board-only faces
v = mesh['v']; f = mesh['f']
keep = v[f][:, :, 2].max(1) < 0.0 if which == 'under' else np.ones(len(f), bool)
m2 = dict(mesh); m2['f'] = f[keep]
Kr = np.array([[2000, 0, 600], [0, 2000, 600], [0, 0, 1.]])
Rr = Renderer(m2, Kr, 1200, 1200)
Tr = np.eye(4)
if sign > 0:  # look along -z (camera at +z)
    Tr[:3, :3] = np.diag([1, -1, -1])
Tr[:3, 3] = -Tr[:3, :3] @ np.array([AX[0], AX[1], z0]) + np.array([0, 0, 2.5])
o = Rr.render(Tr[None], depth=True)
ref, wref = rectify(o['rgb'][0].cpu().numpy(), Tr, Kr, P, depth=o['depth'][0].cpu().numpy(), normal_sign=sign)
cv2.imwrite('/work/scratch/frames/rect_ref_%s.png' % which, (ref[..., ::-1] * 255).clip(0, 255).astype(np.uint8))
np.save('/work/scratch/rect_ref_%s.npy' % which, np.concatenate([ref, wref[..., None]], -1))
h = h5py.File('/work/inputs/masks.h5')
cap = cv2.VideoCapture('/work/inputs/video.mp4')
for si, (a, b, st) in enumerate(segs):
    acc = 0; wacc = 0
    for fr in range(a, b, st):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr); ok, im = cap.read(); im = im[:, :, ::-1].astype(np.float32) / 255
        od = R.render(Tc[fr][None], rgb=False, depth=True)
        vm = (h['object'][fr] & ~h['person'][fr]).astype(np.float32)
        vm = cv2.erode(vm, np.ones((5, 5), np.uint8))
        rr, w = rectify(im, Tc[fr], K, P, depth=od["depth"][0].cpu().numpy(), valid_mask=vm, normal_sign=sign, min_facing=0.08)
        # per-frame photometric normalisation (gain) on valid pixels
        if w.sum() < 100: continue
        g = (rr * w[..., None]).sum((0, 1)) / w.sum()
        acc = acc + rr / g * w[..., None]; wacc = wacc + w
    img = acc / np.maximum(wacc, 1e-6)[..., None]
    np.save('/work/scratch/rect_%s_%s_%d.npy' % (tag, which, si), np.concatenate([img, wacc[..., None]], -1))
    vis = (img / np.percentile(img[wacc > 0], 99) * 0.9).clip(0, 1)
    cv2.imwrite('/work/scratch/frames/rect_%s_%s_%d.png' % (tag, which, si), (vis[..., ::-1] * 255).astype(np.uint8))
    print('seg', a, b, 'coverage', (wacc > 0).mean())

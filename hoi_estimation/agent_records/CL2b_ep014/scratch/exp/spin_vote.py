import sys; sys.path.insert(0, '/work/outputs')
from common import *
from match import *
import h5py
poses = sys.argv[1]; frames = eval(sys.argv[2]); kind = sys.argv[3] if len(sys.argv) > 3 else 'superpoint'
spins = [0, 72, 144, -144, -72]
D = load_inputs('/work/inputs'); mesh = load_mesh('/work/inputs')
Tw = D['rec']['T_world_mesh'].astype(np.float64) if poses == 'base' else np.load(poses)['T_world_mesh']
Tc = np.linalg.inv(D['Twc']) @ Tw
R = Renderer(mesh, D['K'], 1152, 1536)
mt = Matcher(kind)
h = h5py.File('/work/inputs/masks.h5')
cap = cv2.VideoCapture('/work/inputs/video.mp4')
ax_p = np.array([0.006, 0.004, 0.0])


def spinT(T, deg):
    S = np.eye(4); S[:3, :3] = axis_angle([0, 0, 1], np.radians(deg)); S[:3, 3] = ax_p - S[:3, :3] @ ax_p
    return T @ S


tot = np.zeros(len(spins))
for f in frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read(); im = im[:, :, ::-1].astype(np.float32) / 255
    om = h['object'][f]; pm = h['person'][f]
    keep = cv2.dilate(om.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    cnt = []
    for s in spins:
        o = R.render(spinT(Tc[f], s)[None])
        rm = o['mask'][0].cpu().numpy(); rgb = o['rgb'][0].cpu().numpy()
        box = crop_box(rm | om)
        tr, A = to_tensor_crop(im * keep[..., None], box, 640)
        tq, _ = to_tensor_crop(rgb.astype(np.float32), box, 640)
        k0, k1, sc = mt.match(tq, tr)
        d = np.linalg.norm(k0 - k1, axis=1) / A[0, 0]
        cnt.append(int((d < 6).sum()))
    cnt = np.array(cnt); tot += cnt
    print(f, dict(zip(spins, cnt)), flush=True)
print('total', dict(zip(spins, tot)))

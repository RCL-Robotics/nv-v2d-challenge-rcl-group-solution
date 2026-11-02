import sys; sys.path.insert(0, '/work/outputs')
from common import *
from match import *
import h5py
poses = sys.argv[1]; frames = [int(x) for x in sys.argv[2].split(',')]
kind = sys.argv[3] if len(sys.argv) > 3 else 'superpoint'
spins = [int(x) for x in sys.argv[4].split(',')] if len(sys.argv) > 4 else [0, 72, 144, -144, -72]
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


def spin_of(Tref, T):
    Rr = Tref[:3, :3].T @ T[:3, :3]
    rv = cv2.Rodrigues(Rr)[0][:, 0]
    return np.degrees(np.arctan2(Rr[1, 0], Rr[0, 0])), np.degrees(np.linalg.norm(rv))


for f in frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read(); im = im[:, :, ::-1].copy()
    om = h['object'][f]; pm = h['person'][f]
    for s in spins:
        T0 = spinT(Tc[f], s)
        r = register(mt, R, im, om, pm, T0, D['K'])
        if r is None or r['T'] is None:
            print(f, s, 'fail', r and (r['n_match'], r['n_inl'])); continue
        sp, ang = spin_of(Tc[f], r['T'])
        print(f, 'init spin', s, 'match', r['n_match'], 'inl', r['n_inl'], 'disp %.1f' % r['med_disp'],
              '-> spin vs ref %.1f  total rot %.1f  dt(cm) %s' % (sp, ang, np.round((r['T'][:3, 3] - Tc[f][:3, 3]) * 100, 1)))

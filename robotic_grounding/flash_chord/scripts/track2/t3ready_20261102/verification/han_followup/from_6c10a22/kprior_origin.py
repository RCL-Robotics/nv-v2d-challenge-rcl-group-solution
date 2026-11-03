import json, sys, numpy as np
d = json.load(open(sys.argv[1]))['per_episode']
KP = np.array([v['K_pipeline'] for v in d.values()]); K = np.array([v['K'] for v in d.values()])
fx, fy, cx, cy = KP[:, 0, 0], KP[:, 1, 1], KP[:, 0, 2], KP[:, 1, 2]
print('n', len(KP), 'fx==fy', bool(np.allclose(fx, fy)))
for name, a in [('f', fx), ('cx', cx), ('cy', cy)]:
    print(f"{name}: mean {a.mean():.3f} median {np.median(a):.3f} min {a.min():.2f} max {a.max():.2f}")
print('released K (agent camera.json): f', K[:,0,0].min(), K[:,0,0].max(), 'cx', K[:,0,2].min(), K[:,0,2].max(), 'cy', K[:,1,2].min(), K[:,1,2].max())
print('image centre', 1536/2, 1152/2)

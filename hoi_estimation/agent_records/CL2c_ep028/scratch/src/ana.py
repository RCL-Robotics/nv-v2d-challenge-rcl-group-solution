import numpy as np
from opt import Rb,tb,rot_angle_deg
d=np.load('../cands.npz'); R,t,L=d['R'],d['t'],d['L']
F,G=L.shape
best=L.argmin(1)
ang=rot_angle_deg(R[np.arange(F),best],Rb)
# head-end direction in camera (mesh -z axis) of best vs baseline
for s in range(0,F,20):
    sl=slice(s,s+20)
    bc=np.bincount(best[sl],minlength=G)
    print(s,'best counts',bc,'medL %.2f'%np.median(L[sl].min(1)),'base(h0) %.2f'%np.median(L[sl,0]),'ang2base %.0f'%np.median(ang[sl]))

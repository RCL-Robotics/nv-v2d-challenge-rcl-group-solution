import numpy as np
from viterbi import viterbi
from opt import Rb,tb,rot_angle_deg
d=np.load('../cands.npz'); R,t,L=d['R'],d['t'],d['L']
F=len(L)
p=viterbi(R,t,L)
Rp=R[np.arange(F),p]; tp=t[np.arange(F),p]
a=rot_angle_deg(Rp,Rb)
dth=rot_angle_deg(Rp[1:],Rp[:-1])
np.save('../vit_path.npy',p)
for s in range(0,F,10):
    print(s, 'h',p[s:s+10], 'ang2base %4.0f'%np.median(a[s:s+10]), 'maxstep %5.1f'%dth[s:s+10].max(), 'L %.2f'%L[np.arange(s,min(F,s+10)),p[s:s+10]].mean())

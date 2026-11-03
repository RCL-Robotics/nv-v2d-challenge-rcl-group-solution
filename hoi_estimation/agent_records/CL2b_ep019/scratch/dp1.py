import sys; sys.path.insert(0,'/work/outputs')
from common import *; from decode import *
d=np.load('pass1.npz'); L=d['loss']; T=d['T']
F=len(T)
static=np.zeros(F,bool); static[0:113]=True; static[769:]=True
path=viterbi([T[f] for f in range(F)],[L[f]/0.2 for f in range(F)],static=static)
P=T[np.arange(F),path]; np.save('dp1.npy',P)
b=L.argmin(1)
print('frames where DP != argmin:',(np.array(path)!=b).sum())
dr=rot_angle(P[1:,:3,:3],P[:-1,:3,:3]); print('rot steps >15', np.where(dr>15)[0], dr[dr>15].round())
for s in range(0,F,40): print(s,''.join(str(x) for x in path[s:s+40]))

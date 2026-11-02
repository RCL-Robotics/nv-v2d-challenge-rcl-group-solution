import sys; sys.path.insert(0,'/work/outputs')
from hoi_fit import *
S=setup(); pm=S['pm']; par=S['par_base']
res=np.load(sys.argv[1]); T=S['Tb'].copy()
for fi in range(len(T)):
    if res[fi,0,3]==0: continue
    j=np.argmin(res[fi,:,3]); T[fi]=pm.pose(*res[fi,j,:3])
    o=np.argsort(res[fi,:,3]); d=lambda x: (np.degrees(x-par[fi,0])+180)%360-180
    print(fi, ' '.join(f"{d(res[fi,j,0]):7.1f}/{res[fi,j,3]:.3f}/({(res[fi,j,1]-par[fi,1])*100:.1f},{(res[fi,j,2]-par[fi,2])*100:.1f})" for j in o[:4]))
np.save(sys.argv[2],T)

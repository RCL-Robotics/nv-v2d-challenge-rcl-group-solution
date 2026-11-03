exec(open('/work/scratch/dev.py').read().split('info={}')[0])
import pickle
from hyp import *; from constrained import *
from stages import *
cs0=pickle.load(open('/work/scratch/cache/A_sole.pkl','rb'))
sole0=PlaneModel(V,SOLE_N,n_c,D_FIT)
kf=list(range(0,660,5)); ps=viterbi_plane(cs0,kf)
sel=np.array([[cs0[f][k][i] for k in ("psi","a","b")] for f,i in zip(kf,ps)])
R,t=sole0.pose(sel[:,0],sel[:,1],sel[:,2])
fn=unit(SOLE_N)
print('init angles',[round(float(np.degrees(np.arccos(-(Ri@fn)@n_c))),2) for Ri in R[:5]])
print('scores',[round(float(cs0[f]['score'][i]),3) for f,i in zip(kf[:10],ps[:10])])
print('psi deg',np.degrees(sel[:20,0]).round(1))
R2,t2,s,c=fit_batch(fit,R[:20],t[:20],kf[:20],data,iters=120,lr_r=0.006,lr_t=0.002,w_col=0.3)
print('after',[round(float(np.degrees(np.arccos(-(Ri@fn)@n_c))),2) for Ri in R2], s.round(3), c.round(3))

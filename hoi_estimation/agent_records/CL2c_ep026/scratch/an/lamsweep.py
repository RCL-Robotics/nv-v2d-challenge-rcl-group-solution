import sys; sys.path.insert(0,'/work/outputs')
import solve
from common import *; from render import Renderer as R2; from scorer import Scorer as S2, crop_for
r,cam,pl,m=load(); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
Tb=r['T_world_mesh'].astype(float); F=len(Tb)
d=np.load('/work/scratch/run2_diag.npz'); meas=d['meas']; X=d['X']; fs=d['fscore']; h0=float(d['h0'])
phases=[(int(a),int(b),'moving' if mv else 'static') for a,b,mv in d['phases']]
M=solve.FloorModel(pl,(Tb[0]@np.diag([-1,1,-1,1.]))[:3,:3],Tb[0][:3,3],h0); M.set_mesh(np.asarray(m.vertices)[::10])
R=R2(m,K); msk=h5py.File('/work/inputs/masks.h5'); S=S2(R,'/work/scratch/frames',msk)
seq=X[:,0].copy()
mu=meas.copy()
for (s,e,k) in phases:
    if k=='moving':
        y=np.unwrap(meas[s:e,0]); y+=np.round((X[s,0]-y[0])/(2*np.pi))*2*np.pi; mu[s:e,0]=y
mu[np.isnan(mu)]=0
sp={(s,e):X[s] for (s,e,k) in phases if k=='static'}
wts=np.where(np.isnan(fs),0,np.maximum(fs-0.8,0.05))
held=[]
w2=wts.copy()
for (s,e,k) in phases:
    if k=='moving':
        idx=list(range(s+2,e-2,4)); held+=idx; w2[idx]=0
held=np.array(held)
cache={}
def evalX(Xs):
    tot=[]
    for t in held:
        T=M.pose(*Xs[t]); sc=S.score(t,(Tcw@T)[None],crop_for(msk['object'][t])); tot.append(sc[0][0]+sc[1][0])
    return np.mean(tot)
# per-frame (unsmoothed) on held-out frames = upper bound reference
print('per-frame own fit (not held out): %.4f'%evalX(np.where(np.isnan(meas),X,mu)))
# linear interpolation from neighbours
Xi=mu.copy()
for t in held: Xi[t]=0.5*(mu[t-1]+mu[t+1])
print('neighbour interp: %.4f'%evalX(Xi))
for lam in [0.3,1,3,10,30,100,300,1000]:
    Xs=solve.smooth_trajectory(F,phases,mu,w2,sp,[lam]*5)
    print('lam %6.1f held-out score %.4f'%(lam,evalX(Xs)))

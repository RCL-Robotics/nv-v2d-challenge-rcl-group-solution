import sys; sys.path.insert(0,'/work/outputs'); import solve
from common import *
r,cam,pl,m=load(); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc); Tb=r['T_world_mesh'].astype(float); F=len(Tb)
d=np.load('/work/scratch/run2_diag.npz'); meas=d['meas']; X=d['X']; fs=d['fscore']; h0=float(d['h0'])
phases=[(int(a),int(b),'moving' if mv else 'static') for a,b,mv in d['phases']]
M=solve.FloorModel(pl,(Tb[0]@np.diag([-1,1,-1,1.]))[:3,:3],Tb[0][:3,3],h0); M.set_mesh(np.asarray(m.vertices)[::10])
mu=meas.copy()
for (s,e,k) in phases:
    if k=='moving':
        y=np.unwrap(meas[s:e,0]); y+=np.round((X[s,0]-y[0])/(2*np.pi))*2*np.pi; mu[s:e,0]=y
mu[np.isnan(mu)]=0
sp={(s,e):X[s] for (s,e,k) in phases if k=='static'}
wts=np.where(np.isnan(fs),0,np.maximum(fs-0.8,0.05))
for name,tilt in [('run2b',True),('run2b_flat',False)]:
    mm=mu.copy()
    if not tilt: mm[:,3:]=0
    Xs=solve.smooth_trajectory(F,phases,mm,wts,sp,[3.0]*5)
    T=np.stack([M.pose(*Xs[t]) for t in range(F)])
    Tc=Tcw[None]@T; s=np.median((Tcw[None]@Tb)[:,2,3])/np.median(Tc[:,2,3]); Tc[:,:3,3]*=s
    np.savez(f'/work/scratch/{name}.npz',T_world_mesh=Twc[None]@Tc)
    print(name, 'max tilt deg', np.degrees(np.abs(Xs[:,3:]).max(0)))

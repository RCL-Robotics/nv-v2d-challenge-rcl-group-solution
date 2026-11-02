import sys; sys.path.insert(0,'/work/outputs'); import solve
from common import *; from render import Renderer as R2; from scorer import Scorer as S2, crop_for
r,cam,pl,m=load(); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc); Tb=r['T_world_mesh'].astype(float)
d=np.load('/work/scratch/run2_diag.npz'); meas=d['meas']; h0=float(d['h0'])
M=solve.FloorModel(pl,(Tb[0]@np.diag([-1,1,-1,1.]))[:3,:3],Tb[0][:3,3],h0); M.set_mesh(np.asarray(m.vertices)[::10])
R=R2(m,K); msk=h5py.File('/work/inputs/masks.h5'); S=S2(R,'/work/scratch/frames',msk)
for t in [462,470,476]:
    crop=crop_for(msk['object'][t])
    def sf(poses): 
        sc=S.score(t,np.stack([Tcw@P for P in poses]),crop); return sc[0]+sc[1]
    x=meas[t,:3]
    for roll in [-6,-3,0,3,6]:
        for dh in [-0.01,0,0.01]:
            def sf2(poses,dh=dh):
                return sf([np.vstack([np.hstack([P[:3,:3],(P[:3,3]+dh*M.n)[:,None]]),[0,0,0,1]]) for P in poses])
            b,bs=solve.local_search(sf2,M,x,solve.GRIDS[1:],tilt=(0.0,np.radians(roll)))
            print(t,'roll',roll,'dh(extra)',dh,'score %.3f'%bs, 'dyaw %.1f da %.1f db %.1f cm'%(np.degrees(b[0]-x[0]),100*(b[1]-x[1]),100*(b[2]-x[2])))

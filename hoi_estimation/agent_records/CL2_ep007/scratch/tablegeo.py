import json,numpy as np
from scipy.optimize import least_squares
cam=json.load(open('inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam'])
pl=np.array(json.load(open('inputs/ground_plane.json'))['plane'])
n_c=Twc[:3,:3].T@pl[:3]; d_c=pl[3]+pl[:3]@Twc[:3,3]   # plane n_c.x + d_c = 0 in camera; n_c points up
import sys
dfloor=float(sys.argv[1]) if len(sys.argv)>1 else 0.0  # shift of floor plane (m, + = floor lower)
d_c=d_c+dfloor
def floorpt(u,v):
    r=np.linalg.inv(K)@[u,v,1]; s=-d_c/(n_c@r); return s*r
cast=np.array([floorpt(*p) for p in [(746,911),(659,879),(839,879),(701,830),(801,835)]])
# fit circle in plane: center c (on plane), radius
def res(p):
    c=p[:3]; return np.r_[np.linalg.norm(cast-c,axis=1)-p[3], 10*(n_c@c+d_c)]
p0=np.r_[cast.mean(0),0.35]; sol=least_squares(res,p0); c=sol.x[:3]; print('caster circle center',c.round(3),'R',sol.x[3].round(3),'resid',np.abs(sol.fun[:5]).max().round(3))
# table top: points (u,v) on rim of top surface
rim=np.array([(632,655),(848,652),(740,632),(740,674)],float)
def proj(P): p=P@K.T; return p[:,:2]/p[:,2:3]
# horizontal basis
e1=np.cross(n_c,[0,0,1.]); e1/=np.linalg.norm(e1); e2=np.cross(n_c,e1)
def res2(p):
    h,r=p; C=c+h*n_c
    ang=np.linspace(0,2*np.pi,720,endpoint=False); P=C+r*(np.cos(ang)[:,None]*e1+np.sin(ang)[:,None]*e2)
    uv=proj(P)
    return np.r_[uv[:,0].min()-rim[0,0], uv[:,0].max()-rim[1,0], uv[:,1].min()-rim[2,1], uv[:,1].max()-rim[3,1]]
s2=least_squares(res2,[0.9,0.4]); h,r=s2.x; print('table height %.3f radius %.3f'%(h,r),'resid px',s2.fun.round(1))
C=c+h*n_c; print('table top center cam',C.round(3))
np.save('scratch/table_geo.npy',np.r_[n_c, -(n_c@C)])  # plane n.x + d = 0
# iron contact at frame 0: sole bottom seen around u=720,v=655 -> ray-plane
r=np.linalg.inv(K)@[720,655,1]; s=(n_c@C)/(n_c@r); print('iron contact point at (720,655):',(s*r).round(3))

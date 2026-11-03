import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoilib import *
ep=Episode('/work/inputs')
d=np.load('carry_profile.npz'); fr=list(d['frames']); g=d['grid']; iou=d['iou'].reshape(len(fr),len(g)); Tf=d['Tf'].reshape(len(fr),len(g),4,4)
R0=ep.T_base[0,:3,:3]
sig_iou=float(sys.argv[1]) if len(sys.argv)>1 else 0.01
sig_step=float(sys.argv[2]) if len(sys.argv)>2 else 6.0
drift_occ=float(sys.argv[3]) if len(sys.argv)>3 else 85.0
X=np.arange(-360,360,2.5)  # absolute twist states
F0,F1=116,345
unary=np.zeros((F1-F0+1,len(X)))
for k,i in enumerate(fr):
    if i<F0 or i>F1: continue
    tws=np.array([twist_swing(R0.T@(ep.T_world_cam@Tf[k,j])[:3,:3])[0] for j in range(len(g))])
    # periodic interpolation of iou over twist mod 90
    o=np.argsort(tws%90); xs=np.concatenate([(tws%90)[o]-90,(tws%90)[o],(tws%90)[o]+90]); ys=np.tile(iou[k][o],3)
    prof=np.interp(X%90,xs,ys)
    unary[i-F0]=-0.5*((prof-prof.max())/sig_iou)**2
# anchors: A static twist ~ -1.3 at frame 116 (strong), B static: -68 mod 90 at 345
def wrapd(a): return (a+45)%90-45
unary[0]+= -0.5*((X-(-1.3))/3.0)**2
unary[-1]+= -0.5*(wrapd(X-(-68.6))/3.0)**2
# occluded frames: no unary.  transitions
n=F1-F0+1
r=np.load('/work/inputs/reconstruction.npz'); Kp=r['human_keypoints_world'].astype(np.float64)
mode=sys.argv[4] if len(sys.argv)>4 else 'occ'
up=np.array([0,-1.,0])
if mode=='wrist': vv=Kp[:,62]-Kp[:,41]; yy=np.degrees(np.arctan2(vv[:,0],vv[:,2]))
else:
    fc=np.cross(up,(Kp[:,5]-Kp[:,6])+(Kp[:,9]-Kp[:,10])); yy=np.degrees(np.arctan2(fc[:,0],fc[:,2]))
yy=np.degrees(np.unwrap(np.radians(yy)))
from scipy.ndimage import uniform_filter1d
yy=uniform_filter1d(yy,5)
dface=-np.diff(yy,prepend=yy[0])  # predicted twist increment per frame
logp=np.zeros((n,len(X))); bp=np.zeros((n,len(X)),int)
D=X[None,:]-X[:,None]  # to - from
alpha=unary[0].copy(); A=[alpha]
for t in range(1,n):
    i=F0+t
    mu=(drift_occ/51.0 if 197<=i<=247 else 0.0) if mode=='occ' else dface[i]
    tr=-0.5*((D-mu)/sig_step)**2
    m=alpha[:,None]+tr
    mx=m.max(0); alpha=mx+np.log(np.exp(m-mx).sum(0))+unary[t]   # forward (sum)
    alpha-=alpha.max(); A.append(alpha)
post=np.exp(A[-1]-A[-1].max()); post/=post.sum()
for c in [-158.6,-68.6,21.4,111.4,201.4,291.4]:
    s=np.abs(X-c)<20; print('B branch %6.1f  P=%.3f'%(c,post[s].sum()))
# Viterbi path
alpha=unary[0].copy(); BP=[]
for t in range(1,n):
    i=F0+t; mu=(drift_occ/51.0 if 197<=i<=247 else 0.0) if mode=='occ' else dface[i]
    m=alpha[:,None]-0.5*((D-mu)/sig_step)**2
    BP.append(m.argmax(0)); alpha=m.max(0)+unary[t]
path=[alpha.argmax()]
for b in BP[::-1]: path.append(b[path[-1]])
path=X[np.array(path[::-1])]
np.save('vit_path.npy',path)
print(' '.join('%d:%.0f'%(F0+t,path[t]) for t in range(0,n,5)))

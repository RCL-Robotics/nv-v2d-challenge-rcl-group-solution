import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, json, trimesh
from render_utils import load_mesh
cam=json.load(open('/work/inputs/camera.json')); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'])
m,_=load_mesh('/work/inputs/object_mesh.glb'); V=np.asarray(m.vertices)[::10]; c=np.asarray(m.vertices).mean(0)
P=np.load('run2.npz')['T_world_mesh']; Tc=Tcw[None]@P
a,b=202,559
zc=lambda T: (T[...,:3,:3]@c+T[...,:3,3])[...,2]
s_start=zc(Tc[a-1])/zc(Tc[a]); s_end=zc(Tc[b+1])/zc(Tc[b])
print('anchor ratios start %.4f end %.4f  (= %.1f / %.1f cm)'%(s_start,s_end,(s_start-1)*zc(Tc[a])*100,(s_end-1)*zc(Tc[b])*100))
t=np.linspace(0,1,b-a+1); s=s_start+(s_end-s_start)*t
Tr=Tc.copy(); Tr[a:b+1,:3,3]*=s[:,None]
def floor(Tcam):
    W=np.einsum('fij,nj->fni',(Twc[None]@Tcam)[:,:3,:3],V)+(Twc[None]@Tcam)[:,None,:3,3]
    return (W@pl[:3]+pl[3]).min(1)*100
for nm,X in [('run2',Tc),('ramp',Tr)]:
    h=floor(X)
    print(nm,'floor height median: S1 %.2f moving %.2f S2 %.2f | moving std %.2f | moving first/last 30: %.2f %.2f'%(np.median(h[:202]),np.median(h[a:b+1]),np.median(h[560:]),h[a:b+1].std(),np.median(h[a:a+30]),np.median(h[b-29:b+1])),
          ' moving median z diff vs base cm %.2f'%(np.median(X[a:b+1,2,3])*100-np.median((Tcw[None]@np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float))[a:b+1,2,3])*100))
target=np.median(np.r_[floor(Tc[:202]),floor(Tc[560:])])
best=None
for k in np.linspace(0.96,1.02,121):
    X=Tc.copy(); X[a:b+1,:3,3]*=k; hm=np.median(floor(X[a:b+1]))
    if best is None or abs(hm-target)<best[0]: best=(abs(hm-target),k)
k=best[1]; print('floor cue: uniform scale %.4f -> median depth shift %.2f cm (target floor h %.2f)'%(k,(k-1)*np.median(Tc[a:b+1,2,3])*100,target))
# per-segment agreement: floor cue in thirds of the moving phase vs ramp
for lo,hi in [(a,a+120),(a+120,a+240),(a+240,b+1)]:
    best=None
    for kk in np.linspace(0.95,1.03,161):
        X=Tc.copy(); X[lo:hi,:3,3]*=kk; hm=np.median(floor(X[lo:hi]))
        if best is None or abs(hm-target)<best[0]: best=(abs(hm-target),kk)
    print('  frames %d-%d: floor cue shift %.2f cm, ramp shift %.2f cm'%(lo,hi-1,(best[1]-1)*np.median(Tc[lo:hi,2,3])*100,np.median((s[lo-a:hi-a]-1)*Tc[lo:hi,2,3])*100))

import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, trimesh
from hoi_lib import *
cam,K,Twc,rec=load_inputs('/work/inputs')
mesh=load_mesh('/work/inputs'); fd=FrameData('/work/inputs')
v=np.asarray(mesh.vertices); f=np.asarray(mesh.faces)
keep=(v[f][:,:,2].min(1)>0.085)
hm=trimesh.Trimesh(v,f[keep],process=False); hm.visual=mesh.visual
Rh=Renderer(mesh); Rh.f=torch.tensor(f[keep].astype(np.int32),device='cuda')
T=rec['T_world_mesh'].astype(float); Tc=np.linalg.inv(Twc)@T
a0,a1=int(sys.argv[1]),int(sys.argv[2]); fr=list(range(a0,a1))
R0=rot_mean(Tc[fr,:3,:3]); t0=np.median(Tc[fr,:3,3],0)
hc=project(K,R0@[0,0,0.125]+t0); s=0.2*K[0,0]/t0[2]
box=(hc[0]-s/2,hc[1]-s/2,hc[0]+s/2,hc[1]+s/2); hw=(200,200)
med=np.median(np.stack([fd.crop(i,box,hw)[0] for i in fr]),0)
lum=med.mean(-1)
# brightness threshold: Otsu inside crop
l8=(lum*255).astype(np.uint8); th,_=cv2.threshold(l8,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
bright=l8>th
print('otsu',th)
angs=np.arange(0,360,10)
best=[]
res=[]
for a in angs:
    X0=np.eye(4); X0[:3,:3]=R0@rot_about_axis([0,0,1],np.radians(a)); X0[:3,3]=t0
    # small translation search to align head
    bestiou=0
    for dx in np.linspace(-0.012,0.012,7):
      for dy in np.linspace(-0.012,0.012,7):
        X=X0.copy(); X[:3,3]+= [dx,dy,0]
        _,m,_=Rh.render(torch.tensor(X[None],device='cuda'),K,box,hw); m=m[0].cpu().numpy()
        # compare only in upper part rows where head is (exclude neck/body region below head bottom)
        iou=(m&bright).sum()/max((m|bright).sum(),1)
        if iou>bestiou: bestiou=iou; bm=m
    res.append(bestiou); print(a,'%.3f'%bestiou)
cv2.imwrite(sys.argv[3],np.hstack([(np.dstack([lum]*3)*255).astype(np.uint8),np.dstack([bright*255]*3).astype(np.uint8)]))
print('windowed')
win=np.zeros_like(bright); win[15:115,25:135]=True
for a in angs:
    X0=np.eye(4); X0[:3,:3]=R0@rot_about_axis([0,0,1],np.radians(a)); X0[:3,3]=t0
    bi=0
    for dx in np.linspace(-0.025,0.025,11):
      for dy in np.linspace(-0.025,0.025,11):
        X=X0.copy(); X[:3,3]+= [dx,dy,0]
        _,m,_=Rh.render(torch.tensor(X[None],device='cuda'),K,box,hw); m=m[0].cpu().numpy()&win
        b=bright&win; iou=(m&b).sum()/max((m|b).sum(),1)
        if iou>bi: bi=iou; bd=(dx,dy)
    print(a,'%.3f'%bi,np.round(np.array(bd)*100,1))

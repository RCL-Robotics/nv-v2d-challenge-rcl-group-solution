import sys, time; sys.path.insert(0,'/work/scratch')
from bfit import *
m,tex=load_mesh(); K,Twc,Kp=load_cam()
r=np.load(INP+'/reconstruction.npz'); Tw=r['T_world_mesh'].astype(np.float64)
Tc=np.linalg.inv(Twc)[None]@Tw
R=Renderer(m,tex,CROP,CROP); bf=BatchFitter(R)
t0=time.time()
with h5py.File(INP+'/masks.h5') as f:
    OM=f['object'][:]; PM=f['person'][:]
Ks=[];tg=[];wt=[]
for i in range(700):
    ys,xs=np.nonzero(OM[i]); kc,x0,y0=crop_K(K,xs.mean(),ys.mean()); Ks.append(kc)
    tg.append(crop_img(OM[i],x0,y0)); p=cv2.dilate(PM[i].astype(np.uint8),np.ones((5,5)))
    wt.append(1-crop_img(p,x0,y0))
tg=np.stack(tg).astype(np.float32); wt=np.stack(wt).astype(np.float32)
np.savez_compressed('crops.npz',Ks=np.stack(Ks),tg=tg.astype(bool),wt=wt.astype(bool))
Tf=[];L=[];L0=[]
for s in range(0,700,50):
    sl=slice(s,s+50)
    T,per,_=bf.fit(Tc[sl],Ks[sl],tg[sl],wt[sl],iters=150)
    _,per0,_=bf.fit(Tc[sl],Ks[sl],tg[sl],wt[sl],iters=0)
    Tf.append(T);L.append(per);L0.append(per0)
Tf=np.concatenate(Tf);L=np.concatenate(L);L0=np.concatenate(L0)
np.savez('fit_free.npz',T=Tf,loss=L,loss0=L0)
print('time',time.time()-t0)

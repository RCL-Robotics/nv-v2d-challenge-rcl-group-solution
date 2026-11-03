import sys; sys.path.insert(0,'/work/scratch')
from bfit import *
m,tex=load_mesh(); K,Twc,Kp=load_cam()
r=np.load(INP+'/reconstruction.npz'); Tw=r['T_world_mesh'].astype(np.float64)
Tc=np.linalg.inv(Twc)[None]@Tw
c=(m.bounds[0]+m.bounds[1])/2
R=Renderer(m,tex,CROP,CROP); bf=BatchFitter(R)
ids=list(range(0,196,4))+list(range(588,700,4))
with h5py.File(INP+'/masks.h5') as f:
    om=f['object'][ids]; pm=f['person'][ids]
Ks=[];tg=[];wt=[]
for j,i in enumerate(ids):
    ys,xs=np.nonzero(om[j]); kc,x0,y0=crop_K(K,xs.mean(),ys.mean()); Ks.append(kc)
    tg.append(crop_img(om[j],x0,y0)); p=cv2.dilate(pm[j].astype(np.uint8),np.ones((5,5)))
    wt.append(1-crop_img(p,x0,y0))
tg=np.stack(tg).astype(np.float32); wt=np.stack(wt).astype(np.float32)
T0=Tc[ids]
def centre(T): return T[:,:3,:3]@c+T[:,:3,3]
for fd in [False,True]:
    T,per,_=bf.fit(T0,Ks,tg,wt,iters=150,fix_depth=fd)
    for nm,sel in [('A',np.array(ids)<300),('B',np.array(ids)>300)]:
        pb=centre(T0[sel]); pf=centre(T[sel])
        print('fixdepth',fd,nm,'base: c std(cm)',(pb.std(0)*100).round(2),'rotspread',rotdist(T0[sel,:3,:3],rot_mean(T0[sel,:3,:3])[None]).mean().round(2),
              '| fit: c std',(pf.std(0)*100).round(2),'rotspread',rotdist(T[sel,:3,:3],rot_mean(T[sel,:3,:3])[None]).mean().round(2),
              'mean dz(cm)',((pf-pb)[:,2].mean()*100).round(1),'rot fit-vs-basemean',rotdist(rot_mean(T[sel,:3,:3]),rot_mean(T0[sel,:3,:3])).round(2), 'loss',per[sel].mean().round(4))

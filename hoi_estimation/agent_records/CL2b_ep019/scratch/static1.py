import sys,time; sys.path.insert(0,'/work/outputs')
from common import *; from fit import *; from decode import *
import h5py
K,Twc,_=load_camera(); f=h5py.File('/work/inputs/masks.h5')
mesh=load_mesh()[4]; pts=sample_points(mesh,1200)
d=np.load('pass1.npz'); L=d['loss']; T=d['T']
res={}
for a,b in [(769,886)]:
    fr=np.linspace(a,b-1,16).round().astype(int)
    data={i:FrameData(f['object'][i],f['person'][i],n_pix=800) for i in fr}
    cand=[]
    for i in np.linspace(a,b-1,12).round().astype(int):
        for h in np.argsort(L[i])[:3]: cand.append(T[i,h])
    # unique
    U=[]
    for c in cand:
        if all(rot_angle(c[:3,:3],u[:3,:3])>4 for u in U): U.append(c)
    print(a,b,'n unique cands',len(U))
    T0=[];fid=[];sh=[]
    for g,u in enumerate(U):
        for i in fr: T0.append(u); fid.append(i); sh.append(g)
    t=time.time()
    Tg,LL=fit_poses(K,np.array(T0),fid,data,pts,iters=150,share=sh)
    tot=LL['loss'].reshape(len(U),len(fr))
    print('time',time.time()-t)
    o=np.argsort(tot.mean(1))
    for g in o[:6]: print(g,'mean loss %.3f med %.3f'%(tot[g].mean(),np.median(tot[g])),'rot vs best %.1f'%rot_angle(Tg[g,:3,:3],Tg[o[0],:3,:3]),'z %.3f'%Tg[g,2,3])
    res[a]=Tg[o[0]]

print(np.round(fr)); print(tot[19].round(2)); print(tot[5].round(2))
print('areas',[data[i].area for i in fr])
np.save('staticC_alt.npy',np.stack([Tg[19],Tg[5]]))

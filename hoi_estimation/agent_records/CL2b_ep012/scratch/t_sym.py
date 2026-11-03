import sys, time; sys.path.insert(0,'/work/scratch')
from bfit import *
m,tex=load_mesh(); K,Twc,Kp=load_cam()
r=np.load(INP+'/reconstruction.npz'); Tw=r['T_world_mesh'].astype(np.float64)
Tc=np.linalg.inv(Twc)[None]@Tw
S=np.load('Tsym_y.npy')
R=Renderer(m,tex,CROP,CROP); bf=BatchFitter(R)
d=np.load('crops.npz'); Ks=d['Ks']; tg=d['tg'].astype(np.float32); wt=d['wt'].astype(np.float32)
Tf=np.load('fit_free.npz')['T']
Ts=[];L=[]
for s in range(0,700,50):
    sl=slice(s,s+50)
    T,per,_=bf.fit(Tf[sl]@S,Ks[sl],tg[sl],wt[sl],iters=150)
    Ts.append(T);L.append(per)
Ts=np.concatenate(Ts);L=np.concatenate(L)
np.savez('fit_sym.npz',T=Ts,loss=L)
Lb=np.load('fit_free.npz')['loss']
for s in range(0,700,25): print(s, 'base %.4f sym %.4f  frac base better %.2f  rot sym-vs-(base@S) %.1f'%(Lb[s:s+25].mean(),L[s:s+25].mean(),(Lb[s:s+25]<L[s:s+25]).mean(), rotdist(Ts[s,:3,:3],(Tf[s]@S)[:3,:3])))
print('overall frac base better',(Lb<L).mean(), 'mean rel diff',((L-Lb)/Lb).mean())

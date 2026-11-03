from pano import *
import h5py, time
src=sys.argv[1]
T=np.load(src)['T_world_mesh'] if src!='base' else rec['T_world_mesh'].astype(np.float64)
Tc=np.linalg.inv(Twc)@T
P=h5py.File('/work/inputs/masks.h5','r')['person']
ms=np.load('/work/scratch/mstats.npz')
def wyaw(Tw): x=Tw[:3,0]; return np.degrees(np.arctan2(x[2],x[0]))
A=np.full((980,NPHI),np.nan); cov=np.zeros(980)
t=time.time()
for i in range(980):
    if ms['area'][i]<300: continue
    sm,wt=pano(i,Tc[i],P[i])
    sc=corr_shift(sm,wt,-1)
    cov[i]=(wt.sum(0)>0).mean()
    for s in range(NPHI):
        ay=wyaw(Twc@Tc[i]@yawrot(np.radians(-s*360/NPHI)))%360
        A[i,int(round(ay/3))%NPHI]=sc[s]
print(time.time()-t)
np.savez(sys.argv[2],A=A,cov=cov)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.figure(figsize=(10,20)); Z=(A-np.nanmean(A,1,keepdims=True))/np.nanstd(A,1,keepdims=True)
plt.imshow(np.nan_to_num(Z),aspect='auto',extent=[0,360,980,0],cmap='viridis',vmin=-2.5,vmax=2.5)
x=T[:,:3,0]; by=np.degrees(np.arctan2(x[:,2],x[:,0]))%360
plt.plot(by,np.arange(980),'r.',ms=1); plt.yticks(range(0,980,20)); plt.grid(alpha=0.3)
plt.savefig(sys.argv[3],dpi=50)

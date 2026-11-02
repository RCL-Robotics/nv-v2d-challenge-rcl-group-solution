from pano import *
import h5py
T=np.load('/work/scratch/fitA.npz')['T_world_mesh']; Tc=np.linalg.inv(Twc)@T
P=h5py.File('/work/inputs/masks.h5','r')['person']
rows=[]
d=np.load('/work/scratch/texpano.npz'); tp=np.where(d['wt']>0,d['sm']/np.maximum(d['wt'],1),0)
def show(im,s=2.0):
    return cv2.resize(np.clip(im*s+128,0,255).astype(np.uint8),(NPHI*8,NH*8),interpolation=cv2.INTER_NEAREST)
rows.append(show(-tp,3))
for a,b,st in [(0,124,4),(280,410,4),(600,604,1),(660,664,1)]:
    S=np.zeros((NH,NPHI)); W=np.zeros((NH,NPHI))
    for i in range(a,b,st):
        sm,wt=pano(i,Tc[i],P[i]); S+=sm; W+=wt
    im=np.where(W>0,S/np.maximum(W,1),0); r=show(im,6); r[np.repeat(np.repeat(W==0,8,0),8,1)]=0
    cv2.putText(r,f'{a}',(3,15),0,0.5,255,1); rows.append(r)
cv2.imwrite('/work/scratch/vis/pano_avg.png',np.vstack(rows))

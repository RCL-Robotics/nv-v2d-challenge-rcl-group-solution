import h5py, numpy as np
f=h5py.File('/work/inputs/masks.h5','r')
F=708; out=np.zeros((F,8))
for s in range(0,F,23):
    o=f['object'][s:s+23]; p=f['person'][s:s+23]
    for j in range(o.shape[0]):
        m=o[j]; ys,xs=np.nonzero(m)
        if len(xs)==0: out[s+j]=[0]*8; continue
        out[s+j]=[len(xs),xs.mean(),ys.mean(),xs.min(),xs.max(),ys.min(),ys.max(),(m&p[j]).sum()]
np.save('maskstats.npy',out)

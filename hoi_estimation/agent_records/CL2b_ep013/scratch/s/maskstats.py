import numpy as np,h5py
f=h5py.File('/work/inputs/masks.h5')
F=922; st=np.zeros((F,8))
for i in range(F):
    om=f['object'][i]; pm=f['person'][i]
    ys,xs=np.nonzero(om)
    if len(xs)==0: st[i]=np.nan; st[i,0]=0; continue
    st[i]=[len(xs),xs.mean(),ys.mean(),xs.min(),xs.max(),ys.min(),ys.max(),0]
    # person pixels adjacent to object bbox
    x0,x1,y0,y1=xs.min(),xs.max(),ys.min(),ys.max()
    st[i,7]=pm[max(y0-10,0):y1+10,max(x0-10,0):x1+10].mean()
np.save('/work/scratch/maskstats.npy',st)

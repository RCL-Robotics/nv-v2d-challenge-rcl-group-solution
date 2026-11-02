import numpy as np, h5py
f=h5py.File('/work/inputs/masks.h5')
O=f['object']; P=f['person']; refE=O[714]; ref0=O[0]
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']
for i in list(range(560,600,1)):
    o=O[i]; vis=~P[i]
    a=o&vis; b=refE&vis; print(i,'iou end %.3f'%((a&b).sum()/max((a|b).sum(),1)), 'base pos',np.round(T[i,:3,3],3))

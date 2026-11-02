import sys, numpy as np, json, h5py
sys.path.insert(0,'/work/outputs')
from hoilib import *
R=128
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam'])
Tb=np.linalg.inv(Twc)[None]@np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float)
H5=h5py.File('/work/inputs/masks.h5')
F=len(Tb); boxes=np.zeros((F,3)); obs=np.zeros((F,R,R),np.float32); val=np.zeros((F,R,R),np.float32); Kc=np.zeros((F,3,3))
area=np.zeros(F)
for i in range(F):
    om=H5['object'][i]; pm=H5['person'][i]
    p=K@Tb[i,:3,3]; uv=p[:2]/p[2]
    box=crop_box_from_mask(om,uv,min_size=64,scale=1.8)
    boxes[i]=box; Kc[i]=crop_K(K,box,R)
    o=resample_crop(om.astype(np.float32),box,R); pr=resample_crop(pm.astype(np.float32),box,R)
    obs[i]=o; val[i]=1-np.clip(pr-o,0,1); area[i]=om.sum()
np.savez('/work/scratch/crops.npz',boxes=boxes,obs=obs,val=val,Kc=Kc,area=area)

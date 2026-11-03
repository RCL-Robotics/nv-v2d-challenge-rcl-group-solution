import sys, time; sys.path.insert(0,'/work/outputs')
import numpy as np, json, h5py
import solve
from solve import *
solve.T0=time.time()
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam'])
Tc=np.linalg.inv(Twc)[None]@np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float)
mesh,tex=load_mesh('/work/inputs/object_mesh.glb'); R=Renderer(mesh,tex,K); c=R.v.mean(0).double().cpu().numpy()
f=h5py.File('/work/inputs/masks.h5'); a,b=202,560
T=fit_moving(SilFitter(R,K,fat_px=FAT_PX), f['object'][a:b], f['person'][a:b], Tc[a:b], c)
np.save(sys.argv[1],T)
S,dev=robust_smooth(T,c,2.0); print('final dev>3deg frames', np.where(dev>3)[0]+a, 'dev p50/p90/max %.2f %.2f %.2f'%(np.median(dev),np.percentile(dev,90),dev.max()))

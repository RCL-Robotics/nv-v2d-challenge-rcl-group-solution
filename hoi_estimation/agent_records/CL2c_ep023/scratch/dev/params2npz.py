import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoi_lib import *
inp='/work/inputs'
cam,K,Twc,rec,plane=load_inputs(inp); mesh=load_mesh(inp)
fm=FloorModel(mesh,plane,np.array([-0.28,-0.02,-0.96]),h0=0.014)
z=np.load(sys.argv[1]); ids=z['ids']; r=z['r0']
th=np.unwrap(r[:,0]); F=946; t=np.arange(F)
P=np.stack([np.interp(t,ids,th),np.interp(t,ids,r[:,1]),np.interp(t,ids,r[:,2])],1)
np.savez(sys.argv[2],T_world_mesh=np.array([fm.pose(*p) for p in P]))

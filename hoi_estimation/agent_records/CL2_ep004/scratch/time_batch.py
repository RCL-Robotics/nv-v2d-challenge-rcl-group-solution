import sys, time; sys.path.insert(0,'/work/outputs')
import numpy as np, torch
from bat_lib import *; from fitter import *
s=np.load('/work/scratch/simp.npz'); FR=FastRenderer(s['vs'],s['fs'],s['cs'])
K=np.array([[800.,0,768],[0,800,576],[0,0,1]])
T0=np.tile(np.eye(4),(1,1,1)); T0[0,:3,3]=[0,0,2.5]
d=dict(M=np.zeros((RES,RES),np.float32),P=np.zeros((RES,RES),np.float32),D=np.zeros((RES,RES),np.float32),I=np.zeros((RES,RES,3),np.float32))
for B in [12,48,96]:
    torch.cuda.synchronize(); t0=time.time()
    fit_batch(FR,K,np.repeat(T0,B,0),[(568,376,400.)]*B,[d]*B,iters=50)
    torch.cuda.synchronize(); print(B,(time.time()-t0)/50*1000,'ms/iter', torch.cuda.max_memory_allocated()/1e9)

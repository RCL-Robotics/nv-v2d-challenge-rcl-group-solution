import numpy as np,json,torch,cv2
from pathlib import Path
from refine import Renderer,make_data,to_torch,optimize
torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(0);np.random.seed(0)
a=np.load('/work/inputs/reconstruction.npz');c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K']);T=np.linalg.inv(c['T_world_cam'])@a['T_world_mesh'];ii=np.array([0,250,550,680]);r=Renderer('/work/inputs/object_mesh.glb',K);d=to_torch(make_data(Path('/work/inputs'),ii,K,T));outs=[]
for k in range(2):
 torch.manual_seed(0);np.random.seed(0);rr,tt,ss,st=optimize(r,T[ii,:3,:3],T[ii,:3,3],d,steps=85,texture=True,rot_reg=.012,shift_reg=.004);outs.append((rr,tt,ss))
print('determinism max differences',[np.max(np.abs(x-y)) for x,y in zip(outs[0],outs[1])]);

import sys; sys.path.insert(0,'/work/outputs')
from hoi_lib import *
import time
K,Twc,rec=load_inputs('/work/inputs')
R=MeshRenderer('/work/inputs/object_mesh.glb')
Tb=mesh_to_cam(Twc,rec['T_world_mesh'].astype(np.float64))
T=torch.tensor(Tb[:64],dtype=torch.float32,device=DEV)
crops=[(700,550,192,144)]*64
for ss in [1,2]:
  for it in range(3):
    torch.cuda.synchronize(); t=time.time()
    out=render_soft(R,T,K,crops,(192,144),ss=ss)
    torch.cuda.synchronize(); print(ss,time.time()-t)
print(torch.cuda.max_memory_allocated()/1e9)

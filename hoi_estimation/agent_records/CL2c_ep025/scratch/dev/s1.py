import sys, time, pickle, json
sys.path.insert(0,'/work/outputs')
from hoi_lib import *
t0=time.time()
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
D=pickle.load(open("D.pkl","rb")); F=D.F
print('data',time.time()-t0)

R=Renderer('/work/inputs/object_mesh.glb'); RF=Refiner(R,K,D.S)
r=np.load('/work/inputs/reconstruction.npz'); Tb=Tcw[None]@r['T_world_mesh'].astype(np.float64)
Tf=r['T_cam_mesh_foundationpose'].astype(np.float64)
FX=np.diag([1,-1,-1.])
jobs=[]
for i in range(F):
    for Rs in [Tb[i,:3,:3],Tb[i,:3,:3]@FX,Tf[i,:3,:3],Tf[i,:3,:3]@FX]: jobs.append((i,Rs,Tb[i,:3,3]))
tt=lambda a:torch.tensor(np.array(a),dtype=torch.float32,device=dev)
BS=192; out=[]
torch.cuda.synchronize(); t1=time.time()
for s in range(0,len(jobs),BS):
    J=jobs[s:s+BS]; idx=[j[0] for j in J]
    Rm,t,ls,lc=RF.optimize(tt([j[1] for j in J]),tt([j[2] for j in J]),*D.batch(idx),iters=40)
    for k,j in enumerate(J): out.append((j[0],Rm[k].cpu().numpy().astype(np.float64),t[k].cpu().numpy().astype(np.float64),ls[k].item(),lc[k].item()))
torch.cuda.synchronize(); print('S1 gpu',time.time()-t1)
pickle.dump(out,open('s1.pkl','wb'))

import time, pickle
from refine import *
R=Renderer('/work/inputs/object_mesh.glb'); K,Twc,Kp=load_cam(); Tcw=np.linalg.inv(Twc)
r=np.load('/work/inputs/reconstruction.npz'); Tb=Tcw[None]@r['T_world_mesh'].astype(np.float64)
Tf=r['T_cam_mesh_foundationpose'].astype(np.float64)
D=Data(); RF=Refiner(R,K); F=D.F
flips={'id':np.eye(3),'fx':np.diag([1,-1,-1.]),'fy':np.diag([-1,1,-1.]),'fz':np.diag([-1,-1,1.])}
jobs=[]
for i in range(F):
    for src,Rs in [('b',Tb[i,:3,:3]),('f',Tf[i,:3,:3])]:
        for fn,Fm in flips.items(): jobs.append((i,src+fn,Rs@Fm,Tb[i,:3,3]))
tt=lambda a:torch.tensor(np.array(a),dtype=torch.float32,device=dev)
res=[]; BS=128; t1=time.time()
for s in range(0,len(jobs),BS):
    J=jobs[s:s+BS]; dd=[]
    for j in J:
        p=K@j[3]; dd.append(D.get(j[0],p[:2]/p[2]))
    img=tt([d[0] for d in dd]);om=tt([d[1] for d in dd]);val=tt([d[2] for d in dd]);x0=tt([d[3] for d in dd]);y0=tt([d[4] for d in dd])
    Rm,t,ls,lc=RF.optimize(tt([j[2] for j in J]),tt([j[3] for j in J]),img,om,val,x0,y0,D.S,iters=60)
    for k,j in enumerate(J): res.append(dict(f=j[0],src=j[1],R=Rm[k].cpu().numpy(),t=t[k].cpu().numpy(),ls=ls[k].item(),lc=lc[k].item()))
    if s==0: torch.cuda.synchronize(); print('first batch',time.time()-t1,flush=True)
print('total',time.time()-t1)
pickle.dump(res,open('stageA.pkl','wb'))

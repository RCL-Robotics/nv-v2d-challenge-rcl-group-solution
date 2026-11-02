import sys,time
from refine import *
R=Renderer('/work/inputs/object_mesh.glb'); K,Twc,Kp=load_cam(); Tcw=np.linalg.inv(Twc)
r=np.load('/work/inputs/reconstruction.npz'); Tb=Tcw[None]@r['T_world_mesh'].astype(np.float64)
D=Data(); RF=Refiner(R,K)
frames=[int(a) for a in sys.argv[1].split(',')]
flips={'id':np.eye(3),'fx':np.diag([1,-1,-1.]),'fy':np.diag([-1,1,-1.]),'fz':np.diag([-1,-1,1.])}
batch=[]
for i in frames:
    p=K@Tb[i,:3,3]; uv=p[:2]/p[2]
    img,om,val,x0,y0,n=D.get(i,uv)
    for fn,Fm in flips.items():
        batch.append((i,fn,Tb[i,:3,:3]@Fm,Tb[i,:3,3],img,om,val,x0,y0))
tt=lambda a:torch.tensor(np.array(a),dtype=torch.float32,device=dev)
R0=tt([b[2] for b in batch]);t0=tt([b[3] for b in batch]);img=tt([b[4] for b in batch]);om=tt([b[5] for b in batch]);val=tt([b[6] for b in batch]);x0=tt([b[7] for b in batch]);y0=tt([b[8] for b in batch])
torch.cuda.synchronize();t1=time.time()
ls0,lc0,_=RF.losses(R0,t0,img,om,val,x0,y0,192)
Rm,t,ls,lc=RF.optimize(R0,t0,img,om,val,x0,y0,192)
torch.cuda.synchronize();print('time',time.time()-t1, 'mem',torch.cuda.max_memory_allocated()/1e9)
for k,b in enumerate(batch):
    ang=np.degrees(np.arccos(np.clip((np.trace(b[2].T@Rm[k].cpu().numpy())-1)/2,-1,1)))
    print(b[0],b[1],'init sil %.3f col %.3f -> sil %.3f col %.3f  dR %.1f dt'%(ls0[k],lc0[k],ls[k],lc[k],ang),(t[k].cpu().numpy()-b[3]).round(3))
np.savez('test_ref_out.npz',R=Rm.cpu().numpy(),t=t.cpu().numpy(),frames=[b[0] for b in batch])

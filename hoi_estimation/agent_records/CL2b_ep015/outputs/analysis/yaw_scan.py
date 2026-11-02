# Analysis script (not used by solve.py). Needs the evidence cache built by an earlier scratch script (ev320.pkl).
import sys; sys.path.insert(0,'/work/outputs')
from evidence import *
import pickle, time
K,Kp,Twc=load_camera('/work/inputs'); m=load_mesh('/work/inputs/object_mesh.glb'); r=Renderer(m,K)
rr=np.load('/work/inputs/reconstruction.npz'); Tc0=np.linalg.inv(Twc)@rr['T_world_mesh'].astype(np.float64)
ev=pickle.load(open('/work/scratch/ev320.pkl','rb'))
cxy=np.array([0.005,-0.004,0])
def Rz(phi):
    T=np.eye(4); c,s=np.cos(phi),np.sin(phi); T[:2,:2]=[[c,-s],[s,c]]; T[:3,3]=cxy-T[:3,:3]@cxy; return T
step=int(sys.argv[1]) if len(sys.argv)>1 else 4
iters=int(sys.argv[2]) if len(sys.argv)>2 else 0
yaws=np.radians(np.arange(0,360,10)); idx=np.arange(0,715,step)
t0=time.time()
pairs=[(i,j) for i in idx for j in range(len(yaws))]
IOU=np.zeros((len(idx),len(yaws))); PH=np.zeros_like(IOU); LS=np.zeros_like(IOU); TT=np.zeros((len(idx),len(yaws),4,4))
B=72
for b in range(0,len(pairs),B):
    pb=pairs[b:b+B]; ids=np.array([p[0] for p in pb])
    Tc=np.array([Tc0[i]@Rz(yaws[j]) for i,j in pb])
    T,iou,ls,lp=score_and_refine(r,ev,ids,Tc,iters=iters,lam=0.3)
    for k,(i,j) in enumerate(pb):
        ii=i//step; IOU[ii,j]=iou[k]; PH[ii,j]=lp[k]; LS[ii,j]=ls[k]; TT[ii,j]=T[k]
print('time',time.time()-t0)
np.savez(f'yawscan_s{step}_it{iters}.npz',IOU=IOU,PH=PH,LS=LS,idx=idx,yaws=yaws,T=TT)

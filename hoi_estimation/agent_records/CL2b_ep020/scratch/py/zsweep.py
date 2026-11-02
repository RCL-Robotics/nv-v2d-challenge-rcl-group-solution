from common import *
J=np.load('/work/scratch/joint_free.npz'); th=J['th']; uvz=J['uvz']
D=DiffFitter(Rd,K)
dev='cuda'; Bc=torch.tensor(Rcw@B,device=dev,dtype=torch.float32); Kinv=torch.tensor(np.linalg.inv(K),device=dev,dtype=torch.float32)
zg=np.arange(-0.12,0.1201,0.005)
best=np.zeros(553); curv=np.zeros(553)
for s in range(0,553,79):
    fr=list(range(s,min(553,s+79)))
    D.load_frames([frame(i) for i in fr],[OM[i] for i in fr],[PM[i] for i in fr])
    S=np.zeros((len(zg),len(fr)))
    with torch.no_grad():
        R=Bc[None]@rotz_t(torch.tensor(th[fr],device=dev,dtype=torch.float32))
        ray=torch.tensor(np.c_[uvz[fr,:2],np.ones(len(fr))],device=dev,dtype=torch.float32)@Kinv.T
        for k,dz in enumerate(zg):
            t=ray*torch.tensor(uvz[fr,2]+dz,device=dev,dtype=torch.float32)[:,None]
            iou,_=D.losses(R,t,torch.arange(len(fr)),w_ncc=0)
            S[k]=iou.cpu().numpy()
    k=S.argmax(0)
    # parabolic refinement
    for j,i in enumerate(fr):
        kk=min(max(k[j],1),len(zg)-2); y0,y1,y2=S[kk-1,j],S[kk,j],S[kk+1,j]
        den=y0-2*y1+y2; off=0.5*(y0-y2)/den if den<0 else 0
        best[i]=uvz[i,2]+zg[kk]+off*0.005
np.save('/work/scratch/zbest_iou.npy',best)
for i in range(0,553,8): print(i,'iou-z %.3f  base z %.3f'%(best[i],Tcm[i,2,3]))

from common import *
fr=list(range(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])))
D=DiffFitter(Rd,K)
D.load_frames([frame(i) for i in fr],[OM[i] for i in fr],[PM[i] for i in fr])
dev='cuda'
Bc=torch.tensor(Rcw@B,device=dev,dtype=torch.float32)
grid=np.radians(np.arange(-180,180,3.))
S=np.zeros((len(grid),len(fr)))
with torch.no_grad():
    for gi,g in enumerate(grid):
        th=torch.tensor(yaw_base[fr]+g,device=dev,dtype=torch.float32)
        Rcm=Bc[None]@rotz_t(th); tcm=torch.tensor(Tcm[fr,:3,3]+[-0.006,0.002,0],device=dev,dtype=torch.float32)
        iou,ncc=D.losses(Rcm,tcm,torch.arange(len(fr)))
        S[gi]=(iou+0.5*ncc).cpu().numpy()
np.save('/work/scratch/flip_%s_%s.npy'%(sys.argv[1],sys.argv[2]),S)
for j,i in enumerate(fr):
    k=np.argmax(S[:,j]); d=np.abs((np.degrees(grid-grid[k])+180)%360-180); k2=np.argmax(np.where(d>30,S[:,j],-9))
    print(i,'best %4d (%.3f)  @0 %.3f  2nd %4d (%.3f)'%(np.degrees(grid[k]),S[k,j],S[60,j],np.degrees(grid[k2]),S[k2,j]))

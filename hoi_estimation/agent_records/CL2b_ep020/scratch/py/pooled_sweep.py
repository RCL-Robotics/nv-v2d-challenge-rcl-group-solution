from common import *
a,b,step=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]); dx,dy=float(sys.argv[4]),float(sys.argv[5])
fr=list(range(a,b+1,step))
D=DiffFitter(Rd,K)
D.load_frames([frame(i) for i in fr],[OM[i] for i in fr],[PM[i] for i in fr])
th0=np.median(yaw_base[a:b+1]); tc0=np.median(Tcm[a:b+1,:3,3],0)+[dx,dy,0]
dev='cuda'
Bc=torch.tensor(Rcw@B,device=dev,dtype=torch.float32)
grid=np.radians(np.arange(-180,180,2.))
res=[]
with torch.no_grad():
    for g in grid:
        th=torch.tensor(th0+g,device=dev,dtype=torch.float32)
        Rcm=(Bc@rotz_t(th))[None].expand(len(fr),3,3); tcm=torch.tensor(tc0,device=dev,dtype=torch.float32)[None].expand(len(fr),3)
        iou,ncc=D.losses(Rcm,tcm,torch.arange(len(fr)))
        iou2,ncc2=D.losses(Rcm,tcm,torch.arange(len(fr)),feat='rgb')
        res.append((iou.mean().item(),ncc.mean().item(),ncc2.mean().item()))
res=np.array(res)
for j in range(0,len(grid),3): print('%4d iou %.3f dog %.3f rgb %.3f'%(np.degrees(grid[j]),*res[j]))
k=np.argmax(res[:,1]); print('best dog',np.degrees(grid[k]), 'best rgb', np.degrees(grid[np.argmax(res[:,2])]),'best iou',np.degrees(grid[np.argmax(res[:,0])]))

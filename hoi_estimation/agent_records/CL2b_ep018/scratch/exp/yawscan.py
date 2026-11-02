import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch
from hoi_lib import *
D=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs/object_mesh.glb')
center=(mesh.bounds[0]+mesh.bounds[1])/2
R=Renderer(mesh,D['K'],D['W'],D['H'],scale=0.5)
Tf=np.load('proto_a.npz')['T']; Tcb=np.linalg.inv(D['T_wc'])[None]@D['rec']['T_world_mesh'].astype(float)
VID=np.load('vid_half.npy',mmap_mode='r'); OM=np.load('om_half.npy',mmap_mode='r'); PM=np.load('pm_half.npy',mmap_mode='r'); BB=np.load('bb.npy')
def Rz(th):
    c,s=np.cos(th),np.sin(th); M=np.eye(4); M[:2,:2]=[[c,-s],[s,c]]; return M
C=np.eye(4); C[:3,3]=center
for s,e in [(0,252),(462,659),(720,858)]:
    frames=list(range(s,e,6))
    bb=BB[frames]; cen=np.stack([(bb[:,0]+bb[:,1])/2,(bb[:,2]+bb[:,3])/2],1)
    off,(omc,pmc,imc)=make_crops([OM[frames],PM[frames],VID[frames]],cen)
    valid=torch.tensor(~pmc|omc,dtype=torch.float32,device='cuda'); vo=torch.tensor(omc&~pmc,dtype=torch.float32,device='cuda')
    img=torch.tensor(imc,dtype=torch.float32,device='cuda')/255; tg=torch.tensor(omc,device='cuda')
    offt=torch.tensor(off,dtype=torch.float32,device='cuda')
    for lab,TT in [('fit',Tf),('base',Tcb)]:
        line=[]
        for th in [-4,-2,-1,0,1,2,4]:
            T=TT[frames]@C@Rz(np.radians(th))@np.linalg.inv(C)
            rgb,m=render_crop(R,torch.tensor(T,dtype=torch.float32,device='cuda'),offt,192,256)
            ncc=ncc_score(rgb,m,img,vo).mean().item()
            iou=((m&tg).float()*valid).sum()/(((m|tg).float()*valid).sum())
            line.append('%d:%.3f/%.3f'%(th,ncc,iou.item()))
        print(s,lab,' '.join(line))

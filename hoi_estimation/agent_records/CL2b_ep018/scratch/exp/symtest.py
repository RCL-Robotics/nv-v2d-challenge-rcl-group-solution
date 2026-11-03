import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch
from hoi_lib import *
D=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs/object_mesh.glb')
center=(mesh.bounds[0]+mesh.bounds[1])/2
R=Renderer(mesh,D['K'],D['W'],D['H'],scale=0.5)
Tc=np.linalg.inv(D['T_wc'])[None]@D['rec']['T_world_mesh'].astype(np.float64)
VID=np.load('vid_half.npy',mmap_mode='r'); OM=np.load('om_half.npy',mmap_mode='r'); PM=np.load('pm_half.npy',mmap_mode='r'); BB=np.load('bb.npy')
def Rz(th):
    c,s=np.cos(th),np.sin(th); M=np.eye(4); M[:2,:2]=[[c,-s],[s,c]]; return M
C=np.eye(4); C[:3,3]=center
for frames in [range(0,250,10),range(300,460,10),range(470,650,10),range(660,720,5),range(720,858,10)]:
    frames=list(frames)
    bb=BB[frames]; cen=np.stack([(bb[:,0]+bb[:,1])/2,(bb[:,2]+bb[:,3])/2],1)
    off,(omc,pmc,imc)=make_crops([OM[frames],PM[frames],VID[frames]],cen)
    valid=torch.tensor(omc&~pmc,dtype=torch.float32,device='cuda')
    img=torch.tensor(imc,dtype=torch.float32,device='cuda')/255
    offt=torch.tensor(off,dtype=torch.float32,device='cuda')
    sc=[]
    for th in [0,90,180,270]:
        T=Tc[frames]@C@Rz(np.radians(th))@np.linalg.inv(C)
        rgb,m=render_crop(R,torch.tensor(T,dtype=torch.float32,device='cuda'),offt,192,256)
        sc.append(ncc_score(rgb,m,img,valid).cpu().numpy())
    sc=np.array(sc)
    print(frames[0],frames[-1],'mean ncc per hyp',np.round(sc.mean(1),3),'frac frames hyp0 best',(sc.argmax(0)==0).mean())

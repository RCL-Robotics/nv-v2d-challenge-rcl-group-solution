import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch
from hoi_lib import *
D=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs/object_mesh.glb')
center=(mesh.bounds[0]+mesh.bounds[1])/2
R=Renderer(mesh,D['K'],D['W'],D['H'],scale=0.5)
VID=np.load('vid_half.npy',mmap_mode='r'); OM=np.load('om_half.npy',mmap_mode='r'); PM=np.load('pm_half.npy',mmap_mode='r'); BB=np.load('bb.npy')
Tcb=np.linalg.inv(D['T_wc'])[None]@D['rec']['T_world_mesh'].astype(float)
cands=[('base',Tcb)]+[(f,np.load(f)['T']) for f in sys.argv[1:]]
phases=[('S1',0,252),('M1',252,456),('S2',456,659),('M2',659,720),('S3',720,858)]
res={}
for lab,T in cands:
    iou=np.zeros(858); ncc=np.zeros(858)
    for s in range(0,858,64):
        fr=list(range(s,min(s+64,858)))
        bb=BB[fr]; cen=np.stack([(bb[:,0]+bb[:,1])/2,(bb[:,2]+bb[:,3])/2],1)
        off,(omc,pmc,imc)=make_crops([OM[fr],PM[fr],VID[fr]],cen)
        valid=torch.tensor(~pmc|omc,device='cuda'); vo=torch.tensor(omc&~pmc,dtype=torch.float32,device='cuda')
        img=torch.tensor(imc,dtype=torch.float32,device='cuda')/255; tg=torch.tensor(omc,device='cuda')
        rgb,m=render_crop(R,torch.tensor(T[fr],dtype=torch.float32,device='cuda'),torch.tensor(off,dtype=torch.float32,device='cuda'),192,256)
        iou[fr]=(((m&tg)&valid).sum((1,2)).float()/((m|tg)&valid).sum((1,2)).float()).cpu().numpy()
        ncc[fr]=ncc_score(rgb,m,img,vo).cpu().numpy()
    pc=T[:,:3,:3]@center+T[:,:3,3]
    dt=np.linalg.norm(np.diff(pc,axis=0),axis=1); acc=np.linalg.norm(pc[2:]-2*pc[1:-1]+pc[:-2],axis=1)
    dR=np.array([rot_angle(T[i,:3,:3],T[i+1,:3,:3]) for i in range(857)])
    line=[lab[:12].ljust(12)]
    for n,s,e in phases:
        line.append('%s iou %.3f ncc %.3f dt %.1fmm acc %.1fmm dR %.2f |'%(n,iou[s:e].mean(),ncc[s:e].mean(),1000*np.median(dt[s:e-1]),1000*np.median(acc[s:e-2]),np.median(dR[s:e-1])))
    print(' '.join(line))

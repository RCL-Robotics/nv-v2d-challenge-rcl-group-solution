import sys, numpy as np, json, h5py, torch, cv2, time
sys.path.insert(0,'/work/outputs')
from hoilib import *
from scipy.ndimage import gaussian_filter1d
torch.manual_seed(0)
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
gp=json.load(open('/work/inputs/ground_plane.json'))['plane']; n=np.array(gp[:3]); down=-n/np.linalg.norm(n)
Tw=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float); F=len(Tw)
fr=np.load('/work/scratch/frames_all.npy',mmap_mode='r'); H5=h5py.File('/work/inputs/masks.h5')
mesh=TexMesh('/work/inputs/object_mesh.glb'); cr=np.load('/work/scratch/crops.npz')
yf=YawFrame(down)
dec=[yf.decompose(Tw[i,:3,:3]) for i in range(F)]
psi_b=np.array([d[1] for d in dec]); dvec=np.array([d[2] for d in dec])
FL=(0,150); DK=(452,F); CA=(150,452)
def circmean(a): return np.arctan2(np.sin(a).mean(),np.cos(a).mean())
def static_pose(a,b):
    psi=circmean(psi_b[a:b]); R=yf.compose(down,psi)
    tc=np.median((Tcw[None]@Tw[a:b])[:,:3,3],axis=0)
    T=np.eye(4); T[:3,:3]=Tcw[:3,:3]@R; T[:3,3]=tc; return T
Tfl=static_pose(*FL); Tdk=static_pose(*DK)
print('static yaw floor %.1f desk %.1f'%(np.degrees(circmean(psi_b[slice(*FL)])),np.degrees(circmean(psi_b[slice(*DK)]))))
# ---- colour model from static phases
P_np,N_np=surface_points(mesh); P=torch.tensor(P_np,device='cuda'); N=torch.tensor(N_np,device='cuda')
def frame_box(i,pad=24):
    om=H5['object'][i]; ys,xs=np.nonzero(om)
    if len(xs)<30: return None
    return (max(0,xs.min()-pad),max(0,ys.min()-pad),min(1536,xs.max()+pad),min(1152,ys.max()+pad))
def blurred(i): return cv2.GaussianBlur(np.asarray(fr[i]).astype(np.float32)/255.,(0,0),1.0)
def accumulate_model(frames, Tcam):
    S=torch.zeros(len(P_np),3,device='cuda'); C=torch.zeros(len(P_np),device='cuda')
    for i in frames:
        box=frame_box(i)
        if box is None: continue
        col,vis=sample_point_colors(mesh,P,N,torch.tensor(Tcam[i][None],dtype=torch.float32,device='cuda'),K,blurred(i),H5['object'][i],H5['person'][i],box)
        if vis.sum()<40: continue
        S+=(col[0]*vis[0,:,None]); C+=vis[0].float()
    return S,C
Tst=np.zeros((F,4,4)); Tst[FL[0]:FL[1]]=Tfl; Tst[DK[0]:DK[1]]=Tdk
t=time.time()
S1,C1=accumulate_model(range(FL[0],FL[1],2),Tst); S2,C2=accumulate_model(range(DK[0],DK[1],2),Tst)
Tbc=Tcw[None]@Tw
S3,C3=accumulate_model(range(330,445,2),Tbc)
S=S1+S2+S3; C=C1+C2+C3; model=S/C.clamp(min=1)[:,None]
print('model coverage all',(C>=2).sum().item())
print('model points: floor %d desk %d both %d  (%.1fs)'%((C1>=2).sum(),(C2>=2).sum(),((C1>=2)&(C2>=2)).sum(),time.time()-t))
# ---- yaw scan on carry
nb=36; psis=np.arange(nb)*2*np.pi/nb
dsm=gaussian_filter1d(dvec,3,axis=0); dsm/=np.linalg.norm(dsm,axis=1,keepdims=True)
tw=gaussian_filter1d(Tw[:,:3,3],2,axis=0)
IoU=np.full((F,nb),np.nan); PH=np.full((F,nb),np.nan); NP=np.zeros((F,nb))
t=time.time()
for i in range(*CA):
    Th=np.zeros((nb,4,4))
    for k,ps in enumerate(psis):
        Twm=np.eye(4); Twm[:3,:3]=yf.compose(dsm[i],ps); Twm[:3,3]=tw[i]; Th[k]=Tcw@Twm
    Thr,iou=fit_silhouettes(mesh,np.repeat(cr['Kc'][i][None],nb,0),np.repeat(cr['obs'][i][None],nb,0),np.repeat(cr['val'][i][None],nb,0),Th,iters=40,lr_r=0.0,lr_uv=0.8)
    IoU[i]=iou; Th=Thr
    Tt=torch.tensor(Th,dtype=torch.float32,device='cuda')
    with torch.no_grad():
        box=frame_box(i)
        if box is not None:
            col,vis=sample_point_colors(mesh,P,N,Tt,K,blurred(i),H5['object'][i],H5['person'][i],box)
            c,nn=model_corr(col,vis,model,C)
            PH[i]=c.cpu().numpy(); NP[i]=nn.cpu().numpy()
print('scan time %.1f'%(time.time()-t))
np.savez('/work/scratch/yawscan.npz',IoU=IoU,PH=PH,NP=NP,psis=psis,psi_b=psi_b)

from rend import *
r,cam=load(); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
Tc=Tcw[None]@r['T_world_mesh'].astype(float)
M=Mesh(); MK=Masks(); S=192
gp=np.array(json.load(open(INP+'/ground_plane.json'))['plane']); n=gp[:3]; d=gp[3]
upc=Tcw[:3,:3]@n
def run(a,b,maskkind,step=3,fix_up=False):
    ids=list(range(a,b,step)); fr=read_frames(ids)
    i0=ids[len(ids)//2]; p=Tc[i0,:3,3]; uv=K@p; uv=uv[:2]/uv[2]
    x0,y0=int(uv[0]-S/2),int(uv[1]-S/2)
    O=[];V=[]
    for i in ids:
        o,pm=MK.crop(i,x0,y0,S)
        if maskkind=='red':
            hsv=cv2.cvtColor(fr[i][y0:y0+S,x0:x0+S],cv2.COLOR_BGR2HSV)
            red=(hsv[...,1]>110)&((hsv[...,0]<12)|(hsv[...,0]>165))
            near=cv2.dilate(o.astype(np.uint8),np.ones((9,9),np.uint8))>0
            o=red&near
            o=cv2.morphologyEx(o.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))>0
        pmd=cv2.dilate(pm.astype(np.uint8),np.ones((5,5),np.uint8))>0
        O.append(o); V.append(~pmd|o)
    O=torch.tensor(np.array(O),dtype=torch.float32,device=DEV); V=torch.tensor(np.array(V),dtype=torch.float32,device=DEV)
    # single pose: share by fitting on the mean mask with valid weights
    Om=(O*V).sum(0,keepdim=True)/V.sum(0,keepdim=True).clamp(min=1); Vm=(V.mean(0,keepdim=True)>0.5).float()
    R0=Tc[i0:i0+1,:3,:3].copy()
    if fix_up:
        # rotate so mesh y aligns with floor up
        yb=R0[0][:,1]; ax=np.cross(yb,upc); s_=np.linalg.norm(ax); ang=np.arcsin(s_)
        import scipy.spatial.transform as st
        R0[0]=st.Rotation.from_rotvec(ax/s_*ang).as_matrix()@R0[0]
    R,t,iou,_=fit_sil(M,K,R0,uv[None],np.array([p[2]]),Om,Vm,np.array([[x0,y0]]),S,iters=(100,100,150),fit_rot=not fix_up)
    tilt=np.degrees(np.arccos(R[0][:,1]@upc))
    W=(M.v.cpu().numpy()@R[0].T+t[0])@Twc[:3,:3].T+Twc[:3,3]; h=(W@n+d).min()
    print(f'{a}-{b} {maskkind:5s} fix_up={fix_up}: z_fit {t[0,2]:.4f} z_base_med {np.median(Tc[a:b,2,3]):.4f} dz {100*(t[0,2]-np.median(Tc[a:b,2,3])):.2f}cm iou {iou[0]:.3f} tilt {tilt:.1f} bottom-above-plane {h*100:.1f}cm  mask area {Om.sum().item():.0f}')
for a,b in [(0,160),(795,870)]:
    for mk in ['sam','red']:
        for fu in [False,True]:
            run(a,b,mk,fix_up=fu)

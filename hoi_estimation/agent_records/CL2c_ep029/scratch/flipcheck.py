from traj import *
import cv2
d=np.load('run1_half_notilt.npz'); Tn=d['Tw']; s=float(d['s'])
dh=np.load('dec_handle.npz'); handR=SilRenderer(dh['V'],dh['F'])
crop=(500,520,400,520); x0,y0,W,H=crop; Pm=bodyR.proj(K,x0,y0,W,H)
def scaled_cam(Tw_, s):
    # scale about contact centre in world
    Tc_=Tcw@Tw_; Rc=Tc_[:3,:3]; tc=Tc_[:3,3]
    cmc=Rc@C_M+tc
    return Rc*s, cmc-Rc@C_M*s
def scores(i, Tw_, s):
    Rc,tc=scaled_cam(Tw_,s); Rc=tt(Rc)[None]; tc=tt(tc)[None]
    with torch.no_grad():
        mb,zb=bodyR.render(Rc,tc,Pm[None],H,W); mh,zh=handR.render(Rc,tc,Pm[None],H,W)
    mb=mb[0].cpu().numpy()>0.5; mh=(mh[0].cpu().numpy()>0.5)&~mb
    o=OM[i][y0:y0+H,x0:x0+W]; p=PM[i][y0:y0+H,x0:x0+W]
    iou=(mb&o&~p).sum()/((mb|o)&~p).sum()
    g=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg',0)[y0:y0+H,x0:x0+W]
    e=cv2.Canny(g,30,90)>0
    dt=cv2.distanceTransform((~e).astype(np.uint8),cv2.DIST_L2,3)
    he=(cv2.Canny(mh.astype(np.uint8)*255,50,150)>0)&~p&~o
    ch=np.minimum(dt[he],10).mean() if he.sum()>20 else np.nan
    return iou,ch
flipR=lambda T: T@np.r_[np.c_[R.from_rotvec(U_M*np.pi).as_matrix(),np.zeros(3)],[[0,0,0,1]]]
for i in [20,60,180,300,420,118,236,358]:
    T=Tn[i]
    # flip about the up axis through the contact centre
    Rf=R.from_rotvec(NRM*np.pi).as_matrix(); cw=T[:3,:3]@C_M+T[:3,3]
    Tf=np.eye(4); Tf[:3,:3]=Rf@T[:3,:3]; Tf[:3,3]=cw-Tf[:3,:3]@C_M
    a=scores(i,T,s); b=scores(i,Tf,s); c=scores(i,Tw[i],1.0)
    print(i,'ours iou %.3f handle-chamfer %.2fpx | yaw+180 iou %.3f chamfer %.2fpx | base(s=1) iou %.3f chamfer %.2f'%(a+b+c))

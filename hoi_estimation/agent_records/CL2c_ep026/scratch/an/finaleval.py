"""validation table: baseline vs result. Objective terms (IoU, NCC) + independent checks (wheel colour, edge)"""
from common import *; from render import Renderer; from scorer import *; import sys, torch, cv2, json
r,cam,pl,m=load(); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
R=Renderer(m,K); msk=h5py.File('/work/inputs/masks.h5')
n=pl[:3]/np.linalg.norm(pl[:3]); d=pl[3]/np.linalg.norm(pl[:3])
Tb=r['T_world_mesh'].astype(float); F=len(Tb)
trajs={'baseline':Tb,'result':np.load(sys.argv[1])['T_world_mesh']}
phases=[(0,281,'S1 static'),(281,357,'M1 rotate'),(357,399,'S2 static'),(399,510,'M2 rotate'),(510,669,'S3 static')]
def ncc(a,b,w):
    if w.sum()<20: return np.nan
    a=a[w]-a[w].mean(); b=b[w]-b[w].mean(); return (a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-9)
res={}
STEP=3
for name,T in trajs.items():
    Tc=Tcw[None]@T; out=np.full((F,4),np.nan)
    for i in range(0,F,STEP):
        ob=msk['object'][i]; per=msk['person'][i]
        x0,y0,w,h=crop_for(ob)
        im=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')[y0:y0+h,x0:x0+w][...,::-1].astype(np.float32)/255
        pc=per[y0:y0+h,x0:x0+w]; oc=ob[y0:y0+h,x0:x0+w]
        col,mk,_=R.render(Tc[i],crop=(x0,y0,w,h),res=(2*h,2*w))
        col=cv2.resize(col[0].cpu().numpy(),(w,h),interpolation=cv2.INTER_AREA); mk=cv2.resize(mk[0].cpu().numpy(),(w,h),interpolation=cv2.INTER_AREA)
        mb=mk>0.5
        iou=(mb&oc&~pc).sum()/max(((mb|oc)&~pc).sum(),1)
        g=im.mean(-1); gr=col.mean(-1)
        nc=ncc(gr,g,(mk>0.99)&~pc)
        Gi=im[...,1]-0.5*(im[...,0]+im[...,2]); Gr=col[...,1]-0.5*(col[...,0]+col[...,2])
        green=(mk>0.9)&(Gr>0.15)&~pc; grey=(mk>0.9)&(Gr<0.03)&(gr>0.2)&(gr<0.6)&(np.abs(col[...,0]-col[...,2])<0.05)&~pc
        wc=Gi[green].mean()-Gi[grey].mean() if green.sum()>5 and grey.sum()>5 else np.nan
        gg=cv2.GaussianBlur(g,(3,3),0); gm=np.hypot(cv2.Sobel(gg,cv2.CV_32F,1,0),cv2.Sobel(gg,cv2.CV_32F,0,1))
        e=(cv2.morphologyEx(mb.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3)))>0)&~pc
        out[i]=[iou,nc,wc,gm[e].mean() if e.sum()>10 else np.nan]
    res[name]=(out,Tc)
Tcb=Tcw[None]@Tb; v=m.vertices[::25]
rows=[]
hdr='| phase | traj | IoU | NCC | wheel-colour (+ = consistent) | edge grad | dz mean cm | dz median cm | median rot diff to baseline deg | mean frame-to-frame rot deg | mean frame-to-frame trans cm | lowest wheel height cm |'
print(hdr); print('|'+'---|'*12)
for a,b,pn in phases+[(0,F,'ALL')]:
    for name,(out,Tc) in res.items():
        T=Twc[None]@Tc; o=out[a:b]
        dz=(Tc[a:b,2,3]-Tcb[a:b,2,3])*100
        rd=rotang(Tc[a:b,:3,:3],Tcb[a:b,:3,:3]); drr=rotang(Tc[a+1:b,:3,:3],Tc[a:b-1,:3,:3])
        dt=np.linalg.norm(Tc[a+1:b,:3,3]-Tc[a:b-1,:3,3],axis=1)*100
        wh=np.mean([np.sort((v@T[i,:3,:3].T+T[i,:3,3])@n+d)[:10].mean() for i in range(a,b,5)])*100
        print(f'| {pn} | {name} | {np.nanmean(o[:,0]):.3f} | {np.nanmean(o[:,1]):.3f} | {np.nanmean(o[:,2]):+.3f} ({np.nanmean(o[:,2]>0)*100:.0f}% frames >0) | {np.nanmean(o[:,3]):.1f} | {dz.mean():+.2f} | {np.median(dz):+.2f} | {np.median(rd):.1f} | {drr.mean():.2f} | {dt.mean():.2f} | {wh:+.1f} |')

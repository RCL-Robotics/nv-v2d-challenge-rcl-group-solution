from common import *; from render import Renderer; from scorer import *; from param import *; import torch, cv2
r,cam,pl,m=load(); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
R=Renderer(m,K); msk=h5py.File('/work/inputs/masks.h5')
fp=FloorParam(pl[:3]); Tb=r['T_world_mesh'].astype(float); Ry=np.diag([-1,1,-1,1.])
o=np.load('perframe.npy')
ss=np.arange(0.90,1.101,0.01)
for a,b in [(0,285),(353,404),(507,669)]:
    acc=np.zeros(len(ss)); accm=np.zeros(len(ss)); cnt=0
    for i in range(a,b,6):
        T=fp.apply(Tb[i]@Ry,*o[i,0,:3]); Tc=Tcw@T
        crop=crop_for(msk['object'][i]); x0,y0,w,h=crop
        im=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg').astype(np.float32)
        g=cv2.GaussianBlur(im.mean(-1),(3,3),0)
        gm=np.hypot(cv2.Sobel(g,cv2.CV_32F,1,0),cv2.Sobel(g,cv2.CV_32F,0,1))[y0:y0+h,x0:x0+w]
        per=msk['person'][i][y0:y0+h,x0:x0+w]; ob=msk['object'][i][y0:y0+h,x0:x0+w].astype(np.uint8)
        mb=(cv2.morphologyEx(ob,cv2.MORPH_GRADIENT,np.ones((3,3)))>0)
        P=np.stack([Tc]*len(ss)); P[:,:3,3]*=ss[:,None]
        _,mk,_=R.render(P,crop=crop); mk=mk.cpu().numpy().astype(np.uint8)
        for k in range(len(ss)):
            e=(cv2.morphologyEx(mk[k],cv2.MORPH_GRADIENT,np.ones((3,3)))>0)&~per
            acc[k]+=gm[e].mean(); 
            # distance of rendered contour to mask contour
            dt=cv2.distanceTransform((~mb).astype(np.uint8),cv2.DIST_L2,3)
            accm[k]+=dt[e].mean()
        cnt+=1
    print(a,b,'edge grad vs scale', ' '.join('%.2f:%.1f'%(s,x/cnt) for s,x in zip(ss,acc)),' best %.2f'%ss[acc.argmax()],' | mask-contour dist best %.2f'%ss[accm.argmin()])

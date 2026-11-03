from common import *; from render import Renderer; from scorer import *; from param import *; import torch
r,cam,pl,m=load(); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
R=Renderer(m,K); msk=h5py.File('/work/inputs/masks.h5'); S=Scorer(R,'/work/scratch/frames',msk)
fp=FloorParam(pl[:3]); n=pl[:3]; d=pl[3]
Tb=r['T_world_mesh'].astype(float); Ry=np.diag([-1,1,-1,1.])
o=np.load('perframe.npy')
v=m.vertices[::10]
for a,b in [(0,285),(353,404),(507,669)]:
    fr=list(range(a,b,4))
    ss=np.arange(0.90,1.101,0.01); acc=np.zeros((len(ss),2)); sfloor=[]; zb=[]
    for i in fr:
        T=fp.apply(Tb[i]@Ry,*o[i,0,:3]); Tc=Tcw@T; zb.append((Tcw@Tb[i])[2,3])
        P=np.stack([Tc]*len(ss)); P[:,:3,3]*=ss[:,None]
        iou,ncc=S.score(i,P,crop_for(msk['object'][i])); acc[:,0]+=iou; acc[:,1]+=ncc
        # floor contact scale
        hs=[]
        for s in ss:
            Tw=Twc@P[list(ss).index(s)]; p=v@Tw[:3,:3].T+Tw[:3,3]; hs.append(np.sort(p@n+d)[:20].mean())
        sfloor.append(np.interp(0,np.array(hs)[::-1],ss[::-1]))
        zr=Tc[2,3]
    acc/=len(fr)
    zr=np.median([ (Tcw@fp.apply(Tb[i]@Ry,*o[i,0,:3]))[2,3] for i in fr]); zbm=np.median(zb)
    print(a,b,'refined z %.3f baseline z %.3f'%(zr,zbm))
    print('  IoU vs scale:',' '.join('%.2f:%.4f'%(s,x) for s,x in zip(ss,acc[:,0])))
    k=acc[:,0].argmax(); print('  best IoU scale %.2f -> dz %.3f m; ncc best scale %.2f'%(ss[k],(ss[k]-1)*zr, ss[acc[:,1].argmax()]))
    sf=np.median(sfloor); print('  floor contact scale %.3f -> dz %.3f m'%(sf,(sf-1)*zr))

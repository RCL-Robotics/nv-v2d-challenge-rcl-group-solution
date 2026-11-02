import sys, time; sys.path.insert(0,'/work/outputs')
from lib import *
exec(open('/work/scratch/code/upright.py').read())
Fi=Fitter(R,K,D,c0)
phases=[(0,133),(210,273),(345,408),(474,531),(600,642),(687,780)]
step=int(sys.argv[1]) if len(sys.argv)>1 else 15
angs=np.radians(np.arange(0,360,step))
def scores(Tf, fr):
    # Tf (Hn,4,4) one pose per hyp, evaluated on frames fr
    Hn=len(Tf); out={k:np.zeros(Hn) for k in ['iou','ncc','err','cerr','wncc']}
    for t in fr:
        img,om,valid,_=Fi.batch(np.array([t]*Hn))
        T=torch.tensor(Tf,device='cuda',dtype=torch.float32)
        rgb,m,sil=Fi.render_pairs(T,np.array([t]*Hn))
        iou,ncc=Fi.losses(T,np.array([t]*Hn),(img,om,valid,None) if False else Fi.batch(np.array([t]*Hn)))
        g=blur(gray(rgb)); gi=blur(gray(img))
        reg=m*valid; n=reg.sum((1,2)).clamp_min(1)
        # affine fit gray
        gm=(g*reg).sum((1,2))/n; im_=(gi*reg).sum((1,2))/n
        a=((g-gm[:,None,None])*(gi-im_[:,None,None])*reg).sum((1,2))/(((g-gm[:,None,None])**2*reg).sum((1,2))+1e-8)
        pred=a[:,None,None]*(g-gm[:,None,None])+im_[:,None,None]
        err=((pred-gi).abs()*reg).sum((1,2))/n
        cerr=0
        for c in range(3):
            r=rgb[...,c]; ic=img[...,c]; rm=(r*reg).sum((1,2))/n; imc=(ic*reg).sum((1,2))/n
            ac=((r-rm[:,None,None])*(ic-imc[:,None,None])*reg).sum((1,2))/(((r-rm[:,None,None])**2*reg).sum((1,2))+1e-8)
            cerr=cerr+(((ac[:,None,None]*(r-rm[:,None,None])+imc[:,None,None]-ic).abs())*reg).sum((1,2))/n
        # variance-weighted ncc in common region
        gz=g*m; wv=gz.std(0,keepdim=True)*valid; U=(m.sum(0,keepdim=True)>0).float()*valid
        gt=gi; 
        x=(gz-(gz*wv).sum((1,2),keepdim=True)/wv.sum()); y=(gt-(gt*wv).sum((1,2),keepdim=True)/wv.sum())
        wn=(x*y*wv).sum((1,2))/((x*x*wv).sum((1,2))*(y*y*wv).sum((1,2))).sqrt().clamp_min(1e-8)
        for k,vv in [('iou',iou),('ncc',ncc),('err',err),('cerr',cerr),('wncc',wn)]: out[k]+=vv.detach().cpu().numpy()/len(fr)
    return out
for (a,b) in phases:
    fr=np.arange(a,b); fr=fr[np.argsort([(D['pm'][i]&~D['om'][i])[40:120,40:120].sum() for i in fr])][:12]
    Tm=Tc[fr[0]].copy(); Tm[:3,3]=np.median(Tc[a:b,:3,3],0); Tu=upright(Tm)
    T0=spin(Tu,c0,u,angs)
    Tf,iou,ncc=Fi.fit(T0,[list(fr)]*len(T0),iters=40,free_rot=False)
    s=scores(Tf,fr)
    # absolute handle azimuth in camera (atan2(z,x)) of mesh +x projected horizontally
    hx=np.einsum('hij,j->hi',Tf[:,:3,:3],np.array([1.,0,0]))
    az=np.degrees(np.arctan2(hx[:,2],hx[:,0]))
    print('phase',a,b)
    print('  az  ',' '.join('%4d'%x for x in az))
    for k in s: print('  %-4s'%k,' '.join('%4d'%(100*x) for x in s[k]),' best az',int(az[np.argmax(s[k]) if k not in('err','cerr') else np.argmin(s[k])]))

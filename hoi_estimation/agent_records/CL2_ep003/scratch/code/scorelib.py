import sys; sys.path.insert(0,"/work/outputs")
from lib import *
def scores(FI, Tf, fr):
    # Tf (Hn,4,4) one pose per hyp, evaluated on frames fr
    Hn=len(Tf); out={k:np.zeros(Hn) for k in ['iou','ncc','err','cerr','wncc']}
    for t in fr:
        img,om,valid,_=FI.batch(np.array([t]*Hn))
        T=torch.tensor(Tf,device='cuda',dtype=torch.float32)
        rgb,m,sil=FI.render_pairs(T,np.array([t]*Hn))
        iou,ncc=FI.losses(T,np.array([t]*Hn),(img,om,valid,None) if False else FI.batch(np.array([t]*Hn)))
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

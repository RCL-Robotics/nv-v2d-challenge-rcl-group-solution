import sys; sys.path.insert(0,'/work/outputs')
from common import *; from fitcore import *
def viz(fit,d,frames,R,t,out,ncol=4):
    tiles=[]
    for k,i in enumerate(frames):
        P=fit.proj_mats(d['roi'][[i]],(160,160))
        m,rgb,_=fit.render(torch.tensor(R[k:k+1],dtype=torch.float32,device='cuda'),torch.tensor(t[k:k+1],dtype=torch.float32,device='cuda'),P,(160,160),rgb=True)
        m=m[0].detach().cpu().numpy(); rgb=rgb[0].detach().cpu().numpy()
        img=d['rgb'][i]; ov=img*(1-0.6*m[...,None])+0.6*rgb
        ob=(d['mask'][i]>0.5).astype(np.uint8)
        a=(np.hstack([img,ov,rgb+(1-m[...,None])*img*0.3])*255).astype(np.uint8)[...,::-1].copy()
        cnt,_=cv2.findContours(ob,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for off in (0,160): 
            c2=[c+[off,0] for c in cnt]; cv2.drawContours(a,c2,-1,(0,255,0),1)
        cv2.putText(a,str(i),(3,14),0,0.45,(0,255,255),1)
        tiles.append(cv2.resize(a,None,fx=1.5,fy=1.5))
    while len(tiles)%ncol: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(out,np.vstack([np.hstack(tiles[k:k+ncol]) for k in range(0,len(tiles),ncol)]))

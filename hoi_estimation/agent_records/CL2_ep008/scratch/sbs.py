import sys, numpy as np, cv2, torch
from common import *
from render import Renderer, scaleK
def sbs(Tcams, idxs, out, names=None):
    # full res crops; Tcams: list of (F,4,4)
    cap=cv2.VideoCapture('/work/inputs/video.mp4')
    R=Renderer(1536,1152)
    rows=[]
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,img=cap.read()
        ms=[]; rends=[]
        for T in Tcams:
            o=R.render(T[i:i+1],K); m=o['mask'][0].cpu().numpy()
            rgb=(o['rgb'][0].cpu().numpy()*255).astype(np.uint8)[...,::-1]
            r=img.copy()//3; r[m]=rgb[m]; rends.append(r); ms.append(m)
        allm=np.any(ms,0)|(np.abs(img.astype(int)-img.astype(int)).sum(-1)<0)
        ys,xs=np.nonzero(np.any(ms,0)); 
        y0,y1,x0,x1=max(ys.min()-40,0),min(ys.max()+40,1152),max(xs.min()-40,0),min(xs.max()+40,1536)
        tiles=[img[y0:y1,x0:x1]]+[r[y0:y1,x0:x1] for r in rends]
        h=300; tiles=[cv2.resize(t,(int(t.shape[1]*h/t.shape[0]),h)) for t in tiles]
        row=np.hstack(tiles); cv2.putText(row,str(i),(5,25),0,0.8,(0,0,255),2); rows.append(row)
    w=max(r.shape[1] for r in rows); rows=[np.pad(r,((0,0),(0,w-r.shape[1]),(0,0))) for r in rows]
    cv2.imwrite(out,np.vstack(rows))
if __name__=='__main__':
    idxs=[int(x) for x in sys.argv[1].split(',')]
    sbs([Tc],idxs,sys.argv[2])

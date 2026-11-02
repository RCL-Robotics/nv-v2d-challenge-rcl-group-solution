import sys; sys.path.insert(0,'/work/outputs')
import cv2, numpy as np, h5py, pickle
def extend(img, om, pm, H_rng=(95,125), S_min=60, dil=2):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    B=(hsv[...,0]>=H_rng[0])&(hsv[...,0]<=H_rng[1])&(hsv[...,1]>=S_min)&~pm
    near=cv2.dilate(om.astype(np.uint8),np.ones((101,101),np.uint8))>0
    U=(B|om|(pm&near)).astype(np.uint8)  # person pixels near the mask (the hand) may connect barrel and handle
    Ud=cv2.dilate(U,np.ones((2*dil+1,2*dil+1),np.uint8))
    n,lab=cv2.connectedComponents(Ud)
    keep=np.unique(lab[om&(Ud>0)]); keep=keep[keep>0]
    return om | (B & np.isin(lab,keep))
if __name__=='__main__':
    cap=cv2.VideoCapture('/work/inputs/video.mp4'); f=h5py.File('/work/inputs/masks.h5')
    tiles=[]
    for i in [int(x) for x in sys.argv[1:]]:
        cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read()
        om=f['object'][i]; pm=f['person'][i]
        ys,xs=np.nonzero(om); cx,cy=int(xs.mean()),int(ys.mean())
        x0=max(0,min(1536-200,cx-100)); y0=max(0,min(1152-200,cy-100))
        sl=(slice(y0,y0+200),slice(x0,x0+200))
        E=extend(im[sl],om[sl],pm[sl])
        c=im[sl].copy()
        for m,col in [(om[sl],(255,0,255)),(E,(0,255,0))]:
            cs,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(c,cs,-1,col,1)
        c=cv2.resize(c,(400,400),interpolation=cv2.INTER_NEAREST); cv2.putText(c,str(i),(5,25),0,0.8,(0,0,255),2); tiles.append(c)
    while len(tiles)%5: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite('/work/scratch/ext.jpg',np.vstack([np.hstack(tiles[k:k+5]) for k in range(0,len(tiles),5)]))

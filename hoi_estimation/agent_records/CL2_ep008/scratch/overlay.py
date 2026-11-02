import sys, numpy as np, cv2, torch
from common import *
from render import Renderer, scaleK
def overlay(Tcam, frames_idx, out, crop=(270,500,240,470), cols=8):
    fr=np.load('/work/scratch/frames_half.npy', mmap_mode='r')
    R=Renderer(768,576); Kh=scaleK(K,0.5)
    tiles=[]
    for i in frames_idx:
        o=R.render(Tcam[i:i+1],Kh)
        rgb=(o['rgb'][0].cpu().numpy()*255).astype(np.uint8)[...,::-1]; m=o['mask'][0].cpu().numpy()
        img=np.array(fr[i]).copy()
        a=img.copy(); a[m]=(0.5*img[m]+0.5*rgb[m]).astype(np.uint8)
        cnts,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(a,cnts,-1,(0,255,0),1)
        y0,y1,x0,x1=crop
        t=np.hstack([img[y0:y1,x0:x1],a[y0:y1,x0:x1]]); cv2.putText(t,str(i),(5,20),0,0.6,(0,0,255),2); tiles.append(t)
    while len(tiles)%cols: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(out,np.vstack([np.hstack(tiles[i:i+cols]) for i in range(0,len(tiles),cols)]))
if __name__=='__main__':
    overlay(Tc, list(range(0,708,24)), 'ov_base.jpg', cols=6)

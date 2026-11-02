import cv2, numpy as np, h5py, sys
c=cv2.VideoCapture('/work/inputs/video.mp4')
f=h5py.File('/work/inputs/masks.h5')
idx=[int(x) for x in sys.argv[1].split(',')]
tiles=[];i=0
want=set(idx); frames={}
j=0
while len(frames)<len(want):
    ok,im=c.read()
    if not ok: break
    if j in want: frames[j]=im
    j+=1
for i in idx:
    im=frames[i]; om=f['object'][i]
    ys,xs=np.nonzero(om)
    if len(xs)==0: cy,cx=576,768
    else: cy,cx=int(ys.mean()),int(xs.mean())
    s=120
    crop=im[max(cy-s,0):cy+s,max(cx-s,0):cx+s]
    crop=cv2.resize(crop,(360,360),interpolation=cv2.INTER_CUBIC)
    cv2.putText(crop,str(i),(5,25),0,0.8,(0,255,0),2); tiles.append(crop)
while len(tiles)%6: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[2],np.vstack([np.hstack(tiles[k:k+6]) for k in range(0,len(tiles),6)]))

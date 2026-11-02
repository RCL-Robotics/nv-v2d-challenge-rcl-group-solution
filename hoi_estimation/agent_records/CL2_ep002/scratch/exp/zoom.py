import numpy as np, cv2, sys
ms=np.load('/work/scratch/maskstats.npz')
fr=[int(x) for x in sys.argv[1].split(',')]; tiles=[]
S=int(sys.argv[3]) if len(sys.argv)>3 else 110
for i in fr:
    img=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')
    cx,cy=ms['cx'][i],ms['cy'][i]
    if np.isnan(cx): cx,cy=float(sys.argv[4]),float(sys.argv[5])
    x0=int(cx-S/2); y0=int(cy-S/2)
    c=cv2.resize(img[y0:y0+S,x0:x0+S],(400,400),interpolation=cv2.INTER_CUBIC)
    cv2.putText(c,str(i),(5,25),0,0.8,(0,255,255),2); tiles.append(c)
while len(tiles)%4: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[2], np.vstack([np.hstack(tiles[k:k+4]) for k in range(0,len(tiles),4)]))

import cv2,numpy as np,sys
a,b,s=map(int,sys.argv[1:4]); out=sys.argv[4]
x0,y0,x1,y1=[int(v) for v in (sys.argv[5].split(',') if len(sys.argv)>5 else '120,180,360,320'.split(','))]
tiles=[]
for i in range(a,b,s):
    im=cv2.imread(f'/work/scratch/crops/{i:05d}.jpg')[y0:y1,x0:x1]
    im=cv2.resize(im,None,fx=1.5,fy=1.5); cv2.putText(im,str(i),(3,18),0,0.6,(0,255,255),2); tiles.append(im)
n=6
while len(tiles)%n: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out,np.vstack([np.hstack(tiles[k:k+n]) for k in range(0,len(tiles),n)]))

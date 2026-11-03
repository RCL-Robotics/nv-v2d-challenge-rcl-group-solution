import pickle, numpy as np, cv2, sys
from viz import R, K
Rs,ts,un=pickle.load(open('cache/s2.pkl','rb'))
f=int(sys.argv[1]); h=int(sys.argv[2])
img=cv2.imread(f'/work/scratch/frames/{f:04d}.jpg'); c=np.load('/work/scratch/maskstats.npz')['c'][f]; S=110
x0=int(c[0]-S/2); y0=int(c[1]-S/2); crop=img[y0:y0+S,x0:x0+S]
tiles=[crop]
for k in range(0,24,2):
    j=h*24+k; T=np.eye(4); T[:3,:3]=Rs[f][j]; T[:3,3]=ts[f][j]
    col,mm,_=R.render(T,K,S,S,x0,y0); col=(col[0].cpu().numpy()[...,::-1]*255).astype(np.uint8); mm=mm[0].cpu().numpy()
    o=crop.copy(); o[mm]=col[mm]; cv2.putText(o,'%d %.2f'%(k*15,un[f][j]),(2,12),0,0.4,(0,0,255),1); tiles.append(o)
tiles.append(crop)
cv2.imwrite(sys.argv[3],cv2.resize(np.vstack([np.hstack(tiles[:7]),np.hstack(tiles[7:])]),None,fx=2,fy=2))

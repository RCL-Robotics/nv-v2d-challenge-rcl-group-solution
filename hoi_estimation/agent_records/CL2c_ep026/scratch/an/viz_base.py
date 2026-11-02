from common import *; from render import Renderer; import cv2, torch, sys
r,cam,pl,m=load(); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam'])
R=Renderer(m,K)
T=r['T_world_mesh'].astype(np.float64)
key=sys.argv[1] if len(sys.argv)>1 else 'base'
if key!='base': T=np.load(key)['T_world_mesh']
Tc=np.linalg.inv(Twc)[None]@T
ids=[0,150,260,300,330,360,390,430,470,500,530,600]
tiles=[]
for i in ids:
    im=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')
    col,mask,_=R.render(Tc[i])
    col=(col[0].cpu().numpy()[...,::-1]*255).astype(np.uint8); mk=mask[0].cpu().numpy()>0
    ov=im.copy(); ov[mk]=(0.4*im[mk]+0.6*col[mk]).astype(np.uint8)
    cnt,_=cv2.findContours(mk.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    ys,xs=np.nonzero(mk); cy,cx=int(ys.mean()),int(xs.mean())
    c1=im[cy-80:cy+80,cx-120:cx+120].copy(); c2=ov[cy-80:cy+80,cx-120:cx+120].copy()
    cv2.drawContours(c1,[cc-np.array([cx-120,cy-80]) for cc in cnt],-1,(0,0,255),1)
    t=np.hstack([cv2.resize(c1,(480,320)),cv2.resize(c2,(480,320))]); cv2.putText(t,str(i),(5,30),0,1,(0,255,0),2); tiles.append(t)
rows=[np.hstack(tiles[k:k+2]) for k in range(0,12,2)]
cv2.imwrite(f'/work/scratch/viz/ov_{sys.argv[2] if len(sys.argv)>2 else "base"}.jpg',np.vstack(rows))

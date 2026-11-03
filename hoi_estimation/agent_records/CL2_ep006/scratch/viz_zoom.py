import sys; sys.path.insert(0,'/work/outputs')
from hoi_render import *
import cv2, json
# args: out.jpg frames poseA.npy[,poseB.npy...]  ; draws contours colors red=baseline, green, blue, ...
out=sys.argv[1]; frames=[int(x) for x in sys.argv[2].split(',')]; Ps=[np.load(p) for p in sys.argv[3].split(',')]
Tb=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam'])
m=load_mesh('/work/inputs/object_mesh.glb'); R=Renderer(m,1152,1536)
cols=[(0,0,255),(0,255,0),(255,128,0),(255,0,255)]
tiles=[]
for i in frames:
    im=cv2.imread(f'/work/scratch/frames/{i+1:04d}.jpg')
    ov=im.copy()
    for T,col in zip([Tb]+Ps,cols):
        _,mk,_=R.render(np.linalg.inv(Twc)@T[i],K); mk=mk[0].cpu().numpy().astype(np.uint8)
        cnt,_=cv2.findContours(mk,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE); cv2.drawContours(ov,cnt,-1,col,1)
    ys,xs=np.nonzero(mk); cx,cy=int(xs.mean()),int(ys.mean())
    c=ov[max(0,cy-150):cy+150,max(0,cx-150):cx+150]; c=cv2.resize(c,(600,600),interpolation=cv2.INTER_NEAREST)
    cv2.putText(c,str(i),(5,25),0,0.8,(0,0,255),2); tiles.append(c)
nc=3
while len(tiles)%nc: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out,np.vstack([np.hstack(tiles[k:k+nc]) for k in range(0,len(tiles),nc)]))

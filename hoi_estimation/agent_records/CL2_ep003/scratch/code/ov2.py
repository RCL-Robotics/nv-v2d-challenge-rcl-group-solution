import sys; sys.path.insert(0,'/work/outputs')
from lib import *
src=sys.argv[1]; out=sys.argv[2]; idx=[int(x) for x in sys.argv[3].split(',')]
m,v,f,uv,tex=load_mesh('/work/inputs/object_mesh.glb')
cam,K,Twc,rec=load_inputs('/work/inputs')
if src=='base': Tw=rec['T_world_mesh']
else: Tw=np.load(src)['T_world_mesh']
Tc=np.linalg.inv(Twc)[None]@Tw
R=Renderer(v,f,uv,tex)
cap=cv2.VideoCapture('/work/inputs/video.mp4'); i=0; tiles=[]
while i<=max(idx):
    ok,im=cap.read()
    if i in idx:
        rgb,mask,_=R.render(Tc[i:i+1],K,W,H)
        rgb=(rgb[0].cpu().numpy()[...,::-1]*255).astype(np.uint8); mk=mask[0].cpu().numpy()
        ys,xs=np.nonzero(mk); cy,cx=(int(ys.mean()),int(xs.mean())) if len(ys) else (500,800)
        h=70; sl=(slice(max(cy-h,0),cy+h),slice(max(cx-h,0),cx+h))
        a=im[sl].copy(); b=a.copy(); b[mk[sl]]=rgb[sl][mk[sl]]
        c=np.hstack([a,b]); c=cv2.resize(c,(560,280),interpolation=cv2.INTER_NEAREST); cv2.putText(c,str(i),(5,25),0,0.8,(0,0,255),2); tiles.append(c)
    i+=1
while len(tiles)%3: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out,np.vstack([np.hstack(tiles[k:k+3]) for k in range(0,len(tiles),3)]))

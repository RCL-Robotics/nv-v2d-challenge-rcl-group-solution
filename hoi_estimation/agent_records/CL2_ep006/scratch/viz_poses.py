import sys; sys.path.insert(0,'/work/outputs')
from hoi_render import *
import json, cv2
Tc=np.load(sys.argv[1]); out_path=sys.argv[2]; frames=[int(x) for x in sys.argv[3].split(',')]
m=load_mesh('/work/inputs/object_mesh.glb')
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'])
R=Renderer(m,1152,1536)
out=[]
for i in frames:
    rgb,mask,d=R.render(Tc[i],K)
    rgb=(rgb[0].cpu().numpy()*255).astype(np.uint8)[...,::-1]; mk=mask[0].cpu().numpy()
    im=cv2.imread(f'frames/{i+1:04d}.jpg')
    ov=im.copy(); ov[mk]=(0.5*im[mk]+0.5*rgb[mk]).astype(np.uint8)
    cnt,_=cv2.findContours(mk.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(ov,cnt,-1,(0,255,0),1)
    crop=lambda a:a[420:740,700:1060]
    t=np.hstack([crop(im),crop(ov),crop(rgb)]); cv2.putText(t,str(i),(5,25),0,0.8,(0,0,255),2); out.append(t)
while len(out)%2: out.append(np.zeros_like(out[0]))
cv2.imwrite(out_path,np.vstack([np.hstack(out[i:i+2]) for i in range(0,len(out),2)]))

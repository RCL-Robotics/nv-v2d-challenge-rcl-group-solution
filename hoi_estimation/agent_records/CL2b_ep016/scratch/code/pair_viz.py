import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2
from hoi_lib import *
cam,K,Twc,rec=load_inputs('/work/inputs')
Rd=Renderer(load_mesh('/work/inputs')); fd=FrameData('/work/inputs')
src=sys.argv[1]
Tw=rec['T_world_mesh'].astype(float) if src=='base' else np.load(src)['T_world_mesh']
Tc=np.linalg.inv(Twc)@Tw; Tb=np.linalg.inv(Twc)@rec['T_world_mesh'].astype(float)
frames=[int(x) for x in sys.argv[2].split(',')] if ',' in sys.argv[2] else list(range(*[int(x) for x in sys.argv[2].split(':')]))
tiles=[]
for i in frames:
    box=frame_box(K,Tb[i],scale=1.3); hw=(160,160)
    img,om,pm=fd.crop(i,box,hw)
    rgb,m,_=Rd.render(torch.tensor(Tc[i][None],device='cuda'),K,box,hw)
    r=img.copy(); mk=m[0].cpu().numpy(); r[mk]=rgb[0].cpu().numpy()[mk]
    t=(np.hstack([img,r])[...,::-1]*255).astype(np.uint8).copy(); cv2.putText(t,str(i),(3,15),0,0.5,(0,255,0),1); tiles.append(t)
nc=int(sys.argv[4]) if len(sys.argv)>4 else 5
while len(tiles)%nc: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[3],np.vstack([np.hstack(tiles[k:k+nc]) for k in range(0,len(tiles),nc)]))

import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, torch
from hoilib import *
ep=Episode('/work/inputs'); fr=ep.frames()
rd=Renderer(ep.mesh)
want=[int(a) for a in sys.argv[2].split(',')]
src=sys.argv[1]
T = ep.Tc_base if src=='base' else ep.T_cam_world[None]@np.load(src)['T_world_mesh']
tiles=[]
for i in want:
    c=proj(ep.K,T[i:i+1,:3,3])[0]
    crop=crop_box(c,110); S=256
    m,rgb=rd.render(T[i:i+1],ep.K,crop,S,color=True)
    m=m[0].cpu().numpy(); rgb=rgb[0].cpu().numpy()
    im=cut(fr[i],crop,S).astype(np.float32)/255
    om=cut(ep.obj_mask[i],crop,S,cv2.INTER_NEAREST); pm=cut(ep.person_mask[i],crop,S,cv2.INTER_NEAREST)
    a=im.copy(); a=np.where(m[...,None]>0.5, 0.5*a+0.5*rgb, a)
    cs,_=cv2.findContours(om.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    a=(a*255).astype(np.uint8)[:,:,::-1].copy()
    cv2.drawContours(a,cs,-1,(0,255,0),1)
    cs,_=cv2.findContours(pm.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(a,cs,-1,(255,0,0),1)
    b=(im*255).astype(np.uint8)[:,:,::-1].copy()
    cv2.putText(b,str(i),(5,20),0,0.6,(0,255,0),2)
    tiles.append(np.hstack([b,a]))
while len(tiles)%3: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[3],np.vstack([np.hstack(tiles[k:k+3]) for k in range(0,len(tiles),3)]))

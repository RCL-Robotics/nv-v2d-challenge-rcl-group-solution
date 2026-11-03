import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, h5py
from hoi_lib import *
inp='/work/inputs'
cam,K,Twc,rec,plane=load_inputs(inp); mesh=load_mesh(inp)
R=Renderer(mesh,K,1536,1152)
mf=h5py.File(inp+'/masks.h5')
ids=[int(x) for x in sys.argv[1].split(',')]
P=np.load(sys.argv[2])['T_world_mesh'] if len(sys.argv)>2 else rec['T_world_mesh']
tiles=[]
for i in ids:
    m=R.render(np.linalg.inv(Twc)@P[i].astype(np.float64),need_tex=False)['mask'][0].cpu().numpy()
    om=mf['object'][i]; pm=mf['person'][i]
    img=np.zeros((1152,1536,3),np.uint8); img[m]=(0,0,255); img[om]=(0,255,0); img[m&om]=(0,255,255); img[pm&~om]=(80,80,80)
    ys,xs=np.nonzero(om|m); cy,cx=int(ys.mean()),int(xs.mean()); s=120
    t=img[max(0,cy-s):cy+s,max(0,cx-s):cx+s]; t=cv2.resize(t,(480,480),interpolation=cv2.INTER_NEAREST)
    cv2.putText(t,str(i),(5,25),0,0.8,(255,255,255),2); tiles.append(t)
cv2.imwrite('/work/scratch/frames/cmp.jpg',np.concatenate(tiles,1))

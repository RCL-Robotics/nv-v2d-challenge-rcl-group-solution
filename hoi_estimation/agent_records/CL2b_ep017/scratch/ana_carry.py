import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoilib import *
d=np.load('carry_fit.npz'); frames=d['frames']; Tf=d['Tf'].reshape(len(frames),-1,4,4); iou=d['iou'].reshape(len(frames),-1)
import json
ep=Episode('/work/inputs')
R0=ep.T_base[0,:3,:3]
for k,i in enumerate(frames):
    if i%2: continue
    s=[]
    for h in range(Tf.shape[1]):
        Tw=ep.T_world_cam@Tf[k,h]
        tw,sw=twist_swing(R0.T@Tw[:3,:3])
        s.append('%4.0f(%3.0f)/%2.0f %.3f'%(tw,(tw+45)%90-45,sw,iou[k,h]))
    twb,swb=twist_swing(R0.T@ep.T_base[i,:3,:3])
    print(i,'area %5d'%d['area'][k],'base %4.0f/%2.0f |'%(twb,swb),' | '.join(s))

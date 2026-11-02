import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoilib import *
ep=Episode('/work/inputs')
d=np.load('carry_profile.npz'); fr=d['frames']; g=d['grid']; iou=d['iou'].reshape(len(fr),len(g)); Tf=d['Tf'].reshape(len(fr),len(g),4,4)
R0=ep.T_base[0,:3,:3]
for k,i in enumerate(fr):
    tws=[twist_swing(R0.T@(ep.T_world_cam@Tf[k,j])[:3,:3])[0] for j in range(len(g))]
    tws=np.array(tws)%90
    j=iou[k].argmax(); 
    prof=''.join(' .:-=+*#%@'[min(9,int(max(0,(iou[k,jj]-iou[k].max()+0.02))/0.02*9))] for jj in range(len(g)))
    print(i,'best %5.1f'%tws[j],'iou %.3f range %.3f'%(iou[k].max(),iou[k].max()-iou[k].min()),prof)

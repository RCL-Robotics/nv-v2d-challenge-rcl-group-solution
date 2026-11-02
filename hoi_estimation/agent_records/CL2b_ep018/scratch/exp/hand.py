import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoi_lib import *
from scipy.spatial import cKDTree
D=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs/object_mesh.glb')
P,_=trimesh.sample.sample_surface(mesh,20000,seed=0)
kp=D['rec']['human_keypoints_world']; cp=D['rec']['contact_prob_right_left']
Tb=D['rec']['T_world_mesh'].astype(float); Tf=D['T_wc'][None]@np.load(sys.argv[1])['T']
tree=cKDTree(P)
def hd(T,i,idx):
    q=(kp[i,idx]-T[i,:3,3])@T[i,:3,:3]  # to mesh frame
    return tree.query(q)[0].min()
R_=list(range(21,42)); L_=list(range(42,63))
for s,e in [(260,456),(456,659),(659,720),(720,740)]:
    rb=[];rf=[]
    for i in range(s,e):
        for idx,c in [(R_,cp[i,0]),(L_,cp[i,1])]:
            if c>0.5: rb.append(hd(Tb,i,idx)); rf.append(hd(Tf,i,idx))
    print(s,e,len(rb),'median hand-mesh dist base %.3f fit %.3f'%(np.median(rb),np.median(rf)), 'mean %.3f %.3f'%(np.mean(rb),np.mean(rf)))

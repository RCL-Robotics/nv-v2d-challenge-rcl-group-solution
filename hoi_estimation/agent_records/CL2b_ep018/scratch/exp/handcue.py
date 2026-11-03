import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoi_lib import *
from scipy.spatial import cKDTree
D=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs/object_mesh.glb')
P,_=trimesh.sample.sample_surface(mesh,30000,seed=0); tree=cKDTree(P)
center=(mesh.bounds[0]+mesh.bounds[1])/2
kp=D['rec']['human_keypoints_world']; cp=D['rec']['contact_prob_right_left']
Tw=np.load(sys.argv[1])['T_world_mesh']; Tcw=np.linalg.inv(D['T_wc'])
hands=[list(range(21,42)),list(range(42,63))]
shifts=np.linspace(-0.12,0.12,49)
for nm,s,e in [('S1',0,252),('M1a',252,350),('M1b',350,456),('S2',456,659),('M2a',659,680),('M2b',680,700),('M2c',700,720),('S3',720,858)]:
    cost=np.zeros(len(shifts)); n=0
    for i in range(s,e):
        Tc=Tcw@Tw[i]; pc=Tc[:3,:3]@center+Tc[:3,3]; ray=pc/pc[2]
        for h in range(2):
            if cp[i,h]<0.5: continue
            q=(Tcw[:3,:3]@kp[i,hands[h]].T).T+Tcw[:3,3]  # cam frame keypoints
            for j,ds in enumerate(shifts):
                t=Tc[:3,3]+ray*ds
                qm=(q-t)@Tc[:3,:3]
                dd=tree.query(qm)[0]
                cost[j]+=np.sort(dd)[:5].mean()   # 5 closest hand keypoints touch the surface
            n+=1
    if n: print(nm,'n',n,'best shift %.3f'%shifts[np.argmin(cost)], 'cost curve',np.round(cost[::4]/n*100,1))

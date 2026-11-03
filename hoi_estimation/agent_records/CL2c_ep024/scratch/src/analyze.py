import sys; sys.path.insert(0,'/work/scratch/src')
from fitlib import *
f=sys.argv[1]
cam,K,Twc,rec=load_inputs()
Tc=np.linalg.inv(Twc)@rec['T_world_mesh'].astype(float)
d=np.load(f); T=d['T']
rd=np.degrees(np.linalg.norm(Rot.from_matrix(np.transpose(Tc[:,:3,:3],(0,2,1))@T[:,:3,:3]).as_rotvec(),axis=1))
zr=T[:,2,3]/Tc[:,2,3]
for a in range(0,748,25): print(a, 'rotdiff mean %.1f max %.1f'%(rd[a:a+25].mean(),rd[a:a+25].max()), 'z ratio %.3f'%zr[a:a+25].mean(), 'iou %.3f'%d['iou'][a:a+25].mean())

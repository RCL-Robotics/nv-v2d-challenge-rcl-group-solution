import sys; sys.path.insert(0,'/work/outputs')
from hoi_lib import *
from scipy.spatial.transform import Rotation as Rr
K,Twc,rec=load_inputs('/work/inputs')
B=mesh_to_cam(Twc,rec['T_world_mesh'].astype(np.float64)); C=rec['T_cam_mesh_coconet'].astype(np.float64); P=rec['T_cam_mesh_foundationpose'].astype(np.float64)
def rd(A,Bm): return np.degrees(Rr.from_matrix(np.einsum('nji,njk->nik',A[:,:3,:3],Bm[:,:3,:3])).magnitude())
r1=rd(B,C); r2=rd(B,P); r3=rd(C,P)
for a in range(0,901,30): print(a,'b-c %.1f b-fp %.1f c-fp %.1f'%(r1[a:a+30].mean(),r2[a:a+30].mean(),r3[a:a+30].mean()), 'zB %.3f zC %.3f zP %.3f'%(B[a,2,3],C[a,2,3],P[a,2,3]))

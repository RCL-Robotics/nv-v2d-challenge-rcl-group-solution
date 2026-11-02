import os, numpy as np,sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hoi_extraction", "scripts"))
from eval_object_pose import quat_wxyz_to_R
for i in range(30):
  r=np.load(f'ref_poses/episode_{i:06d}.npz')['pose_xyz_wxyz'].astype(float); t=np.load(f'tier2_poses/episode_{i:06d}.npz')['pose_xyz_wxyz'].astype(float)
  A=quat_wxyz_to_R(r[:1,3:])[0];B=quat_wxyz_to_R(t[:1,3:])[0]
  d=np.linalg.norm(t[:,:3]-t[0,:3],axis=1); s=np.argmax(d>0.03) ; dr=np.linalg.norm(r[:,:3]-r[0,:3],axis=1); sr=np.argmax(dr>0.01)
  print(i,'f0 rot err %.1f'%np.degrees(np.arccos(np.clip((np.trace(A.T@B)-1)/2,-1,1))),'pos %.1fcm'%(100*np.linalg.norm(r[0,:3]-t[0,:3])),'static t2(3cm)',s,'gt(1cm)',sr,len(t))

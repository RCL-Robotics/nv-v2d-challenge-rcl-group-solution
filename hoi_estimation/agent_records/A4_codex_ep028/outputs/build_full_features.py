import numpy as np,json
p='/work/inputs/';a=np.load(p+'reconstruction.npz');c=json.load(open(p+'camera.json'));K=np.array(c['K']);base=np.linalg.inv(c['T_world_cam'])@a['T_world_mesh'];obs=json.load(open('/work/outputs/observations.json'));a=np.load('/work/outputs/feature_tracks.npz');head=a['head'].copy();tip=a['tip'].copy();quality=a['quality'].copy();line=a['blade_line'].copy();p3=np.array([[-.00386614,-.0140458,-.44275588],[.00168146,.01075304,.99291701]])
for i in range(660):
 xy=(p3@base[i,:3,:3].T+base[i,:3,3])@K.T;xy=xy[:,:2]/xy[:,2:];cs=np.array([x for x in obs[i]['orange'] if x[2]>=6]);
 if not len(cs):continue
 for j,target in enumerate([head,tip]):
  dis=np.linalg.norm(cs[:,:2]-xy[j],axis=1);dis[cs[:,2]<(60 if j==0 else 6)]=1e5;k=np.argmin(dis)
  if dis[k]<60:target[i]=cs[k,:2];quality[i,j]=cs[k,2]
np.savez('/work/outputs/feature_tracks_full.npz',head=head,tip=tip,quality=quality,blade_line=line)
Q=base.copy();Q[660:]=np.load('/work/outputs/refined_candidate.npz')['T_cam_mesh'][660:];np.savez('/work/outputs/full_init.npz',T_cam_mesh=Q)
print('landmarks',np.isfinite(head[:,0]).sum(),np.isfinite(tip[:,0]).sum())

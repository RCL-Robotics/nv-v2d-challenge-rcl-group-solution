import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, json, trimesh
from handframe import hand_frames
Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
pl=json.load(open('/work/inputs/ground_plane.json'))['plane']; n=np.array(pl[:3]); d=pl[3]
m=trimesh.load('/work/inputs/object_mesh.glb', force='scene', process=False).to_geometry(); v=np.asarray(m.vertices)[::5]
Tw=np.load(sys.argv[1])['T_world_mesh']; Tc=np.linalg.inv(Twc)@Tw
rec=np.load('/work/inputs/reconstruction.npz'); kp=rec['human_keypoints_world'].astype(np.float64)
def floor_min(Tcam):
    w=(Twc[:3,:3]@((Tcam[:3,:3]@v.T).T+Tcam[:3,3]).T).T+Twc[:3,3]; return (w@n+d).min()
for name,(a,b) in [('start',(0,169)),('end',(592,715))]:
    i=(a+b)//2; T=Tc[i]
    for target in [0.0,0.018]:
        # find depth scale s (move along ray) s.t. floor_min = target
        ss=np.linspace(0.9,1.1,2001); vals=[]
        for s in ss:
            T2=T.copy(); T2[:3,3]*=s; vals.append(floor_min(T2))
        vals=np.array(vals); k=np.argmin(np.abs(vals-target))
        print(name,'floor target %.3f: depth shift %.1f cm (cur floor dist %.2f cm)'%(target,(ss[k]-1)*T[2,3]*100, 100*floor_min(T)))
# hand cue: handle (top of mesh) vs right-hand grip point (mean of finger bases)
handle=np.array([0,0,0.32])
shifts=[]
for i in range(200,590):
    g=kp[i,[28,32,36,40]].mean(0); gc=np.linalg.inv(Twc)[:3,:3]@g+np.linalg.inv(Twc)[:3,3]
    hc=Tc[i,:3,:3]@handle+Tc[i,:3,3]
    # move object along its ray by dz*ray: choose s minimizing |s*hc_offset ...| approx: shift of handle along the ray of object center
    ray=Tc[i,:3,3]/Tc[i,2,3]
    s=np.dot(gc-hc,ray)/np.dot(ray,ray); shifts.append(s)
shifts=np.array(shifts); print('hand cue: depth shift along ray median %.1f cm, IQR [%.1f, %.1f]'%(100*np.median(shifts),100*np.percentile(shifts,25),100*np.percentile(shifts,75)))
ref=np.load('/work/inputs/frame0_reference_human.npz')['vertices_world']; hv=rec['human_vertices_world'][0].astype(np.float64)
Tcw=np.linalg.inv(Twc); zr=(Tcw[:3,:3]@ref.T).T[:,2]+Tcw[2,3]; zh=(Tcw[:3,:3]@hv.T).T[:,2]+Tcw[2,3]
print('frame-0 human depth: reconstructed median z %.3f vs reference %.3f (diff %.1f cm)'%(np.median(zh),np.median(zr),100*(np.median(zh)-np.median(zr))))

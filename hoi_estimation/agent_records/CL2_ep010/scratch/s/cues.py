"""Depth cues: (1) hand contact, (2) floor contact, evaluated as function of a depth shift along viewing rays."""
import sys, json, numpy as np, trimesh
from scipy.spatial import cKDTree
cam=json.load(open('/work/inputs/camera.json')); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
plane=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'])
mesh=trimesh.load('/work/inputs/object_mesh.glb', force='scene', process=False).to_geometry(); c=mesh.bounds.mean(0)
P,_=trimesh.sample.sample_surface_even(mesh,20000,seed=0) if hasattr(trimesh.sample,'sample_surface_even') else (mesh.vertices[::10],None)
tree=cKDTree(P)
rec=np.load('/work/inputs/reconstruction.npz'); kp=rec['human_keypoints_world']; cp=rec['contact_prob_right_left']
Tw=np.load(sys.argv[1])['T_world_mesh']; F=len(Tw)
Tc=Tcw[None]@Tw; pc=Tc[:,:3,:3]@c+Tc[:,:3,3]
def shifted(Tc_i, pc_i, dz):
    s=(pc_i[2]+dz)/pc_i[2]; T=Tc_i.copy(); T[:3,3]=pc_i*s-T[:3,:3]@c; return Twc@T
shifts=np.arange(-0.12,0.121,0.01)
phases=[('S0',0,166),('S1',198,260),('M',261,596),('S2',597,793)]
hands={'R':list(range(21,42)),'L':list(range(42,63))}
for ph,a,b in phases:
    fr=np.arange(a,b+1,3)
    # hand cue
    res=[]
    for dz in shifts:
        dd=[]
        for i in fr:
            Tws=shifted(Tc[i],pc[i],dz); Tinv=np.linalg.inv(Tws)
            for h,(idx) in enumerate([hands['R'],hands['L']]):
                if cp[i,h]<0.9: continue
                k=kp[i,idx]; km=k@Tinv[:3,:3].T+Tinv[:3,3]
                dd.append(np.min(tree.query(km)[0]))   # closest hand keypoint to desk surface
        res.append(np.median(dd) if dd else np.nan)
    res=np.array(res)
    # floor cue: lowest point height vs floor; target height = 0.014 (true frame-0 human sole offset)
    fl=[]
    for dz in shifts:
        hs=[]
        for i in fr:
            Tws=shifted(Tc[i],pc[i],dz); h=((mesh.vertices[::20]@Tws[:3,:3].T+Tws[:3,3])@plane[:3]+plane[3]).min(); hs.append(h)
        fl.append(np.median(hs))
    fl=np.array(fl)
    if np.all(np.isnan(res)): hb='n/a'
    else: hb=f'{shifts[np.nanargmin(res)]*100:+.0f}cm (min med dist {np.nanmin(res)*100:.1f}cm, at 0: {res[len(shifts)//2]*100:.1f}cm)'
    fz=np.interp(0.014,fl[::-1],shifts[::-1]) if fl[0]>fl[-1] else np.nan
    print(f'{ph}: hand-contact best shift {hb}; floor: lowest point at shift0 {fl[len(shifts)//2]*100:.1f}cm above plane, shift giving +1.4cm: {fz*100:+.1f}cm')

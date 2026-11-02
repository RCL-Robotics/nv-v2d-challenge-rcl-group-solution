import sys,json; sys.path.insert(0,'/work/outputs')
from lib import *
m,v,f,uv,tex=load_mesh('/work/inputs/object_mesh.glb'); cam,K,Twc,rec=load_inputs('/work/inputs'); c0,u=body_axis(v)
Tcw=np.linalg.inv(Twc); V=Tcw[None]@np.load('v2/object_poses_world.npz')['T_world_mesh']
log=json.load(open('v2/object_poses_world_log.json')); Kp=np.array(cam['K_pipeline'])
gp=json.load(open('/work/inputs/ground_plane.json'))['plane']; n_up=Tcw[:3,:3]@np.array(gp[:3]); n_up/=np.linalg.norm(n_up)
if n_up[1]>0: n_up=-n_up
f5=h5py.File('/work/inputs/masks.h5'); A=h5py.File('/pipeline/depth/moge2/aligned.h5')['aligned/front_stereo_camera_left']
rng=np.random.default_rng(0); normals=[]
for (a,b) in log['static_phases'][1:]:
    for i in range(a,b,6):
        d=cv2.imdecode(A[i],cv2.IMREAD_UNCHANGED).astype(np.float32)/1000; om=f5['object'][i]; pm=f5['person'][i]
        ys,xs=np.nonzero(om); yb=ys.max(); xc=int(xs.mean())
        Y,X=np.mgrid[yb+3:yb+60,xc-100:xc+100]; ok=~pm[Y,X]&~om[Y,X]
        z=d[Y,X][ok]; P=np.stack([(X[ok]-Kp[0,2])/Kp[0,0]*z,(Y[ok]-Kp[1,2])/Kp[1,1]*z,z],1)
        best=None
        for _ in range(200):
            s=P[rng.choice(len(P),3,replace=False)]; n=np.cross(s[1]-s[0],s[2]-s[0]); n/=np.linalg.norm(n)+1e-12
            inl=np.abs((P-s[0])@n)<0.01
            if best is None or inl.sum()>best.sum(): best=inl
        Q=P[best]; n=np.linalg.svd(Q-Q.mean(0))[2][2]; n=n if n@n_up>0 else -n; normals.append(n)
normals=np.array(normals); nm=np.median(normals,0); nm/=np.linalg.norm(nm)
print('table normal vs floor normal: median angle %.1f deg, per-frame angles'%np.degrees(np.arccos(nm@n_up)), np.round(np.degrees(np.arccos(np.clip(normals@n_up,-1,1)))).astype(int))
for s in log['static']:
    a,b=s['phase']; ax=V[a][:3,:3]@u
    # lateral (image x) and depth components of tilt relative to n_up
    d=ax-n_up*(ax@n_up)
    print(s['phase'],'tilt %.1f'%s['tilt_deg'],'components x %.1f z %.1f deg'%(np.degrees(np.arcsin(d[0])),np.degrees(np.arcsin(d[2]))))

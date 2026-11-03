import sys,json; sys.path.insert(0,'/work/outputs')
from lib import *
m,v,f,uv,tex=load_mesh('/work/inputs/object_mesh.glb'); cam,K,Twc,rec=load_inputs('/work/inputs'); c0,u=body_axis(v)
Tcw=np.linalg.inv(Twc); V1=Tcw[None]@np.load('v1/object_poses_world.npz')['T_world_mesh']
log=json.load(open('v1/object_poses_world_log.json')); Kp=np.array(cam['K_pipeline'])
gp=json.load(open('/work/inputs/ground_plane.json'))['plane']; n_up=Tcw[:3,:3]@np.array(gp[:3]); n_up/=np.linalg.norm(n_up)
hb=((v-c0)@u).min(); bot=c0+u*hb; hv=v[v[:,0]>0.035].mean(0)
f5=h5py.File('/work/inputs/masks.h5'); A=h5py.File('/pipeline/depth/moge2/aligned.h5')['aligned/front_stereo_camera_left']
kp=np.einsum('ij,fkj->fki',Tcw[:3,:3],rec['human_keypoints_world'].astype(np.float64))+Tcw[:3,3]
cp=rec['contact_prob_right_left'].max(1); rng=np.random.default_rng(0)
for s in log['static']:
    a,b=s['phase']; T=V1[a]; zc=(T[:3,:3]@c0+T[:3,3])[2]; pb=T[:3,:3]@bot+T[:3,3]
    zs=[]
    for i in range(a,b,2):
        z=table_cue_z(A[i],f5['object'][i],f5['person'][i],Kp,K,pb,n_up,rng)
        if z is not None: zs.append(z+(zc-pb[2]))
    fr=np.array([i for i in range(max(0,a-10),min(781,b+10)) if cp[i]>0.5])
    hs=hand_cue_shift(np.repeat(T[None],len(fr),0),fr,kp,c0,hv) if len(fr) else None
    print(a,b,'z %.3f'%zc,'table n=%d med %.3f (%+.1f) iqr %.1fcm'%(len(zs),np.median(zs),100*(np.median(zs)-zc),100*np.subtract(*np.percentile(zs,[75,25]))) if zs else 'table none','| hand %+.1f resid %.1f n=%d'%(100*hs[0],100*hs[1],len(fr)) if hs else '')

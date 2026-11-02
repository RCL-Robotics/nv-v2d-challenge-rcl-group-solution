import sys; sys.path.insert(0,'/work/outputs')
from lib import *
from scipy.spatial.transform import Rotation as Rr
m,v,f,uv,tex=load_mesh('/work/inputs/object_mesh.glb'); cam,K,Twc,rec=load_inputs('/work/inputs'); c0,u=body_axis(v)
Tcw=np.linalg.inv(Twc)
kp=rec['human_keypoints_world'].astype(np.float64)
phases=[(0,130),(214,269),(349,399),(480,527),(605,641),(689,781)]
logo=[135,125,120,120,115,115]
Rst=[]
for k in range(6):
    Tf=np.load(f'medfit_{k}.npy'); hx=np.einsum('hij,j->hi',Tf[:,:3,:3],np.array([1.,0,0])); az=np.degrees(np.arctan2(hx[:,2],hx[:,0]))
    j=int(np.argmin(np.abs(((az-100)+180)%360-180))); T=spin(Tf[j],c0,u,np.radians([logo[k]-az[j]]))[0]
    Rst.append(Twc[:3,:3]@T[:3,:3])   # world rotation
def kabsch(A,B):
    a=A-A.mean(0); b=B-B.mean(0); U,S,Vt=np.linalg.svd(a.T@b); d=np.sign(np.linalg.det(Vt.T@U.T))
    return Vt.T@np.diag([1,1,d])@U.T
for k in range(5):
    e=phases[k][1]-1; s=phases[k+1][0]
    for name,sl in [('R',slice(21,42)),('L',slice(42,63))]:
        Rh=kabsch(kp[e,sl],kp[s,sl])
        pred=Rh@Rst[k]
        err=np.degrees(Rr.from_matrix(pred.T@Rst[k+1]).magnitude())
        # also identity (no rotation) baseline
        err0=np.degrees(Rr.from_matrix(Rst[k].T@Rst[k+1]).magnitude())
        # also mid-carry consistency: hand rotation magnitude
        mags=[np.degrees(Rr.from_matrix(kabsch(kp[e,sl],kp[t,sl])).magnitude()) for t in range(e,s,6)]
        print(k,name,'pred err %.1f  static diff %.1f'%(err,err0),'hand rot mags',np.round(mags).astype(int))

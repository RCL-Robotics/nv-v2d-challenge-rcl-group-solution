import sys; sys.path.insert(0,'/work/outputs')
from lib import *
from scipy.spatial.transform import Rotation as Rr, Slerp
m,v,f,uv,tex=load_mesh('/work/inputs/object_mesh.glb'); cam,K,Twc,rec=load_inputs('/work/inputs'); c0,u=body_axis(v)
Tcw=np.linalg.inv(Twc); Tb=rec['T_world_mesh'].astype(np.float64)
kp=rec['human_keypoints_world'].astype(np.float64)
phases=[(0,130),(214,269),(349,399),(480,527),(605,641),(689,781)]
logo=[135,125,120,120,115,115]
Tst=[]
for k in range(6):
    Tf=np.load(f'medfit_{k}.npy'); hx=np.einsum('hij,j->hi',Tf[:,:3,:3],np.array([1.,0,0])); az=np.degrees(np.arctan2(hx[:,2],hx[:,0]))
    j=int(np.argmin(np.abs(((az-100)+180)%360-180))); T=spin(Tf[j],c0,u,np.radians([logo[k]-az[j]]))[0]
    a,b=phases[k]; T[:3,3]*=np.median((Tcw[None]@Tb[a:b])[:,2,3])/T[2,3]   # depth re-anchor
    Tst.append(Twc@T)
def rigid(A,B):
    ca,cb=A.mean(0),B.mean(0); a=A-ca; b=B-cb; U,S,Vt=np.linalg.svd(a.T@b); d=np.sign(np.linalg.det(Vt.T@U.T))
    R=Vt.T@np.diag([1,1,d])@U.T; T=np.eye(4); T[:3,:3]=R; T[:3,3]=cb-R@ca; return T
out=Tb.copy()
for k in range(6):
    a,b=phases[k]; out[a:b]=Tst[k]
for k in range(5):
    e=phases[k][1]-1; s=phases[k+1][0]
    ctr=lambda T: T[:3,:3]@c0+T[:3,3]
    sl=min([slice(21,42),slice(42,63)],key=lambda q: np.linalg.norm(kp[e,q].mean(0)-ctr(Tst[k])))
    for t in range(e+1,s):
        Tf_=rigid(kp[e,sl],kp[t,sl])@Tst[k]; Tbk=rigid(kp[s,sl],kp[t,sl])@Tst[k+1]
        w=(t-e)/(s-e)
        R=Slerp([0,1],Rr.from_matrix([Tf_[:3,:3],Tbk[:3,:3]]))(w).as_matrix()
        p=(1-w)*ctr(Tf_)+w*ctr(Tbk)
        T=np.eye(4); T[:3,:3]=R; T[:3,3]=p-R@c0; out[t]=T
np.savez('pred_handrigid.npz',T_world_mesh=out)
# compare with baseline in carries
for k in range(5):
    e=phases[k][1]-1; s=phases[k+1][0]
    ts=range(e+1,s,5)
    ax=[np.degrees(np.arccos(np.clip((out[t][:3,:3]@u)@(Tb[t][:3,:3]@u),-1,1))) for t in ts]
    dp=[np.linalg.norm((out[t][:3,:3]@c0+out[t][:3,3])-(Tb[t][:3,:3]@c0+Tb[t][:3,3]))*100 for t in ts]
    print(k,'axis diff',np.round(ax).astype(int)); print('  pos diff cm',np.round(dp).astype(int))

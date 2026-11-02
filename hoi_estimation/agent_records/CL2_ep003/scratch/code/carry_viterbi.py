import sys,time; sys.path.insert(0,'/work/outputs')
from lib import *
m,v,f,uv,tex=load_mesh('/work/inputs/object_mesh.glb'); cam,K,Twc,rec=load_inputs('/work/inputs'); c0,u=body_axis(v)
Tcw=np.linalg.inv(Twc); D=dict(np.load('crops160.npz')); R=Renderer(v,f,uv,tex)
P=Tcw[None]@np.load('pred_handrigid.npz')['T_world_mesh']; B=Tcw[None]@rec['T_world_mesh'].astype(np.float64)
FP=rec['T_cam_mesh_foundationpose'].astype(np.float64); CO=rec['T_cam_mesh_coconet'].astype(np.float64)
phases=[(0,130),(214,269),(349,399),(480,527),(605,641),(689,781)]
F=len(B); out=P.copy()
for k in range(5):
    a=phases[k][1]-1; b=phases[k+1][0]+1; fr=np.arange(a,b); N=len(fr)
    cands=[]; yd=[]
    for t in fr:
        cs=[P[t]]
        for src in (B,FP,CO):
            T=src[t].copy(); T[:3,3]=B[t][:3,3]   # rotation from source, position from baseline
            Ty,d=yaw_match(T,P[t][:3,:3],c0,u); Ty[:3,3]=B[t][:3,3]-Ty[:3,:3]@c0+T[:3,:3]@c0; cs.append(Ty)
        cands.append(np.stack(cs))
    cands=np.stack(cands)  # N,C,4,4
    C=cands.shape[1]
    iou=iou_scores(R,K,D,cands.reshape(-1,4,4),np.repeat(fr,C)).reshape(N,C)
    prior=geodesic_deg(cands[:,:,:3,:3],P[fr][:,None,:3,:3])
    data=-iou*20+prior/15.0
    def trans(t):
        return geodesic_deg(cands[t-1][:,None,:3,:3],cands[t][None,:,:3,:3])/5.0
    path=viterbi(data,trans)
    sel=cands[np.arange(N),path]
    out[fr]=sel
    print(k,fr[0],fr[-1],'mean iou: hand %.3f base %.3f fp %.3f co %.3f  chosen %.3f'%(*iou.mean(0),iou[np.arange(N),path].mean()),'choice counts',np.bincount(path,minlength=C))
np.savez('pred_carryV.npz',T_world_mesh=Twc[None]@out)

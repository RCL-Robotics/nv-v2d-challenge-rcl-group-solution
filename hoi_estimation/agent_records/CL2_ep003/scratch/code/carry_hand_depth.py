import sys,json; sys.path.insert(0,'/work/outputs')
from lib import *
m,v,f,uv,tex=load_mesh('/work/inputs/object_mesh.glb'); cam,K,Twc,rec=load_inputs('/work/inputs'); c0,u=body_axis(v)
Tcw=np.linalg.inv(Twc); B=Tcw[None]@rec['T_world_mesh'].astype(np.float64)
V1=Tcw[None]@np.load(sys.argv[1])['T_world_mesh']
kp=np.einsum('ij,fkj->fki',Tcw[:3,:3],rec['human_keypoints_world'].astype(np.float64))+Tcw[:3,3]
cp=rec['contact_prob_right_left'].max(1); hv=v[v[:,0]>0.035].mean(0)
def hand_shift(T,fr):
    best=None
    for sft in np.arange(-0.3,0.3,0.005):
        dd=[]
        for j,i in enumerate(fr):
            Ti=T[j]; cen=Ti[:3,:3]@c0+Ti[:3,3]; s=(cen[2]+sft)/cen[2]
            hc=(Ti[:3,:3]@hv+Ti[:3,3])+cen*(s-1)
            hands=[kp[i,21:42].mean(0),kp[i,42:63].mean(0)]; h=min(hands,key=lambda q:np.linalg.norm(q-cen))
            dd.append(np.linalg.norm(hc-h))
        if best is None or np.mean(dd)<best[0]: best=(np.mean(dd),sft)
    return best
for a,b in [(130,214),(269,349),(399,480),(527,605),(641,689)]:
    fr=np.arange(a,b); fr=fr[cp[fr]>0.5]
    zc=np.median([(T[:3,:3]@c0+T[:3,3])[2] for T in V1[fr]])
    r=hand_shift(V1[fr],fr)
    # segment halves
    h1=hand_shift(V1[fr[:len(fr)//2]],fr[:len(fr)//2]); h2=hand_shift(V1[fr[len(fr)//2:]],fr[len(fr)//2:])
    print('carry',a,b,'n',len(fr),'z_med %.3f'%zc,'hand shift %+.1fcm resid %.1f | halves %+.1f %+.1f -> z %.3f'%(100*r[1],100*r[0],100*h1[1],100*h2[1],zc+r[1]))

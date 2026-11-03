from common import *; from param import *
import trimesh
r,cam,pl,m=load(); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
fp=FloorParam(pl[:3]); Tb=r['T_world_mesh'].astype(float); Ry=np.diag([-1,1,-1,1.])
o=np.load('perframe.npy'); kp=r['human_keypoints_world']; cp=r['contact_prob_right_left']
pq=trimesh.proximity.ProximityQuery(m)
ss=np.arange(0.92,1.061,0.01)
for a,b in [(286,352),(353,404),(405,506)]:
    res=[]
    for i in range(a,b,3):
        T=fp.apply(Tb[i]@Ry,*o[i,0,:3]); Tc=Tcw@T
        row=[]
        for s in ss:
            P=Tc.copy(); P[:3,3]*=s; Tw=Twc@P
            hk=np.concatenate([kp[i,21:42],kp[i,42:63]])
            loc=(hk-Tw[:3,3])@Tw[:3,:3]
            _,dist,_=pq.on_surface(loc)
            row.append([np.sort(dist[:21])[:3].mean(),np.sort(dist[21:])[:3].mean()])
        res.append(row)
    res=np.array(res)  # frames, scales, 2
    tot=np.median(res.min(-1),0)
    print(a,b,'median closest-hand dist vs scale:',' '.join('%.2f:%.3f'%(s,x) for s,x in zip(ss,tot)))
    print('   z ~ %.3f, argmin scale %.2f'%((Tcw@Tb[a])[2,3],ss[tot.argmin()]))

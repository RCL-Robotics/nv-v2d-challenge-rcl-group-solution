# validation metrics: python evalv.py name=file.npz ...
from rend import *
sys.path.insert(0,'/work/outputs'); from solve import occluder_region
import sys as _s
r,cam=load(); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
gp=np.array(json.load(open(INP+'/ground_plane.json'))['plane']); n=gp[:3]; d=gp[3]
Tb=r['T_world_mesh'].astype(float); F=len(Tb)
runs={'base':Tb}
for a in _s.argv[1:]:
    k,f=a.split('='); runs[k]=np.load(f)['T_world_mesh']
PH=[('A chair static',0,190),('carry',190,672),('B floor+hand',672,709),('slide',709,762),('C floor static',762,870),('ALL',0,870)]
M=Mesh(); MK=Masks(); S=192
ker=np.ones((5,5),np.uint8)
# masks crops cached around baseline/new positions: use full 192 crop centered on mask bbox center
bbc=np.load('mask_bb.npy')
cent=np.stack([(bbc[:,2]+bbc[:,3])/2,(bbc[:,4]+bbc[:,5])/2],1)
xy0=np.stack([np.clip(cent[:,0]-S/2,0,1536-S),np.clip(cent[:,1]-S/2,0,1152-S)],1).astype(int)
O=[];Vv=[]
kp_c=r['human_keypoints_world'].astype(float)@Tcw[:3,:3].T+Tcw[:3,3]
zb=(Tcw[None]@r['T_world_mesh'].astype(float))[:,2,3]
cap=cv2.VideoCapture(INP+'/video.mp4')
for i in range(F):
    ok,im=cap.read(); br=im[xy0[i,1]:xy0[i,1]+S,xy0[i,0]:xy0[i,0]+S].max(2)>=45
    o,p=MK.crop(i,*xy0[i],S); ign=occluder_region(p,br,i,kp_c,zb[i]); O.append(o); Vv.append(~ign|o)
O=np.array(O); Vv=np.array(Vv)
Vs=M.v.cpu().numpy()[::5]
kp=r['human_keypoints_world'].astype(float); hmid=(kp[:,21:42].mean(1)+kp[:,42:63].mean(1))/2
res={}
for name,T in runs.items():
    Tc=Tcw[None]@T
    iou=np.zeros(F)
    for b in range(0,F,96):
        ids=np.arange(b,min(F,b+96))
        a,_=render_sil(M,torch.tensor(Tc[ids,:3,:3],dtype=torch.float32,device=DEV),torch.tensor(Tc[ids,:3,3],dtype=torch.float32,device=DEV),K,torch.tensor(xy0[ids],dtype=torch.float32,device=DEV),S)
        a=a.cpu().numpy()>0.5; o=O[ids]; v=Vv[ids]
        iou[ids]=(a&o&v).sum((1,2))/np.maximum(((a|o)&v).sum((1,2)),1)
    p=T[:,:3,3]
    acc=np.r_[0,np.linalg.norm(p[2:]-2*p[1:-1]+p[:-2],axis=1)*100,0]
    angv=np.r_[0,rotang(T[:-1,:3,:3],T[1:,:3,:3])]
    W=np.einsum('fij,vj->fvi',T[:,:3,:3],Vs)+p[:,None]; hmin=(W@n+d).min(1)*100
    dz=(Tc[:,2,3]-(Tcw[None]@Tb)[:,2,3])*100
    hd=np.linalg.norm(p-hmid,axis=1)*100
    res[name]=dict(iou=iou,acc=acc,angv=angv,hmin=hmin,dz=dz,hd=hd)
print('%-16s %-6s %6s %7s %8s %8s %9s %8s %8s %8s'%('phase','run','IoU','IoU<.8','acc cm','angv deg','floor cm','dz mean','dz med','hand sd'))
for ph,a,b in PH:
    for name in runs:
        q=res[name]; sl=slice(a,b)
        print('%-16s %-6s %6.3f %7d %8.3f %8.2f %9.1f %8.2f %8.2f %8.2f'%(ph,name,q['iou'][sl].mean(),(q['iou'][sl]<0.8).sum(),np.median(q['acc'][sl]),np.median(q['angv'][sl]),np.percentile(q['hmin'][sl],5),q['dz'][sl].mean(),np.median(q['dz'][sl]),q['hd'][sl].std()))
np.savez('eval_last.npz',**{k+'_'+m:v for k,q in res.items() for m,v in q.items()})

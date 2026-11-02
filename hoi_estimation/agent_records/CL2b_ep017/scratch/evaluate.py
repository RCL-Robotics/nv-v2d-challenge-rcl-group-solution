import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch, cv2
from hoilib import *
ep=Episode('/work/inputs'); rd=Renderer(ep.mesh)
res=np.load(sys.argv[1])['T_world_mesh']; a_end=int(sys.argv[2]); b_start=int(sys.argv[3])
F=ep.F; Tc_new=ep.T_cam_world[None]@res; Tc_b=ep.Tc_base
def ious(Tc):
    out=np.full(F,np.nan)
    for t in range(F):
        if ep.obj_mask[t].sum()<150: continue
        ctr=proj(ep.K,(Tc_b[t]@np.array([0,0,0.09,1]))[None,:3])[0]; h=int(0.35*ep.K[0,0]/Tc_b[t,2,3]); crop=crop_box(ctr,h)
        m,_=rd.render(Tc[t:t+1],ep.K,crop,192); m=(m[0]>0.5).float()
        om=torch.tensor(cut(ep.obj_mask[t].astype(np.uint8),crop,192,cv2.INTER_NEAREST),device='cuda').float()
        pm=torch.tensor(cut(ep.person_mask[t].astype(np.uint8),crop,192,cv2.INTER_NEAREST),device='cuda').float()
        v=1-pm*(1-om)
        out[t]=((m*om*v).sum()/(((m+om-m*om)*v).sum()).clamp_min(1)).item()
    return out
ib=ious(Tc_b); inw=ious(Tc_new)
def rstep(T):
    return np.array([np.degrees(Rot.from_matrix(T[t,:3,:3].T@T[t+1,:3,:3]).magnitude()) for t in range(F-1)])
def tstep(T): return np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1)*100
rb,rn=rstep(ep.T_base),rstep(res); tb,tn=tstep(ep.T_base),tstep(res)
R0=ep.T_base[0,:3,:3]
twb=np.array([twist_swing(R0.T@ep.T_base[t,:3,:3])[0] for t in range(F)])
twn=np.array([twist_swing(R0.T@res[t,:3,:3])[0] for t in range(F)])
# ADD-like difference between baseline and ours (cm), to see where we changed things
pts=ep.mesh.sample(3000, seed=0) if 'seed' in ep.mesh.sample.__code__.co_varnames else ep.mesh.sample(3000)
ph=np.hstack([pts,np.ones((len(pts),1))])
addd=np.array([np.linalg.norm((res[t]@ph.T-ep.T_base[t]@ph.T)[:3],axis=0).mean()*100 for t in range(F)])
phases=[('floor',0,a_end+1),('carried',a_end+1,b_start),('table',b_start,F),('all',0,F)]
print('| phase | frames | IoU base | IoU ours | n vis | med rot step base/ours (deg) | p95 rot step base/ours | med trans step base/ours (cm) | dz mean / median (cm) | mean ADD(base,ours) cm |')
print('|---|---|---|---|---|---|---|---|---|---|')
for n,a,b in phases:
    s=slice(a,b); ss=slice(a,max(a,b-1))
    dz=(Tc_new[s,2,3]-Tc_b[s,2,3])*100
    vb=~np.isnan(ib[s])
    print(f'| {n} | {a}-{b-1} | {np.nanmean(ib[s]):.3f} | {np.nanmean(inw[s]):.3f} | {vb.sum()} | {np.median(rb[ss]):.2f} / {np.median(rn[ss]):.2f} | {np.percentile(rb[ss],95):.1f} / {np.percentile(rn[ss],95):.1f} | {np.median(tb[ss]):.2f} / {np.median(tn[ss]):.2f} | {dz.mean():.2f} / {np.median(dz):.2f} | {addd[s].mean():.1f} |')
np.savez(sys.argv[4] if len(sys.argv)>4 else 'eval.npz',ib=ib,inw=inw,twb=twb,twn=twn,addd=addd,rb=rb,rn=rn)
if len(sys.argv)>5:
    for t in range(0,F,6): print(t,'iou b %.3f n %.3f'%(ib[t],inw[t]),'tw b %.0f n %.0f'%(twb[t],twn[t]),'ADDdiff %.1f'%addd[t],'z b %.3f n %.3f'%(Tc_b[t,2,3],Tc_new[t,2,3]))

from traj import *
OMall=OM[:]; PMall=PM[:]
moving,feat=detect_phases(OMall)
seg,k,yaw0,p0,Rb,cw0,sids=init_from_baseline(moving)
crop=make_crop(OMall); x0,y0,W,H=crop; print('crop',crop)
Og=torch.tensor(OMall[:,y0:y0+H,x0:x0+W],device='cuda').float(); Vg=1-torch.tensor(PMall[:,y0:y0+H,x0:x0+W],device='cuda').float()
mode=sys.argv[1]
data=np.arange(F) if mode=='all' else np.arange(0,F,2)
tr=Traj(seg,k,yaw0,p0,Rb,cw0)
optimize(tr,Og,Vg,crop,data,iters=int(sys.argv[2]),fit_tilt=(sys.argv[3]=="tilt"))
print('max mem GB',torch.cuda.max_memory_allocated()/1e9)
Twn=world_poses(tr)
np.savez(f'/work/scratch/run1_{mode}_{sys.argv[3]}.npz',Tw=Twn,s=torch.exp(tr.ls).item(),tilt=tr.tilt.detach().cpu().numpy(),seg=seg)
Ib=eval_iou(None,Og,Vg,crop,Tcams=Tc)
Is=eval_iou(tr,Og,Vg,crop,scale=True); I1=eval_iou(tr,Og,Vg,crop,scale=False)
ph=[(0,103),(103,135),(135,220),(220,252),(252,340),(340,376),(376,467)]
ev=np.arange(F)%2==1
for a,b in ph:
    sl=np.zeros(F,bool); sl[a:b]=True
    sel=sl&ev if mode!='all' else sl
    print(a,b,'iou base %.3f ours(s) %.3f ours(s=1) %.3f'%(Ib[sel].mean(),Is[sel].mean(),I1[sel].mean()))

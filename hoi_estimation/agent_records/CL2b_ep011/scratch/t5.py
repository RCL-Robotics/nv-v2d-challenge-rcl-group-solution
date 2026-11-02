exec(open('setup.py').read())
import itertools
a,b=469,644
idx=np.arange(a,b+1); N=len(idx)
vel=lambda T: rot_angle_deg(T[1:,:3,:3],T[:-1,:3,:3])
S0=score_traj(fit,Tb[idx],idx)
train=(np.arange(N)%2==0)
for lam_r,lam_p in [(30,0.5),(100,0.5),(300,0.5)]:
    t=time.time()
    Tr=fit_traj(fit,Tb[idx],idx,lam_acc_r=lam_r,lam_prior_r=lam_p,img_mask=train)
    S=score_traj(fit,Tr,idx)
    te=~train
    dr_=rot_angle_deg(Tr[:,:3,:3],Tb[idx,:3,:3])
    print('lam_r',lam_r,'lam_p',lam_p,'time %.1f'%(time.time()-t),
      'HELDOUT iou base %.3f fit %.3f | r2 base %.3f fit %.3f || train iou %.3f->%.3f | drot mean %.1f | rot speed med %.2f mean %.2f (base %.2f %.2f)'%(
      S0['siou'][te].mean(),S['siou'][te].mean(),S0['ncc'][te].mean(),S['ncc'][te].mean(),S0['siou'][train].mean(),S['siou'][train].mean(),dr_.mean(),
      np.median(vel(Tr)),vel(Tr).mean(),np.median(vel(Tb[idx])),vel(Tb[idx]).mean()))
    np.save(f'm3_traj_{lam_r}.npy',Tr)

exec(open('setup.py').read())
a,b=469,644
idx=np.arange(a,b+1); N=len(idx)
vel=lambda T: rot_angle_deg(T[1:,:3,:3],T[:-1,:3,:3])
S0=score_traj(fit,Tb[idx],idx)
train=(np.arange(N)%2==0); te=~train
for kw in [dict(iters=120,lr_r=0.02,lr_t=0.003,lam_acc_r=30,lam_prior_r=0.1),dict(iters=120,lr_r=0.02,lr_t=0.003,lam_acc_r=10,lam_acc_t=1000,lam_prior_r=0.1)]:
    t=time.time()
    Tr=fit_traj(fit,Tb[idx],idx,img_mask=train,**kw)
    S=score_traj(fit,Tr,idx)
    dr_=rot_angle_deg(Tr[:,:3,:3],Tb[idx,:3,:3])
    print(kw,'time %.1f'%(time.time()-t),
      'HELDOUT iou base %.3f fit %.3f | r2 base %.3f fit %.3f || train iou %.3f | drot mean %.1f | rot speed med %.2f mean %.2f'%(
      S0['siou'][te].mean(),S['siou'][te].mean(),S0['ncc'][te].mean(),S['ncc'][te].mean(),S['siou'][train].mean(),dr_.mean(),
      np.median(vel(Tr)),vel(Tr).mean()))
    for s in range(0,N,20): print('  ',idx[s],'iou %.2f->%.2f drot %.1f'%(S0['siou'][s:s+20].mean(),S['siou'][s:s+20].mean(),dr_[s:s+20].mean()))

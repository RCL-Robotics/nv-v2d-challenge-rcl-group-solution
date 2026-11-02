exec(open('setup.py').read())
phases={'A':(0,78),'B':(200,303),'C':(402,468),'D1':(645,669),'D2':(692,746),'E':(845,900)}
for name,(a,b) in phases.items():
    T0=mean_pose(Tb[a:b+1]); t=time.time()
    out,fr=solve_static(fit,T0,range(a,b+1),PM=PM,OM=OM)
    L=np.stack([o[3] for o in out]); wins=np.bincount(L.argmin(0),minlength=4)
    print(name,'%.1fs'%(time.time()-t),' | '.join('f%d L %.3f iou %.3f r2 %.3f'%(k,o[0],o[2]['siou'].mean(),o[2]['ncc'].mean()) for k,o in enumerate(out)),'wins',wins)
    np.savez(f'st_{name}.npz',T=np.stack([o[1] for o in out]),L=L,fr=fr)

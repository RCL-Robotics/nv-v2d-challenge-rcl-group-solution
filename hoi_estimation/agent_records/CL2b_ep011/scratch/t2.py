exec(open('setup.py').read())
phases={'A':(0,78),'B':(200,303),'C':(402,468),'D1':(645,669),'D2':(692,746),'E':(845,900)}
for name,(a,b) in phases.items():
    fr=pick_frames(PM,OM,range(a,b+1),16)
    T0=mean_pose(Tb[a:b+1])
    t=time.time()
    res=[]
    for k,Fl in enumerate(SYM_FLIPS):
        Tf,terms=fit.fit((T0@Fl)[None],fr,pid=np.zeros(len(fr),int),iters=80)
        loss=terms['sil']+0.3*(1-terms['ncc'])
        res.append((k,loss.mean(),terms['siou'].mean(),terms['ncc'].mean(),loss,Tf[0]))
    L=np.stack([r[4] for r in res])
    wins=np.bincount(L.argmin(0),minlength=4)
    print(name, 'time %.1f'%(time.time()-t), ' | '.join('f%d L %.3f iou %.3f r2 %.3f'%r[:4] for r in res), 'wins',wins,
          'drot vs T0 %s'%[round(float(rot_angle_deg(T0[:3,:3]@SYM_FLIPS[r[0]][:3,:3], r[5][:3,:3])),1) for r in res])
    np.save(f'static_{name}.npy',np.stack([r[5] for r in res]))

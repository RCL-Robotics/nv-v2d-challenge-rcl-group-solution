exec(open('setup.py').read())
import sys; sys.path.insert(0,'/work/outputs'); from solve import STATIC
d1=np.load('run1_diag.npz'); d2=np.load('run2_diag.npz')
for n,a,b in STATIC:
    idx=np.arange(a,b+1)
    Tm=mean_pose(Tb[a:b+1])
    res=[]
    for lab,T in [('base/frame',Tb[idx]),('base mean',np.repeat(Tm[None],len(idx),0)),('run1',np.repeat(d1['static_'+n][None],len(idx),0)),('run2',np.repeat(d2['static_'+n][None],len(idx),0))]:
        S=score_traj(fit,T,idx)
        res.append('%s iou %.3f r2 %.3f'%(lab,S['siou'].mean(),S['ncc'].mean()))
    print(n,' | '.join(res))

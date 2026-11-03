exec(open('setup.py').read())
t=time.time()
idx=list(range(0,901,1))
sc={}
for k,Fl in enumerate(SYM_FLIPS):
    T=Tb@Fl
    res={'siou':[], 'ncc':[], 'sil':[]}
    for a in range(0,901,64):
        s=fit.score(T[a:a+64], idx[a:a+64])
        for kk in res: res[kk].append(s[kk])
    sc[k]={kk:np.concatenate(v) for kk,v in res.items()}
print('score time',time.time()-t)
np.savez('flipscores.npz',**{f'{k}_{kk}':v for k in sc for kk,v in sc[k].items()})
for a in range(0,901,30):
    print(a, ' | '.join('f%d iou %.2f ncc %.2f'%(k,sc[k]['siou'][a:a+30].mean(),sc[k]['ncc'][a:a+30].mean()) for k in sc))

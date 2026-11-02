exec(open('setup.py').read())
from scipy.spatial.transform import Rotation as Rr
a,b=469,644
idx=np.arange(a,b+1)
t=time.time()
Tr=np.zeros((len(idx),4,4)); TT={}
for s in range(0,len(idx),32):
    ii=idx[s:s+32]
    Tf,terms=fit.fit(Tb[ii],ii,iters=50)
    Tr[s:s+32]=Tf
    for k,v in terms.items(): TT.setdefault(k,[]).append(v)
TT={k:np.concatenate(v) for k,v in TT.items()}
sb=fit.score(Tb[idx[:64]],idx[:64])
print('time',time.time()-t)
S0={k:[] for k in TT}
for s in range(0,len(idx),64):
    sc=fit.score(Tb[idx[s:s+64]],idx[s:s+64])
    for k in S0: S0[k].append(sc[k])
S0={k:np.concatenate(v) for k,v in S0.items()}
dr_=rot_angle_deg(Tr[:,:3,:3],Tb[idx,:3,:3])
for s in range(0,len(idx),10):
    sl=slice(s,s+10)
    print(idx[s],'base iou %.2f r2 %.2f | fit iou %.2f r2 %.2f | drot %.1f'%(S0['siou'][sl].mean(),S0['ncc'][sl].mean(),TT['siou'][sl].mean(),TT['ncc'][sl].mean(),dr_[sl].mean()))
vel=lambda T: rot_angle_deg(T[1:,:3,:3],T[:-1,:3,:3])
print('rot speed base median %.2f mean %.2f, fit median %.2f mean %.2f'%(np.median(vel(Tb[idx])),vel(Tb[idx]).mean(),np.median(vel(Tr)),vel(Tr).mean()))
np.save('m3_fit.npy',Tr)

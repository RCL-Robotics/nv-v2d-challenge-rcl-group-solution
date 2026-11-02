import sys, time, pickle; sys.path.insert(0,'/work/outputs')
from hoi_fit import *
from hoi_validate import *
S=setup(); pm=S['pm']; F_=len(S['Tb'])
r=np.load('exp4.npz'); th,a,b,dh=r['th'],r['a'],r['b'],float(r['dh'])
T4=np.stack([pm.pose(th[i],a[i],b[i]) for i in range(F_)]); T4[:,:3,3]-=dh*pm.up
np.save('exp4_T.npy',T4)
Tv=np.load('vit_T.npy')
import os
if os.path.exists('flowcheck.pkl'): fc_tracks=pickle.load(open('flowcheck.pkl','rb'))
grays=read_gray_frames('/work/inputs/video.mp4')
t0=time.time(); fc=FlowCheck(S['mesh'],S['K'],S['Twc'],grays,'/work/inputs/masks.h5',lag=4,step=2); print('tracks',len(fc.tracks),time.time()-t0)
res={}
for name,T in [('base',S['Tb']),('viterbi',Tv),('exp4',T4)]:
    e=fc.errors(T); res[name]=e
pickle.dump(res,open('flow_res.pkl','wb'))
phases=[('S1',0,164),('M1',164,228),('M2',228,320),('M3',320,348),('Mfail',348,384),('M4',384,469),('S2',469,546)]
print('phase   n   motion(px) | median flow err px: base viterbi exp4')
for nm,s,e in phases:
    ks=[t for t in res['base'] if s<=t<e and t in res['exp4'] and t in res['viterbi']]
    if not ks: continue
    mo=np.median([res['base'][t][1] for t in ks])
    print(nm,len(ks),round(mo,2),' '.join(f"{np.median([res[n][t][0] for t in ks]):.2f}" for n in ['base','viterbi','exp4']))

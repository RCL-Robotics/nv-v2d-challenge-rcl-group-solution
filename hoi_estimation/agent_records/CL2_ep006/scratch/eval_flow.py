import sys, time, pickle, os; sys.path.insert(0,'/work/outputs')
from hoi_fit import *
from hoi_validate import *
S=setup(); F_=len(S['Tb'])
grays=read_gray_frames('/work/inputs/video.mp4')
fc=FlowCheck(S['mesh'],S['K'],S['Twc'],grays,'/work/inputs/masks.h5',lag=4,step=2)
names=['base']+sys.argv[1].split(',')
Ts=[S['Tb']]+[np.load(n) for n in names[1:]]
res={n:fc.errors(T) for n,T in zip(names,Ts)}
phases=[('S1',0,164),('M1',164,228),('M2',228,320),('M3',320,348),('Mfail',348,384),('M4',384,469),('S2',469,546),('ALL',0,546)]
print('phase n motion |',' '.join(names))
for nm,s,e in phases:
    ks=[t for t in res['base'] if s<=t<e and all(t in res[n] for n in names)]
    if not ks: continue
    print(nm,len(ks),round(np.median([res['base'][t][1] for t in ks]),2),'|',' '.join(f"{np.mean([min(res[n][t][0],10) for t in ks]):.2f}" for n in names))
pickle.dump(res,open('flow_last.pkl','wb'))

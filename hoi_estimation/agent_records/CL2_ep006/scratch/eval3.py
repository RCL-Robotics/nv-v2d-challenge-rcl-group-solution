import sys, time, pickle; sys.path.insert(0,'/work/outputs')
from hoi_fit import *
from hoi_validate import *
S=setup(); pm=S['pm']; F_=len(S['Tb'])
G=np.load('grid_v2.npy'); m=np.load('motion2.npy')
seg=detect_phases(m,S['rec']['contact_prob_right_left'].max(1))
grays=read_gray_frames('/work/inputs/video.mp4')
fc=FlowCheck(S['mesh'],S['K'],S['Twc'],grays,'/work/inputs/masks.h5',lag=4,step=2)
phases=[('S1',0,164),('M1',164,228),('M2',228,320),('M3',320,348),('Mfail',348,384),('M4',384,469),('S2',469,546),('ALL',0,546)]
eb=fc.errors(S['Tb'])
out={}
for pw in [0.0,0.1,0.3,1.0,3.0]:
    th,a,b,*_=decode_trajectory(G,seg,prior_th=S['par_base_raw'][:,0],prior_valid=S['good'].astype(float),prior_w=pw)
    T=np.stack([pm.pose(th[i],a[i],b[i]) for i in range(F_)])
    e=fc.errors(T); out[pw]=(th,e)
    np.save(f'dec_pw{pw}.npy',T)
    print('prior_w',pw, ' '.join(f"{nm}:{np.mean([min(e[t][0],10) for t in e if s<=t<x]):.2f}" for nm,s,x in phases))
print('baseline', ' '.join(f"{nm}:{np.mean([min(eb[t][0],10) for t in eb if s<=t<x and t in out[0.0][1]]):.2f}" for nm,s,x in phases))
for i in range(216,330,8): print(i, ' '.join('%.1f'%np.degrees(out[pw][0][i]) for pw in out), 'base %.1f'%np.degrees(S['par_base_raw'][i,0]))

import sys; sys.path.insert(0,'/work/outputs')
from hoi_fit import *
from hoi_validate import *
S=setup(); pm=S['pm']; F_=len(S['Tb'])
G=np.load('grid_v2.npy'); m=np.load('motion2.npy')
seg=detect_phases(m,S['rec']['contact_prob_right_left'].max(1))
grays=read_gray_frames('/work/inputs/video.mp4')
fc=FlowCheck(S['mesh'],S['K'],S['Twc'],grays,'/work/inputs/masks.h5',lag=4,step=2)
phases=[('S1',0,164),('M1',164,228),('M2',228,320),('M3',320,348),('Mfail',348,384),('M4',384,469),('S2',469,546),('ALL',0,546)]
for lth in [3.0,10.0]:
  for lab in [8.0,50.0,200.0,1000.0]:
    th,a,b,*_=decode_trajectory(G,seg,prior_th=S['par_base_raw'][:,0],prior_valid=S['good'].astype(float),prior_w=0.3,lam_ab=lab,lam_th=lth)
    T=np.stack([pm.pose(th[i],a[i],b[i]) for i in range(F_)])
    Tr,_=reanchor_depth(T,S['Tb'],S['good'],seg,S['Twc'])
    e=fc.errors(Tr)
    print('lam_th',lth,'lam_ab',lab, ' '.join(f"{nm}:{np.mean([min(e[t][0],10) for t in e if s<=t<x]):.3f}" for nm,s,x in phases))

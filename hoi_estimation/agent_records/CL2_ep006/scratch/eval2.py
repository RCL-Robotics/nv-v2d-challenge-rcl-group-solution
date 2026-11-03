import sys, time, pickle; sys.path.insert(0,'/work/outputs')
from hoi_fit import *
from hoi_validate import *
S=setup(); pm=S['pm']; F_=len(S['Tb'])
G=np.load(sys.argv[1]); m=np.load('motion2.npy')
seg=detect_phases(m,S['rec']['contact_prob_right_left'].max(1))
print('static', [(s,np.nonzero(seg==s)[0][[0,-1]]) for s in [0,1]])
th,a,b,path,thr,abr=decode_trajectory(G,seg)
T=np.stack([pm.pose(th[i],a[i],b[i]) for i in range(F_)])
Tr,r=reanchor_depth(T,S['Tb'],S['good'],seg,S['Twc'])
np.save(sys.argv[2],Tr); np.save(sys.argv[2].replace('.npy','_noanchor.npy'),T)
yb=np.degrees(S['par_base_raw'][:,0])
for i in range(0,F_,12): print(i, 'yaw base %.1f new %.1f raw %.1f'%(yb[i],np.degrees(th[i]),np.degrees(thr[i])), 'ratio %.3f'%r[i])

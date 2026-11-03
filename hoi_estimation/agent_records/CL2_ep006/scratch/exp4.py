import sys, time; sys.path.insert(0,'/work/outputs')
from hoi_fit import *
torch.manual_seed(0)
S=setup(); pm=S['pm']
E=Evidence(S); sr=SilRenderer(decimate(S['mesh']),S['K'],S['Twc'],scale=0.5)
F_=len(S['Tb'])
C=np.load('grid_all.npy'); path=np.load('vit_path.npy')
th0=np.radians(path*3.0); a0=C[np.arange(F_),path,3]; b0=C[np.arange(F_),path,4]
m=motion_score(E.G,E.MO,E.MP); np.save('motion2.npy',m)
seg=-np.ones(F_,int)
e1=np.nonzero(m>1.3)[0]; e1=e1[e1>5][0]  # first moving frame
cp=S['rec']['contact_prob_right_left'].max(1); last_contact=np.nonzero(cp>0.5)[0].max()
s2=last_contact+np.nonzero(m[last_contact:]<1.3)[0][0]
seg[:e1]=0; seg[s2:]=1
print('static phases',0,e1-1,'and',s2,F_-1)
t0=time.time()
th,a,b,dh=joint_refine(sr,E,pm,th0,a0,b0,seg,iters=int(sys.argv[1]),log=print)
print('time',time.time()-t0)
np.savez('exp4.npz',th=th,a=a,b=b,dh=dh,seg=seg)

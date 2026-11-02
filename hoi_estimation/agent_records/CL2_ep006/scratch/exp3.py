import sys, time; sys.path.insert(0,'/work/outputs')
from hoi_fit import *
torch.manual_seed(0)
S=setup(); pm=S['pm']; fm=FreeDepthModel(pm,S['Twc'])
E=Evidence(S); sr=SilRenderer(decimate(S['mesh']),S['K'],S['Twc'],scale=0.5)
F_=len(S['Tb'])
base=np.array([fm.params_from_pose(T) for T in S['Tb']])
# interpolate non-plausible baseline frames for the translation init
good=S['good']; idx=np.arange(F_)
for k in range(1,4): base[~good,k]=np.interp(idx[~good],idx[good],base[good,k])
t0=time.time()
# pass 0: depth ratio from a sparse set of plausible frames (yaw free, near baseline)
fr0=[i for i in range(0,F_,15) if good[i]]
TH,XYZ,CO=fit_frames(sr,E,fm,fr0,np.array([[S['par_base'][i,0]] for i in fr0]),base[fr0,1:],iters=150,fit_th=True)
ratio=np.median(XYZ[:,0,2]/base[fr0,3]); print('depth ratio image/baseline',ratio, np.percentile(XYZ[:,0,2]/base[fr0,3],[10,50,90]), time.time()-t0)
xyz0=base[:,1:].copy(); xyz0[:,2]*=ratio
frames=list(range(F_)) if sys.argv[1]=='all' else [int(x) for x in sys.argv[1].split(',')]
NG=120; ths=np.arange(NG)*2*np.pi/NG
TH,XYZ,CO=fit_frames(sr,E,fm,frames,np.tile(ths,(len(frames),1)),xyz0[frames],iters=40)
out=np.zeros((F_,NG,6),np.float32); out[frames,:,:3]=CO; out[frames,:,3:]=XYZ
np.save(sys.argv[2],out); np.save('depth_ratio.npy',ratio)
print('time',time.time()-t0)

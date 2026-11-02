import sys, pickle; sys.path.insert(0,'/work/scratch')
from common import *
from bat_lib import Renderer
ep=Episode('/work/inputs'); C=pickle.load(open('/work/scratch/run2/stageA.pkl','rb')); ep.__dict__.update(C['ep'])
D=np.load('/work/scratch/run2/decoded.npz'); T=D['T']
Q=transported_frames(T); Rf=Renderer(ep.mesh)
angs,S,valid,sil=roll_scores(ep,Rf,T,Q,list(range(ep.F)),n_ang=36)
np.savez('/work/scratch/rollmap2.npz',angs=angs,S=S,valid=valid,sil=sil,Q=Q)
cur=np.degrees(np.arctan2(np.einsum('fi,fi->f',Q[:,:,1],T[:,:3,0]),np.einsum('fi,fi->f',Q[:,:,0],T[:,:3,0])))%360
for ls in [0,30,60,120]:
    k=decode_roll(S,valid,[(0,110),(706,802)],sil=sil,lam_sil=ls); phi=np.degrees(angs[k])
    d=np.abs((phi-cur+180)%360-180)
    print('lam_sil',ls,'static1 %.0f static2 %.0f (cur %.0f %.0f)  moving: median |change vs run2| %.0f'%(phi[50],phi[750],cur[50],cur[750],np.median(d[111:706])))

import sys; sys.path.insert(0,'/work/scratch')
from common import *
ep=Episode('/work/inputs')
d=np.load('/work/scratch/rollmap2.npz'); S=d['S']; sil=d['sil']; Q=d['Q']; valid=d['valid']; angs=d['angs']; n=len(angs)
k=decode_roll(S,valid,[(0,110),(706,802)],sil=sil)
for a,b in [(0,110),(706,802)]:
    fr=np.arange(a,b+1)
    i=fr[0]; Rb=ep.Tc_base[fr].mean(0)[:3,:3]
    phb=np.arctan2(Q[i][:,1]@ep.Tc_base[i,:3,0],Q[i][:,0]@ep.Tc_base[i,:3,0])%(2*np.pi)
    kb=int(np.round(phb/(2*np.pi/n)))%n; kd=k[i]
    St=S[fr].sum(0); Ss=sil[fr].sum(0)
    print((a,b),'decoded bin',kd,'baseline bin',kb,'tex: %.2f vs %.2f'%(St[kd],St[kb]),'sil: %.2f vs %.2f'%(Ss[kd],Ss[kb]), 'keep' if (St[kd]<St[kb] and Ss[kd]<Ss[kb]) else 'use baseline roll')

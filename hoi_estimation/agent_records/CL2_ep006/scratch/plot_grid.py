import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
C=np.load(sys.argv[1]); par=np.load('par_base.npy') if False else None
from hoi_fit import setup
S=setup(); par=S['par_base']
frames=[int(x) for x in sys.argv[2].split(',')]
fig,axs=plt.subplots(3,4,figsize=(16,9)); 
ths=np.degrees(np.arange(C.shape[1])*2*np.pi/C.shape[1])
for ax,fi in zip(axs.flat,frames):
    ax.plot(ths,C[fi,:,0],label='total'); ax.plot(ths,C[fi,:,1],label='sil'); ax.plot(ths,C[fi,:,2],label='fb')
    ax.axvline(np.degrees(par[fi,0])%360,color='k',ls='--'); ax.set_title(str(fi)); ax.set_ylim(0,1.2)
axs.flat[0].legend(); plt.tight_layout(); plt.savefig(sys.argv[3],dpi=60)

import numpy as np, sys
from scipy.spatial.transform import Rotation as Ro
for p in sys.argv[1:]:
    T=np.load(p)['Tc']
    a=T[:,:3,2]; ax=np.degrees(np.arccos(np.clip((a[:-1]*a[1:]).sum(1),-1,1)))
    rel=np.einsum('fji,fjk->fik',T[:-1,:3,:3],T[1:,:3,:3]); rv=Ro.from_matrix(rel).as_rotvec()
    roll=np.degrees(np.abs(rv[:,2]))
    print(p,'axis step med %.2f p95 %.1f max %.1f | roll step med %.2f p95 %.1f'%(np.median(ax[111:705]),np.percentile(ax[111:705],95),ax.max(),np.median(roll[111:705]),np.percentile(roll[111:705],95)))
    o=np.argsort(-ax)[:25]; print('  big axis jumps:',sorted([(int(i),int(ax[i])) for i in o]))

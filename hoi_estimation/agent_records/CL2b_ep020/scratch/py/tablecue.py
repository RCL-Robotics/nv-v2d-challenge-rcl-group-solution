from common import *
J=np.load('/work/scratch/joint_fixz.npz'); th=J['th']; uvz=J['uvz']
zs=np.load('/work/scratch/zbest_iou.npy')
d=json.load(open('/work/inputs/ground_plane.json'))['plane'][3]
bot_m=np.array([center[0],center[1],v[:,2].max()])
Kinv=np.linalg.inv(K)
def bottom_height(i,z):
    Rwm=B@np.array([[np.cos(th[i]),-np.sin(th[i]),0],[np.sin(th[i]),np.cos(th[i]),0],[0,0,1]])
    t=z*(Kinv@np.r_[uvz[i,:2],1]); pb=Rcw@Rwm@bot_m+t
    return nup@(Twc[:3,:3]@pb+Twc[:3,3])+d, pb
def z_for_height(i,h):
    # height linear in z (t scales along ray, rotation fixed)
    h1,_=bottom_height(i,3.0); h2,_=bottom_height(i,3.1)
    return 3.0+(h-h1)*(0.1)/(h2-h1)
occ=np.array([PM[i][545:657,690:818].sum() for i in range(553)])
for name,(a,b) in dict(S1=(0,130),S1b=(140,210),P=(282,308),S2a=(428,460),S2=(465,552)).items():
    fr=np.arange(a,b+1)
    zsil=np.median(zs[fr])
    hs=[bottom_height(i,zsil)[0] for i in fr]
    print(name,'sil z %.4f  bottom height at sil z %.4f  base z med %.4f  uv %s'%(zsil,np.median(hs),np.median(Tcm[fr,2,3]),np.round(np.median(uvz[fr,:2],0),2)))
    hs_ref=0.8137
    print('    z for height %.4f: %.4f'%(hs_ref, np.median([z_for_height(i,hs_ref) for i in fr])))
from scipy.ndimage import median_filter, gaussian_filter1d
zt=np.array([z_for_height(i,0.8137) for i in range(553)])
np.save('/work/scratch/ztable.npy',zt)
zsm=gaussian_filter1d(median_filter(zs,9,mode='nearest'),3,mode='nearest'); ztm=gaussian_filter1d(median_filter(zt,9,mode='nearest'),3,mode='nearest')
for i in range(200,470,6): print(i,'sil %.3f table %.3f  | smoothed sil %.3f table %.3f  base %.3f occ %d'%(zs[i],zt[i],zsm[i],ztm[i],Tcm[i,2,3],occ[i]))

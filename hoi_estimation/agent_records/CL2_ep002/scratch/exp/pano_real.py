from pano import *
import h5py
src=sys.argv[1]; a,b,st=[int(x) for x in sys.argv[2:5]]; sign=float(sys.argv[5])
T=np.load(src)['T_world_mesh'] if src!='base' else rec['T_world_mesh'].astype(np.float64)
Tc=np.linalg.inv(Twc)@T
P=h5py.File('/work/inputs/masks.h5','r')['person']
def wyaw(Tw): x=Tw[:3,0]; return np.degrees(np.arctan2(x[2],x[0]))
out=[]
for i in range(a,b,st):
    sm,wt=pano(i,Tc[i],P[i])
    sc=corr_shift(sm,wt,sign)
    top=np.argsort(-sc)[:3]
    absy=[(wyaw(Twc@Tc[i]@yawrot(np.radians(-s*360/NPHI))))%360 for s in top]
    out.append(sc)
    print(i, f'baseyaw {wyaw(T[i])%360:6.1f}', 'top shifts',top*3, 'abs yaw', np.round(absy,0), 'sc',sc[top].round(2), 'cov',(wt.sum(0)>0).mean().round(2))
np.save(f'/work/scratch/pscore_{a}_{b}.npy',np.array(out))

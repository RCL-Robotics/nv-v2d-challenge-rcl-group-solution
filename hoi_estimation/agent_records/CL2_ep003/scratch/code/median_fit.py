import sys; sys.path.insert(0,'/work/outputs'); sys.path.insert(0,'/work/scratch/code')
from lib import *
from scorelib import scores
exec(open('/work/scratch/code/upright.py').read())
f5=h5py.File('/work/inputs/masks.h5')
S=np.load('static_median.npz')
img=[];om=[];pm=[]
for k,(a,b) in enumerate(S['phases']):
    x0,y0=S['off'][k]
    oms=np.stack([f5['object'][i][y0:y0+160,x0:x0+160] for i in range(a,b)])
    pms=np.stack([f5['person'][i][y0:y0+160,x0:x0+160] for i in range(a,b)])
    ok=~(pms&~oms); frac=(oms&ok).sum(0)/np.maximum(ok.sum(0),1)
    img.append(np.nan_to_num(S['med'][k][...,::-1]).astype(np.uint8)); om.append(frac>0.5); pm.append(ok.sum(0)==0)
MD=dict(img=np.stack(img),om=np.stack(om),pm=np.stack(pm),off=S['off'].astype(float))
np.savez('median_crops.npz',**MD)
Fi=Fitter(R,K,MD,c0)
step=int(sys.argv[1]); targets=np.arange(0,360,step)
for k,(a,b) in enumerate(S['phases']):
    Tm=Tc[a].copy(); Tm[:3,3]=np.median(Tc[a:b,:3,3],0); Tu=upright(Tm)
    hx=Tu[:3,:3]@np.array([1.,0,0]); az0=np.degrees(np.arctan2(hx[2],hx[0]))
    T0=spin(Tu,c0,u,np.radians(targets-az0))
    Tf,iou,ncc=Fi.fit(T0,[[k]]*len(T0),iters=80,free_z=True)
    s=scores(Fi,Tf,[k])
    hx=np.einsum('hij,j->hi',Tf[:,:3,:3],np.array([1.,0,0])); az=np.degrees(np.arctan2(hx[:,2],hx[:,0]))
    tilt=[np.degrees(np.arccos(np.clip((Tq[:3,:3]@u)@nc,-1,1))) for Tq in Tf]
    print('phase',a,b)
    print('  az  ',' '.join('%4d'%x for x in az)); print('  tilt',' '.join('%4d'%x for x in tilt)); print('  dz  ',' '.join('%4d'%x for x in 100*(Tf[:,2,3]-T0[:,2,3])))
    for kk in s: print('  %-4s'%kk,' '.join('%4d'%(100*x) for x in s[kk]))
    np.save(f'medfit_{k}.npy',Tf)

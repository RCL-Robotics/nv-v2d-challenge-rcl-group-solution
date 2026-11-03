import sys; sys.path.insert(0,'/work/scratch/src')
from fitlib import *
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
cam,K,Twc,rec=load_inputs()
Tcw=np.linalg.inv(Twc)
Tc=Tcw@rec['T_world_mesh'].astype(float)
kp=rec['human_keypoints_world'].astype(float)
kpc=np.einsum('ij,fkj->fki',Tcw[:3,:3],kp)+Tcw[:3,3]
def anchor(T,ref,fr):
    T=T.copy(); s=np.median(ref[fr,2,3])/np.median(T[fr,2,3]); T[:,:3,3]*=s; return T
def smooth_base(Tc,win=31):
    T=Tc.copy()
    z=savgol_filter(median_filter(Tc[:,2,3],9),win,2)
    T[:,:3,3]*= (z/Tc[:,2,3])[:,None]
    return T
def hand_in_obj(T,fr):
    out=[]
    for i in fr:
        Ti=np.linalg.inv(T[i]); h=kpc[i,21:42].mean(0)
        out.append(Ti[:3,:3]@h+Ti[:3,3])
    return np.array(out)
fr=np.arange(int(sys.argv[2]) if len(sys.argv)>2 else 230,int(sys.argv[3]) if len(sys.argv)>3 else 500)
cands={'base':Tc,'base_smoothz':smooth_base(Tc)}
for f in sys.argv[1].split(','):
    J=np.load(f)['T']; cands[f]=anchor(J,Tc,fr)
    Jz=J.copy(); Jz=anchor(Jz,Tc,fr); zs=smooth_base(Tc)[:,2,3]; Jz[:,:3,3]*=(zs/Jz[:,2,3])[:,None]; cands[f+'+basez']=Jz
for k,T in cands.items():
    h=hand_in_obj(T,fr)
    print('%-40s hand-in-object std (cm) %s  total %.2f'%(k,np.round(h.std(0)*100,2),np.linalg.norm(h.std(0))*100))

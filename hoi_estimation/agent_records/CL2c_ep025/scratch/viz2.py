# viz arbitrary poses: python viz2.py out.jpg file.npz  (npz with R,t,frames)
import sys
from viz import *
d=np.load(sys.argv[2]); fr=d['frames']
pans=[]; 
groups={}
for k,f in enumerate(fr): groups.setdefault(int(f),[]).append(k)
for f,ks in groups.items():
    poses=[]
    for k in ks:
        T=np.eye(4); T[:3,:3]=d['R'][k]; T[:3,3]=d['t'][k]; poses.append(T)
    pans.append(crop_panel(f,poses,[K]*len(poses),S=160,scale=1.5))
cv2.imwrite(sys.argv[1],np.vstack(pans))

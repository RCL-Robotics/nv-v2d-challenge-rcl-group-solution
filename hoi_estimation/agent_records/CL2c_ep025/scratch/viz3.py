# python viz3.py out.jpg poses.npz frames [S] [scale]  -> image | selected | spin+90 | spin+180 | spin-90
import sys
from viz import *
d=np.load(sys.argv[2]); fr=[int(a) for a in sys.argv[3].split(',')]
S=int(sys.argv[4]) if len(sys.argv)>4 else 110; sc=float(sys.argv[5]) if len(sys.argv)>5 else 3
pans=[]
def Rz(a):
    c,s=np.cos(np.radians(a)),np.sin(np.radians(a)); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
for f in fr:
    poses=[]
    for a in [0,90,180,270]:
        T=np.eye(4); T[:3,:3]=d['R'][f]@Rz(a); T[:3,3]=d['t'][f]; poses.append(T)
    pans.append(crop_panel(f,poses,[K]*4,S=S,scale=sc))
cv2.imwrite(sys.argv[1],np.vstack(pans))

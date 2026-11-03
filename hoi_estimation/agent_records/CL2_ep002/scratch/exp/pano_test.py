from pano import *
import h5py
src=sys.argv[1]; fr=[int(x) for x in sys.argv[2].split(',')]
T=np.load(src)['T_world_mesh']; Tc=np.linalg.inv(Twc)@T
P=h5py.File('/work/inputs/masks.h5','r')['person']
tiles=[]
for i in fr:
    sm,wt=pano(i,Tc[i],P[i])
    im=np.where(wt>0,sm/np.maximum(wt,1),0)
    im=np.clip(im*6+128,0,255).astype(np.uint8); im[wt==0]=0
    tiles.append(cv2.resize(im,(NPHI*6,NH*6),interpolation=cv2.INTER_NEAREST))
    cv2.putText(tiles[-1],str(i),(3,20),0,0.6,255,1)
cv2.imwrite(sys.argv[3],np.vstack(tiles))

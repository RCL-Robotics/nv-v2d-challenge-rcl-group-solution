from pano import *
sm,wt=texture_pano()
np.savez('/work/scratch/texpano.npz',sm=sm,wt=wt)
im=np.where(wt>0,sm/np.maximum(wt,1),0); im=np.clip(im*3+128,0,255).astype(np.uint8)
cv2.imwrite('/work/scratch/vis/texpano.png',cv2.resize(im,(NPHI*6,NH*6),interpolation=cv2.INTER_NEAREST))

import os, cv2,numpy as np
d=os.environ["V2D_ROOT"] + '/support/ep001'
a=cv2.imread(d+'/background.png')[550:1150,500:950]; o=np.load(d+'/occupancy.npy').astype(np.float32)[550:1150,500:950]
r=cv2.imread(d+'/fit_render_mask_keep.png',0); h,w=r.shape; w3=w//3
rr=cv2.resize(r[:,:w3],(1536,1152))[550:1150,500:950]; mm=cv2.resize(r[:,w3:2*w3],(1536,1152))[550:1150,500:950]
b=a.copy(); b[...,2]=np.maximum(b[...,2],rr); b[...,1]=np.maximum(b[...,1],mm//2)
cv2.imwrite(os.environ["V2D_ROOT"] + '/support/vis/dbg_ep1.jpg',np.concatenate([a,b,cv2.cvtColor((o*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)],1))

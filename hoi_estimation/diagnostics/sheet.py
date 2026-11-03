import os, sys, cv2,numpy as np
S=sys.argv[1]; es="01 02 03 06 07 09 11 13 16 17 20 25 27 28".split()
ims=[cv2.imread(f"{S}/f_{e}.jpg") for e in es]
for im,e in zip(ims,es): cv2.putText(im,"ep"+e,(10,40),cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,255,255),3)
ims+= [np.zeros_like(ims[0])]*2
rows=[np.concatenate(ims[i:i+4],1) for i in range(0,16,4)]
cv2.imwrite(os.environ["V2D_ROOT"] + "/support/vis/sheet_frame0.jpg",np.concatenate(rows,0))

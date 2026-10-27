import cv2,numpy as np
S="/tmp/claude-1003/-home-hanzhou-Projects-v2d/55432ff5-7f95-4b79-9c90-93c1bcd7201c/scratchpad/sup"; es="01 02 03 06 07 09 11 13 16 17 20 25 27 28".split()
ims=[cv2.imread(f"{S}/f_{e}.jpg") for e in es]
for im,e in zip(ims,es): cv2.putText(im,"ep"+e,(10,40),cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,255,255),3)
ims+= [np.zeros_like(ims[0])]*2
rows=[np.concatenate(ims[i:i+4],1) for i in range(0,16,4)]
cv2.imwrite("/mnt/nas/Group/hanzhou/v2d/support/vis/sheet_frame0.jpg",np.concatenate(rows,0))

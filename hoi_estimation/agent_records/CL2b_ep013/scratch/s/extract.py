import cv2, numpy as np
c=cv2.VideoCapture('/work/inputs/video.mp4'); fr=[]
while True:
    ok,im=c.read()
    if not ok: break
    fr.append(cv2.resize(im,(768,576),interpolation=cv2.INTER_AREA))
fr=np.stack(fr); print(fr.shape); np.save('/work/scratch/frames_half.npy',fr)

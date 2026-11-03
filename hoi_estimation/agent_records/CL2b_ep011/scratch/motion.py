import numpy as np, h5py, cv2
f=h5py.File('/work/inputs/masks.h5'); OM=f['object'][:]; PM=f['person'][:]
cap=cv2.VideoCapture('/work/inputs/video.mp4'); G=[]
while True:
    ok,fr=cap.read()
    if not ok: break
    G.append(cv2.GaussianBlur(cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY),(5,5),0).astype(np.float32))
G=np.stack(G); F=len(G)
k=3
d=np.full(F,np.nan); cen=np.full((F,2),np.nan); area=OM.reshape(F,-1).sum(1)
for t in range(F):
    ys,xs=np.nonzero(OM[t])
    if len(xs): cen[t]=[xs.mean(),ys.mean()]
for t in range(k,F):
    reg=(OM[t]|OM[t-k]); reg=cv2.dilate(reg.astype(np.uint8),np.ones((5,5)))>0
    reg&=~PM[t]&~PM[t-k]
    if reg.sum()>200: d[t]=np.abs(G[t]-G[t-k])[reg].mean()
np.savez('motion.npz',d=d,cen=cen,area=area)
for t in range(0,F,5): print(t, '%.2f'%d[t], np.round(cen[t],1), area[t], int(PM[t][cv2.dilate(OM[t].astype(np.uint8),np.ones((15,15)))>0].sum()))

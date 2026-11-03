import numpy as np, cv2, h5py
f=h5py.File('/work/inputs/masks.h5'); OM=f['object']; PM=f['person']
cap=cv2.VideoCapture('/work/inputs/video.mp4'); G=[]
while True:
    ok,fr=cap.read()
    if not ok: break
    G.append(cv2.cvtColor(cv2.resize(fr,(768,576)),cv2.COLOR_BGR2GRAY))
G=np.array(G).astype(np.float32); np.save('/work/scratch/gray_half.npy',G.astype(np.uint8))
F=len(G)
def ds(m): return cv2.resize(m.astype(np.uint8),(768,576),interpolation=cv2.INTER_NEAREST)>0
om=np.array([ds(OM[i]) for i in range(F)]); pm=np.array([ds(PM[i]) for i in range(F)])
np.save('/work/scratch/om_half.npy',om); np.save('/work/scratch/pm_half.npy',pm)
k=np.ones((9,9),np.uint8)
for ref in [0,F-1]:
    reg=cv2.dilate(om[ref].astype(np.uint8),k)>0
    out=[]
    for i in range(F):
        v=reg&~cv2.dilate(pm[i].astype(np.uint8),k).astype(bool)&~cv2.dilate(pm[ref].astype(np.uint8),k).astype(bool)
        out.append(np.abs(G[i]-G[ref])[v].mean() if v.sum()>100 else np.nan)
    out=np.array(out)
    print('ref',ref, ' '.join(f'{i}:{out[i]:.1f}' for i in range(0,F,10)))
# mask iou with neighbor (centroid motion)
c=np.array([np.argwhere(m).mean(0) if m.any() else [np.nan,np.nan] for m in om])
print('centroid', ' '.join(f'{i}:{c[i,0]:.0f},{c[i,1]:.0f}' for i in range(0,F,10)))

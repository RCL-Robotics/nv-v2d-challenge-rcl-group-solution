import numpy as np, cv2
G=np.load('gray_half.npy').astype(np.float32); om=np.load('om_half.npy'); pm=np.load('pm_half.npy')
F=len(G); k=np.ones((9,9),np.uint8)
def d(i,j):
    reg=(cv2.dilate(om[i].astype(np.uint8),k)>0)|(cv2.dilate(om[j].astype(np.uint8),k)>0)
    p=cv2.dilate((pm[i]|pm[j]).astype(np.uint8),np.ones((15,15),np.uint8))>0
    v=reg&~p
    return np.abs(G[i]-G[j])[v].mean()
for ref in [150,200,250]:
    print('ref',ref,' '.join(f'{i}:{d(i,ref):.1f}' for i in range(140,300,5)))
print('ref',700,' '.join(f'{i}:{d(i,700):.1f}' for i in range(560,794,8)))
print('step5',' '.join(f'{i}:{d(i,i-5):.1f}' for i in range(5,794,5)))

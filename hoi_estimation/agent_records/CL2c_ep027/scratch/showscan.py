import numpy as np, cv2
d=np.load('/work/scratch/yawscan.npz'); IoU=d['IoU']; PH=d['PH']; psis=d['psis']; pb=d['psi_b']
F=len(IoU); rows=[i for i in range(F) if not np.isnan(IoU[i]).all()]
a=rows[0]; b=rows[-1]+1
def norm(x):
    x=np.nan_to_num(x[a:b],nan=np.nanmin(x[a:b])); lo=np.percentile(x,2,axis=1,keepdims=True); hi=x.max(1,keepdims=True); return (x-lo)/(hi-lo+1e-9)
I=norm(IoU); Ph=np.nan_to_num(PH[a:b],nan=0); Ph=np.clip((Ph+0.2)/1.0,0,1)
def img(x):
    im=(np.clip(x,0,1)*255).astype(np.uint8); im=cv2.applyColorMap(im,cv2.COLORMAP_VIRIDIS)
    im=cv2.resize(im,(36*8,(b-a)*2),interpolation=cv2.INTER_NEAREST)
    for r in range(b-a):
        k=int(round((pb[a+r]%(2*np.pi))/(2*np.pi)*36))%36
        cv2.circle(im,(k*8+4,r*2+1),1,(0,0,255),-1)
    for r in range(0,b-a,25): cv2.putText(im,str(a+r),(0,r*2+8),0,0.3,(255,255,255),1)
    return im
cv2.imwrite('/work/scratch/viz/scan.png',np.hstack([img(I),np.zeros(((b-a)*2,10,3),np.uint8),img(Ph)]))

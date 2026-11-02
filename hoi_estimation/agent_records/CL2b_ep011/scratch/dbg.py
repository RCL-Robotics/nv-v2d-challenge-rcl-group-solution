exec(open('setup.py').read())
import sys
fr=[int(x) for x in sys.argv[2].split(',')]
T=Tb if len(sys.argv)<4 else None
rows=[]
for i in fr:
    p=K@Tb[i,:3,3]; crop=obs.crop_for(i,(p[0]/p[2],p[1]/p[2]))
    img,om,pm,bg=obs.get(i,crop)
    rgb,m,d,nr=render_soft(R,torch.tensor(Tb[i:i+1],dtype=torch.float32,device=DEV),K,[crop],fit.size)
    m=m[0].cpu().numpy(); d=d[0].cpu().numpy()
    occ=(d>bg+0.05)&(m>0.5)
    vis=(img*255).astype(np.uint8)[:,:,::-1].copy()
    a=np.zeros_like(vis); a[...,1]=om*255; a[...,2]=(m>0.5)*255; a[...,0]=pm*120
    b=np.zeros_like(vis); b[occ]=(0,0,255); 
    bgv=np.clip((bg-2.5)/2*255,0,255).astype(np.uint8); bgv=cv2.cvtColor(bgv,cv2.COLOR_GRAY2BGR)
    row=np.hstack([vis,a,np.maximum(b,bgv//2)])
    cv2.putText(row,str(i),(3,15),0,0.5,(0,0,255),1)
    rows.append(cv2.resize(row,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST))
cv2.imwrite(sys.argv[1],np.vstack(rows))

from rend import *
r,cam=load(); K=np.array(cam['K'])
s=np.load('silfit.npz'); R=s['R']; t=s['t']; x0y0=s['x0y0']
M=Mesh(); MK=Masks(); S=192; F=len(R)
vis=np.zeros(F); sa=np.zeros(F)
for b in range(0,F,64):
    ids=list(range(b,min(F,b+64)))
    a,_=render_sil(M,torch.tensor(R[ids],dtype=torch.float32,device=DEV),torch.tensor(t[ids],dtype=torch.float32,device=DEV),K,torch.tensor(x0y0[ids],dtype=torch.float32,device=DEV),S)
    a=a.cpu().numpy()>0.5
    for k,i in enumerate(ids):
        o,p=MK.crop(i,*x0y0[i],S)
        sa[i]=a[k].sum(); vis[i]=(a[k]&~(p&~o)).sum()/max(a[k].sum(),1)
np.savez('vis.npz',vis=vis,sa=sa)
for i in range(0,F,10): print(i,'%.2f'%vis[i],int(sa[i]),'%.2f'%s['iou'][i], end=' | ' if i%50 else '\n')

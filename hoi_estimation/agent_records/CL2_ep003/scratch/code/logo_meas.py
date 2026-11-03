import sys; sys.path.insert(0,'/work/outputs')
from lib import *
m,v,f,uv,tex=load_mesh('/work/inputs/object_mesh.glb'); cam,K,Twc,rec=load_inputs('/work/inputs'); c0,u=body_axis(v)
L=logo_texture(v,f,uv,tex,c0,u).astype(np.float32)
RL=Renderer(v,f,uv,np.repeat(L[...,None],3,2))
MD=np.load('median_crops.npz')
sc=4; K4=K.copy(); K4[:2]*=sc; targets=np.arange(0,360,5)
for k in range(6):
    Tf=np.load(f'medfit_{k}.npy')
    hx=np.einsum('hij,j->hi',Tf[:,:3,:3],np.array([1.,0,0])); az=np.degrees(np.arctan2(hx[:,2],hx[:,0]))
    j=int(np.argmin(np.abs(((az-100)+180)%360-180))); Tb=Tf[j]
    Ts=spin(Tb,c0,u,np.radians(targets-az[j])); off=MD['off'][k]*sc
    lab,mk,_=RL.render(Ts,K4,160*sc,160*sc,crop=np.repeat(off[None],len(Ts),0)); lab=lab[...,0].cpu().numpy(); mk=mk.cpu().numpy()
    # rendered logo: rows and fractional column across body
    rfr=[]
    for q in range(len(Ts)):
        ys,xs=np.nonzero(lab[q]>0.5)
        if len(xs)<30: rfr.append(np.nan); continue
        row=int(np.median(ys)); cols=np.nonzero(mk[q][row])[0]
        rfr.append((xs.mean()-cols.min())/(cols.max()-cols.min()))
    rfr=np.array(rfr)
    # observed: rows around median logo row of the plausible hypotheses
    ys_all=np.nonzero(lab[(targets>=60)&(targets<=160)]>0.5)[1]; r0,r1=np.percentile(ys_all,[5,95]).astype(int)
    g=cv2.cvtColor(MD['img'][k],cv2.COLOR_RGB2GRAY).astype(np.float32)
    bd=cv2.resize(bright_detail(g),None,fx=sc,fy=sc,interpolation=cv2.INTER_LINEAR)
    om=cv2.resize(MD['om'][k].astype(np.uint8),None,fx=sc,fy=sc,interpolation=cv2.INTER_NEAREST)
    band=bd[r0:r1+1]*om[r0:r1+1]; prof=band.sum(0)
    cols=np.nonzero(om[(r0+r1)//2])[0]; lft,rgt=cols.min(),cols.max()
    inner=np.zeros_like(prof); inner[lft+6:rgt-5]=prof[lft+6:rgt-5]
    pk=np.argmax(inner); 
    # centroid of blob around peak above half max
    hm=inner>0.5*inner[pk]; a=pk; b=pk
    while a>0 and hm[a-1]: a-=1
    while b<len(hm)-1 and hm[b+1]: b+=1
    cx=(np.arange(a,b+1)*inner[a:b+1]).sum()/inner[a:b+1].sum()
    ofr=(cx-lft)/(rgt-lft)
    d=np.abs(rfr-ofr); best=targets[np.nanargmin(d)]
    print(k,'obs frac %.2f (blob %d-%d of %d-%d)'%(ofr,a,b,lft,rgt),'render fr:',' '.join('%.2f'%x for x in rfr[10:36]),'-> az',best)

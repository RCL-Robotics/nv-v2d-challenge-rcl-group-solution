import sys; sys.path.insert(0,'/work/outputs')
from lib import *
m,v,f,uv,tex=load_mesh('/work/inputs/object_mesh.glb'); cam,K,Twc,rec=load_inputs('/work/inputs'); c0,u=body_axis(v)
L=logo_texture(v,f,uv,tex,c0,u).astype(np.float32); print('logo texels',L.sum())
RL=Renderer(v,f,uv,np.repeat(L[...,None],3,2))
MD=np.load('median_crops.npz')
sc=4; K4=K.copy(); K4[:2]*=sc
targets=np.arange(0,360,5)
for k in range(6):
    Tf=np.load(f'medfit_{k}.npy')
    hx=np.einsum('hij,j->hi',Tf[:,:3,:3],np.array([1.,0,0])); az=np.degrees(np.arctan2(hx[:,2],hx[:,0]))
    j=int(np.argmin(np.abs(((az-100)+180)%360-180))); Tb=Tf[j]
    Ts=spin(Tb,c0,u,np.radians(targets-az[j]))
    off=MD['off'][k]*sc
    lab,mk,_=RL.render(Ts,K4,160*sc,160*sc,crop=np.repeat(off[None],len(Ts),0))
    lab=lab[...,0].cpu().numpy(); mk=mk.cpu().numpy()
    g=cv2.cvtColor(MD['img'][k],cv2.COLOR_RGB2GRAY).astype(np.float32)
    g4=cv2.resize(g,None,fx=sc,fy=sc,interpolation=cv2.INTER_CUBIC)
    bd=cv2.resize(bright_detail(g),None,fx=sc,fy=sc,interpolation=cv2.INTER_LINEAR)
    sco=[]
    for q in range(len(Ts)):
        w=lab[q]*mk[q]
        if w.sum()<20: sco.append(np.nan); continue
        reg=cv2.erode(mk[q].astype(np.uint8),np.ones((9,9)))>0
        w=w*reg
        if w.sum()<20: sco.append(np.nan); continue
        sco.append((bd*w).sum()/w.sum()-bd[reg].mean())
    sco=np.array(sco)
    print(k,' '.join('%3s'%('-' if np.isnan(x) else int(x*10)) for x in sco),' best',targets[np.nanargmax(sco)])

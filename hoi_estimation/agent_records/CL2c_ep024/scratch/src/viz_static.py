from fitlib import *
cam,K,Twc,rec=load_inputs(); M=Mesh()
Tc=np.linalg.inv(Twc)@rec['T_world_mesh'].astype(float)
x0,y0,w,h,s=680,790,120,100,5
rows=[]
for i,files in [(40,['static_S1_1','static_S1_0']),(720,['static_S2_1','static_S2_0'])]:
    Ts=[Tc[i]]+[np.load(f'an/{f}.npy') for f in files]
    im=cv2.resize(read_frame(i)[y0:y0+h,x0:x0+w],None,fx=s,fy=s,interpolation=cv2.INTER_CUBIC)
    m,rgb,_=M.render(torch.tensor(np.stack(Ts),device=dev),K,x0,y0,w,h,s)
    row=[im]
    for k in range(len(Ts)):
        r=(rgb[k].cpu().numpy()[...,::-1]*255).astype(np.uint8); mm=m[k].cpu().numpy()
        ov=im.copy(); ov[mm]=r[mm]
        c,_=cv2.findContours(mm.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        e=im.copy(); cv2.drawContours(e,c,-1,(0,0,255),1)
        row+= [ov,e]
    rows.append(np.hstack(row))
cv2.imwrite('an/static_cmp.jpg',np.vstack(rows))

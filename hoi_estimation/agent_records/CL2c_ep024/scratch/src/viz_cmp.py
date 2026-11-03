import sys; sys.path.insert(0,'/work/scratch/src')
from fitlib import *
cam,K,Twc,rec=load_inputs(); M=Mesh()
Tc=np.linalg.inv(Twc)@rec['T_world_mesh'].astype(float)
J=np.load(sys.argv[1])['T']
frames=[int(x) for x in sys.argv[2].split(',')]
x0,y0,w,h,s=650,740,170,150,4
rows=[]
for i in frames:
    im=cv2.resize(read_frame(i)[y0:y0+h,x0:x0+w],None,fx=s,fy=s,interpolation=cv2.INTER_CUBIC)
    TJ=J[i].copy(); TJ[:3,3]*=Tc[i,2,3]/TJ[2,3]   # same depth as baseline for display
    row=[im]
    for T,col in [(Tc[i],(0,0,255)),(J[i],(255,0,0))]:
        m,_,_=M.render(torch.tensor(T[None],device=dev),K,x0,y0,w,h,s,need_tex=False)
        c,_=cv2.findContours(m[0].cpu().numpy().astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        e=im.copy(); cv2.drawContours(e,c,-1,col,2); row.append(e)
    cv2.putText(row[0],str(i),(5,30),0,1,(0,255,255),2)
    rows.append(np.hstack(row))
cv2.imwrite(sys.argv[3],np.vstack(rows))

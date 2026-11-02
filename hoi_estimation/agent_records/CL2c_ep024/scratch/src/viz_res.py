import sys; sys.path.insert(0,'/work/scratch/src')
from fitlib import *
cam,K,Twc,rec=load_inputs(); M=Mesh()
Tcw=np.linalg.inv(Twc)
Tb=Tcw@rec['T_world_mesh'].astype(float); Tr=Tcw@np.load(sys.argv[1])['T_world_mesh']
frames=[int(x) for x in sys.argv[2].split(',')] if ',' in sys.argv[2] else list(range(*[int(v) for v in sys.argv[2].split(':')]))
x0,y0,w,h,s=650,740,170,150,2
tiles=[]
for i in frames:
    im=cv2.resize(read_frame(i)[y0:y0+h,x0:x0+w],None,fx=s,fy=s,interpolation=cv2.INTER_CUBIC)
    for T,col in [(Tb[i],(0,0,255)),(Tr[i],(255,128,0))]:
        m,_,_=M.render(torch.tensor(T[None],device=dev),K,x0,y0,w,h,s,need_tex=False)
        c,_=cv2.findContours(m[0].cpu().numpy().astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(im,c,-1,col,1)
    cv2.putText(im,str(i),(5,22),0,0.7,(0,255,255),2); tiles.append(im)
nc=8
while len(tiles)%nc: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[3],np.vstack([np.hstack(tiles[k:k+nc]) for k in range(0,len(tiles),nc)]))

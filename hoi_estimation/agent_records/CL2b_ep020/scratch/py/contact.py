from common import *
J=np.load('/work/scratch/test_out.npz'); Tw=J['T_world_mesh']; Tc=Tcw@Tw
o=Tc[:,:3,3]; pu=o@K.T; pu=pu[:,:2]/pu[:,2:]
def contact_row(img, uc, vguess):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)
    g=cv2.GaussianBlur(g,(3,3),0.7)
    rows=[]
    for c in range(int(round(uc))-6,int(round(uc))+7):
        lo=int(vguess)-8; p=g[lo:lo+17,c]
        k=int(np.argmin(p))
        if 0<k<len(p)-1:
            y0,y1,y2=p[k-1],p[k],p[k+1]; den=y0-2*y1+y2; off=0.5*(y0-y2)/den if den>0 else 0
            rows.append(lo+k+off)
    return np.median(rows) if rows else np.nan
vc=np.array([contact_row(frame(i),pu[i,0]+2,639) for i in range(553)])
np.save('/work/scratch/contact_v.npy',vc)
for a,b in [(0,211),(212,281),(282,310),(311,426),(427,552)]:
    s=vc[a:b+1]; print(a,b,'median %.2f std %.2f p10 %.2f p90 %.2f'%(np.nanmedian(s),np.nanstd(s),np.nanpercentile(s,10),np.nanpercentile(s,90)))
print(' '.join('%d:%.1f'%(i,vc[i]) for i in range(200,440,4)))

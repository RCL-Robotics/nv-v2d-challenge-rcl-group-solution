from common import *
# vertical intensity profile in central columns; locate cooker->table transition near the bottom with subpixel accuracy
def edge_rows(i, cols, rlo=625, rhi=650):
    im=frame(i).astype(np.float32)
    # use blue channel minus red: cooker is white (b~r), table is tan (r>>b)
    ch=im[:,:,0]-im[:,:,2]
    out=[]
    for c in cols:
        p=ch[rlo:rhi,c-1:c+2].mean(1)
        g=np.diff(p)  # going down: white (b-r ~0) -> tan (b-r negative): g negative
        k=np.argmin(g)
        if 0<k<len(g)-1:
            y0,y1,y2=g[k-1],g[k],g[k+1]; den=y0-2*y1+y2; off=0.5*(y0-y2)/den if den!=0 else 0
        else: off=0
        out.append(rlo+k+0.5+off)
    return np.array(out)
for name,fr in dict(S1=range(0,130,4),S1b=range(140,210,4),P=range(282,310,2),S2=range(465,552,4)).items():
    for cols in [range(740,772,2)]:
        e=np.array([edge_rows(i,cols) for i in fr])
        print(name,'bottom edge row median %.2f  per-col mean %s'%(np.median(e),np.round(np.median(e,0),1)))

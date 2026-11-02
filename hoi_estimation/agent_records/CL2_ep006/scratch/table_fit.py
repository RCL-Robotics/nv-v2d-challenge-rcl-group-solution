import cv2, numpy as np, json
from scipy.optimize import least_squares
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
pl=json.load(open('/work/inputs/ground_plane.json'))['plane']; n=np.array(pl[:3]); nn=np.linalg.norm(n); n=n/nn; d=pl[3]/nn
cf=np.load('caster_center.npy')
e1=np.cross(n,[0,0,1.]); e1/=np.linalg.norm(e1); e2=np.cross(n,e1)
def proj(P):
    pc=P@Tcw[:3,:3].T+Tcw[:3,3]; return (pc[:,:2]/pc[:,2:])*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
def circle(H,R,dx,dz,N=720):
    a=np.linspace(0,2*np.pi,N,endpoint=False)
    c=cf+H*n+dx*e1+dz*e2
    return c+R*(np.cos(a)[:,None]*e1+np.sin(a)[:,None]*e2)
pts_up=[];pts_lo=[]
for f in [0,540]:
    im=cv2.imread(f'frames/{f+1:04d}.jpg'); x0,y0=680,620; c=im[y0:800,x0:1080]
    hsv=cv2.cvtColor(c,cv2.COLOR_BGR2HSV)
    m=((hsv[...,0]>8)&(hsv[...,0]<30)&(hsv[...,1]>40)&(hsv[...,2]>120)).astype(np.uint8)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    nl,lab,st,_=cv2.connectedComponentsWithStats(m); k=1+np.argmax(st[1:,4]); m=(lab==k).astype(np.uint8)
    # per column: topmost & bottommost wood pixel
    cols=np.nonzero(m.any(0))[0]
    for x in cols[3:-3]:
        ys=np.nonzero(m[:,x])[0]; pts_up.append((x+x0,ys.min()+y0,f)); pts_lo.append((x+x0,ys.max()+y0,f))
pts_up=np.array(pts_up,float); pts_lo=np.array(pts_lo,float)
def resid(p, pts, H_off=0.0):
    H,R,dx,dz=p[:4]
    uv=proj(circle(H+H_off,R,dx,dz))
    # for each point, distance to curve (vertical distance at same column on the relevant arc)
    out=[]
    for x,y,_ in pts:
        dd=np.hypot(uv[:,0]-x,uv[:,1]-y); out.append(dd.min())
    return np.array(out)
def fun(p,mode):
    r=resid(p,pts_up)
    if mode=='both': r=np.r_[r,resid(p,pts_lo,-p[4])]
    return r
for mode in ['up','both']:
    p0=[0.85,0.38,0,0,0.02]
    sol=least_squares(fun,p0 if mode=='both' else p0[:4],args=(mode,),loss='soft_l1',f_scale=1.5)
    print(mode,'H,R,dx,dz,(t)',sol.x.round(4),'rms',np.sqrt(np.mean(sol.fun**2)).round(2), 'median',np.median(np.abs(sol.fun)).round(2))
    np.save(f'table_{mode}.npy',sol.x)
sol2=least_squares(lambda q: resid([q[0],q[1],0,0],pts_up),[0.85,0.38],loss='soft_l1',f_scale=1.5)
print('centered fit H,R',sol2.x.round(4), np.median(np.abs(sol2.fun)).round(2))
p=np.load('table_up.npy')
for f in [0,540]:
    im=cv2.imread(f'frames/{f+1:04d}.jpg')
    for pp,col in [(p,(0,0,255)),([sol2.x[0],sol2.x[1],0,0],(0,255,0))]:
        uv=proj(circle(*pp)).astype(np.int32); cv2.polylines(im,[uv],True,col,1)
    for x,y,ff in pts_up:
        if ff==f: im[int(y),int(x)]=(255,0,0)
    cv2.imwrite(f'viz/tablefit{f}.jpg',cv2.resize(im[620:800,680:1080],None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST))

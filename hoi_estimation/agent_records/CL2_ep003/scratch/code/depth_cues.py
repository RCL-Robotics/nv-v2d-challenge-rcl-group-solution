import sys,json; sys.path.insert(0,'/work/outputs')
from lib import *
m,v,f,uv,tex=load_mesh('/work/inputs/object_mesh.glb'); cam,K,Twc,rec=load_inputs('/work/inputs'); c0,u=body_axis(v)
Tcw=np.linalg.inv(Twc); B=Tcw[None]@rec['T_world_mesh'].astype(np.float64)
Kp=np.array(cam['K_pipeline'])
res=np.load('v1/object_poses_world.npz')['T_world_mesh']; V1=Tcw[None]@res
log=json.load(open('v1/object_poses_world_log.json'))
hb=((v-c0)@u).min(); bot=c0+u*hb
f5=h5py.File('/work/inputs/masks.h5'); fd=h5py.File('/pipeline/depth/moge2/aligned.h5')
A=fd['aligned/front_stereo_camera_left']
kp=np.einsum('ij,fkj->fki',Tcw[:3,:3],rec['human_keypoints_world'].astype(np.float64))+Tcw[:3,3]
cp=rec['contact_prob_right_left'].max(1)
hv=v[v[:,0]>0.035].mean(0)
def backproj(us,vs,z,KK): return np.stack([(us-KK[0,2])/KK[0,0]*z,(vs-KK[1,2])/KK[1,1]*z,z],-1)
for k,s in enumerate(log['static']):
    a,b=s['phase']; T=V1[a]
    zc=(T[:3,:3]@c0+T[:3,3])[2]
    pb=T[:3,:3]@bot+T[:3,3]  # bottom centre (camera), at anchored depth
    zt=[];zo=[]
    for i in range(a,b,3):
        d=cv2.imdecode(A[i],cv2.IMREAD_UNCHANGED).astype(np.float32)/1000
        om=f5['object'][i]; pm=f5['person'][i]
        ys,xs=np.nonzero(om); yb=ys.max(); xc=xs.mean()
        # table points: rows yb+3..yb+40, cols xc-60..xc+60, not person/object
        Y,X=np.mgrid[yb+3:yb+40,int(xc)-60:int(xc)+60]
        ok=~pm[Y,X]&~om[Y,X]
        P=backproj(X[ok].astype(float),Y[ok].astype(float),d[Y,X][ok],Kp)
        # plane fit (least squares, z = a x + b y + c robustified)
        Am=np.c_[P[:,0],P[:,1],np.ones(len(P))]; coef=np.linalg.lstsq(Am,P[:,2],rcond=None)[0]
        for _ in range(3):
            r=np.abs(Am@coef-P[:,2]); keep=r<np.percentile(r,80); coef=np.linalg.lstsq(Am[keep],P[keep,2],rcond=None)[0]
        # ray through bottom centre pixel (in K_pipeline pixel coords use the same pixel)
        pix=K@pb; pix=pix[:2]/pix[2]
        ray=np.array([(pix[0]-Kp[0,2])/Kp[0,0],(pix[1]-Kp[1,2])/Kp[1,1],1.0])
        # z = a*x+b*y+c with x=ray0*z,y=ray1*z
        z=coef[2]/(1-coef[0]*ray[0]-coef[1]*ray[1]); zt.append(z)
        # moge on object: median of object pixels (front surface) + radius 4cm
        zo.append(np.median(d[om&~pm]))
    z_table_centre=np.median(zt)+ (zc-pb[2])   # centre vs bottom-centre depth offset of the pose
    z_obj=np.median(zo)+0.038
    # hand cue: contact frames within phase +-10 frames
    fr=[i for i in range(max(0,a-10),min(len(B),b+10)) if cp[i]>0.5]
    zh=None
    if fr:
        best=None
        for sft in np.arange(-0.3,0.3,0.005):
            dd=[]
            for i in fr:
                Ti=V1[i] if a<=i<b else T
                cen=Ti[:3,:3]@c0+Ti[:3,3]; sc=(cen[2]+sft)/cen[2]
                hc=(Ti[:3,:3]@hv+Ti[:3,3])+cen*(sc-1)
                hands=[kp[i,21:42].mean(0),kp[i,42:63].mean(0)]; hsel=min(hands,key=lambda h:np.linalg.norm(h-cen))
                dd.append(np.linalg.norm(hc-hsel))
            if best is None or np.mean(dd)<best[0]: best=(np.mean(dd),sft)
        zh=best
    print('phase',a,b,'z_base(anchored) %.3f'%zc,'| table %.3f (%+.1fcm)'%(z_table_centre,100*(z_table_centre-zc)),
          '| moge_obj %.3f (%+.1fcm)'%(z_obj,100*(z_obj-zc)),'| img %+.1fcm'%s['image_fit_dz_cm'],
          '| hand shift %+.1fcm (resid %.1fcm, n=%d)'%(100*zh[1],100*zh[0],len(fr)) if zh else '')

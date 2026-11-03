import sys; sys.path.insert(0, '/work/outputs')
from common import *
poses=sys.argv[1]; f=int(sys.argv[2]); spins=[float(x) for x in sys.argv[3].split(',')]; out=sys.argv[4]
D = load_inputs('/work/inputs'); mesh = load_mesh('/work/inputs')
Tw = D['rec']['T_world_mesh'].astype(np.float64) if poses == 'base' else np.load(poses)['T_world_mesh']
Tc = np.linalg.inv(D['Twc']) @ Tw
R = Renderer(mesh, D['K'], 1152, 1536)
ax_p = np.array([0.0007, 0.0137, 0.0])
def spinT(T, deg):
    S = np.eye(4); S[:3, :3] = axis_angle([0, 0, 1], np.radians(deg)); S[:3, 3] = ax_p - S[:3, :3] @ ax_p
    return T @ S
cap = cv2.VideoCapture('/work/inputs/video.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read(); im=im[:,:,::-1]
# centre of crop: projection of underside centre
zc=float(sys.argv[6]) if len(sys.argv)>6 else -0.08
p = Tc[f][:3,:3]@np.array([0.006,0.004,zc])+Tc[f][:3,3]; q=D['K']@p; u,v=int(q[0]/q[2]),int(q[1]/q[2])
s=int(sys.argv[5]) if len(sys.argv)>5 else 180
tiles=[im[v-s:v+s,u-s:u+s]]
o=R.render(np.stack([spinT(Tc[f],d) for d in spins]))
for i,d in enumerate(spins):
    rgb=(o['rgb'][i].cpu().numpy()*255).astype(np.uint8); m=o['mask'][i].cpu().numpy()
    c=(im*0.3).astype(np.uint8); c[m]=rgb[m]; c=c[v-s:v+s,u-s:u+s].copy(); cv2.putText(c,str(d),(5,25),0,0.8,(255,255,0),2); tiles.append(c)
cv2.imwrite(out,np.concatenate(tiles,1)[:,:,::-1])

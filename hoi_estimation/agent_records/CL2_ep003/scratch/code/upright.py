m,v,f,uv,tex=load_mesh('/work/inputs/object_mesh.glb')
cam,K,Twc,rec=load_inputs('/work/inputs'); c0,u=body_axis(v)
Tc=np.linalg.inv(Twc)[None]@rec['T_world_mesh'].astype(np.float64)
D=dict(np.load('/work/scratch/crops160.npz')); R=Renderer(v,f,uv,tex)
gp=json.load(open('/work/inputs/ground_plane.json'))['plane']; nw=np.array(gp[:3])
nc=np.linalg.inv(Twc)[:3,:3]@nw; nc/=np.linalg.norm(nc)
if nc[1]>0: nc=-nc
def upright(T):
    a=T[:3,:3]@u; ax=np.cross(a,nc); s=np.linalg.norm(ax); ang=np.arcsin(min(s,1))
    Rm=rot_about(ax/s,[ang])[0] if s>1e-9 else np.eye(3)
    out=T.copy(); out[:3,:3]=Rm@T[:3,:3]; cc=T[:3,:3]@c0+T[:3,3]; out[:3,3]=cc-out[:3,:3]@c0; return out

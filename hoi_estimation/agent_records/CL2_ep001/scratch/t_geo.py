exec(open('/work/scratch/common_exp.py').read())
import json
pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']); n,d=pl[:3],pl[3]
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=np.asarray(m.vertices)
def info(Tc,tag):
    Tw=Twc@Tc; X=V@Tw[:3,:3].T+Tw[:3,3]; s=X@n+d
    ax=Tc[:3,:3]@np.array([0,0,1.])
    print(tag,'floor signed dist min %.3f (pctl1 %.3f)'%(s.min(),np.percentile(s,0.5)),'axis_cam',np.round(ax,3),'z %.3f'%Tc[2,3])
for i in [0,60,100]: info(Tb[i],f'base{i}')
info(np.load('scratch/static_0_base_0.0.npy'),'ref0'); info(np.load('scratch/static_0_base_0.5.npy'),'ref0c')
for i in [660,700]: info(Tb[i],f'base{i}')
info(np.load('scratch/static_640_base_0.0.npy'),'ref640')
# sign: which side is "above"? check human feet
hv=rec['human_vertices_world'][0]; print('human min/max dist',(hv@n+d).min(),(hv@n+d).max())

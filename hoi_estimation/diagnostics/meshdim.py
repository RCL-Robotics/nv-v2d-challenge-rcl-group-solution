import trimesh,numpy as np
for n in ['round_table','tall_bar_stool','skinny_wood_chair','white_desk']:
    m=trimesh.load(f'/mnt/nas/Group/hanzhou/v2d/dataset/track_2/tier_1_multiview_caption/mesh/{n}/{n}.glb',force='scene',process=False).to_geometry()
    v=np.asarray(m.vertices); print(n,'min',v.min(0).round(3),'max',v.max(0).round(3),'nv',len(v),'nf',len(m.faces))

import os, trimesh,numpy as np,cv2
m=trimesh.load(os.environ["V2D_ROOT"] + '/dataset/track_2/tier_1_multiview_caption/mesh/tall_bar_stool/tall_bar_stool.glb',force='scene',process=False).to_geometry()
v=np.asarray(m.vertices); out=[]
for a,b in [(0,2),(1,2),(0,1)]:
    im=np.zeros((500,500),np.uint8); p=v[:,[a,b]]; p=(p-p.min(0))/(p.max(0)-p.min(0)).max()*480+10
    for x,y in p[::3].astype(int): im[499-y if b==2 else y,x]=255
    out.append(im)
cv2.imwrite(os.environ["V2D_ROOT"] + '/support/vis/dbg_stool_mesh.png',np.concatenate(out,1))

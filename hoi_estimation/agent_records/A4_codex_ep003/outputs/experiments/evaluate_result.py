import json,sys,numpy as np,torch,cv2
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw
from fit_poses import data,loss_fn
from render_utils import Renderer
p='/work/inputs';result=sys.argv[1] if len(sys.argv)>1 else '/work/outputs/object_poses_world.npz';cam,base,T,orig,img,masks,people=data(p);K=np.array(cam['K']);W=np.array(cam['T_world_cam']);final=np.load(result)['T_world_mesh'];Tc=np.linalg.inv(W)@final;ren=Renderer(p+'/object_mesh.glb',K);F=len(T);details=[]
with torch.no_grad():
 for s in range(0,F,24):
  e=min(F,s+24);m=torch.tensor(masks[s:e],device='cuda');person=torch.tensor(people[s:e],device='cuda');vis=(~person)|m
  stats=[]
  for A in [T,Tc]:
   rgb,mask=ren.render(A[s:e,:3,:3],A[s:e,:3,3],orig[s:e]);pred=mask[...,0]>.5
   iou=((pred&m)&vis).sum((1,2))/(((pred|m)&vis).sum((1,2)))
   stats.append(iou.cpu().numpy())
  details.extend(np.stack(stats,-1))
details=np.array(details)
steps=lambda R:Rotation.from_matrix(R[1:]@R[:-1].transpose(0,2,1)).magnitude()*180/np.pi
bsteps=steps(T[:,:3,:3]);fsteps=steps(Tc[:,:3,:3]);change=Rotation.from_matrix(Tc[:,:3,:3]@T[:,:3,:3].transpose(0,2,1)).magnitude()*180/np.pi
uv=lambda t:(t@K.T)[:,:2]/(t@K.T)[:,2:]
summary={'frames':F,'dtype':str(final.dtype),'all_finite':bool(np.isfinite(final).all()),'camera_distance_max_change_m':float(np.max(np.abs(np.linalg.norm(T[:,:3,3],axis=-1)-np.linalg.norm(Tc[:,:3,3],axis=-1)))),'rotation_determinant_range':[float(np.linalg.det(final[:,:3,:3]).min()),float(np.linalg.det(final[:,:3,:3]).max())],'rotation_orthogonality_max':float(np.max(np.abs(final[:,:3,:3]@final[:,:3,:3].transpose(0,2,1)-np.eye(3)))),'rotation_change_deg_percentiles':np.percentile(change,[0,25,50,75,95,100]).tolist(),'pixel_centre_change_percentiles':np.percentile(np.linalg.norm(uv(Tc[:,:3,3])-uv(T[:,:3,3]),axis=-1),[0,25,50,75,95,100]).tolist(),'baseline_rotation_step_deg_percentiles':np.percentile(bsteps,[50,90,95,99,100]).tolist(),'final_rotation_step_deg_percentiles':np.percentile(fsteps,[50,90,95,99,100]).tolist(),'baseline_steps_over_45':int((bsteps>45).sum()),'final_steps_over_45':int((fsteps>45).sum()),'baseline_visible_mask_IoU_mean':float(details[:,0].mean()),'final_visible_mask_IoU_mean':float(details[:,1].mean()),'mask_iou_improved_frames':int((details[:,1]>details[:,0]).sum())}
json.dump(summary,open('/work/outputs/validation.json','w'),indent=2);print(json.dumps(summary,indent=2));np.savez('/work/outputs/mask_validation.npz',iou=details)
idx=[0,80,160,240,300,420,440,560,640,720];out=Image.new('RGB',(4*256,len(idx)*280),'white');d=ImageDraw.Draw(out)
for n,f in enumerate(idx):
 d.text((0,n*280+3),str(f),fill='black')
 with torch.no_grad():rgb,a=ren.render(np.stack([T[f,:3,:3],Tc[f,:3,:3]]),np.stack([T[f,:3,3],Tc[f,:3,3]]),orig[f])
 colors=rgb.cpu().numpy();alpha=a.cpu().numpy();views=[img[f]]
 for j in range(2):views.append(img[f]*(1-.65*alpha[j])+.65*colors[j])
 views.append(colors[1]+.15*(1-alpha[1]))
 for j,v in enumerate(views):out.paste(Image.fromarray(np.uint8(np.clip(v*255,0,255))).resize((256,256)),(j*256,n*280+24));d.text((j*256+25,n*280+3),['Video','Baseline overlay','Final overlay','Final render'][j],fill='black')
out.save('/work/outputs/comparison.jpg')

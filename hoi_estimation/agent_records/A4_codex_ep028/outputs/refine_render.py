import os,time,json,argparse
os.environ.setdefault('OMP_NUM_THREADS','4')
import cv2,h5py,trimesh,numpy as np,torch,nvdiffrast.torch as dr
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)

def refine(inputs, initfile, outfile, start=660, steps=160):
 timer=time.time();p=Path(inputs);a=np.load(p/'reconstruction.npz');c=json.load(open(p/'camera.json'));K=np.array(c['K']);W=np.array(c['T_world_cam']);base=np.linalg.inv(W)@a['T_world_mesh'];data=np.load(initfile);init=data['T_cam_mesh'];tracks=np.load('/work/outputs/feature_tracks_full.npz');head=tracks['head'];tip=tracks['tip'];m=trimesh.load(p/'object_mesh.glb',force='scene',process=False).to_geometry();fullv=np.array(m.vertices);fullcolor=m.visual.to_color().vertex_colors[:,:3]/255.;m.merge_vertices(merge_tex=True,merge_norm=True);m=m.simplify_quadric_decimation(face_count=12000);v=np.asarray(m.vertices);f=np.asarray(m.faces);colors=fullcolor[cKDTree(fullv).query(v)[1]];ctx=dr.RasterizeCudaContext();device='cuda';ft=torch.tensor(f.astype('int32'),device=device);vt=torch.tensor(v.astype('float32'),device=device);ct=torch.tensor(colors.astype('float32'),device=device);cap=cv2.VideoCapture(str(p/'video.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,start);masks=h5py.File(p/'masks.h5');N=len(base);S=256;ims=[];ob=[];per=[];dists=[];boxes=[];orange=[];orange_dt=[];surface=[]
 for i in range(start,N):
  _,im=cap.read();om=masks['object'][i].astype('uint8');pm=masks['person'][i].astype('uint8');yy,xx=np.where(om);uv=(v@init[i,:3,:3].T+init[i,:3,3])@K.T;uv=uv[:,:2]/uv[:,2:];lo=np.minimum([xx.min(),yy.min()],uv.min(0));hi=np.maximum([xx.max(),yy.max()],uv.max(0));center=(lo+hi)/2;size=max(hi-lo)+80;size=min(size,900);left=int(center[0]-size/2);top=int(center[1]-size/2);sz=int(size);boxes.append([left,top,sz]);
  def crop(x,interp):
   M=np.array([[S/sz,0,-left*S/sz],[0,S/sz,-top*S/sz]],float);return cv2.warpAffine(x,M,(S,S),flags=interp)
  oo=crop(om,cv2.INTER_NEAREST);pp=crop(pm,cv2.INTER_NEAREST);rgb=crop(im,cv2.INTER_LINEAR);ims.append(rgb[:,:,::-1].copy()/255.);ob.append(oo);per.append(pp);allow=oo|pp;dt=cv2.distanceTransform((1-allow).astype('uint8'),cv2.DIST_L2,3);dists.append(np.minimum(dt,30));hsv=cv2.cvtColor(rgb,cv2.COLOR_BGR2HSV);oo2=((hsv[:,:,0]<24)&(hsv[:,:,1]>130)&(hsv[:,:,2]>60))&(cv2.dilate(oo,np.ones((3,3),np.uint8))>0);orange.append(oo2);orange_dt.append(np.minimum(cv2.distanceTransform((1-oo2).astype('uint8'),cv2.DIST_L2,3),30))
 print('prepared',round(time.time()-timer,2),flush=True)
 B=N-start;boxes=np.array(boxes);obs=torch.tensor(np.array(ob),device=device,dtype=torch.float32);persons=torch.tensor(np.array(per),device=device,dtype=torch.float32);dt=torch.tensor(np.array(dists),device=device,dtype=torch.float32);odt=torch.tensor(np.array(orange_dt),device=device,dtype=torch.float32);rgb=torch.tensor(np.array(ims),device=device,dtype=torch.float32);rt=torch.tensor(init[start:,:3,:3],device=device,dtype=torch.float32);tt=torch.tensor(init[start:,:3,3],device=device,dtype=torch.float32);rr=torch.tensor(np.linalg.norm(base[start:,:3,3],axis=1),device=device,dtype=torch.float32);bt=torch.tensor(boxes,device=device,dtype=torch.float32);delta=torch.nn.Parameter(torch.zeros((B,3),device=device));uv0=tt[:,:2]/tt[:,2:];duv=torch.nn.Parameter(torch.zeros((B,2),device=device));opt=torch.optim.Adam([{'params':[delta],'lr':.007},{'params':[duv],'lr':.001}]);hpt=np.array([-.00386614,-.0140458,-.44275588]);tpt=np.array([.00168146,.01075304,.99291701]);lpts=torch.tensor(np.stack([hpt,tpt]),device=device,dtype=torch.float32);lobs=np.stack([head[start:],tip[start:]],axis=1);lvalid=np.isfinite(lobs[:,:,0]);lobs=np.nan_to_num(lobs);lobs_t=torch.tensor(lobs,device=device,dtype=torch.float32);valid_t=torch.tensor(lvalid,device=device,dtype=torch.float32);objarea=obs.sum((1,2)).clamp(min=30);losslog=[]
 def poses():
  x,y,z=delta.unbind(-1);zero=torch.zeros_like(x);skew=torch.stack([zero,-z,y,z,zero,-x,-y,x,zero],-1).reshape(-1,3,3);rot=torch.matrix_exp(skew)@rt;ray=torch.cat([uv0+duv,torch.ones((B,1),device=device)],1);trans=ray/torch.linalg.norm(ray,dim=1,keepdim=True)*rr[:,None];return rot,trans
 for step in range(steps):
  opt.zero_grad();total=0.
  for j in range(0,B,16):
   k=min(j+16,B);rot,trans=poses();rot=rot[j:k];trans=trans[j:k];xyz=torch.einsum('bij,vj->bvi',rot,vt)+trans[:,None];z=xyz[:,:,2];u=K[0,0]*xyz[:,:,0]+K[0,2]*z;vv=K[1,1]*xyz[:,:,1]+K[1,2]*z;box=bt[j:k];clip=torch.stack([2*(u-box[:,None,0]*z)/box[:,None,2]-z,2*(vv-box[:,None,1]*z)/box[:,None,2]-z,.99*z-.01,z],-1);rast,_=dr.rasterize(ctx,clip,ft,resolution=[S,S]);alpha=(rast[:,:,:,3:]>0).float();alpha=dr.antialias(alpha,rast,clip,ft)[:,:,:,0];co,_=dr.interpolate(ct[None].expand(k-j,-1,-1).contiguous(),rast,ft);co=dr.antialias(co,rast,clip,ft);oo=obs[j:k];pp=persons[j:k];vis=alpha*(1-pp*(1-oo));
   # Visible silhouette plus occlusion-aware outside distance.
   miss=((1-alpha)*oo).sum((1,2))/objarea[j:k];extra=(vis*(1-oo)).sum((1,2))/objarea[j:k];outside=(vis*dt[j:k]).sum((1,2))/objarea[j:k];sil=1.6*miss+.75*extra+.25*outside
   # Hue cue discourages rendering the black blade at the orange tip.
   predorange=(co[:,:,:,0]-co[:,:,:,1]*1.18).clamp(0)*vis;closs=(predorange*odt[j:k]).sum((1,2))/objarea[j:k];
   lp=torch.einsum('bij,kj->bki',rot,lpts)+trans[:,None];puv=torch.stack([K[0,0]*lp[:,:,0]/lp[:,:,2]+K[0,2],K[1,1]*lp[:,:,1]/lp[:,:,2]+K[1,2]],-1);err=torch.linalg.norm(puv-lobs_t[j:k],dim=-1);land=(torch.nn.functional.huber_loss(err,torch.zeros_like(err),delta=4,reduction='none')*valid_t[j:k]).sum(1)*.008;
   loss=(sil+.6*closs+land).sum()/B;loss.backward();total+=float(loss.detach())
  rot,trans=poses();reg=2.*delta.square().sum(1).mean()+30*duv.square().sum(1).mean();smooth=25*(rot[2:]-2*rot[1:-1]+rot[:-2]).square().mean()+150*(trans[2:]-2*trans[1:-1]+trans[:-2]).square().mean();(reg+smooth).backward();opt.step();losslog.append(total)
  if step%20==0 or step==steps-1:print(step,'loss',round(total,4),'reg',round(float(reg),4),'seconds',round(time.time()-timer,1),flush=True)
 rot,trans=poses();Q=init.copy();Q[start:,:3,:3]=rot.detach().cpu().numpy();Q[start:,:3,3]=trans.detach().cpu().numpy();np.savez(outfile,T_cam_mesh=Q,loss=np.array(losslog));print('wrote',outfile,'seconds',time.time()-timer,flush=True)
if __name__=='__main__':
 pa=argparse.ArgumentParser();pa.add_argument('--steps',type=int,default=160);pa.add_argument('--start',type=int,default=660);pa.add_argument('--init',default='/work/outputs/endpoint_initial.npz');pa.add_argument('--out',default='/work/outputs/refined_candidate.npz');args=pa.parse_args();refine('/work/inputs',args.init,args.out,steps=args.steps,start=args.start)

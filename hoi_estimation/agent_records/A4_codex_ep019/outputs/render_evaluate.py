"""Exact textured-mesh validation, no new pose estimator or external data."""
import numpy as np, cv2, torch, trimesh, nvdiffrast.torch as dr, json, h5py, time
class Evaluator:
 def __init__(self,inputs='/work/inputs'):
  self.inputs=inputs;m=trimesh.load(inputs+'/object_mesh.glb',force='scene',process=False).to_geometry();m.merge_vertices();device='cuda';self.v=torch.tensor(np.array(m.vertices),dtype=torch.float32,device=device);self.faces=torch.tensor(m.faces.astype('int32'),device=device);self.uv=torch.tensor(np.array(m.visual.uv),dtype=torch.float32,device=device);self.tex=torch.tensor(np.array(m.visual.material.baseColorTexture.convert('RGB'))[::-1].copy()/255,dtype=torch.float32,device=device)[None];self.ctx=dr.RasterizeCudaContext();c=json.load(open(inputs+'/camera.json'));self.K=np.array(c['K']);self.kt=torch.tensor(self.K,dtype=torch.float32,device=device);self.vid=cv2.VideoCapture(inputs+'/video.mp4');self.h=h5py.File(inputs+'/masks.h5');a=np.load(inputs+'/reconstruction.npz');self.base=np.linalg.inv(c['T_world_cam'])[None]@a['T_world_mesh']
 def frame(self,i,poses,save=None):
  self.vid.set(1,int(i));_,im=self.vid.read();mask=np.array(self.h['object'][i],np.uint8);person=np.array(self.h['person'][i],np.uint8);yy,xx=np.where(mask);size=max(xx.max()-xx.min()+1,yy.max()-yy.min()+1)+80;cx=(xx.max()+xx.min())/2;cy=(yy.max()+yy.min())/2;scale=320/size;x0=cx-size/2;y0=cy-size/2;A=np.array([[scale,0,-x0*scale],[0,scale,-y0*scale]]);S=320
  obs=cv2.warpAffine(mask,A,(S,S),flags=cv2.INTER_NEAREST);occ=cv2.warpAffine(person,A,(S,S),flags=cv2.INTER_NEAREST);image=cv2.warpAffine(im,A,(S,S));hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);dil=cv2.dilate(obs,np.ones((3,3),np.uint8));orange=((hsv[:,:,0]>3)&(hsv[:,:,0]<28)&(hsv[:,:,1]>110)&(hsv[:,:,2]>70)&(dil>0)).astype('uint8');occ=occ*(1-dil);dt=cv2.distanceTransform(1-obs,cv2.DIST_L2,5);odt=cv2.distanceTransform(1-orange,cv2.DIST_L2,5)
  Rs=torch.tensor(poses[:,:3,:3],dtype=torch.float32,device='cuda');ts=torch.tensor(poses[:,:3,3],dtype=torch.float32,device='cuda');q=torch.einsum('bij,nj->bni',Rs,self.v)+ts[:,None];xy=q@self.kt.T;xy=xy[...,:2]/xy[...,2:];xy=(xy-torch.tensor([x0,y0],device='cuda',dtype=torch.float32))*scale;z=q[...,2];near=.1;far=10.;clip=torch.stack([(2*xy[...,0]/S-1)*z,(2*xy[...,1]/S-1)*z,((far+near)/(far-near))*z-2*far*near/(far-near),z],-1)
  with torch.no_grad():
   rast,_=dr.rasterize(self.ctx,clip,self.faces,[S,S]);uv,_=dr.interpolate(self.uv[None],rast,self.faces);rgb=dr.texture(self.tex,uv,filter_mode='linear');pred=(rast[...,3]>0).cpu().numpy();rgb=(rgb.cpu().numpy()*255).astype('uint8')
  vals=[];tiles=[]
  for j in range(len(poses)):
   pr=pred[j].astype('uint8');vis=pr*(1-occ);pdt=cv2.distanceTransform(1-pr,cv2.DIST_L2,5);rh=cv2.cvtColor(rgb[j],cv2.COLOR_RGB2HSV);po=((rh[:,:,0]>3)&(rh[:,:,0]<28)&(rh[:,:,1]>90)&(rh[:,:,2]>65)&(pr>0)).astype('uint8');podt=cv2.distanceTransform(1-po,cv2.DIST_L2,5)
   def mean(v,m):return float(np.sum(v*m)/max(1,np.sum(m)))
   # Distances have a 1 px noise allowance in source-image units, then robust linear tails.
   def robust(x):return np.sqrt((x/scale)**2+1)-1
   fw=mean(robust(dt),vis);bw=mean(robust(pdt),obs);ofw=mean(robust(odt),po*(1-occ));obw=mean(robust(podt),orange);intersection=np.sum(vis*obs);iou=intersection/max(1,np.sum(vis|obs));cost=.45*fw+.65*bw+.65*ofw+.9*obw;vals.append([cost,fw,bw,ofw,obw,iou,float(vis.sum()/max(1,pr.sum()))])
   if save:
    rr=rgb[j][:,:,::-1].copy();rr[pr==0]=245;combo=np.concatenate([image,rr],1);cv2.putText(combo,f'{i} cand {j}: {cost:.2f} IoU {iou:.2f}',(4,20),0,.5,(10,40,255),1);tiles.append(combo)
  if save:cv2.imwrite(save,np.concatenate(tiles,0))
  return np.array(vals)
def main(path,out):
 start=time.time();a=np.load(path);ev=Evaluator();allposes=[];metrics=[]
 for idx,i in enumerate(a['indices']):
  cp=np.zeros((a['candidates'].shape[1]+1,4,4));cp[:-1,:3,:]=a['candidates'][idx];cp[:,3,3]=1;cp[-1]=ev.base[i];values=ev.frame(i,cp);allposes.append(cp);metrics.append(values)
  if idx%25==0:print('render',i,'elapsed',round(time.time()-start,1),'best/base',values[:,0].min(),values[-1,0],flush=True)
 np.savez(out,indices=a['indices'],poses=np.array(allposes),metrics=np.array(metrics));print('render evaluation seconds',time.time()-start)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('out');args=p.parse_args();main(args.path,args.out)

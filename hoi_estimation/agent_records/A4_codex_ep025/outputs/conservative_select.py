import numpy as np
from scipy.ndimage import binary_dilation

def angle(A,B):
 return np.arccos(np.clip((np.einsum('...ij,...ij->...',A,B)-1)/2,-1,1))

def choose(c,K):
 R=c['R'];t=c['t'];l=c['scores'];stats=c['stats'];masks=c['masks'];F,N=l.shape
 uv=t[...,:2]/t[...,2:]*[K[0,0],K[1,1]]+K[:2,2];ang=angle(R,R[:,0,None]);axis=np.arccos(np.clip(np.einsum('fni,fi->fn',R[:,:,:,2],R[:,0,:,2]),-1,1));shift=np.linalg.norm(uv-uv[:,0,None],axis=-1)
 pca=np.zeros(F);elong=np.zeros(F)
 for f in range(F):
  y,x=np.where(masks[f]);cov=np.cov(np.stack([x,y]));val,vec=np.linalg.eigh(cov);a=vec[:,-1];p=R[f,0,:,2][None]*np.array([-.1,.1])[:,None]+t[f,0];q=p@K.T;q=q[:,:2]/q[:,2:];d=q[1]-q[0];pca[f]=np.degrees(np.arccos(np.clip(abs(a@d)/np.linalg.norm(d),0,1)));elong[f]=val[1]/max(val[0],1)
 # Silhouette size alone cannot establish out-of-plane tilt at fixed distance.
 # Require an independent projected-axis error, or gross silhouette failure.
 evidence=((pca>12)&(elong>2.2))|((stats[:,0,0]<.55)&(elong>2.2))
 if 'blue_iou' in c:
  bi=c['blue_iou'];bcount=c['blue'].sum((1,2));color_evidence=(bcount>80)&(bi[:,0]<.15)&np.any((bi>.50)&(stats[:,:,0]>.72),axis=1)
  evidence|=color_evidence
 eligible=binary_dilation(evidence,iterations=3)
 gain=l[:,0,None]-l
 valid=(gain>.10)&(stats[:,:,0]>.63)&(stats[:,:,0]>stats[:,0,None,0]+.05)&eligible[:,None]
 valid&=(axis<np.deg2rad(110))|((gain>.40)&(stats[:,:,0]>.82))
 if 'blue_iou' in c:
  # A plainly visible latch rules out rotations that hide it. The silhouette
  # cannot override this independent material cue for an almost round bottle.
  strong=(bcount>80)&np.any((bi>.50)&(stats[:,:,0]>.72),axis=1)
  threshold=np.maximum(.35,np.max(np.where(stats[:,:,0]>.72,bi,0),axis=1)-.25)
  valid&=(~strong[:,None])|(bi>=threshold[:,None])
 valid[:,0]=True
 unary=l+.045*np.minimum(ang**2,5)+.010*(shift/3)**2
 unary=np.where(valid,unary,1e4)
 dp=unary[0].copy();back=np.zeros((F,N),int)
 for i in range(1,F):
  rot=angle(R[i-1,:,None],R[i,None]);ax=np.arccos(np.clip(R[i-1,:,:,2]@R[i,:,:,2].T,-1,1))
  # Cap temporal penalties: an uncorrectable isolated frame must not prevent
  # correction of a neighboring frame with decisive independent image evidence.
  temporal=.10*np.minimum((rot/.4)**2,3)+.08*np.minimum((ax/.3)**2,3)
  du=uv[i,None]-uv[i-1,:,None];base=uv[i,0]-uv[i-1,0];temporal+=.002*np.minimum(np.sum((du-base)**2,axis=-1)/9,20)
  cost=dp[:,None]+temporal;back[i]=np.argmin(cost,axis=0);dp=unary[i]+cost.min(0)
 chosen=np.zeros(F,int);chosen[-1]=np.argmin(dp)
 for i in range(F-1,0,-1):chosen[i-1]=back[i,chosen[i]]
 return chosen,dict(gain=gain,angle=ang,axis=axis,shift=shift,pca=pca,elong=elong,eligible=eligible)

if __name__=='__main__':
 import json,sys
 p=sys.argv[1] if len(sys.argv)>1 else '/work/outputs/det_candidates.npz';c=np.load(p);data={k:c[k] for k in c.files};K=np.array(json.load(open('/work/inputs/camera.json'))['K']);
 from blue_metrics import compute_blue
 from render_utils import Renderer
 data['blue_iou']=compute_blue(data,Renderer('/work/inputs/object_mesh.glb',K))
 np.save('/work/outputs/candidate_blue_iou.npy',data['blue_iou'])
 ch,meta=choose(data,K);ii=np.arange(len(ch));
 np.savez_compressed('/work/outputs/conservative_selected.npz',chosen=ch,R=data['R'][ii,ch],t=data['t'][ii,ch],**meta);sel=ch>0;print('changed',sel.sum());print('iou',np.mean(data['stats'][:,0,0]),np.mean(data['stats'][ii,ch,0]));print('selected angles',np.percentile(meta['angle'][ii,ch][sel]*180/np.pi,[0,25,50,75,90,100]));changes=np.where(np.diff(np.r_[False,sel,False]))[0].reshape(-1,2);print('intervals',[(a,b-1) for a,b in changes]);
 for f in [0,240,250,275,300,315,400,450,470,540,550,570,590,600,610,630,650,660,670,680,690,700,710,720,740,750,800]:print(f,ch[f],round(data['scores'][f,0],3),round(data['scores'][f,ch[f]],3),round(np.degrees(meta['axis'][f,ch[f]]),1),round(meta['shift'][f,ch[f]],1))

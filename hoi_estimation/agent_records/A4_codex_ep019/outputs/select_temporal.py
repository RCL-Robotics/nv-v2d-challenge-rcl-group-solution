"""Select among actual rendered hypotheses, including the exact baseline."""
import numpy as np,json
from scipy.spatial.transform import Rotation

def select(data,inputs='/work/inputs',out='/work/outputs/selected.npz'):
 a=np.load(data);P=a['poses'];M=a['metrics'];F,C=P.shape[:2]
 c=json.load(open(inputs+'/camera.json'));K=np.array(c['K']);CW=np.linalg.inv(c['T_world_cam']);rec=np.load(inputs+'/reconstruction.npz');hands=rec['human_keypoints_world'][a['indices'],21:63]@CW[:3,:3].T+CW[:3,3];hu=hands@K.T;hu=hu[...,:2]/hu[...,2:];
 handle=np.einsum('fcij,j->fci',P[:,:,:3,:3],np.array([.03,0,.51]))+P[:,:,:3,3];hp=handle@K.T;hp=hp[...,:2]/hp[...,2:];hd=np.linalg.norm(hp[:,:,None]-hu[:,None],axis=-1).min(-1)
 R=P[:,:,:3,:3];base=R[:,-1:];angle=np.arccos(np.clip((np.einsum('fcij,fcij->fc',R,base)-1)/2,-1,1));
 # Weak preference for the released reconstruction when image evidence ties.
 emission=M[:,:,0]+.12*angle**2
 # Observed grasp/swing interval. The video shows the cane held from about frame 110
 # through frame 788. This rejects occlusion-induced swaps of base and handle.
 holding=(np.arange(F)>=110)&(np.arange(F)<=788)
 emission+=holding[:,None]*.06*np.maximum(0,hd-15).clip(0,100)
 # Very small displacement prior, to avoid moving a well-explained baseline.
 emission+=.8*np.linalg.norm(P[:,:,:3,3]-P[:,-1:,:3,3],axis=-1)
 costs=emission[0].copy();back=np.zeros((F,C),int)
 for f in range(1,F):
  # Pose continuity in actual mesh coordinates: roll and handle/base identity matter.
  ang=np.arccos(np.clip((np.einsum('aij,bij->ab',R[f-1],R[f])-1)/2,-1,1))
  trans=np.linalg.norm(P[f-1,:,:3,3][:,None]-P[f,:,:3,3][None],axis=-1)
  # Quadratic for small jitter, linear tails so fast observed swings remain possible.
  rotcost=2.5*np.where(ang<.25,ang**2,.5*ang-.0625)
  tcost=3.*np.where(trans<.04,trans**2/.04,2*trans-.04)
  total=costs[:,None]+rotcost+tcost;back[f]=total.argmin(0);costs=emission[f]+total.min(0)
 ids=np.zeros(F,int);ids[-1]=costs.argmin()
 for f in range(F-1,0,-1):ids[f-1]=back[f,ids[f]]
 selected=P[np.arange(F),ids].copy()
 np.savez(out,T_cam_mesh=selected,indices=np.arange(F),selected=ids,metrics=M[np.arange(F),ids],emission=emission)
 dr=np.degrees((Rotation.from_matrix(selected[:-1,:3,:3]).inv()*Rotation.from_matrix(selected[1:,:3,:3])).magnitude())
 print('selected baseline frames',np.sum(ids==C-1));print('rotation step percentiles',np.percentile(dr,[50,90,95,99,100]));print('large steps',np.where(dr>25)[0]+1);print('render cost baseline',np.mean(M[:,-1,0]),'selected',np.mean(M[np.arange(F),ids,0]));print('median cost baseline/selected',np.median(M[:,-1,0]),np.median(M[np.arange(F),ids,0]));return selected
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('data');p.add_argument('--out',default='/work/outputs/selected.npz');args=p.parse_args();select(args.data,out=args.out)

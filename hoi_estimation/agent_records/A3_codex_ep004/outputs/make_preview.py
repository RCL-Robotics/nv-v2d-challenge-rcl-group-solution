"""Make the visual evidence sheet from provided video/masks and final poses."""
import json
import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

root = Path('/work')
K = np.array(json.loads((root/'inputs/camera.json').read_text())['K'])
W = np.array(json.loads((root/'inputs/camera.json').read_text())['T_world_cam'])
base = np.linalg.inv(W) @ np.load(root/'inputs/reconstruction.npz')['T_world_mesh']
final = np.linalg.inv(W) @ np.load(root/'outputs/object_poses_world.npz')['T_world_mesh']
selected = np.load(root/'outputs/selection_diagnostics.npz')
frames = [0,123,185,203,246,267,287,308,329,339,349,370,390,411,431,452,462,470,472,482,524,555,565,596,606,658,668,802]
canvas = Image.new('RGB', (4*384,7*280+48), 'white')
draw = ImageDraw.Draw(canvas)
draw.text((10,8),'Magenta: baseline axis | Yellow: output axis | Green: object mask | Red: handle end | Blue: barrel end',fill='black')
video = cv2.VideoCapture(str(root/'inputs/video.mp4'))
with h5py.File(root/'inputs/masks.h5') as masks:
 for i,f in enumerate(frames):
  video.set(cv2.CAP_PROP_POS_FRAMES,f); ok,img = video.read()
  mask=masks['object'][f].astype('uint8');y,x=np.where(mask);projections=[]
  for pose in (base[f],final[f]):
   xyz=np.array([[0,0,-.20978296],[0,0,.33304057]])@pose[:3,:3].T+pose[:3,3];uv=xyz@K.T;projections.append(uv[:,:2]/uv[:,2:])
  points=np.concatenate(projections);lower=np.minimum(np.array([x.min(),y.min()]),points.min(0))-20;upper=np.maximum(np.array([x.max(),y.max()]),points.max(0))+20
  centre=(lower+upper)/2;width=max(upper[0]-lower[0],(upper[1]-lower[1])*384/250);height=width*250/384
  x0,y0=np.maximum(0,(centre-[width/2,height/2]).astype(int));x1,y1=np.minimum([1536,1152],(centre+[width/2,height/2]).astype(int))
  contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(img,contours,-1,(0,255,0),1)
  for j,uv in enumerate(projections):
   cv2.line(img,tuple(uv[0].astype(int)),tuple(uv[1].astype(int)),(255,0,255) if j==0 else (0,255,255),1 if j==0 else 2)
  for pt,col in zip(projections[1],[(255,0,0),(0,0,255)]):cv2.circle(img,tuple(pt.astype(int)),3,col,-1)
  crop=Image.fromarray(cv2.cvtColor(img[y0:y1,x0:x1],cv2.COLOR_BGR2RGB)).resize((384,250));xx=(i%4)*384;yy=(i//4)*280+48;canvas.paste(crop,(xx,yy));draw.text((xx+4,yy+254),f'Frame {f}: '+('corrected' if selected['changed'][f] else 'baseline retained'),fill='black')
video.release();canvas.save(root/'outputs/evidence.jpg',quality=90)

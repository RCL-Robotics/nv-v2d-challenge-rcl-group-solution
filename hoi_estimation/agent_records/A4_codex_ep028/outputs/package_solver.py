from pathlib import Path
root=Path('/work/outputs')
header='''#!/usr/bin/env python3
"""Offline, seeded pose correction for the released squeegee episode.
Uses only supplied video, masks, camera, mesh and reconstruction.
No network, ground truth, external checkpoints or cached data required.
The mesh origin is its area centroid (within 4e-9 m); the baseline
Euclidean distance from this center to the camera is fixed per frame.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse, json, time, random, hashlib
from pathlib import Path
import cv2, h5py, numpy as np, trimesh, torch
import nvdiffrast.torch as dr
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d, median_filter

# Orange fitting centroids computed from the supplied mesh texture.
HEAD_POINT = np.array([-.00386614,-.0140458,-.44275588])
TIP_POINT = np.array([.00168146,.01075304,.99291701])

'''
s=(root/'extract_observations.py').read_text();s=s[s.index("p=Path('/work/inputs')"):];s=s.replace("p=Path('/work/inputs')","p=Path(inputs)");s=s[:s.index("json.dump(rec")]+"cap.release();f.close();return rec\n";s=s.replace("if i%200==0:print(i,round(time.time()-t,1),flush=True)","if i%200==0:print('Observations',i,round(time.time()-t,1),flush=True)");header+='def observations(inputs):\n'+''.join(' '+l+'\n' for l in s.splitlines())+'\n'
s=(root/'track_features.py').read_text();s=s[s.index('def extract('):s.index("if __name__")];s=s.replace('def extract(inputs, start=660):','def track_features(inputs, obs, start=660):');s=s.replace("obs=json.load(open('/work/outputs/observations.json'));",'');s=s.replace('return head,tip,quality,lines','cap.release();f.close();return head,tip,quality,lines');header+=s+'\n'
s=(root/'endpoint_initial.py').read_text();s=s[s.index("p='/work/inputs/'"):s.index("np.savez('/work/outputs/endpoint_initial.npz'")];first=s.index('\n');s="p=Path(inputs);a=np.load(p/'reconstruction.npz');c=json.load(open(p/'camera.json'));K=np.array(c['K']);W=np.array(c['T_world_cam']);T=np.linalg.inv(W)@a['T_world_mesh'];head=tracks['head'].copy();tip=tracks['tip'].copy();lines=tracks['blade_line'].copy();N=len(T);start=660\n"+s[first+1:];s+='return Q\n';header+='def endpoint_initial(inputs, tracks):\n'+''.join(' '+l+'\n' for l in s.splitlines())+'\n'
s=(root/'build_full_features.py').read_text();s=s[s.index("p='/work/inputs/'"):s.index("np.savez('/work/outputs/feature_tracks_full.npz'")];first=s.index('\n');s="p=Path(inputs);a=np.load(p/'reconstruction.npz');c=json.load(open(p/'camera.json'));K=np.array(c['K']);base=np.linalg.inv(c['T_world_cam'])@a['T_world_mesh'];head=tracks['head'].copy();tip=tracks['tip'].copy();quality=tracks['quality'].copy();line=tracks['blade_line'].copy();p3=np.array([HEAD_POINT,TIP_POINT])\n"+s[first+1:];s+="return dict(head=head,tip=tip,quality=quality,blade_line=line)\n";header+='def complete_tracks(inputs, obs, tracks):\n'+''.join(' '+l+'\n' for l in s.splitlines())+'\n'
s=(root/'refine_render.py').read_text();s=s[s.index('def refine('):s.index("if __name__")];s=s.replace('def refine(inputs, initfile, outfile, start=660, steps=160):','def refine(inputs, init, tracks, start=660, steps=100):');s=s.replace("data=np.load(initfile);init=data['T_cam_mesh'];tracks=np.load('/work/outputs/feature_tracks_full.npz');",'');s=s.replace("np.savez(outfile,T_cam_mesh=Q,loss=np.array(losslog));print('wrote',outfile,'seconds',time.time()-timer,flush=True)","cap.release();masks.close();return Q,losslog,time.time()-timer");header+=s+'\n'
(root/'solve.py').write_text(header)

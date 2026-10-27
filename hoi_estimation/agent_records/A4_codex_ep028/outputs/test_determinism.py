import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import sys,hashlib,cv2,numpy as np
from pathlib import Path
s=Path('/work/outputs/solve.py').read_text().split("if __name__=='__main__':main()")[0]
s=s.replace('for i in range(start,len(base)):','for i in range(start,min(start+5,len(base))):').replace('for j,i in enumerate(range(start,len(base))):','for j,i in enumerate(range(start,min(start+5,len(base)))):')
s=s.replace('  result=least_squares', "  if i==0:np.savez(sys.argv[1]+'.debug.npz',vertices=vertices,dt=dt,initial=initial,obs=obs,target=target,objective=objective(np.zeros(5)),jac=jac(np.zeros(5)))\n  result=least_squares");ns={'sys':sys};exec(s,ns);cv2.setNumThreads(1);cv2.setRNGSeed(0)
a=np.load('/work/outputs/feature_tracks_full.npz');tr={k:a[k] for k in a.files};init=np.load('/work/outputs/endpoint_initial.npz')['T_cam_mesh'];q,l,t=ns['refine_cpu']('/work/inputs',init,tr);np.savez(sys.argv[1],q=q[:5],loss=l);print(hashlib.sha256(q[:5].tobytes()).hexdigest(),l)

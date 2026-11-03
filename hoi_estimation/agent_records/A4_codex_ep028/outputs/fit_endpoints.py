import numpy as np,json
from scipy.optimize import least_squares
a=np.load('/work/inputs/reconstruction.npz');c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K']);T=np.linalg.inv(c['T_world_cam'])@a['T_world_mesh'];obs=json.load(open('/work/outputs/observations.json'));h=-.44275;e=.992917
for i in [0,550,650,690,700,710,720,725,900,950,987]:
 o=np.array([x for x in obs[i]['orange'] if x[2]>50]);
 if len(o)<2:continue
 if i<710:head=o[np.argmax(o[:,1])];tip=o[np.argmin(o[:,1])]
 else:head=o[np.argmin(o[:,1])];tip=o[np.argmax(o[:,1])]
 rays=np.c_[np.array([head[:2],tip[:2]]),np.ones(2)]@np.linalg.inv(K).T;dist=np.linalg.norm(T[i,:3,3]);
 def fun(z):
  hp,ep=rays*z[:,None];ax=(ep-hp)/(e-h);t=hp-h*ax;return [np.linalg.norm(ep-hp)-(e-h),np.linalg.norm(t)-dist]
 for ini in [[2,3.5],[3.5,2]]:
  r=least_squares(fun,ini,bounds=(.5,6));hp,ep=rays*r.x[:,None];ax=(ep-hp)/(e-h);tt=hp-h*ax;print(i,'cost',r.cost,'axis',ax.round(3),'t',tt.round(3))

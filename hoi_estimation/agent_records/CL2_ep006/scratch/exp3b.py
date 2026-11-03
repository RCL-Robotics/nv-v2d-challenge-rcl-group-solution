import sys, time; sys.path.insert(0,'/work/outputs')
from hoi_fit import *
S=setup(); pm=S['pm']; fm=FreeDepthModel(pm,S['Twc'])
E=Evidence(S); sr=SilRenderer(decimate(S['mesh']),S['K'],S['Twc'],scale=0.5)
F_=len(S['Tb']); base=np.array([fm.params_from_pose(T) for T in S['Tb']]); good=S['good']
fr0=[i for i in range(0,F_,15) if good[i]]
TH,XYZ,CO=fit_frames(sr,E,fm,fr0,np.array([[S['par_base'][i,0]] for i in fr0]),base[fr0,1:],iters=150,fit_th=True)
for k,i in enumerate(fr0): print(i, 'ratio',round(XYZ[k,0,2]/base[i,3],3),'dyaw',round(np.degrees(TH[k,0]-S['par_base'][i,0]),1),'cost',CO[k,0].round(3))

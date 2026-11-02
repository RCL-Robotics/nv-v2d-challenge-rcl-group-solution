import sys, pickle; sys.path.insert(0,'/work/scratch')
from common import *
from bat_lib import Renderer
from scipy.spatial.transform import Rotation as Ro
ep=Episode('/work/inputs'); C=pickle.load(open('/work/scratch/run2/stageA.pkl','rb')); ep.__dict__.update(C['ep'])
FR,simp=get_FR(); Rf=Renderer(ep.mesh)
T=ep.Tcw[None]@np.load('/work/scratch/run2/object_poses_world.npz')['T_world_mesh']
for frames in [list(range(0,110,11)),list(range(710,802,9))]:
    i0=frames[0]; angs=np.arange(0,360,15)
    Ts=[]
    for a in angs:
        Tr=T[i0].copy(); Rr=T[i0,:3,:3]@Ro.from_euler('z',a,degrees=True).as_matrix()
        c=T[i0,:3,:3]@BAT_CENTER+T[i0,:3,3]; Tr[:3,:3]=Rr; Tr[:3,3]=c-Rr@BAT_CENTER; Ts.append(Tr)
    Ts=np.stack(Ts)
    iou=np.zeros(len(angs)); dp=np.zeros(len(angs)); tex=np.zeros(len(angs))
    for i in frames:
        sc=score_frames(ep,FR,[i],[Ts])[0]; iou+=sc[:,1]
        dp+=depth_scores(ep,FR,[i],[Ts])[0]
        Q=np.repeat(Ts[:1,:3,:3],ep.F,0); TT=np.repeat(Ts[:1],ep.F,0)
        a_,S,v=roll_scores(ep,Rf,TT,Q,[i],n_ang=24); tex+=S[0]
    n=len(frames)
    rb=Ro.from_matrix(T[i0,:3,:3].T@ep.Tc_base[i0,:3,:3]).as_rotvec(); print('frames',frames[0],'baseline roll offset %.0f deg'%(np.degrees(rb[2])%360), 'axis part %.1f'%np.degrees(np.linalg.norm(rb[:2])))
    for k,a in enumerate(angs): print('  %3d  iouloss %.4f  depthprof %.4f  tex %.3f'%(a,iou[k]/n,dp[k]/n,tex[k]/n))

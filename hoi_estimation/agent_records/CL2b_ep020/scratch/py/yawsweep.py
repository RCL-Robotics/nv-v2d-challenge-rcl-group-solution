from common import *
S=Scorer(Rd,K)
for i in [int(a) for a in sys.argv[1:]]:
    S.set_frame(frame(i),OM[i],PM[i])
    th=np.radians(np.arange(-180,180,2))
    Ts=np.stack([yaw_pose(Tcm[i],up_cam,center,t) for t in th])
    iou,ncc=S.score(Ts)
    o=np.argsort(-ncc)[:5]
    print(i,'iou@0 %.3f ncc@0 %.3f'%(iou[90],ncc[90]),'best ncc',[(int(np.degrees(th[k])),round(ncc[k],3),round(iou[k],3)) for k in o])
    print('   ncc curve every 10deg', ' '.join('%.2f'%x for x in ncc[::5]))

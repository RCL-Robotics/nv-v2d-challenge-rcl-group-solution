from common import *
S=Scorer(Rd,K)
th=np.radians(np.arange(-180,180,2))
for i in [int(a) for a in sys.argv[1:]]:
    S.set_frame(frame(i),OM[i],PM[i])
    Ts=np.stack([yaw_pose(Tcm[i],up_cam,center,t) for t in th])
    line=f'{i:4d}'
    for kind in ['rgb','dog','dogg','chroma']:
        iou,ncc=S.score_feat(Ts,kind)
        k=np.argmax(ncc); 
        # second best peak outside +-30deg
        d=np.abs((np.degrees(th-th[k])+180)%360-180); k2=np.argmax(np.where(d>30,ncc,-9))
        line+=f' | {kind}: best {np.degrees(th[k]):.0f} ({ncc[k]:.3f}) @0 {ncc[90]:.3f} 2nd {np.degrees(th[k2]):.0f} ({ncc[k2]:.3f})'
    print(line)

import os, sys; sys.argv=['fit.py','3']
src=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'support', 'fit.py')).read()
src=src.replace("    iou_grid = best[0]","    import cv2 as _c\n    R_=render(pose(best[1][0],torch.tensor(best[1][1],device=dev),torch.tensor([best[1][2],best[1][3]],device=dev))[0]).cpu().numpy()\n    _c.imwrite(f'{OUT}/vis/dbg_render.png',np.concatenate([R_,mask,occ],1)*255); print('sums',R_.sum(),mask.sum(),(occ<0.3).mean())\n    iou_grid = best[0]")
src=src.replace("for i in (map(int, sys.argv[1].split(\",\")) if len(sys.argv) > 1 else SUPPORT):\n    fit(i)","fit(3)")
exec(compile(src,'fit','exec'))

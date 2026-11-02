import numpy as np,cv2,sys
from opt import *
from viz import panel
T=np.load(sys.argv[2])['T_world_mesh']; Tc=Tcw@T
fr=[int(x) for x in sys.argv[3].split(',')]
rows=[panel(i,Tc[i],'res',S=240) for i in fr]
cv2.imwrite(sys.argv[1],np.vstack([np.hstack(rows[k:k+2]) for k in range(0,len(rows),2)]))

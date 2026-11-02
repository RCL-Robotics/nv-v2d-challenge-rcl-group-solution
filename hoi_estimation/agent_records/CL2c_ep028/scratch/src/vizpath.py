import numpy as np,cv2,sys
from opt import *
from viz import panel
d=np.load(sys.argv[2]); p=np.load(sys.argv[3])
fr=[int(x) for x in sys.argv[4].split(',')]
rows=[panel(i,Rt_to_T(d['R'][i,p[i]],d['t'][i,p[i]]),f'h{p[i]} L%.2f'%d['L'][i,p[i]],S=260) for i in fr]
cv2.imwrite(sys.argv[1],np.vstack([np.hstack(rows[k:k+2]) for k in range(0,len(rows),2)]))

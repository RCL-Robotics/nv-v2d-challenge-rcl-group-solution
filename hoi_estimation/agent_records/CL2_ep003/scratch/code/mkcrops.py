import sys; sys.path.insert(0,'/work/outputs')
from lib import *
C=Crops('/work/inputs',160)
np.savez('crops160.npz',img=(C.img*255).astype(np.uint8),om=C.om,pm=C.pm,off=C.off)

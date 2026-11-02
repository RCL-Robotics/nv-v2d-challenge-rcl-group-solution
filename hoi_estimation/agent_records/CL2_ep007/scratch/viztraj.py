import sys,pickle; sys.path.insert(0,'/work/outputs'); sys.path.insert(0,'/work/scratch')
from common import *; from fitcore import *; from vizfit import viz
K,Kp,Twc,rec=load_inputs('/work/inputs')
fit=Fitter(load_mesh('/work/inputs/object_mesh.glb'),K)
d=pickle.load(open('/work/scratch/cache/crops.pkl','rb'))
T=np.load(sys.argv[1])['T'] if sys.argv[1].endswith('.npz') else pickle.load(open(sys.argv[1],'rb'))['T_fit']
fr=[int(x) for x in sys.argv[2].split(',')]
viz(fit,d,fr,T[fr,:3,:3],T[fr,:3,3],sys.argv[3],ncol=3)

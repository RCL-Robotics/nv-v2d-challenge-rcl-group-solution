# cache full-res frames, masks, chair mask for experiments
import sys; sys.path.insert(0,'/work/outputs')
from lib import *
fr=load_video('/work/inputs',scale=1.0); np.save('/work/scratch/cache/full.npy',fr)
f=h5py.File('/work/inputs/masks.h5','r')
np.save('/work/scratch/cache/obj.npy',f['object'][:]); np.save('/work/scratch/cache/person.npy',f['person'][:])

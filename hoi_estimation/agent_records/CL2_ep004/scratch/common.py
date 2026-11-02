import sys, pickle; sys.path.insert(0,'/work/outputs')
import numpy as np
from stages import *
def get_ep():
    ep=Episode('/work/inputs')
    A=pickle.load(open('/work/scratch/stageA.pkl','rb'))
    ep.wins=A['wins']; ep.datas=pickle.load(open('/work/scratch/datas.pkl','rb'))
    ep.mask_area=A['mask_area']; ep.mask_cent=A['mask_cent']
    d=np.load('/work/scratch/masks_small.npz'); ep.mask_small_o=d['o']; ep.mask_small_p=d['p']
    return ep, A
def get_FR():
    s=np.load('/work/scratch/simp.npz'); return FastRenderer(s['vs'],s['fs'],s['cs']), s

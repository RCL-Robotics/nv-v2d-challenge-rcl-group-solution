import os, h5py,json,numpy as np
f=h5py.File(os.environ["V2D_ROOT"] + '/outputs/cari4d/episode_000001/depth/moge2/aligned.h5')
def show(g,p=''):
    for k in g:
        o=g[k]
        print(p+k, getattr(o,'shape',''), getattr(o,'dtype',''))
show(f); print(dict(f.attrs))
print(json.load(open(os.environ["V2D_ROOT"] + '/outputs/cari4d/episode_000001/depth/moge2/depth_report.json')).keys())

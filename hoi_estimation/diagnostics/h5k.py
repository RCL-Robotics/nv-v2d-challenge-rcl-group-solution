import h5py,json,numpy as np
f=h5py.File('/mnt/nas/Group/hanzhou/v2d/outputs/cari4d/episode_000001/depth/moge2/aligned.h5')
def show(g,p=''):
    for k in g:
        o=g[k]
        print(p+k, getattr(o,'shape',''), getattr(o,'dtype',''))
show(f); print(dict(f.attrs))
print(json.load(open('/mnt/nas/Group/hanzhou/v2d/outputs/cari4d/episode_000001/depth/moge2/depth_report.json')).keys())

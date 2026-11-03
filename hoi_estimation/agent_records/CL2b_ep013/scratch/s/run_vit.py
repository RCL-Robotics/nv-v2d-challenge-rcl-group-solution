import sys; sys.path.insert(0,'/work/scratch/s')
from viterbi import *
a,b=int(sys.argv[1]),int(sys.argv[2]); A,B=sys.argv[3],sys.argv[4]
d=np.load(f'/work/scratch/track2_{a}_{b}.npy',allow_pickle=True).item()
st=np.load('/work/scratch/static_res.npy',allow_pickle=True).item()
cands=[];un=[];labs=['fw','bw','b0','b1','b2','b3']
for t in range(a,b+1):
    cs=[d['fw'][t],d['bw'][t]]+d['bc'][t]
    cands.append([c[0] for c in cs]); un.append(np.array([c[1][0] for c in cs]))
path=viterbi(cands,un,st[A]['T'],st[B]['T'])
sel={t:cands[t-a][k] for t,k in zip(range(a,b+1),path)}
print(''.join(labs[k][0] if labs[k][0]!='b' else labs[k][1] for k in path))
d['vit']=sel; np.save(f'/work/scratch/track2_{a}_{b}.npy',d,allow_pickle=True)

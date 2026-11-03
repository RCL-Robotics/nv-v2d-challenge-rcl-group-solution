import sys
sys.argv=[sys.argv[0]]+sys.argv[1:]
exec(open('/work/scratch/s/evaluate.py').read().split("phases=[")[0])
names=sys.argv[1:]
I=[ious(Tb)]+[ious(np.load(n)['T_world_mesh']) for n in names]
blk=((np.arange(F)//5)%4==2)
for a in range(0,F,20):
    print(a, ' '.join(f'{x[a:a+20].mean():.3f}/{x[a:a+20][blk[a:a+20]].mean():.3f}' for x in I))

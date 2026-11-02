import pickle, numpy as np
res=pickle.load(open('stageA.pkl','rb'))
F=859
by=[[] for _ in range(F)]
for x in res: by[x['f']].append(x)
def ang(R1,R2): return np.degrees(np.arccos(np.clip((np.trace(R1.T@R2)-1)/2,-1,1)))
def axang(R1,R2): return np.degrees(np.arccos(np.clip(R1[:,2]@R2[:,2],-1,1)))
wc=1.0
best=[]
for i in range(F):
    L=[x['ls']+wc*x['lc'] for x in by[i]]; k=int(np.argmin(L)); best.append(by[i][k])
    if i%10==0:
        s=sorted(range(len(L)),key=lambda j:L[j])
        print(i, ' '.join('%s:%.3f/%.3f'%(by[i][j]['src'],by[i][j]['ls'],by[i][j]['lc']) for j in s[:4]), '| 2nd-best ang %.0f ax %.0f'%(ang(by[i][s[0]]['R'],by[i][s[1]]['R']),axang(by[i][s[0]]['R'],by[i][s[1]]['R'])))

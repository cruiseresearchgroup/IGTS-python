import numpy as np
from IGTS import *
#import matplotlib.pyplot as plt

TS_len=420
TS_dim=6
k=3
Integ_TS = np.zeros((TS_dim, TS_len))
print((TS_len/40)-1)
GT_TT = 40 * np.random.randint(1,(TS_len/40)-1, size=k)
GT_TT = GT_TT-1
gt = np.append(np.array(1), np.append(sorted(GT_TT),np.array(TS_len)))
print('GT_TT >>>' , sorted(GT_TT))
Lval=0
for i in range(k):
    for d in range(TS_dim):
        val = np.random.randint(1,6)
        if Lval==val:
            val=val+2
        for j in range(gt[i],gt[i+1],1):
            Integ_TS[d,j] = val
        Lval=val

Integ_TS = Integ_TS /2
mu, sigma = 0, 0.1 
noise = np.random.normal(mu, sigma, [TS_dim,TS_len]) 
Integ_TS = Integ_TS + noise

DP_TT,_ =DP_IGTS(Integ_TS, k,1,1)
print('Dynamic Programming extracted TT >>>' , DP_TT)

# TopDown Approach
TD_TT, IG_TT, knee =TopDown(Integ_TS, k,1,1)
print('Top Down Approach:\n extracted  boundaries>>> {}\n Corresponding IG values >>> {} - knee point : {}'.format(TD_TT,IG_TT,knee))

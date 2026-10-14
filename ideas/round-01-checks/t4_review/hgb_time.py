import numpy as np, time
from sklearn.ensemble import HistGradientBoostingRegressor
rng=np.random.default_rng(0)
for N,d in [(1000,8),(10000,64)]:
    X=rng.standard_normal((N,d)); w=np.abs(rng.standard_normal(d)); w*=d/np.linalg.norm(w)
    y=np.cos(2*np.pi*X@w/10)
    for lr,leaves,msl in [(0.1,31,20),(0.01,63,2)]:
        t=time.time(); HistGradientBoostingRegressor(learning_rate=lr,max_leaf_nodes=leaves,min_samples_leaf=msl,random_state=0).fit(X,y); dt=time.time()-t
        print(N,d,lr,leaves,f'{dt:.2f}s per fit')

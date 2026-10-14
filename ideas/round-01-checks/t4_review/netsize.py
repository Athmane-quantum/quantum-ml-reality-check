# Greedy 0.51-geodesic-angle packing of S_w = {w on sphere: w_j >= 1/d^2} (unit norm), LGM eq. B.1
import numpy as np, sys, time
rng=np.random.default_rng(0)
for d in [1,2,3,4,5,6,8]:
    t=time.time()
    if d==1: print(d,1,'(w = R_w is fixed: nothing to learn)'); continue
    pts=[]
    ncand=200000 if d<=6 else 400000
    X=np.abs(rng.standard_normal((ncand,d))); X/=np.linalg.norm(X,axis=1,keepdims=True)
    X=X[(X>=1/d**2).all(1)]
    cos_thr=np.cos(0.51)
    P=np.zeros((0,d))
    for x in X:
        if P.shape[0]==0 or (P@x).max()<cos_thr:
            P=np.vstack([P,x])
    print(d, P.shape[0], f'{time.time()-t:.1f}s')

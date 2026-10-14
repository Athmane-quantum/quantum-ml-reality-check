import numpy as np, itertools
rng=np.random.default_rng(0)
def iqp_probs(n, th1, th2):
    # |psi> = H^n D H^n |0>, D = exp(i sum th1_i Z_i + i sum th2_ij Z_i Z_j)
    z=np.array(list(itertools.product([1,-1],repeat=n)))  # 2^n x n
    phase = z@th1 + sum(th2[i,j]*z[:,i]*z[:,j] for i in range(n) for j in range(i+1,n))
    amp0 = np.exp(1j*phase)/2**(n/2)   # after H^n|0> and D, in Z basis
    # apply H^n: Walsh-Hadamard
    a=amp0.copy()
    N=2**n; h=1
    while h<N:
        for i in range(0,N,2*h):
            x=a[i:i+h].copy(); y=a[i+h:i+2*h].copy()
            a[i:i+h]=(x+y)/np.sqrt(2); a[i+h:i+2*h]=(x-y)/np.sqrt(2)
        h*=2
    p=np.abs(a)**2; return p/p.sum()
for n in [6,8,10]:
    res={}
    for inst in range(5):
        th1=rng.uniform(0,2*np.pi,n); th2=np.triu(rng.uniform(0,2*np.pi,(n,n)),1)
        p=iqp_probs(n,th1,th2)
        u=np.ones(2**n)/2**n
        res.setdefault('TV_uniform',[]).append(0.5*np.abs(p-u).sum())
        res.setdefault('N*2^n*sum p^2',[]).append(2**n*(p**2).sum())
        for N in [1000,4000,10000,100000]:
            tvs=[]
            for r in range(10):
                c=rng.multinomial(N,p)/N
                tvs.append(0.5*np.abs(c-p).sum())
            res.setdefault(f'TV_hist_N{N}',[]).append(np.mean(tvs))
    print('n=',n,{k:round(float(np.mean(v)),3) for k,v in res.items()})

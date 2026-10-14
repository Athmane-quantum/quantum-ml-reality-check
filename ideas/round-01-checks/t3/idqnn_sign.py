# Check: in the IDQNN of arXiv:2509.09033 (Def. 16) at x = 0^n, flipping the sign of one RZ angle
# equals XOR-flipping the X-basis outcomes of that qubit's neighbours. Pure numpy statevector.
import numpy as np, itertools
def idqnn_p0(theta, edges, n):
    B=np.array(list(itertools.product([0,1],repeat=n)))  # computational basis, qubit 0 = MSB
    Z=1-2*B
    amp=np.ones(2**n)/2**(n/2)                            # |+>^n
    amp=amp*np.exp(-0.5j*(Z@theta))                       # RZ(theta_i)=exp(-i theta Z/2)
    for (i,j) in edges: amp=amp*np.where((B[:,i]==1)&(B[:,j]==1),-1,1)  # CZ
    H=np.array([[1,1],[1,-1]])/np.sqrt(2); Hn=np.array([[1.]])
    for _ in range(n): Hn=np.kron(Hn,H)
    return np.abs(Hn@amp)**2, B                            # X-basis measurement
rows,cols=2,3; n=rows*cols; idx=lambda r,c: r*cols+c
edges=[(idx(r,c),idx(r,c+1)) for r in range(rows) for c in range(cols-1)]+[(idx(r,c),idx(r+1,c)) for r in range(rows-1) for c in range(cols)]
rng=np.random.default_rng(0); th=rng.uniform(0,2*np.pi,n)
p,B=idqnn_p0(th,edges,n)
th2=th.copy(); q=1; th2[q]=-th2[q]
p2,_=idqnn_p0(th2,edges,n)
nb=[j for (i,j) in edges if i==q]+[i for (i,j) in edges if j==q]
mask=np.zeros(n,int); mask[nb]=1
perm=[int(''.join(map(str,(b^mask))),2) for b in B]
print('TV(p, p_signflip)=',0.5*np.abs(p-p2).sum(),' max|p_signflip - p(y xor mask)|=',np.abs(p2-p[perm]).max())
# local statistic used by eq. S.2.1: qubit q in |+>, neighbours |0>: P(y_q=1)=sin^2(theta/2) symmetric in sign
print('P(y=1) for +theta and -theta:',np.sin(th[q]/2)**2,np.sin(-th[q]/2)**2)

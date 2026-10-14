"""If |psi> = N^-1/2 sum_i |x_i>|h(x_i)> is loaded from N classical random examples whose x_{-j} are all distinct,
the reduced state of register j is diagonal, so the Fourier-sampling outcome distribution is uniform: no period info."""
import numpy as np
rng = np.random.default_rng(1)
d, q, M1, M2 = 3, 8, 4.0, 2
grid = (np.arange(q) - q//2)/M1
F = np.fft.fft(np.eye(q))/np.sqrt(q)
for w in [np.array([1.3, 0.7, 0.9]), np.array([0.5, 2.1, 1.1])]:
    h = lambda x: np.floor(np.cos(2*np.pi*(x @ w))*M2)/M2
    # draw N examples with distinct x_{-0}
    others = rng.choice(q*q, size=6, replace=False)
    idx = np.stack([rng.integers(q, size=6), others // q, others % q], 1)
    X = grid[idx]; y = h(X)
    # reduced state on coordinate 0: rho[a,b] = (1/N) sum_{i,i'} [x_{i,-0}=x_{i',-0}][y_i=y_i'] |a_i><b_i'|
    N = len(X); rho = np.zeros((q, q))
    for i in range(N):
        for k in range(N):
            if np.all(idx[i, 1:] == idx[k, 1:]) and y[i] == y[k]:
                rho[idx[i, 0], idx[k, 0]] += 1/N
    probs = np.real(np.diag(F @ rho @ F.conj().T))
    print("w =", w, " Fourier-sampling distribution on coord 0:", np.round(probs, 4))

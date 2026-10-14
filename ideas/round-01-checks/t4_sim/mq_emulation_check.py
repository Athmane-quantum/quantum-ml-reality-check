"""Check: <h|O_j (x) I|h> for the QSQ Fourier-sampling observable of Lewis-Gilboa-McClean (eq. D.22)
equals E_{x_-j ~ D_-j}[ sum_y <phi_y|O_j|phi_y> ], computable from membership queries along one fiber.
Also checks the shift-correlation identity <h| S_{k,-T}^dag-free correlation |h> form (D.52) numerically for a
simplified observable (label-register 'value' operator times shift)."""
import numpy as np, itertools
rng = np.random.default_rng(0)
d, nq = 2, 3                      # 2 coordinates, 3 qubits each -> q = 8 grid points per coordinate
q = 2**nq
M1, M2 = 4.0, 2                   # grid spacing 1/M1, labels rounded down to multiples of 1/M2
w = np.array([1.3, 0.7])          # unknown weights (positive orthant)
beta = np.array([0.6, 0.4])       # periodic activation: sum_j beta_j cos(2 pi j y)
grid = (np.arange(q) - q//2) / M1
def g(x):  # x: (..., d)
    y = x @ w
    return sum(b*np.cos(2*np.pi*(j+1)*y) for j, b in enumerate(beta))
def h(x):
    return np.floor(g(x)*M2)/M2
labels = np.arange(-M2, M2+1)/M2          # possible label values
L = len(labels); lab_idx = {round(v*M2): i for i, v in enumerate(labels)}
sigma = 3.0                                # wide discrete Gaussian, product over coordinates
Dj = np.exp(-grid**2/(2*sigma**2)); Dj /= Dj.sum()
# exact state |h> = sum_x sqrt(D(x)) |x1>|x2>|h(x)>
psi = np.zeros((q, q, L))
for i1, i2 in itertools.product(range(q), range(q)):
    x = np.array([grid[i1], grid[i2]])
    psi[i1, i2, lab_idx[round(h(x)*M2)]] = np.sqrt(Dj[i1]*Dj[i2])
assert abs(np.sum(psi**2)-1) < 1e-12
F = np.fft.fft(np.eye(q))/np.sqrt(q)       # QFT matrix (sign convention irrelevant for |.|^2)
Oj = F.conj().T @ np.diag(np.arange(q)/q) @ F
for j in range(d):
    # exact QSQ value: apply O_j on coordinate j
    exact = np.real(np.einsum('abl,ca,cbl->', psi.conj(), Oj, psi) if j == 0 else
                    np.einsum('abl,cb,acl->', psi.conj(), Oj, psi))
    # classical fiber formula (exact sum over x_-j), uses only membership queries along fibers
    def fiber_value(i_other):
        xs = np.zeros((q, d)); xs[:, j] = grid; xs[:, 1-j] = grid[i_other]
        hv = h(xs); tot = 0.0
        for yv in np.unique(hv):
            phi = np.where(hv == yv, np.sqrt(Dj), 0.0)
            tot += np.real(phi.conj() @ Oj @ phi)
        return tot
    fib = np.array([fiber_value(i) for i in range(q)])
    classical_exact = fib @ Dj
    # Monte Carlo over x_-j (what an MQ learner would do): 2000 fibers -> tolerance ~ 1/sqrt(2000)
    samp = rng.choice(q, size=2000, p=Dj)
    mc = fib[samp].mean()
    print(f"coord {j}: exact QSQ={exact:.6f}  fiber-formula={classical_exact:.6f}  MC(2000 fibers)={mc:.4f}  "
          f"MQ calls={2000*q}")
# Shift-correlation observable: expectation of (S_{k,-T} on x register) (x) (value operator V on label, rank-1 |v><v|)
# For uniform D, LGM (D.52) states it equals mean_x h(x) h(x+T e_k)/M2^2 (with their A_m); here we check the
# general form <h| (S_{k,a} (x) |v><v|) |h> = sum_x sqrt(D(x)D(x+a e_k)) v[h(x)] v[h(x+a e_k)] (cyclic shift).
v = labels/np.linalg.norm(labels)
for k, a in [(0, 3), (1, 5)]:
    S = np.roll(np.eye(q), a, axis=0)       # |x> -> |x+a mod q>
    Ok = np.kron(S if k == 0 else np.eye(q), S if k == 1 else np.eye(q))
    big = np.kron(Ok, np.outer(v, v))
    vec = psi.reshape(-1)
    exact = vec @ big @ vec
    tot = 0.0
    for i1, i2 in itertools.product(range(q), range(q)):
        x = np.array([grid[i1], grid[i2]]); j1, j2 = (i1, i2)
        if k == 0: s1, s2 = (i1 + a) % q, i2
        else: s1, s2 = i1, (i2 + a) % q
        xs = np.array([grid[s1], grid[s2]])
        tot += np.sqrt(Dj[i1]*Dj[i2]*Dj[s1]*Dj[s2]) * v[lab_idx[round(h(x)*M2)]] * v[lab_idx[round(h(xs)*M2)]]
    print(f"shift obs k={k}, a={a}: exact={exact:.6f}  pairwise-MQ formula={tot:.6f}")

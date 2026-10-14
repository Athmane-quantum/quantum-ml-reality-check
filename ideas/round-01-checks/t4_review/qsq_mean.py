# LGM eq. D19/E70: O = QFT^-1 diag(l/q) QFT (x) I. A QSQ returns <h|O|h> (+-tau), i.e. the MEAN of the
# Fourier-sampling distribution / q, not a sample. Show it carries ~no period information (d=1, uniform D).
import numpy as np
q=256
F=np.fft.fft(np.eye(q))/np.sqrt(q)
x=np.arange(q)-q//2
for w in [0.11,0.137,0.19,0.23,0.31]:          # period S=1/w grid points
    h=np.floor(np.cos(2*np.pi*w*x)*4)/4          # labels rounded to 1/M2, M2=4
    probs=np.zeros(q)
    for y in np.unique(h):
        phi=np.where(h==y,1/np.sqrt(q),0.0)
        probs+=np.abs(F@phi)**2
    mean=(np.arange(q)/q)@probs
    peaks=np.argsort(probs)[-4:]
    print(f"S={1/w:6.2f}  QSQ value <O>={mean:.4f}   (top Fourier outcomes y={sorted(peaks)}, ~k*q/S={[round(k*q*w) for k in range(1,3)]})")

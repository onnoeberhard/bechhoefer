import numpy as np
import matplotlib.pyplot as plt
from oetils import init_plotting

a = 0.9
y = lambda k: 1 - a**k  # via sp.apart, z-correspondence, and shift theorem.

# Sampling interval T = 1
P = lambda omega: abs((1 - a)/(np.exp(1j * omega) - a))**2

# Plot
_ = init_plotting(latex=True, sans=True)
k = np.arange(0, 101)
fig, ax = plt.subplots()
ax.set_title('Step response')
ax.set_xlabel('Time $t = kT$')
ax.axhline(1, c='k', ls='--')
ax.plot(k, y(k), ds='steps', label='$y(t)$')
ax.legend()
fig.savefig('ex-05-06-a.pdf')

omega = np.logspace(-3, 1.6, 10_000)
fig, ax = plt.subplots()
ax.set_title('Power density')
ax.set_xlabel(r'Angle $\omega T$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(omega, P(omega), label=r'$|G(i\omega T)|^2$')
ax.axvline((1 - a)/np.sqrt(1), c='k', ls='--', label='Corner frequency')
ax.legend()
fig.savefig('ex-05-06-b.pdf')

import numpy as np
import matplotlib.pyplot as plt
from oetils import init_plotting, bootstrap_cis

N = 1000  # Number of coins
n_exp = 5000  # Number of random measurement matrices
ss = [0.2, 0.4, 0.6]  # Noise scale
Ms = list(range(1, 21))  # Number of measurements
p = np.zeros((len(ss), len(Ms), n_exp))

for i, s in enumerate(ss):
    for j, M in enumerate(Ms):
        rng = np.random.default_rng(42)
        Phi = (rng.uniform(size=(n_exp, M, N)) < 0.5).astype(int)  # Measurement matrix
        y0 = Phi[..., 0] + rng.normal(0, s, (n_exp, M))   # Fake coin: index 0
        e = np.sum((Phi - y0[..., None])**2, 1)  # Estimate fake coin from measurement y0
        p[i, j] = (e[:, 0] == e.min(1))/(e == e.min(1)[:, None]).sum(1)

# Plot results
_ = init_plotting(latex=True, sans=True)
fig, ax = plt.subplots()
ax.set_title('Compressed sensing of counterfeit coin')
ax.set_xlabel('Measurements $M$')
ax.set_ylabel('Probability of success')
rng = np.random.default_rng(42)
for i, s in enumerate(ss):
    med, err, *_ = bootstrap_cis(p[i], rng)
    ax.errorbar(Ms, med, err, fmt='.-', label=fr'$\sigma = {s:.1f}$')
ax.legend()
fig.savefig('ex-05-04.pdf')

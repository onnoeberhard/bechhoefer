import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from oetils import init_plotting

init_plotting(latex=True, sans=True)
ts = np.linspace(0, 5, 1000)
ks = np.arange(50)
ts_ = np.linspace(0, 5, 50)
s, t = sp.S('s, t')
z, k = sp.S('z, k')

def step(G, K):
    '''Compute closed-loop step response for discrete controller K'''
    T = G * K / (1 + G * K)
    N, D = T.as_numer_denom()
    N = [float(x) for x in N.as_poly(z).all_coeffs()]
    D = [float(x) for x in D.as_poly(z).all_coeffs()]
    N = [0] * (len(D) - len(N)) + N  # 0-pad to give correct degree coeffs
    return lfilter(N, D, np.ones(50))

# - PI control -
# Continuous system
G = 1 / (1 + s)
K = 1 + 1 / s
T = G * K / (1 + G * K)
r = sp.laplace_transform(sp.Heaviside(t), t, s)[0]
y = (T * r).simplify()
y = sp.inverse_laplace_transform(y, s, t)
y0 = sp.lambdify(t, y)(ts)

# Discrete system
T_ = sp.S(1)/10
G = (1 - sp.exp(-T_)) / (z - sp.exp(-T_))
Kb = K.subs(s, (1 - z**-1) / T_)
Kt = K.subs(s, 2 / T_ * (1 - z**-1) / (1 + z**-1))
yb = step(G, Kb)
yt = step(G, Kt)

# Plot
fig, ax = plt.subplots()
ax.set_title("Discretized PI control of first-order system")
ax.set_xlabel("Time $t = kT$")
ax.axhline(1, ls='--', c='k', label='Reference point')
ax.plot(ts, y0, label='Continuous control')
ax.plot(ts_, yb, label='Backward Euler', ds='steps-post')
ax.plot(ts_, yt, label='Tustin', ds='steps-post')
ax.legend()
ylim = ax.get_ylim()
fig.savefig('ex-05-13-a.pdf')

# - PID control -
# Continuous system
G = 1 / (1 + s)
K = K + s/10
T = G * K / (1 + G * K)
y = (T * r).simplify()
y = sp.inverse_laplace_transform(y, s, t)
y0 = sp.lambdify(t, y)(ts)

# Discrete system
G = (1 - sp.exp(-T_)) / (z - sp.exp(-T_))
Kb = K.subs(s, (1 - z**-1) / T_)
Kt = K.subs(s, 2 / T_ * (1 - z**-1) / (1 + z**-1))
yb = step(G, Kb)
yt = step(G, Kt)

# Plot
fig, ax = plt.subplots()
ax.set_title("Discretized PID control of first-order system")
ax.set_xlabel("Time $t = kT$")
ax.axhline(1, ls='--', c='k', label='Reference point')
ax.plot(ts, y0, label='Continuous control')
ax.plot(ts_, yb, label='Backward Euler', ds='steps-post')
ax.plot(ts_, yt, label='Tustin', ds='steps-post')
ax.legend()
ax.set_ylim(ylim)
fig.savefig('ex-05-13-b.pdf')

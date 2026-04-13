import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import oetils
import scipy as sp
import sympy as sy
oetils.init_plotting()

def f(x, u, A, B, C):
    '''Compute next state and observation'''
    x = A @ x + B * u
    return x, C @ x

# System and model parameters
seed = 42
system = {'zeta': 0.1, 'omega': 5}
T = 0.01    # System time scale
N = 50_000  # Input length

# General continuous second order system
zeta, omega, s = sy.symbols('zeta, omega, s')
Ac = sy.Matrix([[0, 1], [-omega**2, -2*zeta*omega]])
Bc = sy.Matrix([0, 1])
Cc = sy.Matrix([1, 0])

# ZOH discretized system
A = jnp.array(sy.exp(Ac.subs(system) * T), float)
B = jnp.array(Ac.subs(system).inv() @ (A - sy.eye(2)) @ Bc, float).flatten()
C = jnp.array(Cc, float).flatten()
f_ = jax.jit(lambda x, u: f(x, u, A, B, C))

# Sample white noise signal (DRBS should be similar)
rng = jax.random.key(seed)
u = jax.random.normal(rng, N)

# Simulate dynamics
x = jnp.zeros(2)
x, y = jax.lax.scan(f_, x, u)

# Estimate transfer function (with and without Hann apodization)
xf = sp.fft.rfftfreq(N, T)
fu = sp.fft.rfft(u * sp.signal.windows.hann(N))
fy = sp.fft.rfft(y * sp.signal.windows.hann(N))
G = fy / fu
fu_ = sp.fft.rfft(u)
fy_ = sp.fft.rfft(y)
G_ = fy_ / fu_

# Plot
oetils.init_plotting(latex=True, sans=True)
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].set_title("System identification of a second-order ZOH system")
ax[0].set_ylabel(r"Gain $\lvert G(\mathrm{i}\omega)\rvert$")
ax[1].set_ylabel(r"Phase $\angle G(\mathrm{i}\omega)$")
ax[1].set_xlabel(r"Angular frequency $\omega$")
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].plot(2*np.pi*xf, abs(fy_ / fu_), 'C2', label='Estimate without Hann apodization')
ax[1].plot(2*np.pi*xf, np.angle(fy_ / fu_), 'C2')
G = Cc.T @ (s*sy.eye(2) - Ac).inv() @ Bc
G = sy.lambdify(s, G.subs(system))(2j*np.pi*xf).flatten()
ax[0].plot(2*np.pi*xf, abs(G), 'C1', label='True transfer function of continuous system')
ax[1].plot(2*np.pi*xf, np.angle(G), 'C1')
ax[0].plot(2*np.pi*xf, abs(fy / fu), label='Estimate without Hann apodization')
ax[1].plot(2*np.pi*xf, np.angle(fy / fu))
ax[0].legend()
fig.savefig('ex-06-04.pdf')

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

Ye = 0.5
km_to_eV_inv = 5.07e9

def matter_potential(rho):
    return 7.56e-14 * rho * Ye

theta12 = np.radians(33.44)
theta13 = np.radians(8.57)
theta23 = np.radians(45.0)
delta_cp = np.radians(-90)
dm21 = 7.42e-5
dm31 = 2.517e-3

def PMNS():
    c12, s12 = np.cos(theta12), np.sin(theta12)
    c13, s13 = np.cos(theta13), np.sin(theta13)
    c23, s23 = np.cos(theta23), np.sin(theta23)
    d = delta_cp
    return np.array([
        [c12*c13, s12*c13, s13*np.exp(-1j*d)],
        [-s12*c23 - c12*s23*s13*np.exp(1j*d),
         c12*c23 - s12*s23*s13*np.exp(1j*d),
         s23*c13],
        [s12*s23 - c12*c23*s13*np.exp(1j*d),
         -c12*s23 - s12*c23*s13*np.exp(1j*d),
         c23*c13]
    ], dtype=complex)

def hamiltonian(E, rho):
    U = PMNS()
    M2 = np.diag([0, dm21, dm31])
    E_eV = E * 1e9
    H_vac = U @ M2 @ np.conjugate(U.T) / (2 * E_eV)
    V = matter_potential(rho)
    H_mat = np.diag([V, 0, 0])
    return H_vac + H_mat

R = 6371

def prem_density(x, L):
    y = x - L/2
    r = np.sqrt(R**2 - (L/2)**2 + y**2)
    if r < 1220:
        return 13.0
    elif r < 3480:
        return 11.0
    elif r < 5700:
        return 5.0
    else:
        return 3.3

def prem_path_average(L):
    return 4.289

def evolve(E, L, use_prem=True, steps=400):
    dx = L / steps
    state = np.array([0, 1, 0], dtype=complex)
    rho_avg = prem_path_average(L) 
    for i in range(steps):
        x = i * dx
        rho = prem_density(x, L) if use_prem else rho_avg
        H = hamiltonian(E, rho)
        U_step = expm(-1j * H * dx * km_to_eV_inv)
        state = U_step @ state
    return state

def probability(E, L, use_prem=True):
    final = evolve(E, L, use_prem)
    return np.abs(final[0])**2

energies = np.linspace(2, 6, 200)
L = 7000

rho_avg = prem_path_average(L)
print(f"Path-averaged density for L={L} km: {rho_avg:.3f} g/cm³")

P_const = [probability(E, L, False) for E in energies]
P_prem  = [probability(E, L, True)  for E in energies]

def flux(E):
    return np.exp(-E/3)

def cross_section(E):
    return E

T = 1e5
f_sys = 0.05

def events(E, L, use_prem=True):
    P = probability(E, L, use_prem)
    return flux(E) * cross_section(E) * P * T

N_const = np.array([events(E, L, False) for E in energies])
N_prem  = np.array([events(E, L, True)  for E in energies])

def total_error(N):
    stat = np.sqrt(N)
    sys  = f_sys * N
    return np.sqrt(stat**2 + sys**2)

err_const = total_error(N_const)
err_prem  = total_error(N_prem)

plt.figure(figsize=(8, 6))
plt.plot(energies, N_const, '--', label=f"Constant density ($\\rho$ = {rho_avg:.2f} g/cm³)")
plt.plot(energies, N_prem, label="PREM profile")
plt.fill_between(energies, N_const - err_const, N_const + err_const, alpha=0.2)
plt.fill_between(energies, N_prem - err_prem,   N_prem + err_prem,   alpha=0.2)
plt.xlabel("Energy (GeV)")
plt.ylabel("Event Rate (arb. units)")
plt.title(r"Oscillation with Statistical + Systematic Errors (Normal Ordering)  $\delta_{CP}$ = -90°")
plt.legend()
plt.grid()
plt.show()

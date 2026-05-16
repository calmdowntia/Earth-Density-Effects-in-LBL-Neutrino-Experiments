#Updated Chi Square

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import expm

matplotlib.rcParams.update({
    "text.usetex": False,          
    "font.family":       "serif",
    "font.serif":        ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size":         14,
    "axes.labelsize":    16,
    "axes.titlesize":    16,
    "xtick.labelsize":   13,
    "ytick.labelsize":   13,
    "legend.fontsize":   12,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "0.5",
    "lines.linewidth":   2.0,
    "axes.linewidth":    1.2,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.major.size":  6,
    "ytick.major.size":  6,
    "xtick.minor.size":  3,
    "ytick.minor.size":  3,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.top":         True,
    "ytick.right":       True,
    "figure.dpi":        200,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

theta12, theta13, theta23 = np.radians(33.44), np.radians(8.57), np.radians(49.2)
dm21, dm31 = 7.42e-5, 2.517e-3
delta_true = -np.pi / 2

E_vals     = np.linspace(2.0, 6.0, 40)
dE         = E_vals[1] - E_vals[0]
delta_scan = np.linspace(-np.pi, np.pi, 120)
NORM       = 1e4

def flux(E):           return np.exp(-E / 3)
def cross_section(E):  return E
def efficiency(E):     return 0.8

def PMNS(delta):
    c12, s12 = np.cos(theta12), np.sin(theta12)
    c13, s13 = np.cos(theta13), np.sin(theta13)
    c23, s23 = np.cos(theta23), np.sin(theta23)
    return np.array([
        [c12*c13,  s12*c13,  s13*np.exp(-1j*delta)],
        [-s12*c23 - c12*s23*s13*np.exp(1j*delta),
          c12*c23 - s12*s23*s13*np.exp(1j*delta), s23*c13],
        [ s12*s23 - c12*c23*s13*np.exp(1j*delta),
         -c12*s23 - s12*c23*s13*np.exp(1j*delta), c23*c13]
    ])

def Hamiltonian(E, delta, rho, anti=False):
    U  = PMNS(delta)
    M2 = np.diag([0, dm21, dm31])
    H_vac = U @ M2 @ np.conjugate(U.T)
    A = 7.6e-5 * rho * E * (-1 if anti else 1)
    return (H_vac + np.diag([A, 0, 0])) / (2 * E)

R = 6371.0

def prem_density_at_r(r):
    if   r < 1220: return 13.0
    elif r < 3480: return 11.0
    elif r < 5700: return 5.0
    else:          return 3.3

def prem_density_profile(L, n_steps=300):
    dx = L / n_steps
    profile = []
    for i in range(n_steps):
        y = i * dx - L / 2
        r = np.sqrt(max(R**2 - (L/2)**2 + y**2, 0))
        profile.append(prem_density_at_r(r))
    return np.array(profile), dx

def precompute_profile(L):
    return prem_density_profile(L)

def prob_prem(E, delta, rhos, dx, anti=False):
    state = np.array([0, 1, 0], dtype=complex)
    for rho in rhos:
        H = Hamiltonian(E, delta, rho, anti)
        state = expm(-1j * 1.27 * H * dx) @ state
    return np.abs(state[0])**2

def prob_const(E, delta, L, rho, anti=False):
    H = Hamiltonian(E, delta, rho, anti)
    return np.abs(expm(-1j * 1.27 * H * L)[0, 1])**2

def rates_prem(delta, L, rhos, dx, anti=False):
    return np.array([
        NORM * flux(E) * cross_section(E) * efficiency(E)
        * prob_prem(E, delta, rhos, dx, anti) * dE
        for E in E_vals
    ])

def rates_const(delta, L, rho, anti=False):
    return np.array([
        NORM * flux(E) * cross_section(E) * efficiency(E)
        * prob_const(E, delta, L, rho, anti) * dE
        for E in E_vals
    ])

def chi_square(test, true):
    return np.sum((test - true)**2 / (true + 1.0))

selected = [3000, 5000, 9000]
colors = ["#2166AC", "#D6604D", "#1A7D3B"]
deg = np.degrees(delta_scan)
results = {}

for L in selected:
    print(f"Computing L = {L} km …")
    rhos, dx = precompute_profile(L)
    rho_avg  = np.mean(rhos)

    true_nu  = rates_prem(delta_true, L, rhos, dx, anti=False)
    true_anu = rates_prem(delta_true, L, rhos, dx, anti=True)

    chi2_const = np.zeros(len(delta_scan))
    chi2_prem  = np.zeros(len(delta_scan))

    for j, d in enumerate(delta_scan):
        tc_nu  = rates_const(d, L, rho_avg, anti=False)
        tc_anu = rates_const(d, L, rho_avg, anti=True)
        chi2_const[j] = chi_square(tc_nu,  true_nu) + chi_square(tc_anu, true_anu)

        tp_nu  = rates_prem(d, L, rhos, dx, anti=False)
        tp_anu = rates_prem(d, L, rhos, dx, anti=True)
        chi2_prem[j]  = chi_square(tp_nu,  true_nu) + chi_square(tp_anu, true_anu)

    chi2_const = (chi2_const - chi2_const.min()) / chi2_const.max()
    chi2_prem  = (chi2_prem  - chi2_prem.min())  / chi2_prem.max()

    results[L] = (chi2_const, chi2_prem)

fig, ax = plt.subplots(figsize=(9.0, 6.0))  

for i, L in enumerate(selected):
    chi2_const, chi2_prem = results[L]
    label_L = f"{L:,} km"

    ax.plot(deg, chi2_const, linestyle="--", color=colors[i],
            linewidth=2.2, label=f"Const. density,  $L = {label_L}$")

    ax.plot(deg, chi2_prem, linestyle="-", color=colors[i],
            linewidth=2.2, label=f"PREM, $L = {label_L}$",
            alpha=0.85)

    ax.fill_between(deg, chi2_const, chi2_prem,
                    alpha=0.10, color=colors[i])

ax.axvline(-90, color="black", linestyle="--", linewidth=1.6,
           label=r"True $\delta_{CP} = -90°$", zorder=5)

ax.set_xlabel(r"$\delta_{CP}$ (degrees)", labelpad=8)
ax.set_ylabel(r"$\Delta\chi^2$ (normalised)", labelpad=8)
ax.set_xlim(-180, 180)
ax.set_ylim(-0.04, 1.18)
ax.set_xticks(np.arange(-180, 181, 45))
ax.set_yticks(np.arange(0, 1.1, 0.2))

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

legend_elements = []
for i, L in enumerate(selected):
    label_L = f"{L:,}"
    legend_elements.append(
        Line2D([0], [0], color=colors[i], lw=2.2, linestyle="--",
               label=f"Const.,  $L={label_L}$ km"))
    legend_elements.append(
        Line2D([0], [0], color=colors[i], lw=2.2, linestyle="-",
               label=f"PREM,  $L={label_L}$ km"))

leg = ax.legend(
    handles=legend_elements,
    ncol=2,
    loc="upper right",
    fontsize=11,
    framealpha=0.92,
    edgecolor="0.45",
    handlelength=2.4,
    columnspacing=1.0,
    handletextpad=0.6,
    borderpad=0.7,
)

ax.grid(True, which="major", linestyle=":", linewidth=0.7, alpha=0.55)
plt.tight_layout(pad=0.9)
outfile="figure_chi_squared.png"
plt.savefig(
    outfile,
    dpi=300
)
print(f"Saved → {outfile}")
plt.show()

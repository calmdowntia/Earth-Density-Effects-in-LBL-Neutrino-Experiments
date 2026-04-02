
# Earth Density Effects in Long-Baseline Neutrino Experiments

This repository contains the simulation and analysis code used in the study:

**"Earth-Density Effects in Long Baseline Neutrino Experiments"**

## Overview

This project investigates the impact of Earth matter density modelling on neutrino oscillation measurements in long-baseline experiments. In particular, it compares the commonly used constant density approximation with the realistic PREM (Preliminary Reference Earth Model) profile.

The analysis is performed within a full three-flavour neutrino oscillation framework, including matter effects (MSW effect), and focuses on the reconstruction of the CP-violating phase δCP.

## Contents

### Figure 1 — Bias vs Baseline
**File:** `fig1_biasVSbaseline.py`

Simulates νμ → νe appearance event rates across baselines from 1000 km to 12000 km and computes the bias in reconstructed δCP arising from the constant density approximation.

---

### Figure 2 — Δχ² Profiles
**File:** `fig2_chi_squared_profiles.py`

Generates Δχ² distributions as a function of δCP for different baselines, comparing constant density and PREM profiles to study degeneracies and shifts in best-fit values.

---

### Figure 3 — Event Rate Spectra (7000 km)
**File:** `fig3_osc_prob_7000km.py`

Computes predicted νμ → νe appearance event rates as a function of energy (2–6 GeV) at a baseline of 7000 km, highlighting spectral distortions due to realistic Earth density variations.

---

## Methodology

- Three-flavour neutrino oscillation framework using PMNS matrix
- Numerical propagation via matrix exponentiation
- Matter effects included through MSW potential
- Earth density modelled using PREM-based multi-layer profile
- Statistical analysis performed using Poisson χ²

## Requirements

- Python 3.x
- numpy
- scipy
- matplotlib

Install dependencies using:
pip install numpy scipy matplotlib


## Usage

Run individual scripts:
python fig1_biasVSbaseline.py
python fig2_chi_squared_profiles.py
python fig3_osc_prob_7000km.py


## Notes

- All results are generated via numerical simulation (no experimental data)
- The constant density approximation uses path-averaged density values
- PREM model is implemented using a simplified multi-layer Earth structure

## Author

Tia Pandit  
Department of Physics, Kirori Mal College, University of Delhi

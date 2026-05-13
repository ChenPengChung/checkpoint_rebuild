#!/usr/bin/env python3
"""Unit tests for 7-point Lagrange interpolation and rho mass correction."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from interp_checkpoint import (
    lagrange7_weights,
    lagrange7_weights_vectorized,
    apply_rho_mass_correction,
    GridConfig,
    BFR,
    fill_ghost,
)

PASS = 0
FAIL = 0

def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        print(f'  PASS: {name}')
        PASS += 1
    else:
        print(f'  FAIL: {name}  {detail}')
        FAIL += 1


# ═══════════════════════════════════════════════════════════════
# Test A: Lagrange weight properties
# ═══════════════════════════════════════════════════════════════
print('=== Test A: Lagrange-7 weight properties ===')

# A1: Partition of unity (weights sum to 1 for any t)
for t in [0.0, 0.25, 0.5, 0.75, 1.0, 0.123, 0.999]:
    w = lagrange7_weights(t)
    check(f'partition of unity at t={t}', abs(sum(w) - 1.0) < 1e-14,
          f'sum={sum(w)}')

# A2: Kronecker delta (at integer nodes, w[m]=1 for that node, 0 elsewhere)
# t=0 -> stencil node 3 should be 1
w = lagrange7_weights(0.0)
check('Kronecker at t=0 (node 3)', abs(w[3] - 1.0) < 1e-14 and
      all(abs(w[m]) < 1e-14 for m in range(7) if m != 3))

w = lagrange7_weights(1.0)
check('Kronecker at t=1 (node 4)', abs(w[4] - 1.0) < 1e-14 and
      all(abs(w[m]) < 1e-14 for m in range(7) if m != 4))

# A3: Vectorized weights match scalar
t_arr = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 0.123])
w_vec = lagrange7_weights_vectorized(t_arr)
for idx, t in enumerate(t_arr):
    w_sc = lagrange7_weights(t)
    check(f'vectorized == scalar at t={t}',
          np.allclose(w_vec[idx], w_sc, atol=1e-14))

# ═══════════════════════════════════════════════════════════════
# Test B: Polynomial exactness (degree 6)
# ═══════════════════════════════════════════════════════════════
print('\n=== Test B: Polynomial exactness ===')

# 1D test: f(x) = x^p for p=0..6 on uniform nodes {0,1,2,3,4,5,6}
# Interpolation at s = t + 3 should be exact
nodes = np.arange(7, dtype=np.float64)
for p in range(7):
    f_nodes = nodes ** p
    max_err = 0.0
    for t in np.linspace(0, 1, 20):
        w = lagrange7_weights(t)
        s = t + 3.0
        exact = s ** p
        interp = np.dot(w, f_nodes)
        max_err = max(max_err, abs(interp - exact))
    check(f'polynomial p={p} exact', max_err < 1e-10,
          f'max_err={max_err:.2e}')

# Degree 7 should NOT be exact
f_nodes = nodes ** 7
errs = []
for t in np.linspace(0.1, 0.9, 10):
    w = lagrange7_weights(t)
    s = t + 3.0
    errs.append(abs(np.dot(w, f_nodes) - s**7))
check('polynomial p=7 NOT exact (expected)', max(errs) > 1e-3,
      f'max_err={max(errs):.2e}')

# ═══════════════════════════════════════════════════════════════
# Test C: Mass correction matching solver
# ═══════════════════════════════════════════════════════════════
print('\n=== Test C: Mass correction (ReduceRhoSum_Kernel) ===')

# Create a small test grid
cfg = GridConfig(nx=9, ny=9, nz=9, jp=1, gamma=2.0, alpha=0.5,
                 grid_dat='test.dat')
rho = np.ones((cfg.NY6, cfg.NZ6, cfg.NX6), dtype=np.float64)
# Perturb interior
rho[BFR:BFR+2, BFR:BFR+2, BFR:BFR+2] = 1.05  # heavier patch

ni = cfg.NX6 - 7
nj = cfg.NY6 - 7
nk = cfg.NZ6 - 6
N_interior = ni * nj * nk
rho_sum_before = np.sum(rho[BFR:BFR+nj, BFR:BFR+nk, BFR:BFR+ni])
mean_before = rho_sum_before / N_interior

rho_modify, mean_b, mean_a = apply_rho_mass_correction(rho, cfg)

rho_sum_after = np.sum(rho[BFR:BFR+nj, BFR:BFR+nk, BFR:BFR+ni])
check('mean rho == 1.0 after correction',
      abs(rho_sum_after / N_interior - 1.0) < 1e-14,
      f'mean={rho_sum_after / N_interior}')
check('rho_modify formula matches solver',
      abs(rho_modify - (N_interior * 1.0 - rho_sum_before) / N_interior) < 1e-14)

# Verify interior range matches solver ReduceRhoSum_Kernel:
#   i (periodic): ni = NX6-7 = NX-1  →  [3, NX6-4)  excludes periodic duplicate
#   j (periodic): nj = NY6-7 = NY-1  →  [3, NY6-4)  excludes periodic duplicate
#   k (wall):     nk = NZ6-6 = NZ    →  [3, NZ6-3)  includes both walls
check('ni = NX6-7 = NX-1 (periodic: excludes duplicate)', ni == cfg.NX - 1)
check('nj = NY6-7 = NY-1 (periodic: excludes duplicate)', nj == cfg.NY - 1)
check('nk = NZ6-6 = NZ   (wall: includes both walls)',    nk == cfg.NZ)

# Closed-interval last index = N6-4 for all directions (3 ghost each side)
# But half-open upper bound differs: periodic [3, N6-4), wall [3, N6-3)
check('i last index = NX6-5, upper bound NX6-4 (periodic)', 3 + ni - 1 == cfg.NX6 - 5)
check('j last index = NY6-5, upper bound NY6-4 (periodic)', 3 + nj - 1 == cfg.NY6 - 5)
check('k last index = NZ6-4, upper bound NZ6-3 (wall)',     3 + nk - 1 == cfg.NZ6 - 4)


# ═══════════════════════════════════════════════════════════════
# Test D: 3D Lagrange interpolation on polynomial field (same grid)
# ═══════════════════════════════════════════════════════════════
print('\n=== Test D: 3D Lagrange interpolation (same-grid identity) ===')

from interp_checkpoint import (
    interpolate_lagrange7_3d_with_mapping,
    interpolate_phys_3d_with_mapping,
    precompute_phys_mapping_2d,
    build_grid_xyz,
    LX, LY, LZ,
)

# Use a very small grid for speed
cfg_small = GridConfig(nx=13, ny=17, nz=13, jp=1, gamma=2.0, alpha=0.5,
                       grid_dat='')

NX, NY, NZ = cfg_small.NX, cfg_small.NY, cfg_small.NZ
NX6, NY6, NZ6 = cfg_small.NX6, cfg_small.NY6, cfg_small.NZ6

# Simple uniform grid for testing (no curvature)
y_1d = np.linspace(0, LY, NY)
z_1d = np.linspace(0, LZ, NZ)
y2d = np.zeros((NY6, NZ6))
z2d = np.zeros((NY6, NZ6))
for j in range(NY):
    for k in range(NZ):
        y2d[BFR+j, BFR+k] = y_1d[j]
        z2d[BFR+j, BFR+k] = z_1d[k]

# Fill ghost for grid arrays
for g in range(BFR):
    y2d[g, :] = y2d[BFR, :] - (BFR - g) * (y_1d[1] - y_1d[0])
    y2d[NY6-1-g, :] = y2d[NY6-1-BFR, :] + (BFR - g) * (y_1d[1] - y_1d[0])
    z2d[:, g] = z2d[:, BFR]
    z2d[:, NZ6-1-g] = z2d[:, NZ6-1-BFR]

# Build field: f(x,y,z) = 1 + 0.5*sin(2*pi*x/LX) + 0.3*y/LY + 0.2*z/LZ
x_1d = np.linspace(0, LX, NX)
field_old = np.zeros((NY6, NZ6, NX6), dtype=np.float64)
for j in range(NY):
    for k in range(NZ):
        for i in range(NX):
            y_val = y_1d[j]
            z_val = z_1d[k]
            x_val = x_1d[i]
            field_old[BFR+j, BFR+k, BFR+i] = (1.0 + 0.5*np.sin(2*np.pi*x_val/LX)
                                                + 0.3*y_val/LY + 0.2*z_val/LZ)
fill_ghost(field_old, cfg_small)

# Same-grid mapping: every new point maps to itself
mapping = precompute_phys_mapping_2d(y2d, z2d, y2d, z2d, cfg_small, cfg_small)

# Lagrange-7 interpolation on same grid should be exact (or very close)
field_lag7 = interpolate_lagrange7_3d_with_mapping(field_old, mapping)
interior = (slice(BFR, BFR+NY), slice(BFR, BFR+NZ), slice(BFR, BFR+NX))
err_lag7 = np.max(np.abs(field_lag7[interior] - field_old[interior]))
check('same-grid Lagrange-7 identity (max err)',
      err_lag7 < 1e-10, f'err={err_lag7:.2e}')

# Bilinear for comparison
field_bilin = interpolate_phys_3d_with_mapping(field_old, mapping)
err_bilin = np.max(np.abs(field_bilin[interior] - field_old[interior]))
check('same-grid bilinear identity (max err)',
      err_bilin < 1e-10, f'err={err_bilin:.2e}')


# ═══════════════════════════════════════════════════════════════
# Test E: Verify mass correction integrates with Lagrange output
# ═══════════════════════════════════════════════════════════════
print('\n=== Test E: Mass correction on interpolated field ===')

# After Lagrange interpolation, perturb rho and verify correction
rho_test = field_lag7.copy()
rho_test[BFR:BFR+NY, BFR:BFR+NZ, BFR:BFR+NX] *= 1.001  # slight perturbation

ni = NX6 - 7; nj = NY6 - 7; nk = NZ6 - 6
N_int = ni * nj * nk
int_sl = (slice(BFR, BFR+nj), slice(BFR, BFR+nk), slice(BFR, BFR+ni))
mean_before_corr = np.sum(rho_test[int_sl]) / N_int

rho_mod, mb, ma = apply_rho_mass_correction(rho_test, cfg_small)
mean_after_corr = np.sum(rho_test[int_sl]) / N_int

check('mass correction restores mean=1.0',
      abs(mean_after_corr - 1.0) < 1e-14,
      f'mean_after={mean_after_corr}')
check('rho_modify is additive correction',
      abs(rho_mod - (1.0 - mean_before_corr)) < 1e-14)


# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
print(f'\n{"="*50}')
print(f'Results: {PASS} passed, {FAIL} failed')
if FAIL:
    sys.exit(1)
print('All tests passed.')

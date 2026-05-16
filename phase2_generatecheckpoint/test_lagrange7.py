#!/usr/bin/env python3
"""Unit tests for 7-point Lagrange interpolation and rho mass correction."""

import sys
import os
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from interp_checkpoint import (
    lagrange7_weights,
    lagrange7_weights_vectorized,
    clamp_wall_macros,
    apply_rho_mass_correction,
    apply_Ub_correction,
    chapman_enskog_fneq_q,
    compute_Ub,
    GridConfig,
    BFR,
    E,
    fill_ghost,
    stitch_y,
    enforce_periodic_physical_duplicates,
    compare_grid_dat_coords,
    derive_solver_grid_dat,
    validate_solver_grid_match,
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
# Test C: Wall-preserving full-domain mass correction
# ═══════════════════════════════════════════════════════════════
print('\n=== Test C: Wall-preserving full-domain mass correction ===')

# Create a small test grid
cfg = GridConfig(nx=9, ny=9, nz=9, jp=1, gamma=2.0, alpha=0.5,
                 grid_dat='test.dat')
rho = np.ones((cfg.NY6, cfg.NZ6, cfg.NX6), dtype=np.float64)
# Perturb non-wall interior
rho[BFR:BFR+2, BFR+1:BFR+3, BFR:BFR+2] = 1.05  # heavier patch

ni = cfg.NX6 - 7
nj = cfg.NY6 - 7
nk = cfg.NZ6 - 6
N_full = ni * nj * nk
N_adjust = ni * nj * (nk - 2)
full_sl = (slice(BFR, BFR+nj), slice(BFR, BFR+nk), slice(BFR, BFR+ni))
rho_sum_before = np.sum(rho[full_sl])
mean_before = rho_sum_before / N_full

rho_modify, mean_b, mean_a = apply_rho_mass_correction(rho, cfg)

rho_sum_after = np.sum(rho[full_sl])
check('mean rho == 1.0 after correction',
      abs(rho_sum_after / N_full - 1.0) < 1e-14,
      f'mean={rho_sum_after / N_full}')
check('rho_modify uses non-wall rows only',
      abs(rho_modify - (N_full * 1.0 - rho_sum_before) / N_adjust) < 1e-14)
check('bottom wall rho remains exactly 1',
      np.all(rho[BFR:BFR+nj, BFR, BFR:BFR+ni] == 1.0))
check('top wall rho remains exactly 1',
      np.all(rho[BFR:BFR+nj, BFR+nk-1, BFR:BFR+ni] == 1.0))

# Verify interior range matches solver ReduceRhoSum_Kernel:
#   i (periodic): ni = NX6-7 = NX-1  →  [3, NX6-4)  excludes periodic duplicate
#   j (periodic): nj = NY6-7 = NY-1  →  [3, NY6-4)  excludes periodic duplicate
#   k (wall):     nk = NZ6-6 = NZ    →  [3, NZ6-3)  includes both walls
check('ni = NX6-7 = NX-1 (periodic: excludes duplicate)', ni == cfg.NX - 1)
check('nj = NY6-7 = NY-1 (periodic: excludes duplicate)', nj == cfg.NY - 1)
check('nk = NZ6-6 = NZ   (wall: includes both walls)',    nk == cfg.NZ)
check('N_adjust excludes both wall rows', N_adjust == ni * nj * (cfg.NZ - 2))

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
# Test E: Boundary Lagrange interpolation uses cubic ghost extrapolation
# ═══════════════════════════════════════════════════════════════
print('\n=== Test E: Boundary Lagrange interpolation ===')

cfg_old_refine = GridConfig(nx=9, ny=9, nz=9, jp=1, gamma=2.0, alpha=0.5,
                            grid_dat='')
cfg_new_refine = GridConfig(nx=17, ny=17, nz=17, jp=1, gamma=2.0, alpha=0.5,
                            grid_dat='')

def build_uniform_grid(cfg_local):
    y1 = np.linspace(0, LY, cfg_local.NY)
    z1 = np.linspace(0, LZ, cfg_local.NZ)
    y = np.zeros((cfg_local.NY6, cfg_local.NZ6))
    z = np.zeros((cfg_local.NY6, cfg_local.NZ6))
    y[BFR:BFR+cfg_local.NY, BFR:BFR+cfg_local.NZ] = y1[:, np.newaxis]
    z[BFR:BFR+cfg_local.NY, BFR:BFR+cfg_local.NZ] = z1[np.newaxis, :]
    for g in range(BFR):
        y[g, :] = y[BFR, :] - (BFR - g) * (y1[1] - y1[0])
        y[cfg_local.NY6-1-g, :] = y[cfg_local.NY6-1-BFR, :] + (BFR - g) * (y1[1] - y1[0])
        z[:, g] = z[:, BFR]
        z[:, cfg_local.NZ6-1-g] = z[:, cfg_local.NZ6-1-BFR]
    return y, z

y_old_ref, z_old_ref = build_uniform_grid(cfg_old_refine)
y_new_ref, z_new_ref = build_uniform_grid(cfg_new_refine)
field_linear_z = np.zeros((cfg_old_refine.NY6, cfg_old_refine.NZ6, cfg_old_refine.NX6))
for j in range(cfg_old_refine.NY):
    for k in range(cfg_old_refine.NZ):
        field_linear_z[BFR+j, BFR+k, BFR:BFR+cfg_old_refine.NX] = z_old_ref[BFR+j, BFR+k]
fill_ghost(field_linear_z, cfg_old_refine)
mapping_refine = precompute_phys_mapping_2d(
    y_old_ref, z_old_ref, y_new_ref, z_new_ref, cfg_old_refine, cfg_new_refine)
field_linear_z_refined = interpolate_lagrange7_3d_with_mapping(field_linear_z, mapping_refine)
expected_z = z_new_ref[BFR:BFR+cfg_new_refine.NY, BFR:BFR+cfg_new_refine.NZ]
lag_line = field_linear_z_refined[BFR, BFR:BFR+cfg_new_refine.NZ, BFR]
check('refined-grid Lagrange-7 preserves linear z at wall band',
      np.max(np.abs(lag_line - expected_z[0])) < 1e-12,
      f'max_err={np.max(np.abs(lag_line - expected_z[0])):.2e}')


# ═══════════════════════════════════════════════════════════════
# Test F: Verify macro wall policy and mass correction integration
# ═══════════════════════════════════════════════════════════════
print('\n=== Test F: Macro wall policy and mass correction ===')

# After Lagrange interpolation, perturb rho and verify correction
rho_test = field_lag7.copy()
rho_test[BFR:BFR+NY, BFR:BFR+NZ, BFR:BFR+NX] *= 1.001  # slight perturbation
ux_test = np.ones_like(rho_test)
uy_test = np.ones_like(rho_test)
uz_test = np.ones_like(rho_test)

wall_u_before, wall_rho_before = clamp_wall_macros(
    rho_test, ux_test, uy_test, uz_test, cfg_small)
kt_small = cfg_small.NZ6 - 1 - BFR
check('wall clamp sees non-zero velocity residual', wall_u_before > 0.0)
check('wall clamp sees rho residual', wall_rho_before > 0.0)
check('wall clamp sets bottom velocity to zero',
      np.all(ux_test[:, BFR, :] == 0.0) and np.all(uy_test[:, BFR, :] == 0.0)
      and np.all(uz_test[:, BFR, :] == 0.0))
check('wall clamp sets top velocity to zero',
      np.all(ux_test[:, kt_small, :] == 0.0) and np.all(uy_test[:, kt_small, :] == 0.0)
      and np.all(uz_test[:, kt_small, :] == 0.0))
check('wall clamp sets rho to one',
      np.all(rho_test[:, BFR, :] == 1.0) and np.all(rho_test[:, kt_small, :] == 1.0))

ni = NX6 - 7; nj = NY6 - 7; nk = NZ6 - 6
N_full = ni * nj * nk
N_adjust = ni * nj * (nk - 2)
full_sl = (slice(BFR, BFR+nj), slice(BFR, BFR+nk), slice(BFR, BFR+ni))
mean_before_corr = np.sum(rho_test[full_sl]) / N_full

rho_mod, mb, ma = apply_rho_mass_correction(rho_test, cfg_small)
mean_after_corr = np.sum(rho_test[full_sl]) / N_full

check('mass correction restores mean=1.0',
      abs(mean_after_corr - 1.0) < 1e-14,
      f'mean_after={mean_after_corr}')
check('rho_modify is additive correction',
      abs(rho_mod - (1.0 - mean_before_corr) * N_full / N_adjust) < 1e-14)
check('mass correction preserves wall rho=1',
      np.all(rho_test[:, BFR, :] == 1.0) and np.all(rho_test[:, kt_small, :] == 1.0))


# ═══════════════════════════════════════════════════════════════
# Test G: U_bulk correction preserves walls and matches target
# ═══════════════════════════════════════════════════════════════
print('\n=== Test G: U_bulk correction ===')

uy_bulk = np.zeros_like(field_lag7)
uy_bulk[BFR:BFR+NY, BFR+1:BFR+NZ-1, BFR:BFR+NX] = 0.25
enforce_periodic_physical_duplicates(uy_bulk, cfg_small)
fill_ghost(uy_bulk, cfg_small)

Ub_bulk_before = compute_Ub(uy_bulk, z2d, cfg_small)
Ub_bulk_target = 1.25 * Ub_bulk_before
scale_bulk, Ub_before, Ub_after = apply_Ub_correction(
    Ub_bulk_target, uy_bulk, z2d, cfg_small)

check('U_bulk correction scale matches target ratio',
      abs(scale_bulk - 1.25) < 1e-14,
      f'scale={scale_bulk}')
check('U_bulk correction reaches target',
      abs(Ub_after - Ub_bulk_target) < 1e-14,
      f'Ub_after={Ub_after}, target={Ub_bulk_target}')
check('U_bulk correction leaves wall rows clamped',
      np.all(uy_bulk[:, BFR, :] == 0.0) and np.all(uy_bulk[:, kt_small, :] == 0.0))
check('U_bulk correction scales non-wall interior nodes',
      np.allclose(
          uy_bulk[BFR:BFR+NY, BFR+1:BFR+NZ-1, BFR:BFR+NX],
          0.25 * scale_bulk))


# ═══════════════════════════════════════════════════════════════
# Test H: CE fneq rebuild includes walls and preserves conserved moments
# ═══════════════════════════════════════════════════════════════
print('\n=== Test H: CE fneq full-domain rebuild ===')

rho_int = np.ones((NY, NZ, NX), dtype=np.float64)
zero = np.zeros_like(rho_int)
dudx = np.full_like(rho_int, 0.125)
dvdy = np.full_like(rho_int, -0.05)
dwdz = np.full_like(rho_int, 0.025)
grad = (dudx, zero, zero, zero, dvdy, zero, zero, zero, dwdz)
fneq_stack = np.stack([
    chapman_enskog_fneq_q(rho_int, grad, q, ce_coeff=-0.2)
    for q in range(19)
], axis=0)

mass_mode = np.sum(fneq_stack, axis=0)
mx_mode = np.tensordot(E[:, 0], fneq_stack, axes=(0, 0))
my_mode = np.tensordot(E[:, 1], fneq_stack, axes=(0, 0))
mz_mode = np.tensordot(E[:, 2], fneq_stack, axes=(0, 0))

check('CE fneq is rebuilt on bottom wall row',
      np.any(np.abs(fneq_stack[:, :, 0, :]) > 0.0))
check('CE fneq is rebuilt on top wall row',
      np.any(np.abs(fneq_stack[:, :, -1, :]) > 0.0))
check('CE fneq preserves zero density moment',
      np.max(np.abs(mass_mode)) < 1e-14,
      f'max={np.max(np.abs(mass_mode)):.2e}')
check('CE fneq preserves zero x-momentum moment',
      np.max(np.abs(mx_mode)) < 1e-14,
      f'max={np.max(np.abs(mx_mode)):.2e}')
check('CE fneq preserves zero y-momentum moment',
      np.max(np.abs(my_mode)) < 1e-14,
      f'max={np.max(np.abs(my_mode)):.2e}')
check('CE fneq preserves zero z-momentum moment',
      np.max(np.abs(mz_mode)) < 1e-14,
      f'max={np.max(np.abs(mz_mode)):.2e}')


# ═══════════════════════════════════════════════════════════════
# Test I: Rank stitch ignores stale checkpoint ghost rows
# ═══════════════════════════════════════════════════════════════
print('\n=== Test I: Rank stitch and periodic duplicate handling ===')

cfg_rank = GridConfig(nx=9, ny=9, nz=5, jp=2, gamma=2.0, alpha=0.5,
                      grid_dat='test.dat')
per_rank = []
for r in range(cfg_rank.JP):
    a = np.full((cfg_rank.NYD6, cfg_rank.NZ6, cfg_rank.NX6),
                -1000.0 - r, dtype=np.float64)
    for jl in range(cfg_rank.CHUNK):
        local_j = BFR + jl
        global_j = r * cfg_rank.CHUNK + BFR + jl
        a[local_j, :, :] = global_j

    # Poison rows that should not be authoritative in a stitched global field.
    a[:BFR, :, :] = 9000.0 + r
    a[BFR+cfg_rank.CHUNK:, :, :] = 8000.0 + r
    per_rank.append(a)

stitched = stitch_y(per_rank, cfg_rank)
unique_rows = stitched[BFR:BFR+cfg_rank.NY-1, BFR, BFR]
expected_rows = np.arange(BFR, BFR+cfg_rank.NY-1, dtype=np.float64)
physical = stitched[BFR:BFR+cfg_rank.NY, BFR:BFR+cfg_rank.NZ, BFR:BFR+cfg_rank.NX]

check('stitch_y copies only unique physical rows',
      np.array_equal(unique_rows, expected_rows),
      f'rows={unique_rows}')
check('stitch_y reconstructs j periodic duplicate',
      np.array_equal(stitched[BFR+cfg_rank.NY-1, :, :], stitched[BFR, :, :]))
check('stitch_y does not import stale rank ghost rows',
      not np.any(physical >= 8000.0))

dup = np.zeros((cfg_rank.NY6, cfg_rank.NZ6, cfg_rank.NX6), dtype=np.float64)
dup[BFR, :, :] = 1.0
dup[BFR+cfg_rank.NY-1, :, :] = 2.0
dup[:, :, BFR] += 3.0
dup[:, :, BFR+cfg_rank.NX-1] += 4.0
enforce_periodic_physical_duplicates(dup, cfg_rank)
check('enforce_periodic_physical_duplicates syncs j duplicate',
      np.array_equal(dup[BFR+cfg_rank.NY-1, :, :], dup[BFR, :, :]))
check('enforce_periodic_physical_duplicates syncs i duplicate',
      np.array_equal(dup[:, :, BFR+cfg_rank.NX-1], dup[:, :, BFR]))


# ═══════════════════════════════════════════════════════════════
# Test J: NEW grid must match solver runtime grid
# ═══════════════════════════════════════════════════════════════
print('\n=== Test J: Solver grid identity preflight ===')

def write_tiny_grid(path, ni=4, nj=3, delta=0.0):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('TITLE = "tiny"\n')
        f.write('VARIABLES = "x corner"\n')
        f.write('"y corner"\n')
        f.write('ZONE T="tiny"\n')
        f.write(f' I={ni}, J={nj}, K=1,F=POINT\n')
        f.write('DT=(SINGLE SINGLE )\n')
        for j in range(nj):
            for i in range(ni):
                f.write(f'{float(i):.17g} {float(j) + delta:.17g}\n')

with tempfile.TemporaryDirectory() as td:
    a = os.path.join(td, 'newgrid.dat')
    b = os.path.join(td, 'solver_same.dat')
    c = os.path.join(td, 'solver_shifted.dat')
    write_tiny_grid(a)
    write_tiny_grid(b)
    write_tiny_grid(c, delta=1e-6)
    cfg_grid = GridConfig(nx=5, ny=4, nz=3, jp=1, gamma=3.7, alpha=0.5,
                          grid_dat=a)

    same = compare_grid_dat_coords(a, b, cfg_grid)
    diff = compare_grid_dat_coords(a, c, cfg_grid)
    check('grid coordinate compare exact match',
          same['max_abs'] == 0.0 and same['count'] == 12)
    check('grid coordinate compare detects mismatch',
          diff['max_abs_y'] > 0.0 and diff['max_abs'] > 0.0)

    info_ok = validate_solver_grid_match(a, b, cfg_grid, tol=0.0, fatal=False)
    info_bad = validate_solver_grid_match(a, c, cfg_grid, tol=0.0, fatal=False)
    check('validate_solver_grid_match reports ok',
          info_ok is not None and info_ok['ok'])
    check('validate_solver_grid_match reports mismatch',
          info_bad is not None and not info_bad['ok'])

    vh = os.path.join(td, 'variables.h')
    with open(vh, 'w', encoding='utf-8') as f:
        f.write('#define GRID_DAT_DIR "J_Frohlich"\n')
        f.write('#define GRID_DAT_REF "3.fine grid.dat"\n')
    derived = derive_solver_grid_dat(vh, cfg_grid)
    expected = os.path.join(td, 'J_Frohlich',
                            'adaptive_3.fine grid_I4_J3_g3.70_a0.5.dat')
    check('derive_solver_grid_dat mirrors main.cu naming',
          derived == os.path.abspath(expected),
          f'derived={derived}')


# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
print(f'\n{"="*50}')
print(f'Results: {PASS} passed, {FAIL} failed')
if FAIL:
    sys.exit(1)
print('All tests passed.')

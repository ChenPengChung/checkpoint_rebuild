#!/usr/bin/env python3
"""
Velocity projection utilities for LBM checkpoint interpolation.

After cross-grid interpolation, scalar-wise remapping of velocity components
breaks the solenoidal constraint (div(u) != 0).  This module projects the
interpolated velocity field onto a divergence-free space via Helmholtz-Hodge
decomposition:

    u_proj = u_interp - grad(phi)

where phi solves:

    L(phi) = div(u_interp)

with periodic BCs in x (spanwise) and y (streamwise), and homogeneous Neumann
(dphi/dn = 0) in z (wall-normal).

Solver strategy: FFT in x (uniform + periodic), then sparse-direct solve of
independent 2D (j,k) Helmholtz problems for each x-wavenumber.  For wavenumber
ki, the 2D system is:

    [L_jk + lambda_ki * I] * phi_hat = b_hat

where lambda_ki = -(4/dx^2)*sin^2(pi*ki/NI) and L_jk is the compact
conservative Laplacian in the curvilinear (j,k) plane with face-averaged
metric coefficients.  Each 2D system has (NY-1)*NZ DOFs (~8K) and is solved
with sparse LU factorisation in milliseconds.

Available projection levels:
  - poisson: compact approximate scalar Poisson correction.
  - dg-exact: exact CD2 D*B*G scalar projection checked by the H diagnostic.
  - div-exact: exact minimum-norm velocity correction for D(u*) = 0 to
    roundoff under the same wall/periodic/ghost rules as the H diagnostic.

Grid layout follows interp_checkpoint.py conventions:
  - Arrays shape (NY6, NZ6, NX6) with BFR=3 ghost layers per side
  - Interior slice [BFR:BFR+NY, BFR:BFR+NZ, BFR:BFR+NX]
  - Periodic: x (i, spanwise), y (j, streamwise)
  - Walls:    z (k, wall-normal) at k=BFR (bottom) and k=BFR+NZ-1 (top)
  - Curvilinear (j,k) -> (y,z) with inverse metric (dj_dy, dj_dz, dk_dy, dk_dz)
"""

import os
import sys
import time
import numpy as np
from scipy.sparse import coo_matrix, diags, eye as speye
from scipy.sparse.linalg import splu

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from interp_checkpoint import (
    BFR, LX,
    compute_inverse_metric_2d,
    enforce_periodic_physical_duplicates,
    fill_ghost,
)


# ---------------------------------------------------------------
# CD2 building blocks (for RHS divergence and velocity correction)
# ---------------------------------------------------------------

def _gradient_cd2(phi, dx, dj_dy, dj_dz, dk_dy, dk_dz):
    """CD2 gradient of scalar phi.  All arrays are full (NY6, NZ6, NX6)."""
    dphi_di = np.zeros_like(phi)
    dphi_di[:, :, 1:-1] = (phi[:, :, 2:] - phi[:, :, :-2]) * 0.5

    dphi_dj = np.zeros_like(phi)
    dphi_dj[1:-1, :, :] = (phi[2:, :, :] - phi[:-2, :, :]) * 0.5

    dphi_dk = np.zeros_like(phi)
    dphi_dk[:, 1:-1, :] = (phi[:, 2:, :] - phi[:, :-2, :]) * 0.5

    jy = dj_dy[:, :, np.newaxis]
    jz = dj_dz[:, :, np.newaxis]
    ky = dk_dy[:, :, np.newaxis]
    kz = dk_dz[:, :, np.newaxis]

    gx = dphi_di / dx
    gy = jy * dphi_dj + ky * dphi_dk
    gz = jz * dphi_dj + kz * dphi_dk
    return gx, gy, gz


def _divergence_cd2(vx, vy, vz, dx, dj_dy, dj_dz, dk_dy, dk_dz):
    """CD2 divergence of vector (vx, vy, vz).  All arrays full-sized."""
    dvx_di = np.zeros_like(vx)
    dvx_di[:, :, 1:-1] = (vx[:, :, 2:] - vx[:, :, :-2]) * 0.5

    dvy_dj = np.zeros_like(vy)
    dvy_dj[1:-1, :, :] = (vy[2:, :, :] - vy[:-2, :, :]) * 0.5
    dvy_dk = np.zeros_like(vy)
    dvy_dk[:, 1:-1, :] = (vy[:, 2:, :] - vy[:, :-2, :]) * 0.5

    dvz_dj = np.zeros_like(vz)
    dvz_dj[1:-1, :, :] = (vz[2:, :, :] - vz[:-2, :, :]) * 0.5
    dvz_dk = np.zeros_like(vz)
    dvz_dk[:, 1:-1, :] = (vz[:, 2:, :] - vz[:, :-2, :]) * 0.5

    jy = dj_dy[:, :, np.newaxis]
    jz = dj_dz[:, :, np.newaxis]
    ky = dk_dy[:, :, np.newaxis]
    kz = dk_dz[:, :, np.newaxis]

    return (dvx_di / dx
            + jy * dvy_dj + ky * dvy_dk
            + jz * dvz_dj + kz * dvz_dk)


# ---------------------------------------------------------------
# Phi boundary conditions: periodic x,y + Neumann mirror z
# ---------------------------------------------------------------

def _fill_phi_bc(phi, cfg):
    """Set periodic duplicates, then ghost: X periodic, Z Neumann mirror, Y periodic."""
    enforce_periodic_physical_duplicates(phi, cfg)

    nx6, ny6 = cfg.NX6, cfg.NY6
    kt = BFR + cfg.NZ - 1

    phi[:, :, 2]       = phi[:, :, nx6 - 5]
    phi[:, :, 1]       = phi[:, :, nx6 - 6]
    phi[:, :, 0]       = phi[:, :, nx6 - 7]
    phi[:, :, nx6 - 3] = phi[:, :, 4]
    phi[:, :, nx6 - 2] = phi[:, :, 5]
    phi[:, :, nx6 - 1] = phi[:, :, 6]

    phi[:, BFR - 1, :] = phi[:, BFR + 1, :]
    phi[:, BFR - 2, :] = phi[:, BFR + 2, :]
    phi[:, BFR - 3, :] = phi[:, BFR + 3, :]
    phi[:, kt + 1, :]  = phi[:, kt - 1, :]
    phi[:, kt + 2, :]  = phi[:, kt - 2, :]
    phi[:, kt + 3, :]  = phi[:, kt - 3, :]

    phi[2, :, :]       = phi[ny6 - 5, :, :]
    phi[1, :, :]       = phi[ny6 - 6, :, :]
    phi[0, :, :]       = phi[ny6 - 7, :, :]
    phi[ny6 - 3, :, :] = phi[4, :, :]
    phi[ny6 - 2, :, :] = phi[5, :, :]
    phi[ny6 - 1, :, :] = phi[6, :, :]


# ---------------------------------------------------------------
# FFT-based Poisson solver
# ---------------------------------------------------------------

def _build_2d_laplacian(gjj_2d, gkk_2d, nj, nk):
    """Build 2D compact conservative Laplacian as sparse CSR.

    Diagonal metric only (g^{jj}, g^{kk}); cross terms g^{jk} omitted
    to preserve strict symmetry.  j is periodic, k has Neumann mirror BCs.

    Returns CSR matrix of size (nj*nk, nj*nk).
    """
    n = nj * nk

    j_loc = np.arange(nj)
    k_loc = np.arange(nk)
    J, K = np.meshgrid(j_loc, k_loc, indexing='ij')
    J_flat = J.ravel()
    K_flat = K.ravel()
    P = J_flat * nk + K_flat

    J_abs = BFR + J_flat
    K_abs = BFR + K_flat

    # j-direction face averages (metric includes ghost/periodic-duplicate nodes)
    gjj_jp = (gjj_2d[J_abs, K_abs] + gjj_2d[J_abs + 1, K_abs]) * 0.5
    gjj_jm = (gjj_2d[J_abs, K_abs] + gjj_2d[J_abs - 1, K_abs]) * 0.5

    jp_idx = ((J_flat + 1) % nj) * nk + K_flat
    jm_idx = ((J_flat - 1) % nj) * nk + K_flat

    # k-direction face averages (ghost nodes available for ±1 of interior)
    gkk_kp = (gkk_2d[J_abs, K_abs] + gkk_2d[J_abs, K_abs + 1]) * 0.5
    gkk_km = (gkk_2d[J_abs, K_abs] + gkk_2d[J_abs, K_abs - 1]) * 0.5

    rows_list = []
    cols_list = []
    data_list = []

    # --- Diagonal: -(gjj_p + gjj_m + gkk_p + gkk_m) ---
    diag_val = -(gjj_jp + gjj_jm + gkk_kp + gkk_km)
    rows_list.append(P)
    cols_list.append(P)
    data_list.append(diag_val)

    # --- j+1 and j-1 neighbours (periodic) ---
    rows_list.append(P)
    cols_list.append(jp_idx)
    data_list.append(gjj_jp)

    rows_list.append(P)
    cols_list.append(jm_idx)
    data_list.append(gjj_jm)

    # --- k-direction: separate interior / boundary treatment ---
    interior = (K_flat > 0) & (K_flat < nk - 1)
    bot = K_flat == 0
    top = K_flat == nk - 1

    # Interior k: k+1 and k-1 neighbours
    kp_int = J_flat[interior] * nk + (K_flat[interior] + 1)
    km_int = J_flat[interior] * nk + (K_flat[interior] - 1)
    rows_list.append(P[interior])
    cols_list.append(kp_int)
    data_list.append(gkk_kp[interior])
    rows_list.append(P[interior])
    cols_list.append(km_int)
    data_list.append(gkk_km[interior])

    # Bottom Neumann (k=0): phi_{k=-1} = phi_{k=1} → k=1 gets (gkk_p + gkk_m)
    k1_bot = J_flat[bot] * nk + 1
    rows_list.append(P[bot])
    cols_list.append(k1_bot)
    data_list.append(gkk_kp[bot] + gkk_km[bot])

    # Top Neumann (k=nk-1): phi_{k=nk} = phi_{k=nk-2} → k=nk-2 gets (gkk_p + gkk_m)
    km2_top = J_flat[top] * nk + (nk - 2)
    rows_list.append(P[top])
    cols_list.append(km2_top)
    data_list.append(gkk_kp[top] + gkk_km[top])

    rows_all = np.concatenate(rows_list)
    cols_all = np.concatenate(cols_list)
    data_all = np.concatenate(data_list)

    return coo_matrix((data_all, (rows_all, cols_all)), shape=(n, n)).tocsr()


def _build_j_derivative(nj, nk):
    """Periodic CD2 derivative in the j direction on unique (j,k) DOFs."""
    rows = []
    cols = []
    data = []
    for j in range(nj):
        jp = (j + 1) % nj
        jm = (j - 1) % nj
        for k in range(nk):
            p = j * nk + k
            rows.extend((p, p))
            cols.extend((jp * nk + k, jm * nk + k))
            data.extend((0.5, -0.5))
    n = nj * nk
    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def _build_phi_k_derivative(nj, nk):
    """CD2 k-derivative for phi with Neumann mirror wall ghosts."""
    rows = []
    cols = []
    data = []
    for j in range(nj):
        for k in range(1, nk - 1):
            p = j * nk + k
            rows.extend((p, p))
            cols.extend((j * nk + k + 1, j * nk + k - 1))
            data.extend((0.5, -0.5))
    n = nj * nk
    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def _build_vector_k_derivative(nj, nk):
    """CD2 k-derivative for vector fields with constant-copy wall ghosts."""
    rows = []
    cols = []
    data = []
    for j in range(nj):
        for k in range(nk):
            p = j * nk + k
            if k == 0:
                rows.extend((p, p))
                cols.extend((j * nk + 1, j * nk))
                data.extend((0.5, -0.5))
            elif k == nk - 1:
                rows.extend((p, p))
                cols.extend((j * nk + k, j * nk + k - 1))
                data.extend((0.5, -0.5))
            else:
                rows.extend((p, p))
                cols.extend((j * nk + k + 1, j * nk + k - 1))
                data.extend((0.5, -0.5))
    n = nj * nk
    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def _build_dg_exact_yz_operator(dj_dy, dj_dz, dk_dy, dk_dz, nj, nk):
    """Build the exact 2D part of the CD2 D*B*G operator.

    The matrix matches these runtime steps:
      1. phi uses periodic j ghosts and Neumann mirror k ghosts;
      2. G(phi) is converted to physical y/z components;
      3. wall planes of the correction vector are clamped to zero;
      4. divergence uses periodic j ghosts and constant-copy vector k ghosts.
    """
    s = (slice(BFR, BFR + nj), slice(BFR, BFR + nk))
    jy = dj_dy[s].ravel()
    jz = dj_dz[s].ravel()
    ky = dk_dy[s].ravel()
    kz = dk_dz[s].ravel()

    n = nj * nk
    wall_mask = np.ones((nj, nk), dtype=np.float64)
    wall_mask[:, 0] = 0.0
    wall_mask[:, -1] = 0.0
    w = wall_mask.ravel()

    Dj_phi = _build_j_derivative(nj, nk)
    Dk_phi = _build_phi_k_derivative(nj, nk)
    Dj_vec = Dj_phi
    Dk_vec = _build_vector_k_derivative(nj, nk)

    Jy = diags(jy, 0, shape=(n, n), format='csr')
    Jz = diags(jz, 0, shape=(n, n), format='csr')
    Ky = diags(ky, 0, shape=(n, n), format='csr')
    Kz = diags(kz, 0, shape=(n, n), format='csr')
    W = diags(w, 0, shape=(n, n), format='csr')

    Gy = W @ (Jy @ Dj_phi + Ky @ Dk_phi)
    Gz = W @ (Jz @ Dj_phi + Kz @ Dk_phi)

    A_yz = (Jy @ Dj_vec @ Gy
            + Ky @ Dk_vec @ Gy
            + Jz @ Dj_vec @ Gz
            + Kz @ Dk_vec @ Gz)
    return A_yz.tocsc(), w


class _FftPoissonSolver:
    """Cached FFT(x) + sparse-LU(j,k) approximate Poisson inverse."""

    def __init__(self, dx, gjj_2d, gkk_2d, nj, nk, ni, verbose=True):
        self.dx = dx
        self.nj = nj
        self.nk = nk
        self.ni = ni
        self.verbose = verbose

        t0 = time.time()
        n_modes = ni // 2 + 1
        ki_arr = np.arange(n_modes)
        self.lambda_x = -(1.0 / (dx * dx)) * np.sin(2.0 * np.pi * ki_arr / ni) ** 2

        self.nyquist_modes = {0}
        if ni % 2 == 0 and ni // 2 < n_modes:
            self.nyquist_modes.add(ni // 2)

        L_2d = _build_2d_laplacian(gjj_2d, gkk_2d, nj, nk)
        n_2d = nj * nk
        I_2d = speye(n_2d, format='csc', dtype=np.float64)
        L_csc = L_2d.tocsc()

        self.lu = []
        reg = 1e-12 / (dx * dx)
        for ki_idx in range(n_modes):
            A = L_csc + self.lambda_x[ki_idx] * I_2d
            if ki_idx in self.nyquist_modes:
                A = A + reg * I_2d
            self.lu.append(splu(A))

        if verbose:
            print('      2D Laplacian: {} x {} ({} nnz), {} modes factored in {:.2f}s'.format(
                n_2d, n_2d, L_2d.nnz, n_modes, time.time() - t0))

    def solve(self, b_3d):
        """Apply cached approximate inverse to b_3d shaped (nj,nk,ni)."""
        b_hat = np.fft.rfft(b_3d, axis=2)
        phi_hat = np.zeros_like(b_hat)

        for ki_idx, lu in enumerate(self.lu):
            rhs_re = b_hat[:, :, ki_idx].real.ravel().copy()
            rhs_im = b_hat[:, :, ki_idx].imag.ravel().copy()

            if ki_idx in self.nyquist_modes:
                rhs_re -= np.mean(rhs_re)
                rhs_im -= np.mean(rhs_im)

            sol_re = lu.solve(rhs_re)
            sol_im = lu.solve(rhs_im)

            if ki_idx in self.nyquist_modes:
                sol_re -= np.mean(sol_re)
                sol_im -= np.mean(sol_im)

            phi_hat[:, :, ki_idx] = (sol_re + 1j * sol_im).reshape(
                self.nj, self.nk)

        return np.fft.irfft(phi_hat, n=self.ni, axis=2).real


class _DgExactFftSolver:
    """FFT(x) + sparse-LU(j,k) solver for the exact CD2 D*B*G operator."""

    def __init__(self, dx, dj_dy, dj_dz, dk_dy, dk_dz,
                 nj, nk, ni, verbose=True):
        self.dx = dx
        self.nj = nj
        self.nk = nk
        self.ni = ni
        self.verbose = verbose

        t0 = time.time()
        n_modes = ni // 2 + 1
        ki_arr = np.arange(n_modes)
        self.lambda_x = -(1.0 / (dx * dx)) * np.sin(2.0 * np.pi * ki_arr / ni) ** 2

        self.A_yz, wall_mask = _build_dg_exact_yz_operator(
            dj_dy, dj_dz, dk_dy, dk_dz, nj, nk)
        self.W = diags(wall_mask, 0, shape=(nj * nk, nj * nk), format='csc')

        diag_scale = np.max(np.abs(self.A_yz.diagonal()))
        scale = max(float(diag_scale), 1.0 / (dx * dx), 1.0)
        self.reg = 1.0e-13 * scale
        self.singular_modes = {0}
        if ni % 2 == 0 and ni // 2 < n_modes:
            # CD2 first derivative has zero eigenvalue at Nyquist.
            self.singular_modes.add(ni // 2)

        I_2d = speye(nj * nk, format='csc', dtype=np.float64)
        self.lu = []
        self.op_matrices = []
        for ki_idx, lam in enumerate(self.lambda_x):
            A_op = (self.A_yz + lam * self.W).tocsc()
            A_solve = A_op
            if ki_idx in self.singular_modes:
                A_solve = A_solve + self.reg * I_2d
            self.op_matrices.append(A_op)
            self.lu.append(splu(A_solve.tocsc()))

        if verbose:
            print('      exact D*G 2D operator: {} x {} ({} nnz), {} modes factored in {:.2f}s'.format(
                nj * nk, nj * nk, self.A_yz.nnz, n_modes, time.time() - t0))

    def solve(self, b_3d):
        """Solve A phi = b for b shaped (nj,nk,ni)."""
        b_hat = np.fft.rfft(b_3d, axis=2)
        phi_hat = np.zeros_like(b_hat)

        for ki_idx, lu in enumerate(self.lu):
            rhs_re = b_hat[:, :, ki_idx].real.ravel().copy()
            rhs_im = b_hat[:, :, ki_idx].imag.ravel().copy()

            if ki_idx in self.singular_modes:
                rhs_re -= np.mean(rhs_re)
                rhs_im -= np.mean(rhs_im)

            sol_re = lu.solve(rhs_re)
            sol_im = lu.solve(rhs_im)

            if ki_idx in self.singular_modes:
                sol_re -= np.mean(sol_re)
                sol_im -= np.mean(sol_im)

            phi_hat[:, :, ki_idx] = (sol_re + 1j * sol_im).reshape(
                self.nj, self.nk)

        return np.fft.irfft(phi_hat, n=self.ni, axis=2).real

    def apply(self, phi_3d):
        """Apply the exact D*B*G operator to phi_3d shaped (nj,nk,ni)."""
        phi_hat = np.fft.rfft(phi_3d, axis=2)
        out_hat = np.zeros_like(phi_hat)

        for ki_idx, A in enumerate(self.op_matrices):
            phi_re = phi_hat[:, :, ki_idx].real.ravel()
            phi_im = phi_hat[:, :, ki_idx].imag.ravel()
            out_hat[:, :, ki_idx] = (
                A.dot(phi_re) + 1j * A.dot(phi_im)
            ).reshape(self.nj, self.nk)

        return np.fft.irfft(out_hat, n=self.ni, axis=2).real


def _build_divergence_yz_operators(dj_dy, dj_dz, dk_dy, dk_dz, nj, nk):
    """Build exact y/z divergence blocks for wall-clamped velocity DOFs."""
    s = (slice(BFR, BFR + nj), slice(BFR, BFR + nk))
    jy = dj_dy[s].ravel()
    jz = dj_dz[s].ravel()
    ky = dk_dy[s].ravel()
    kz = dk_dz[s].ravel()

    n = nj * nk
    wall_mask = np.ones((nj, nk), dtype=np.float64)
    wall_mask[:, 0] = 0.0
    wall_mask[:, -1] = 0.0
    w = wall_mask.ravel()

    W = diags(w, 0, shape=(n, n), format='csc')
    Dj = _build_j_derivative(nj, nk).tocsc()
    Dk = _build_vector_k_derivative(nj, nk).tocsc()

    Jy = diags(jy, 0, shape=(n, n), format='csc')
    Jz = diags(jz, 0, shape=(n, n), format='csc')
    Ky = diags(ky, 0, shape=(n, n), format='csc')
    Kz = diags(kz, 0, shape=(n, n), format='csc')

    By = (Jy @ Dj @ W + Ky @ Dk @ W).tocsc()
    Bz = (Jz @ Dj @ W + Kz @ Dk @ W).tocsc()
    return W, By, Bz


class _DivExactFftProjector:
    """FFT(x) exact discrete mass projection via minimum-norm velocity correction."""

    def __init__(self, dx, dj_dy, dj_dz, dk_dy, dk_dz,
                 nj, nk, ni, verbose=True):
        self.dx = dx
        self.nj = nj
        self.nk = nk
        self.ni = ni
        self.verbose = verbose

        t0 = time.time()
        n_modes = ni // 2 + 1
        theta = 2.0 * np.pi * np.arange(n_modes) / ni
        self.alpha_x = 1j * np.sin(theta) / dx

        self.W, self.By, self.Bz = _build_divergence_yz_operators(
            dj_dy, dj_dz, dk_dy, dk_dz, nj, nk)

        A_yz = self.By @ self.By.T + self.Bz @ self.Bz.T
        diag_scale = np.max(np.abs(A_yz.diagonal()))
        scale = max(float(diag_scale), 1.0 / (dx * dx), 1.0)
        self.reg = 1.0e-13 * scale

        n = nj * nk
        I_2d = speye(n, format='csc', dtype=np.float64)
        self.lu = []
        self.A_ops = []
        for alpha in self.alpha_x:
            A = (A_yz + (abs(alpha) ** 2) * self.W).tocsc()
            self.A_ops.append(A)
            self.lu.append(splu((A + self.reg * I_2d).tocsc()))

        if verbose:
            print('      exact D velocity projector: {} x {} ({} nnz), {} modes factored in {:.2f}s'.format(
                n, n, A_yz.nnz, n_modes, time.time() - t0))

    def correction(self, rhs_3d):
        """Return minimum-norm q satisfying D q ~= rhs, arrays shaped (nj,nk,ni)."""
        rhs_hat = np.fft.rfft(rhs_3d, axis=2)
        qx_hat = np.zeros_like(rhs_hat)
        qy_hat = np.zeros_like(rhs_hat)
        qz_hat = np.zeros_like(rhs_hat)

        for ki_idx, lu in enumerate(self.lu):
            rhs_re = rhs_hat[:, :, ki_idx].real.ravel().copy()
            rhs_im = rhs_hat[:, :, ki_idx].imag.ravel().copy()

            lam_re = lu.solve(rhs_re)
            lam_im = lu.solve(rhs_im)
            lam = lam_re + 1j * lam_im

            alpha = self.alpha_x[ki_idx]
            qx_hat[:, :, ki_idx] = (
                np.conjugate(alpha) * (self.W @ lam)
            ).reshape(self.nj, self.nk)
            qy_hat[:, :, ki_idx] = (self.By.T @ lam).reshape(self.nj, self.nk)
            qz_hat[:, :, ki_idx] = (self.Bz.T @ lam).reshape(self.nj, self.nk)

        qx = np.fft.irfft(qx_hat, n=self.ni, axis=2).real
        qy = np.fft.irfft(qy_hat, n=self.ni, axis=2).real
        qz = np.fft.irfft(qz_hat, n=self.ni, axis=2).real
        return qx, qy, qz


def _solve_poisson_fft(b_3d, dx, gjj_2d, gkk_2d, nj, nk, ni, verbose=True):
    """Solve Poisson equation using FFT in x + sparse LU in (j,k).

    For each x-wavenumber ki, factors the real matrix A = L_2d + lambda_ki*I
    once, then forward/back-solves for real and imaginary RHS parts separately.
    This avoids complex-valued LU factorisation (the main bottleneck).

    Parameters
    ----------
    b_3d  : ndarray (nj, nk, ni) — RHS on unique interior DOFs
    dx    : float — uniform x spacing
    gjj_2d, gkk_2d : ndarray (NY6, NZ6) — contravariant diagonal metric
    nj, nk, ni : int — unique DOF counts (NY-1, NZ, NX-1)
    verbose : bool

    Returns
    -------
    phi_3d : ndarray (nj, nk, ni) — solution
    """
    t0 = time.time()
    solver = _FftPoissonSolver(dx, gjj_2d, gkk_2d, nj, nk, ni, verbose=verbose)
    phi_3d = solver.solve(b_3d)
    if verbose:
        print('      FFT solve in {:.2f}s'.format(time.time() - t0))
    return phi_3d


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

class PoissonProjectionError(RuntimeError):
    """Raised when the Poisson solver fails to converge."""
    pass


def divergence_diagnostic(ux, uy, uz, cfg, y_2d, z_2d):
    """Return (div_rms, div_max) on unique interior points using CD2."""
    dx = LX / (cfg.NX - 1)
    dj_dy, dj_dz, dk_dy, dk_dz = compute_inverse_metric_2d(y_2d, z_2d)
    div_full = _divergence_cd2(ux, uy, uz, dx, dj_dy, dj_dz, dk_dy, dk_dz)
    s = (slice(BFR, BFR + cfg.NY - 1),
         slice(BFR, BFR + cfg.NZ),
         slice(BFR, BFR + cfg.NX - 1))
    d = div_full[s]
    return float(np.sqrt(np.mean(d ** 2))), float(np.max(np.abs(d)))


def poisson_project(ux, uy, uz, cfg, y_2d, z_2d,
                    max_outer=80, div_tol=1e-6, verbose=True,
                    clamp_walls=True):
    """Project velocity onto divergence-free space.

    Uses outer Richardson iterations to close the gap between the compact
    Laplacian L (used for the Poisson solve) and the CD2 D*G operators
    (used for divergence/gradient in the velocity correction).

    Each outer iteration:
      1. r = div_CD2(u_current)              — residual divergence
      2. dphi = L^{-1}(r)                    — FFT Poisson solve
      3. u_current -= grad_CD2(dphi)          — velocity correction
      4. optionally re-apply no-slip wall clamp, then periodic BCs

    Iterations stop when div RMS < div_tol or stagnates.

    Parameters
    ----------
    ux, uy, uz : ndarray (NY6, NZ6, NX6), ghost-filled velocity components
    cfg        : GridConfig from interp_checkpoint
    y_2d, z_2d : ndarray (NY6, NZ6), grid coordinates from build_grid_xyz
    max_outer  : maximum outer Richardson iterations
    div_tol    : convergence tolerance on div RMS
    verbose    : print progress
    clamp_walls : keep physical wall planes at no-slip during the projection

    Returns
    -------
    ux_proj, uy_proj, uz_proj : corrected velocity (ghost-filled)
    info : dict with divergence and solver diagnostics
    """
    dx = LX / (cfg.NX - 1)
    dj_dy, dj_dz, dk_dy, dk_dz = compute_inverse_metric_2d(y_2d, z_2d)

    nj = cfg.NY - 1
    nk = cfg.NZ
    ni = cfg.NX - 1
    n_dof = nj * nk * ni
    shape3 = (cfg.NY6, cfg.NZ6, cfg.NX6)

    uniq = (slice(BFR, BFR + nj),
            slice(BFR, BFR + nk),
            slice(BFR, BFR + ni))
    full_int = (slice(BFR, BFR + cfg.NY),
                slice(BFR, BFR + cfg.NZ),
                slice(BFR, BFR + cfg.NX))

    gjj_2d = dj_dy**2 + dj_dz**2
    gkk_2d = dk_dy**2 + dk_dz**2

    # Working copies
    ux_w = ux.copy()
    uy_w = uy.copy()
    uz_w = uz.copy()

    def _enforce_projection_bc():
        if clamp_walls:
            kt_wall = cfg.NZ6 - 1 - BFR
            for arr in (ux_w, uy_w, uz_w):
                arr[:, BFR, :] = 0.0
                arr[:, kt_wall, :] = 0.0
        for arr in (ux_w, uy_w, uz_w):
            enforce_periodic_physical_duplicates(arr, cfg)
            fill_ghost(arr, cfg)

    _enforce_projection_bc()

    # Initial divergence after enforcing the projection boundary constraints.
    div_u = _divergence_cd2(ux_w, uy_w, uz_w, dx, dj_dy, dj_dz, dk_dy, dk_dz)
    b_arr = div_u[uniq].copy()
    div_rms_before = float(np.sqrt(np.mean(b_arr ** 2)))
    div_max_before = float(np.max(np.abs(b_arr)))

    if verbose:
        print('      Poisson projection: {} DOFs ({} x {} x {})'.format(
            n_dof, nj, nk, ni))
        print('      div(u) BEFORE: RMS = {:.6e}, max = {:.6e}'.format(
            div_rms_before, div_max_before))

    t0_total = time.time()
    n_outer = 0
    prev_rms = div_rms_before

    for outer in range(max_outer):
        # Residual divergence
        if outer == 0:
            r_arr = b_arr.copy()
        else:
            div_r = _divergence_cd2(ux_w, uy_w, uz_w,
                                    dx, dj_dy, dj_dz, dk_dy, dk_dz)
            r_arr = div_r[uniq].copy()

        r_rms = float(np.sqrt(np.mean(r_arr ** 2)))
        r_max = float(np.max(np.abs(r_arr)))

        if verbose:
            print('      outer iter {}: div RMS = {:.6e}, max = {:.6e}'.format(
                outer, r_rms, r_max))

        if r_rms < div_tol:
            if verbose:
                print('      converged (div RMS < {:.0e})'.format(div_tol))
            break

        if outer > 0 and r_rms > 0.99 * prev_rms:
            if verbose:
                print('      stagnated (ratio {:.4f}), stopping'.format(
                    r_rms / prev_rms))
            break

        prev_rms = r_rms

        # Remove mean for Neumann compatibility
        r_arr -= np.mean(r_arr)

        # FFT Poisson solve
        try:
            dphi_int = _solve_poisson_fft(
                r_arr, dx, gjj_2d, gkk_2d, nj, nk, ni,
                verbose=(verbose and outer == 0))
        except Exception as e:
            raise PoissonProjectionError(
                'FFT Poisson solver failed at outer iter {}: {}'.format(outer, e))

        dphi_int -= np.mean(dphi_int)

        if not np.all(np.isfinite(dphi_int)):
            raise PoissonProjectionError(
                'Poisson solution contains NaN/Inf at outer iter {}'.format(outer))

        # Embed phi and compute gradient correction
        phi = np.zeros(shape3, dtype=np.float64)
        phi[BFR:BFR + nj, BFR:BFR + nk, BFR:BFR + ni] = dphi_int
        _fill_phi_bc(phi, cfg)

        gx, gy, gz = _gradient_cd2(phi, dx, dj_dy, dj_dz, dk_dy, dk_dz)

        # Apply correction to working velocity
        ux_w[full_int] -= gx[full_int]
        uy_w[full_int] -= gy[full_int]
        uz_w[full_int] -= gz[full_int]

        _enforce_projection_bc()

        n_outer = outer + 1

    dt_total = time.time() - t0_total

    # Final divergence diagnostic
    div_final = _divergence_cd2(ux_w, uy_w, uz_w,
                                dx, dj_dy, dj_dz, dk_dy, dk_dz)
    d_cons = div_final[uniq]
    div_rms_after = float(np.sqrt(np.mean(d_cons ** 2)))
    div_max_after = float(np.max(np.abs(d_cons)))

    wall_excl = (slice(BFR, BFR + nj),
                 slice(BFR + 1, BFR + nk - 1),
                 slice(BFR, BFR + ni))
    d_int = div_final[wall_excl]
    div_rms_interior = float(np.sqrt(np.mean(d_int ** 2)))
    div_max_interior = float(np.max(np.abs(d_int)))

    if verbose:
        print('      {} outer iterations, {:.1f}s total'.format(n_outer, dt_total))
        print('      div(u) AFTER:')
        print('        all points: RMS = {:.6e}, max = {:.6e}'.format(
            div_rms_after, div_max_after))
        print('        interior (excl walls): RMS = {:.6e}, max = {:.6e}'.format(
            div_rms_interior, div_max_interior))
        if div_rms_before > 0:
            print('      reduction: {:.2e}x (all), {:.2e}x (interior)'.format(
                div_rms_after / div_rms_before,
                div_rms_interior / div_rms_before))
        corr_max = float(max(np.max(np.abs(ux_w[full_int] - ux[full_int])),
                             np.max(np.abs(uy_w[full_int] - uy[full_int])),
                             np.max(np.abs(uz_w[full_int] - uz[full_int]))))
        print('      max |du| correction = {:.6e}'.format(corr_max))

    info = {
        'div_rms_before': div_rms_before,
        'div_max_before': div_max_before,
        'div_rms_after': div_rms_after,
        'div_max_after': div_max_after,
        'div_rms_interior': div_rms_interior,
        'div_max_interior': div_max_interior,
        'solver_info': 0,
        'outer_iters': n_outer,
        'solve_time_s': dt_total,
        'n_dof': n_dof,
    }
    return ux_w, uy_w, uz_w, info


def poisson_project_dg_exact(ux, uy, uz, cfg, y_2d, z_2d,
                             max_outer=80, div_tol=1e-10, verbose=True,
                             clamp_walls=True, gmres_restart=50):
    """Project velocity with the exact verification operator D*G.

    This solves the same discrete equation that the H diagnostic checks:

        D(u - G phi) = 0  ->  (D B G) phi = D(B u)

    where B is the linear boundary projection used by the restart field:
    optional no-slip wall clamp plus periodic/ghost fill.  The solve uses the
    x-periodicity to diagonalize the operator into independent 2D sparse
    direct solves.  max_outer and gmres_restart are accepted for CLI/API
    compatibility with the approximate projection path.
    """
    dx = LX / (cfg.NX - 1)
    dj_dy, dj_dz, dk_dy, dk_dz = compute_inverse_metric_2d(y_2d, z_2d)

    nj = cfg.NY - 1
    nk = cfg.NZ
    ni = cfg.NX - 1
    n_dof = nj * nk * ni
    shape3 = (cfg.NY6, cfg.NZ6, cfg.NX6)

    uniq = (slice(BFR, BFR + nj),
            slice(BFR, BFR + nk),
            slice(BFR, BFR + ni))
    full_int = (slice(BFR, BFR + cfg.NY),
                slice(BFR, BFR + cfg.NZ),
                slice(BFR, BFR + cfg.NX))
    wall_excl = (slice(BFR, BFR + nj),
                 slice(BFR + 1, BFR + nk - 1),
                 slice(BFR, BFR + ni))

    def _apply_vector_bc(vx, vy, vz):
        if clamp_walls:
            kt_wall = cfg.NZ6 - 1 - BFR
            for arr in (vx, vy, vz):
                arr[:, BFR, :] = 0.0
                arr[:, kt_wall, :] = 0.0
        for arr in (vx, vy, vz):
            enforce_periodic_physical_duplicates(arr, cfg)
            fill_ghost(arr, cfg)

    def _embed_phi(phi_vec):
        phi = np.zeros(shape3, dtype=np.float64)
        phi_int = phi_vec.reshape(nj, nk, ni)
        phi[BFR:BFR + nj, BFR:BFR + nk, BFR:BFR + ni] = phi_int
        _fill_phi_bc(phi, cfg)
        return phi

    def _grad_projected(phi_vec):
        phi = _embed_phi(phi_vec)
        gx, gy, gz = _gradient_cd2(phi, dx, dj_dy, dj_dz, dk_dy, dk_dz)
        _apply_vector_bc(gx, gy, gz)
        return gx, gy, gz

    ux_w = ux.copy()
    uy_w = uy.copy()
    uz_w = uz.copy()
    _apply_vector_bc(ux_w, uy_w, uz_w)

    div_u = _divergence_cd2(ux_w, uy_w, uz_w, dx, dj_dy, dj_dz, dk_dy, dk_dz)
    rhs_arr = div_u[uniq].copy()
    rhs_mean = float(np.mean(rhs_arr))

    div_rms_before = float(np.sqrt(np.mean(div_u[uniq] ** 2)))
    div_max_before = float(np.max(np.abs(div_u[uniq])))
    rhs_norm = float(np.linalg.norm(rhs_arr.ravel()))
    rms_target = float(div_tol)

    if verbose:
        print('      DG-exact projection: {} DOFs ({} x {} x {})'.format(
            n_dof, nj, nk, ni))
        print('      div(u) BEFORE: RMS = {:.6e}, max = {:.6e}, mean = {:.6e}'.format(
            div_rms_before, div_max_before, rhs_mean))
        print('      direct D*G target: RMS <= {:.3e}'.format(rms_target))

    if rhs_norm == 0.0:
        return ux_w, uy_w, uz_w, {
            'div_rms_before': div_rms_before,
            'div_max_before': div_max_before,
            'div_rms_after': div_rms_before,
            'div_max_after': div_max_before,
            'div_rms_interior': float(np.sqrt(np.mean(div_u[wall_excl] ** 2))),
            'div_max_interior': float(np.max(np.abs(div_u[wall_excl]))),
            'solver_info': 0,
            'outer_iters': 0,
            'solve_time_s': 0.0,
            'n_dof': n_dof,
            'rhs_mean': rhs_mean,
            'method': 'dg-exact',
        }

    t0 = time.time()
    try:
        dg_solver = _DgExactFftSolver(
            dx, dj_dy, dj_dz, dk_dy, dk_dz, nj, nk, ni, verbose=verbose)
        phi_int = dg_solver.solve(rhs_arr)
    except Exception as e:
        raise PoissonProjectionError('DG-exact direct solver failed: {}'.format(e))
    dt_total = time.time() - t0

    phi_int = np.asarray(phi_int, dtype=np.float64)
    phi_int -= np.mean(phi_int)

    if not np.all(np.isfinite(phi_int)):
        raise PoissonProjectionError('DG-exact phi contains NaN/Inf')

    true_res = rhs_arr - dg_solver.apply(phi_int)
    true_res_rms = float(np.sqrt(np.mean(true_res ** 2)))
    true_res_max = float(np.max(np.abs(true_res)))

    gx, gy, gz = _grad_projected(phi_int.ravel())
    ux_w[full_int] -= gx[full_int]
    uy_w[full_int] -= gy[full_int]
    uz_w[full_int] -= gz[full_int]
    _apply_vector_bc(ux_w, uy_w, uz_w)

    div_final = _divergence_cd2(ux_w, uy_w, uz_w,
                                dx, dj_dy, dj_dz, dk_dy, dk_dz)
    d_cons = div_final[uniq]
    div_rms_after = float(np.sqrt(np.mean(d_cons ** 2)))
    div_max_after = float(np.max(np.abs(d_cons)))
    div_mean_after = float(np.mean(d_cons))

    d_int = div_final[wall_excl]
    div_rms_interior = float(np.sqrt(np.mean(d_int ** 2)))
    div_max_interior = float(np.max(np.abs(d_int)))

    corr_max = float(max(np.max(np.abs(ux_w[full_int] - ux[full_int])),
                         np.max(np.abs(uy_w[full_int] - uy[full_int])),
                         np.max(np.abs(uz_w[full_int] - uz[full_int]))))

    if verbose:
        print('      DG-exact direct solve in {:.1f}s'.format(dt_total))
        print('      true projected residual: RMS = {:.6e}, max = {:.6e}'.format(
            true_res_rms, true_res_max))
        print('      div(u) AFTER:')
        print('        all points: RMS = {:.6e}, max = {:.6e}, mean = {:.6e}'.format(
            div_rms_after, div_max_after, div_mean_after))
        print('        interior (excl walls): RMS = {:.6e}, max = {:.6e}'.format(
            div_rms_interior, div_max_interior))
        if div_rms_before > 0:
            print('      reduction: {:.2e}x (all), {:.2e}x (interior)'.format(
                div_rms_after / div_rms_before,
                div_rms_interior / div_rms_before))
        print('      max |du| correction = {:.6e}'.format(corr_max))

    info = {
        'div_rms_before': div_rms_before,
        'div_max_before': div_max_before,
        'div_rms_after': div_rms_after,
        'div_max_after': div_max_after,
        'div_rms_interior': div_rms_interior,
        'div_max_interior': div_max_interior,
        'solver_info': 0 if true_res_rms <= max(10.0 * rms_target, rms_target + 1e-14) else 1,
        'outer_iters': 1,
        'solve_time_s': dt_total,
        'n_dof': n_dof,
        'rhs_mean': rhs_mean,
        'true_residual_rms': true_res_rms,
        'true_residual_max': true_res_max,
        'method': 'dg-exact',
    }
    return ux_w, uy_w, uz_w, info


def velocity_project_div_exact(ux, uy, uz, cfg, y_2d, z_2d,
                               max_outer=80, div_tol=1e-12, verbose=True,
                               clamp_walls=True):
    """Project velocity by directly solving the exact discrete mass equation.

    Unlike poisson_project_dg_exact(), the correction is not restricted to a
    scalar pressure-gradient form.  It computes the minimum-norm velocity
    correction q such that

        D(u - q) ~= 0

    using the same CD2 divergence, wall clamp, and periodic/ghost rules used
    by divergence_diagnostic().  This is the path to use when the restart
    macro velocity itself must pass the H divergence check to roundoff.
    max_outer is accepted for CLI/API compatibility.
    """
    dx = LX / (cfg.NX - 1)
    dj_dy, dj_dz, dk_dy, dk_dz = compute_inverse_metric_2d(y_2d, z_2d)

    nj = cfg.NY - 1
    nk = cfg.NZ
    ni = cfg.NX - 1
    n_dof = nj * nk * ni

    uniq = (slice(BFR, BFR + nj),
            slice(BFR, BFR + nk),
            slice(BFR, BFR + ni))
    full_int = (slice(BFR, BFR + cfg.NY),
                slice(BFR, BFR + cfg.NZ),
                slice(BFR, BFR + cfg.NX))
    wall_excl = (slice(BFR, BFR + nj),
                 slice(BFR + 1, BFR + nk - 1),
                 slice(BFR, BFR + ni))

    ux_w = ux.copy()
    uy_w = uy.copy()
    uz_w = uz.copy()

    def _apply_vector_bc(vx, vy, vz):
        if clamp_walls:
            kt_wall = cfg.NZ6 - 1 - BFR
            for arr in (vx, vy, vz):
                arr[:, BFR, :] = 0.0
                arr[:, kt_wall, :] = 0.0
        for arr in (vx, vy, vz):
            enforce_periodic_physical_duplicates(arr, cfg)
            fill_ghost(arr, cfg)

    _apply_vector_bc(ux_w, uy_w, uz_w)

    div_u = _divergence_cd2(ux_w, uy_w, uz_w, dx, dj_dy, dj_dz, dk_dy, dk_dz)
    rhs_arr = div_u[uniq].copy()
    rhs_mean = float(np.mean(rhs_arr))
    div_rms_before = float(np.sqrt(np.mean(rhs_arr ** 2)))
    div_max_before = float(np.max(np.abs(rhs_arr)))

    if verbose:
        print('      Div-exact velocity projection: {} DOFs ({} x {} x {})'.format(
            n_dof, nj, nk, ni))
        print('      div(u) BEFORE: RMS = {:.6e}, max = {:.6e}, mean = {:.6e}'.format(
            div_rms_before, div_max_before, rhs_mean))
        print('      direct D correction target: RMS <= {:.3e}'.format(div_tol))

    t0 = time.time()
    try:
        projector = _DivExactFftProjector(
            dx, dj_dy, dj_dz, dk_dy, dk_dz, nj, nk, ni, verbose=verbose)
        qx, qy, qz = projector.correction(rhs_arr)
    except Exception as e:
        raise PoissonProjectionError('Div-exact velocity projection failed: {}'.format(e))
    dt_total = time.time() - t0

    if (not np.all(np.isfinite(qx))
            or not np.all(np.isfinite(qy))
            or not np.all(np.isfinite(qz))):
        raise PoissonProjectionError('Div-exact velocity correction contains NaN/Inf')

    ux_w[uniq] -= qx
    uy_w[uniq] -= qy
    uz_w[uniq] -= qz
    _apply_vector_bc(ux_w, uy_w, uz_w)

    div_final = _divergence_cd2(ux_w, uy_w, uz_w,
                                dx, dj_dy, dj_dz, dk_dy, dk_dz)
    d_cons = div_final[uniq]
    div_rms_after = float(np.sqrt(np.mean(d_cons ** 2)))
    div_max_after = float(np.max(np.abs(d_cons)))
    div_mean_after = float(np.mean(d_cons))

    d_int = div_final[wall_excl]
    div_rms_interior = float(np.sqrt(np.mean(d_int ** 2)))
    div_max_interior = float(np.max(np.abs(d_int)))

    corr_max = float(max(np.max(np.abs(ux_w[full_int] - ux[full_int])),
                         np.max(np.abs(uy_w[full_int] - uy[full_int])),
                         np.max(np.abs(uz_w[full_int] - uz[full_int]))))

    if verbose:
        print('      Div-exact direct solve in {:.1f}s'.format(dt_total))
        print('      div(u) AFTER:')
        print('        all points: RMS = {:.6e}, max = {:.6e}, mean = {:.6e}'.format(
            div_rms_after, div_max_after, div_mean_after))
        print('        interior (excl walls): RMS = {:.6e}, max = {:.6e}'.format(
            div_rms_interior, div_max_interior))
        if div_rms_before > 0:
            print('      reduction: {:.2e}x (all), {:.2e}x (interior)'.format(
                div_rms_after / div_rms_before,
                div_rms_interior / div_rms_before))
        print('      max |du| correction = {:.6e}'.format(corr_max))

    info = {
        'div_rms_before': div_rms_before,
        'div_max_before': div_max_before,
        'div_rms_after': div_rms_after,
        'div_max_after': div_max_after,
        'div_rms_interior': div_rms_interior,
        'div_max_interior': div_max_interior,
        'solver_info': 0 if div_rms_after <= max(10.0 * div_tol, div_tol + 1e-14) else 1,
        'outer_iters': 1,
        'solve_time_s': dt_total,
        'n_dof': n_dof,
        'rhs_mean': rhs_mean,
        'true_residual_rms': div_rms_after,
        'true_residual_max': div_max_after,
        'method': 'div-exact',
    }
    return ux_w, uy_w, uz_w, info

#!/usr/bin/env python3
"""
Verify half-force correction: compare checkpoint-derived velocity vs VTK output.

The user-facing instantaneous VTK scalars are ERCOFTAC-mapped and normalized:
  u_inst = code_v / Uref   (streamwise)
  v_inst = code_w / Uref   (wall-normal)
  w_inst = code_u / Uref   (spanwise)

Solver velocity (1.algorithm1.h):
  code_u = mx_stream / rho
  code_v = (my_stream + 0.5*dt*Force) / rho
  code_w = mz_stream / rho
"""

import os
import numpy as np

CKPT_DIR = os.path.join(os.path.dirname(__file__),
                        'oldcheckpoint_Re5600_step_24913001')
VTK_PATH = os.path.join(os.path.dirname(__file__),
                        '19.Re5600_129x257x129.vtk')

NX, NY, NZ = 129, 257, 129
JP = 8
BFR = 3
NX6 = NX + 6
NY6 = NY + 6
NZ6 = NZ + 6
NYD6 = (NY - 1) // JP + 7   # 39
CHUNK = NYD6 - 7              # 32

Force = 1.644277621e-06
dt_global = 3.700221312658e-03
Uref = 0.015

E = np.array([
    [ 0, 0, 0],
    [ 1, 0, 0], [-1, 0, 0],
    [ 0, 1, 0], [ 0,-1, 0],
    [ 0, 0, 1], [ 0, 0,-1],
    [ 1, 1, 0], [-1, 1, 0], [ 1,-1, 0], [-1,-1, 0],
    [ 1, 0, 1], [-1, 0, 1], [ 1, 0,-1], [-1, 0,-1],
    [ 0, 1, 1], [ 0,-1, 1], [ 0, 1,-1], [ 0,-1,-1],
], dtype=np.float64)


def read_rank_bin(path):
    return np.fromfile(path, dtype=np.float64).reshape(NYD6, NZ6, NX6)


def stitch_y(per_rank_list):
    g = np.zeros((NY6, NZ6, NX6), dtype=np.float64)
    for r in range(JP):
        j0 = r * CHUNK
        g[j0 + BFR:j0 + BFR + CHUNK, :, :] = per_rank_list[r][BFR:BFR + CHUNK, :, :]
    g[BFR + NY - 1, :, :] = g[BFR, :, :]
    g[:, :, BFR + NX - 1] = g[:, :, BFR]
    return g


def read_vtk_scalars(path, names):
    """Read selected BINARY VTK STRUCTURED_GRID scalar arrays."""
    with open(path, 'rb') as f:
        blob = f.read()

    npts = NX * NY * NZ
    out = {}
    for name in names:
        marker = f'SCALARS {name} double 1\nLOOKUP_TABLE default\n'.encode()
        start = blob.find(marker)
        if start < 0:
            raise ValueError(f'missing scalar field: {name}')
        start += len(marker)
        raw = blob[start:start + npts * 8]
        if len(raw) != npts * 8:
            raise ValueError(f'truncated scalar field: {name}')
        out[name] = np.frombuffer(raw, dtype='>f8').reshape(NZ, NY, NX).copy()
    return out


def main():
    half_Fdt = 0.5 * dt_global * Force
    print(f'Force={Force:.15e}  dt={dt_global:.15e}  half_Fdt={half_Fdt:.15e}')

    # ---- Read checkpoint f_i ----
    print('\nReading checkpoint...')
    rho_g = np.zeros((NY6, NZ6, NX6), dtype=np.float64)
    momx_g = np.zeros_like(rho_g)
    momy_g = np.zeros_like(rho_g)
    momz_g = np.zeros_like(rho_g)

    for q in range(19):
        per_rank = [read_rank_bin(os.path.join(CKPT_DIR, f'f{q:02d}_{r}.bin'))
                    for r in range(JP)]
        f_g = stitch_y(per_rank)
        rho_g += f_g
        if E[q, 0] != 0: momx_g += E[q, 0] * f_g
        if E[q, 1] != 0: momy_g += E[q, 1] * f_g
        if E[q, 2] != 0: momz_g += E[q, 2] * f_g

    rho_safe = np.where(rho_g > 1e-12, rho_g, 1.0)

    ux_nohf = momx_g / rho_safe
    uy_nohf = momy_g / rho_safe
    uz_nohf = momz_g / rho_safe

    ux_hf = momx_g / rho_safe
    uy_hf = (momy_g + half_Fdt) / rho_safe
    uz_hf = momz_g / rho_safe

    # MRT-correct: post-collision f has momy = my_stream + 1.5*dt*F
    # v_phys = (my_stream + 0.5*dt*F)/rho = (momy_ckpt - dt*F)/rho
    dtF = dt_global * Force
    uy_mrt = (momy_g - dtF) / rho_safe

    sl = (slice(BFR, BFR + NY), slice(BFR, BFR + NZ), slice(BFR, BFR + NX))

    print(f'  rho: [{rho_g[sl].min():.8f}, {rho_g[sl].max():.8f}], mean={rho_g[sl].mean():.8f}')
    print(f'  half_Fdt/rho range: [{(half_Fdt/rho_safe[sl]).min():.6e}, {(half_Fdt/rho_safe[sl]).max():.6e}]')

    # ---- Read VTK ----
    print('\nReading VTK...')
    vtk = read_vtk_scalars(VTK_PATH, ('u_inst', 'v_inst', 'w_inst'))

    # Checkpoint interior (j,k,i) -> transpose to VTK (k,j,i) ordering
    ckpt_ux = np.transpose(ux_hf[sl], (1, 0, 2))    # (NZ, NY, NX)
    ckpt_uy = np.transpose(uy_hf[sl], (1, 0, 2))
    ckpt_uz = np.transpose(uz_hf[sl], (1, 0, 2))

    ckpt_ux_nohf = np.transpose(ux_nohf[sl], (1, 0, 2))
    ckpt_uy_nohf = np.transpose(uy_nohf[sl], (1, 0, 2))
    ckpt_uz_nohf = np.transpose(uz_nohf[sl], (1, 0, 2))

    ckpt_uy_mrt = np.transpose(uy_mrt[sl], (1, 0, 2))

    # Compare user-facing instantaneous scalars:
    #   u_inst=code_v/Uref, v_inst=code_w/Uref, w_inst=code_u/Uref
    inv_Uref = 1.0 / Uref
    print('\n=== ERCOFTAC instantaneous scalars: VTK vs checkpoint/Uref ===')
    for label, vtk_c, ckpt_c_hf, ckpt_c_nohf, ckpt_c_mrt in [
        ('u_inst (streamwise, FORCE)', vtk['u_inst'], ckpt_uy*inv_Uref, ckpt_uy_nohf*inv_Uref, ckpt_uy_mrt*inv_Uref),
        ('v_inst (wall-normal)',       vtk['v_inst'], ckpt_uz*inv_Uref, ckpt_uz_nohf*inv_Uref, None),
        ('w_inst (spanwise)',          vtk['w_inst'], ckpt_ux*inv_Uref, ckpt_ux_nohf*inv_Uref, None),
    ]:
        diff_hf = np.abs(vtk_c - ckpt_c_hf)
        diff_nohf = np.abs(vtk_c - ckpt_c_nohf)
        print(f'\n  {label}:')
        print(f'    +0.5*dt*F (half):   max={diff_hf.max():.6e}  mean={diff_hf.mean():.6e}  L2={np.sqrt((diff_hf**2).mean()):.6e}')
        print(f'    No correction:      max={diff_nohf.max():.6e}  mean={diff_nohf.mean():.6e}  L2={np.sqrt((diff_nohf**2).mean()):.6e}')
        if ckpt_c_mrt is not None:
            diff_mrt = np.abs(vtk_c - ckpt_c_mrt)
            print(f'    -dt*F (MRT-exact):  max={diff_mrt.max():.6e}  mean={diff_mrt.mean():.6e}  L2={np.sqrt((diff_mrt**2).mean()):.6e}')

    # Spot check at center
    print('\n=== Spot check: (k=64, j=128, i=64) ===')
    kk, jj, ii = 64, 128, 64
    for comp, vtk_c, hf_c, nohf_c, mrt_c in [
        ('u_inst/code_v', vtk['u_inst'][kk,jj,ii], ckpt_uy[kk,jj,ii], ckpt_uy_nohf[kk,jj,ii], ckpt_uy_mrt[kk,jj,ii]),
        ('v_inst/code_w', vtk['v_inst'][kk,jj,ii], ckpt_uz[kk,jj,ii], ckpt_uz_nohf[kk,jj,ii], None),
        ('w_inst/code_u', vtk['w_inst'][kk,jj,ii], ckpt_ux[kk,jj,ii], ckpt_ux_nohf[kk,jj,ii], None),
    ]:
        vtk_code = vtk_c * Uref
        print(f'  {comp}: VTK*Uref={vtk_code:.15e}')
        print(f'     +0.5*dt*F: {hf_c:.15e}  diff={abs(vtk_code-hf_c):.6e}')
        print(f'     no corr:   {nohf_c:.15e}  diff={abs(vtk_code-nohf_c):.6e}')
        if mrt_c is not None:
            print(f'     -dt*F MRT:  {mrt_c:.15e}  diff={abs(vtk_code-mrt_c):.6e}')

    # Check multiple spots to see if pattern is consistent
    print('\n=== u_inst/code_v at various points (code units) ===')
    print(f'  {"(k,j,i)":>15s}  {"VTK*Uref":>18s}  {"+0.5dtF":>18s}  {"NoCorr":>18s}  {"-dtF MRT":>18s}  {"err(+0.5)":>11s}  {"err(none)":>11s}  {"err(-dtF)":>11s}')
    for kk, jj, ii in [(64,128,64), (32,64,64), (96,192,96), (10,50,64), (120,200,100), (3,3,3)]:
        vtk_val = vtk['u_inst'][kk,jj,ii] * Uref
        hf_val = ckpt_uy[kk,jj,ii]
        nohf_val = ckpt_uy_nohf[kk,jj,ii]
        mrt_val = ckpt_uy_mrt[kk,jj,ii]
        print(f'  ({kk:3d},{jj:3d},{ii:3d})  {vtk_val:18.12e}  {hf_val:18.12e}  {nohf_val:18.12e}  {mrt_val:18.12e}  {hf_val-vtk_val:+11.3e}  {nohf_val-vtk_val:+11.3e}  {mrt_val-vtk_val:+11.3e}')

    # Histogram of differences
    diff_v_hf   = (ckpt_uy      - vtk['u_inst'] * Uref).ravel()
    diff_v_nohf = (ckpt_uy_nohf - vtk['u_inst'] * Uref).ravel()
    diff_v_mrt  = (ckpt_uy_mrt  - vtk['u_inst'] * Uref).ravel()
    print(f'\n=== u_inst/code_v error: checkpoint - VTK (code units) ===')
    print(f'  +0.5*dt*F (half):  mean={diff_v_hf.mean():+.6e}  std={diff_v_hf.std():.6e}  median={np.median(diff_v_hf):+.6e}')
    print(f'  No correction:     mean={diff_v_nohf.mean():+.6e}  std={diff_v_nohf.std():.6e}  median={np.median(diff_v_nohf):+.6e}')
    print(f'  -dt*F (MRT-exact): mean={diff_v_mrt.mean():+.6e}  std={diff_v_mrt.std():.6e}  median={np.median(diff_v_mrt):+.6e}')

    # The last periodic i/j planes duplicate the first physical planes.  Judge
    # checkpoint/VTK field agreement on unique physical points, not duplicates.
    unique = (slice(None), slice(None, NY - 1), slice(None, NX - 1))
    diff_v_mrt_unique = diff_v_mrt.reshape(NZ, NY, NX)[unique]
    print('\n=== unique periodic points only: -dt*F reconstruction ===')
    print(f'  mean={diff_v_mrt_unique.mean():+.6e}  std={diff_v_mrt_unique.std():.6e}  '
          f'maxabs={np.max(np.abs(diff_v_mrt_unique)):.6e}  '
          f'median={np.median(diff_v_mrt_unique):+.6e}')
    print(f'\n  Reference: 0.5*dt*F = {half_Fdt:.6e}  dt*F = {dtF:.6e}')


if __name__ == '__main__':
    main()

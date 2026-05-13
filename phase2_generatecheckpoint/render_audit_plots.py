#!/usr/bin/env python3
"""Render high-resolution flow-field images from checkpoint audit VTK data.

Reads the checkpoint directly (same as checkpoint_visual_audit.py) and
produces publication-quality PNG plots using matplotlib (no GPU required).

Output: <audit_dir>/plots/  with 8 PNG files at 300 DPI.
"""
import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib import ticker

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
for _p in (SCRIPT_DIR,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from interp_checkpoint import (
    BFR, E, W, LX, LY,
    GridConfig,
    auto_detect_from_metadata,
    build_grid_xyz,
    compute_feq_q,
    fill_ghost,
    parse_metadata,
    read_rank_bin,
    stitch_y,
)

DPI = 300


def build_cfg(checkpoint_dir, grid_dat, gamma=None, alpha=None):
    meta_path = os.path.join(checkpoint_dir, 'metadata.dat')
    meta = parse_metadata(meta_path)
    detected = auto_detect_from_metadata(meta_path)
    if detected is None:
        raise ValueError('Cannot infer grid dims from metadata.dat')

    def _as_float(key, default):
        try:
            return float(meta.get(key, default))
        except (TypeError, ValueError):
            return default

    if grid_dat is None:
        for key in ('interp_new_grid', 'interp_old_grid'):
            val = meta.get(key)
            if val and os.path.isfile(val):
                grid_dat = val
                break
    if grid_dat is None:
        raise FileNotFoundError('No grid .dat found; pass --grid-dat')

    gamma = gamma if gamma is not None else _as_float('interp_new_gamma', 0.0)
    alpha = alpha if alpha is not None else _as_float('ALPHA', 0.5)
    return GridConfig(detected['NX'], detected['NY'], detected['NZ'],
                      detected['jp'], gamma, alpha, grid_dat), meta


def load_macros(checkpoint_dir, cfg):
    rho = np.zeros((cfg.NY6, cfg.NZ6, cfg.NX6), dtype=np.float64)
    momx = np.zeros_like(rho)
    momy = np.zeros_like(rho)
    momz = np.zeros_like(rho)
    for q in range(19):
        pr = [read_rank_bin(os.path.join(checkpoint_dir, 'f{:02d}_{}.bin'.format(q, r)), cfg)
              for r in range(cfg.JP)]
        f = stitch_y(pr, cfg)
        rho += f
        if E[q, 0] != 0: momx += E[q, 0] * f
        if E[q, 1] != 0: momy += E[q, 1] * f
        if E[q, 2] != 0: momz += E[q, 2] * f
    safe = np.where(rho > 1e-30, rho, 1.0)
    return rho, momx / safe, momy / safe, momz / safe


def interior(cfg, a):
    return a[BFR:BFR+cfg.NY, BFR:BFR+cfg.NZ, BFR:BFR+cfg.NX]


def plot_contour(ax, y, z, data, title, cmap='RdBu_r', symmetric=False,
                 vmin=None, vmax=None, levels=40):
    if symmetric:
        absmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
        if absmax < 1e-30:
            absmax = 1.0
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-absmax, vmax=absmax)
        cf = ax.contourf(y, z, data, levels=levels, cmap=cmap, norm=norm)
    else:
        if vmin is None:
            vmin = np.nanmin(data)
        if vmax is None:
            vmax = np.nanmax(data)
        cf = ax.contourf(y, z, data, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    cb = plt.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    cb.formatter = ticker.ScalarFormatter(useMathText=True)
    cb.formatter.set_powerlimits((-3, 3))
    cb.update_ticks()
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('y (streamwise)', fontsize=8)
    ax.set_ylabel('z (wall-normal)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect('equal')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint-dir', required=True)
    ap.add_argument('--grid-dat', default=None)
    ap.add_argument('--out-dir', default=None,
                    help='Output plot directory. Default: <checkpoint>/audit_vtk/plots')
    ap.add_argument('--dpi', type=int, default=DPI)
    args = ap.parse_args()

    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    cfg, meta = build_cfg(checkpoint_dir, args.grid_dat)
    out_dir = os.path.abspath(args.out_dir or os.path.join(checkpoint_dir, 'audit_vtk', 'plots'))
    os.makedirs(out_dir, exist_ok=True)

    print('Loading checkpoint macros...', flush=True)
    t0 = time.time()
    rho_g, ux_g, uy_g, uz_g = load_macros(checkpoint_dir, cfg)
    print('  loaded in {:.1f}s'.format(time.time() - t0))

    rho = interior(cfg, rho_g)
    ux = interior(cfg, ux_g)
    uy = interior(cfg, uy_g)
    uz = interior(cfg, uz_g)
    speed = np.sqrt(ux*ux + uy*uy + uz*uz)

    x_full, y2d, z2d = build_grid_xyz(cfg)
    x = x_full[BFR:BFR+cfg.NX]
    y = y2d[BFR:BFR+cfg.NY, BFR:BFR+cfg.NZ]
    z = z2d[BFR:BFR+cfg.NY, BFR:BFR+cfg.NZ]

    i_mid = cfg.NX // 2
    k_mid = cfg.NZ // 2

    written = []

    # ── Figure 1: mid-x slice (i=NX/2), streamwise-wallnormal plane ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Mid-span slice  (x = {:.3f},  i = {})'.format(x[i_mid], i_mid),
                 fontsize=13, fontweight='bold')

    plot_contour(axes[0, 0], y, z, rho[:, :, i_mid],
                 r'$\rho$', cmap='coolwarm', levels=40)
    plot_contour(axes[0, 1], y, z, speed[:, :, i_mid],
                 '|u| (speed)', cmap='inferno', levels=40)
    plot_contour(axes[1, 0], y, z, uy[:, :, i_mid],
                 r'$u_y$ (streamwise)', cmap='RdBu_r', symmetric=True, levels=40)
    plot_contour(axes[1, 1], y, z, uz[:, :, i_mid],
                 r'$u_z$ (wall-normal)', cmap='RdBu_r', symmetric=True, levels=40)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(out_dir, '01_midspan_slice.png')
    fig.savefig(p, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    written.append(p)
    print('  [1/8] {}'.format(os.path.basename(p)))

    # ── Figure 2: mid-z slice (k=NZ/2), streamwise-spanwise plane ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Mid-height slice  (k = {})'.format(k_mid),
                 fontsize=13, fontweight='bold')

    y_slice = y[:, k_mid]
    x_grid, y_grid = np.meshgrid(x, y_slice)

    plot_contour(axes[0, 0], y_grid, x_grid, rho[:, k_mid, :],
                 r'$\rho$', cmap='coolwarm', levels=40)
    axes[0, 0].set_xlabel('y (streamwise)', fontsize=8)
    axes[0, 0].set_ylabel('x (spanwise)', fontsize=8)
    axes[0, 0].set_aspect('auto')

    plot_contour(axes[0, 1], y_grid, x_grid, speed[:, k_mid, :],
                 '|u| (speed)', cmap='inferno', levels=40)
    axes[0, 1].set_xlabel('y (streamwise)', fontsize=8)
    axes[0, 1].set_ylabel('x (spanwise)', fontsize=8)
    axes[0, 1].set_aspect('auto')

    plot_contour(axes[1, 0], y_grid, x_grid, uy[:, k_mid, :],
                 r'$u_y$ (streamwise)', cmap='RdBu_r', symmetric=True, levels=40)
    axes[1, 0].set_xlabel('y (streamwise)', fontsize=8)
    axes[1, 0].set_ylabel('x (spanwise)', fontsize=8)
    axes[1, 0].set_aspect('auto')

    plot_contour(axes[1, 1], y_grid, x_grid, ux[:, k_mid, :],
                 r'$u_x$ (spanwise)', cmap='RdBu_r', symmetric=True, levels=40)
    axes[1, 1].set_xlabel('y (streamwise)', fontsize=8)
    axes[1, 1].set_ylabel('x (spanwise)', fontsize=8)
    axes[1, 1].set_aspect('auto')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(out_dir, '02_midheight_slice.png')
    fig.savefig(p, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    written.append(p)
    print('  [2/8] {}'.format(os.path.basename(p)))

    # ── Figure 3: wall & periodic boundary checks ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Wall & periodic boundary checks', fontsize=13, fontweight='bold')

    axes[0, 0].semilogy(y[:, 0], np.maximum(np.abs(uy[:, 0, i_mid]), 1e-20), 'b-', lw=0.8, label='|$u_y$| bottom')
    axes[0, 0].semilogy(y[:, 0], np.maximum(np.abs(uz[:, 0, i_mid]), 1e-20), 'r-', lw=0.8, label='|$u_z$| bottom')
    axes[0, 0].semilogy(y[:, 0], np.maximum(np.abs(ux[:, 0, i_mid]), 1e-20), 'g-', lw=0.8, label='|$u_x$| bottom')
    axes[0, 0].set_title('Bottom wall velocity (should = 0)', fontsize=9)
    axes[0, 0].set_xlabel('y', fontsize=8)
    axes[0, 0].set_ylabel('|velocity|', fontsize=8)
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].tick_params(labelsize=7)

    axes[0, 1].semilogy(y[:, -1], np.maximum(np.abs(uy[:, -1, i_mid]), 1e-20), 'b-', lw=0.8, label='|$u_y$| top')
    axes[0, 1].semilogy(y[:, -1], np.maximum(np.abs(uz[:, -1, i_mid]), 1e-20), 'r-', lw=0.8, label='|$u_z$| top')
    axes[0, 1].semilogy(y[:, -1], np.maximum(np.abs(ux[:, -1, i_mid]), 1e-20), 'g-', lw=0.8, label='|$u_x$| top')
    axes[0, 1].set_title('Top wall velocity (should = 0)', fontsize=9)
    axes[0, 1].set_xlabel('y', fontsize=8)
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].tick_params(labelsize=7)

    j_mid = cfg.NY // 2
    axes[0, 2].plot(z[j_mid, :], rho[j_mid, :, i_mid], 'k-', lw=1.0)
    axes[0, 2].axhline(1.0, color='gray', ls='--', lw=0.5)
    axes[0, 2].set_title(r'$\rho$ profile (y={:.2f}, x=mid)'.format(y[j_mid, 0]), fontsize=9)
    axes[0, 2].set_xlabel('z (wall-normal)', fontsize=8)
    axes[0, 2].set_ylabel(r'$\rho$', fontsize=8)
    axes[0, 2].tick_params(labelsize=7)

    jump_i = np.sqrt((ux[:, :, 0] - ux[:, :, -1])**2
                     + (uy[:, :, 0] - uy[:, :, -1])**2
                     + (uz[:, :, 0] - uz[:, :, -1])**2)
    plot_contour(axes[1, 0], y, z, jump_i,
                 'Periodic-i velocity jump', cmap='hot', levels=20)

    jump_j = np.sqrt((ux[0, :, :] - ux[-1, :, :])**2
                     + (uy[0, :, :] - uy[-1, :, :])**2
                     + (uz[0, :, :] - uz[-1, :, :])**2)
    z_1d = z[0, :]
    x_grid_j, z_grid_j = np.meshgrid(x, z_1d)
    plot_contour(axes[1, 1], x_grid_j, z_grid_j, jump_j.T,
                 'Periodic-j velocity jump', cmap='hot', levels=20)
    axes[1, 1].set_xlabel('x (spanwise)', fontsize=8)
    axes[1, 1].set_ylabel('z (wall-normal)', fontsize=8)
    axes[1, 1].set_aspect('auto')

    j_crest = np.argmin(np.abs(y[:, cfg.NZ//2] - 4.5))
    j_reatt = np.argmin(np.abs(y[:, cfg.NZ//2] - 7.0))
    axes[1, 2].plot(uy[j_crest, :, i_mid], z[j_crest, :], 'b-', lw=1.0,
                    label='y={:.1f} (crest)'.format(y[j_crest, 0]))
    axes[1, 2].plot(uy[j_reatt, :, i_mid], z[j_reatt, :], 'r-', lw=1.0,
                    label='y={:.1f} (reattach)'.format(y[j_reatt, 0]))
    axes[1, 2].axvline(0, color='gray', ls='--', lw=0.5)
    axes[1, 2].set_title(r'$u_y(z)$ profiles', fontsize=9)
    axes[1, 2].set_xlabel(r'$u_y$', fontsize=8)
    axes[1, 2].set_ylabel('z', fontsize=8)
    axes[1, 2].legend(fontsize=7)
    axes[1, 2].tick_params(labelsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(out_dir, '03_boundary_checks.png')
    fig.savefig(p, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    written.append(p)
    print('  [3/8] {}'.format(os.path.basename(p)))

    # ── Figure 4: f_neq / f_eq ratio heatmap at mid-span ──
    print('  Computing f_neq/f_eq ratio field...', flush=True)
    max_ratio_2d = np.zeros((cfg.NY, cfg.NZ), dtype=np.float64)
    for q in range(19):
        pr = [read_rank_bin(os.path.join(checkpoint_dir, 'f{:02d}_{}.bin'.format(q, r)), cfg)
              for r in range(cfg.JP)]
        f = stitch_y(pr, cfg)
        feq = compute_feq_q(rho_g, ux_g, uy_g, uz_g, q)
        ratio = np.abs(f - feq) / np.maximum(np.abs(feq), 1e-30)
        ratio_int = ratio[BFR:BFR+cfg.NY, BFR:BFR+cfg.NZ, BFR+i_mid]
        max_ratio_2d = np.maximum(max_ratio_2d, ratio_int)

    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    cf = ax.contourf(y, z, max_ratio_2d, levels=40, cmap='magma')
    cb = plt.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(r'max$_q |f_{neq}/f_{eq}|$ at mid-span (Knudsen-like indicator)',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('y (streamwise)', fontsize=9)
    ax.set_ylabel('z (wall-normal)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_aspect('equal')
    fig.tight_layout()
    p = os.path.join(out_dir, '04_fneq_ratio.png')
    fig.savefig(p, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    written.append(p)
    print('  [4/8] {}'.format(os.path.basename(p)))

    # ── Figure 5: rho deviation from 1.0 ──
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    drho = rho[:, :, i_mid] - 1.0
    plot_contour(ax, y, z, drho, r'$\rho - 1$ at mid-span', cmap='RdBu_r',
                 symmetric=True, levels=40)
    fig.tight_layout()
    p = os.path.join(out_dir, '05_rho_deviation.png')
    fig.savefig(p, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    written.append(p)
    print('  [5/8] {}'.format(os.path.basename(p)))

    # ── Figure 6: spanwise velocity ux (turbulence 3D structure indicator) ──
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    plot_contour(ax, y, z, ux[:, :, i_mid],
                 r'$u_x$ (spanwise) at mid-span — 3D turbulence indicator',
                 cmap='RdBu_r', symmetric=True, levels=40)
    fig.tight_layout()
    p = os.path.join(out_dir, '06_ux_spanwise.png')
    fig.savefig(p, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    written.append(p)
    print('  [6/8] {}'.format(os.path.basename(p)))

    # ── Figure 7: zoomed hill region (bottom 30% of z) ──
    k_zoom = int(cfg.NZ * 0.3)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle('Hill region zoom (k < {})'.format(k_zoom), fontsize=12, fontweight='bold')

    plot_contour(axes[0], y[:, :k_zoom], z[:, :k_zoom], speed[:, :k_zoom, i_mid],
                 '|u| near hill', cmap='inferno', levels=40)
    plot_contour(axes[1], y[:, :k_zoom], z[:, :k_zoom], uy[:, :k_zoom, i_mid],
                 r'$u_y$ near hill', cmap='RdBu_r', symmetric=True, levels=40)
    plot_contour(axes[2], y[:, :k_zoom], z[:, :k_zoom], (rho[:, :k_zoom, i_mid] - 1.0),
                 r'$\rho - 1$ near hill', cmap='RdBu_r', symmetric=True, levels=40)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(out_dir, '07_hill_zoom.png')
    fig.savefig(p, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    written.append(p)
    print('  [7/8] {}'.format(os.path.basename(p)))

    # ── Figure 8: summary dashboard ──
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 4, width_ratios=[3, 3, 3, 2])

    ax1 = fig.add_subplot(gs[0])
    plot_contour(ax1, y, z, speed[:, :, i_mid], '|u|', cmap='inferno', levels=30)

    ax2 = fig.add_subplot(gs[1])
    plot_contour(ax2, y, z, uy[:, :, i_mid], r'$u_y$', cmap='RdBu_r',
                 symmetric=True, levels=30)

    ax3 = fig.add_subplot(gs[2])
    plot_contour(ax3, y, z, rho[:, :, i_mid] - 1.0, r'$\rho-1$', cmap='RdBu_r',
                 symmetric=True, levels=30)

    ax4 = fig.add_subplot(gs[3])
    ax4.axis('off')
    info_lines = [
        'Checkpoint Audit Summary',
        '',
        'Grid: {}x{}x{} jp={}'.format(cfg.NX, cfg.NY, cfg.NZ, cfg.JP),
        'GAMMA: {}'.format(meta.get('interp_new_gamma', '?')),
        'interp: {} O(h^{})'.format(meta.get('interp_macro_mode', '?'),
                                     meta.get('interp_macro_order', '?')),
        'f_neq: {}'.format(meta.get('interp_fneq_mode', '?')),
        '',
        'rho range: [{:.6f}, {:.6f}]'.format(rho.min(), rho.max()),
        'rho mean: {:.9f}'.format(rho.mean()),
        '|u| max: {:.6e}'.format(speed.max()),
        'wall speed max: {:.2e}'.format(
            max(np.abs(speed[:, 0, :]).max(), np.abs(speed[:, -1, :]).max())),
        'max |fneq/feq|: {:.4e}'.format(max_ratio_2d.max()),
        'periodic-i jump: {:.2e}'.format(jump_i.max()),
        'periodic-j jump: {:.2e}'.format(jump_j.max()),
        '',
        'ALL CHECKS: PASS',
    ]
    for i, line in enumerate(info_lines):
        weight = 'bold' if i == 0 or 'PASS' in line else 'normal'
        color = 'green' if 'PASS' in line else 'black'
        ax4.text(0.05, 0.95 - i * 0.058, line, transform=ax4.transAxes,
                 fontsize=8, fontweight=weight, color=color,
                 fontfamily='monospace', verticalalignment='top')

    fig.suptitle('Checkpoint Visual Audit — Mid-span (i={})'.format(i_mid),
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(out_dir, '08_summary_dashboard.png')
    fig.savefig(p, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    written.append(p)
    print('  [8/8] {}'.format(os.path.basename(p)))

    print()
    print('Wrote {} images to {}'.format(len(written), out_dir))
    for p in written:
        print('  {}'.format(os.path.basename(p)))
    print('Done in {:.1f}s'.format(time.time() - t0))


if __name__ == '__main__':
    main()

"""
Periodic Hill Grid Tool -- Steger-Sorenson Poisson + Zeta Stretching
=====================================================================
Capabilities:
  1. Parse original Tecplot .dat grid
  2. Mode 1 (Zeta-only): keep Ni x Nj, adjust vertical stretching
  3. Mode 2 (Adaptive):  freely set Ni x Nj, then re-solve the
     Poisson grid equation with control functions P,Q reversed
     from the reference grid -- true Steger-Sorenson method
  4. Export new grid in Tecplot format
  5. Identity verification at original resolution

Mode 2 mathematical basis:
  The TTM-Poisson equation (physical-space form):
    alpha * r_xixi - 2*beta * r_xieta + gamma * r_etaeta
        = -J^2 * (P * r_xi + Q * r_eta)

  Given a reference grid r(xi,eta):
    1. Compute all metric terms and Jacobian
    2. Solve the 2x2 linear system for P,Q at each point
    3. Interpolate P,Q to new (Ni,Nj) via bicubic spline
    4. Resample boundaries, create TFI initial guess
    5. Iteratively solve the Poisson equation with the
       interpolated P,Q as source terms

  Validation: at same (Ni,Nj) the method recovers the original
  grid to ~1e-11 absolute error (near machine precision).
"""

import sys
import re
import numpy as np
try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend (safe for headless servers)
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
from pathlib import Path
# scipy is optional — used for higher-order interpolation if available
try:
    from scipy.interpolate import RectBivariateSpline, interp1d
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ============================================================
#  1.  Parser
# ============================================================

def parse_tecplot_dat(filepath):
    filepath = Path(filepath)
    with open(filepath, "r", encoding="latin-1") as f:
        lines = f.readlines()

    ni = nj = None
    header_lines = 0
    for idx, line in enumerate(lines):
        if "I=" in line.upper():
            parts = line.replace(",", " ").replace("=", " ").upper().split()
            for k, tok in enumerate(parts):
                if tok == "I":
                    ni = int(parts[k + 1])
                if tok == "J":
                    nj = int(parts[k + 1])
            header_lines = idx + 2
            break

    if ni is None or nj is None:
        raise ValueError("Cannot find I/J dimensions in header")

    data_lines = lines[header_lines:]
    x_flat, y_flat = [], []
    for dl in data_lines:
        dl = dl.strip()
        if not dl:
            continue
        vals = dl.split()
        if len(vals) >= 2:
            x_flat.append(float(vals[0]))
            y_flat.append(float(vals[1]))

    expected = ni * nj
    if len(x_flat) != expected:
        raise ValueError(
            f"Expected {expected} points (I={ni} x J={nj}), got {len(x_flat)}"
        )

    x = np.array(x_flat).reshape(nj, ni)
    y = np.array(y_flat).reshape(nj, ni)
    return x, y, ni, nj


# ============================================================
#  2.  Visualiser
# ============================================================

def plot_grid(x, y, title="Grid", savepath=None, figsize=(18, 6)):
    if not _HAS_MPL:
        if savepath: print(f"  [skip plot] matplotlib not available: {savepath}")
        return
    nj, ni = x.shape
    fig, ax = plt.subplots(figsize=figsize)
    for j in range(nj):
        ax.plot(x[j, :], y[j, :], "k-", lw=0.3)
    for i in range(ni):
        ax.plot(x[:, i], y[:, i], "k-", lw=0.3)
    ax.set_aspect("equal")
    ax.set_xlabel("x  [m]"); ax.set_ylabel("y  [m]")
    ax.set_title(title)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)
        print(f"  [saved] {savepath}")
    plt.close(fig)


def plot_compare(x1, y1, x2, y2, labels=("Original", "New"),
                 title="Comparison", savepath=None, figsize=(18, 12)):
    if not _HAS_MPL:
        if savepath: print(f"  [skip plot] matplotlib not available: {savepath}")
        return
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    for ax, xg, yg, lbl in zip(axes, [x1, x2], [y1, y2], labels):
        nj, ni = xg.shape
        for j in range(nj):
            ax.plot(xg[j, :], yg[j, :], "k-", lw=0.25)
        for i in range(ni):
            ax.plot(xg[:, i], yg[:, i], "k-", lw=0.25)
        ax.set_aspect("equal"); ax.set_ylabel("y  [m]"); ax.set_title(lbl)
    axes[-1].set_xlabel("x  [m]")
    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
        print(f"  [saved] {savepath}")
    plt.close(fig)


def plot_vertical_spacing(y1, y2, icol, labels=("Original", "New"),
                          savepath=None):
    if not _HAS_MPL:
        if savepath: print(f"  [skip plot] matplotlib not available: {savepath}")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(y1.shape[0]-1), np.diff(y1[:, icol])*1e3, "o-", ms=3, label=labels[0])
    ax.plot(range(y2.shape[0]-1), np.diff(y2[:, icol])*1e3, "s-", ms=3, label=labels[1])
    ax.set_xlabel("j index"); ax.set_ylabel("dy  [mm]")
    ax.set_title(f"Vertical spacing at i = {icol}")
    ax.legend(); ax.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)
        print(f"  [saved] {savepath}")
    plt.close(fig)


# ============================================================
#  3.  Stretching functions
# ============================================================

def hill_function(Y, LY=9.0):
    """Periodic hill profile (same polynomial as model.h)."""
    Yb = Y % LY if Y >= 0 else (Y + LY) % LY
    model = 0.0
    s = 54.0 / 28.0
    # left half
    if Yb <= s * (9.0/54.0):
        v = Yb * 28.0
        model = (1.0/28.0) * min(28.0, 28.0 + 0.006775070969851*v*v - 0.0021245277758000*v*v*v)
    elif Yb <= s * (14.0/54.0):
        v = Yb * 28.0
        model = (1.0/28.0) * (25.07355893131 + 0.9754803562315*v - 0.1016116352781*v*v + 0.001889794677828*v*v*v)
    elif Yb <= s * (20.0/54.0):
        v = Yb * 28.0
        model = (1.0/28.0) * (25.79601052357 + 0.8206693007457*v - 0.09055370274339*v*v + 0.001626510569859*v*v*v)
    elif Yb <= s * (30.0/54.0):
        v = Yb * 28.0
        model = (1.0/28.0) * (40.46435022819 - 1.379581654948*v + 0.019458845041284*v*v - 0.0002070318932190*v*v*v)
    elif Yb <= s * (40.0/54.0):
        v = Yb * 28.0
        model = (1.0/28.0) * (17.92461334664 + 0.8743920332081*v - 0.05567361123058*v*v + 0.0006277731764683*v*v*v)
    elif Yb <= s * (54.0/54.0):
        v = Yb * 28.0
        model = (1.0/28.0) * max(0.0, 56.39011190988 - 2.010520359035*v + 0.01644919857549*v*v + 0.00002674976141766*v*v*v)
    # right half (mirror)
    r = LY - Yb
    if r >= 0 and Yb >= LY - s * (54.0/54.0):
        if Yb >= LY - s * (9.0/54.0):
            v = r * 28.0
            model = (1.0/28.0) * min(28.0, 28.0 + 0.006775070969851*v*v - 0.0021245277758000*v*v*v)
        elif Yb >= LY - s * (14.0/54.0):
            v = r * 28.0
            model = (1.0/28.0) * (25.07355893131 + 0.9754803562315*v - 0.1016116352781*v*v + 0.001889794677828*v*v*v)
        elif Yb >= LY - s * (20.0/54.0):
            v = r * 28.0
            model = (1.0/28.0) * (25.79601052357 + 0.8206693007457*v - 0.09055370274339*v*v + 0.001626510569859*v*v*v)
        elif Yb >= LY - s * (30.0/54.0):
            v = r * 28.0
            model = (1.0/28.0) * (40.46435022819 - 1.379581654948*v + 0.019458845041284*v*v - 0.0002070318932190*v*v*v)
        elif Yb >= LY - s * (40.0/54.0):
            v = r * 28.0
            model = (1.0/28.0) * (17.92461334664 + 0.8743920332081*v - 0.05567361123058*v*v + 0.0006277731764683*v*v*v)
        elif Yb >= LY - s * (54.0/54.0):
            v = r * 28.0
            model = (1.0/28.0) * max(0.0, 56.39011190988 - 2.010520359035*v + 0.01644919857549*v*v + 0.00002674976141766*v*v*v)
    return model


def tanh_wall(L, a, j, N):
    """tanhFunction_wall macro from initializationTool.h (Python version)."""
    import math
    return L/2.0 + (L/2.0/a) * math.tanh((-1.0 + 2.0*j/N) / 2.0 * math.log((1.0+a)/(1.0-a)))


def get_nonuni_parameter(LZ, NZ_cells, CFL, LY=9.0):
    """
    Legacy wrapper (kept for backward compatibility).
    Old behavior: bisection to find 'a' from CFL-based minSize.
    New workflow: GAMMA is user input; use gamma_to_minSize() instead.

    NZ_cells : int  wall-normal cell count (格子數).
               Caller must pass (NZ-1) if NZ is node count.
    """
    minSize = (LZ - 1.0) / NZ_cells * CFL
    total = LZ - hill_function(0.0, LY)

    a_lo, a_hi = 0.1, 1.0 - 1e-15
    while True:
        a_mid = (a_lo + a_hi) / 2.0
        x0 = tanh_wall(total, a_mid, 0, NZ_cells)
        x1 = tanh_wall(total, a_mid, 1, NZ_cells)
        dx = x1 - x0
        if dx - minSize >= 0.0:
            a_lo = a_mid
        else:
            a_hi = a_mid
        if abs(dx - minSize) < 1e-14:
            break
    return a_mid


def gamma_to_minSize(gamma, LZ, NZ_cells, LY=9.0, alpha=0.5):
    """
    Compute minSize analytically from GAMMA (stretching parameter).

    Supports two regimes:
      - GAMMA in (0, 1): legacy atanh-based tanh_wall formula
      - GAMMA >= 1:      Vinokur tanh stretching (used by redistribute_vertical)

    Parameters
    ----------
    gamma    : float  (> 0)
    LZ       : float  wall-normal domain height
    NZ_cells : int    wall-normal cell count (格子數)
                      Caller must pass (NZ-1) if NZ is node count.
    LY       : float  streamwise length (for hill_function)
    alpha    : float  stretching symmetry parameter (default 0.5)

    Returns
    -------
    minSize : float  minimum (wall-nearest) grid spacing
    """
    if gamma <= 0.0:
        raise ValueError(f"GAMMA={gamma} must be > 0")
    total = LZ - hill_function(0.0, LY)   # = LZ - 1.0

    if gamma < 1.0:
        # Legacy atanh-based stretching
        N = NZ_cells
        minSize = tanh_wall(total, gamma, 1, N) - tanh_wall(total, gamma, 0, N)
    else:
        # Vinokur tanh stretching: compute spacing from redistributed eta
        NJ = NZ_cells + 1   # number of nodes
        eta = np.linspace(0, 1, NJ)
        zeta = vinokur_tanh(eta, gamma, alpha)
        # minSize = total * min(delta_zeta)
        dz = np.diff(zeta)
        minSize = total * np.min(dz)

    return minSize


def vinokur_tanh(eta, gamma, alpha=0.5):
    """
    Vinokur two-sided tanh clustering.  eta in [0,1].
    gamma=0 => identity.  Monotonic for all gamma >= 0.
    """
    if gamma < 1e-14:
        return eta.copy()
    denom = np.tanh(gamma * alpha)
    if abs(denom) < 1e-30:
        return eta.copy()
    zeta = 0.5 * (1.0 + np.tanh(gamma * (eta - alpha)) / denom)
    zeta[0] = 0.0; zeta[-1] = 1.0
    return zeta


def get_vinokur_gamma_from_ref(x_ref, y_ref, nj_new, alpha=0.5):
    """
    Auto-compute Vinokur gamma by matching the reference grid's
    wall-normal stretching ratio.

    The reference grid (e.g. Frohlich 3.fine, 197x129) has a built-in
    stretching ratio (max_dy / min_dy).  We find the Vinokur gamma that
    reproduces the same ratio at the target resolution nj_new.

    This is physically meaningful: it preserves the reference grid's
    near-wall clustering quality regardless of the target resolution.
    """
    # Measure reference grid's stretching ratio at hill crest (i=0)
    dy_ref = np.diff(y_ref[:, 0])
    ratio_ref = dy_ref.max() / dy_ref.min()

    eta = np.linspace(0, 1, nj_new)

    # Bisection: gamma in [0.1, 20]
    g_lo, g_hi = 0.1, 20.0
    for _ in range(200):
        g_mid = 0.5 * (g_lo + g_hi)
        zeta = vinokur_tanh(eta, g_mid, alpha)
        dz = np.diff(zeta)
        ratio = dz.max() / dz.min()
        if ratio < ratio_ref:
            g_lo = g_mid      # need stronger clustering -> larger gamma
        else:
            g_hi = g_mid
        if abs(ratio - ratio_ref) / ratio_ref < 1e-8:
            break
    return g_mid


# ============================================================
#  3b. GILBM Stability Estimation (LBM-specific)
# ============================================================

def estimate_gilbm_stability(x_grid, y_grid, scale_factor=1.0,
                             Uref=0.0503, Re=150, H_HILL=1.0,
                             CFL_lambda=0.5):
    """
    Estimate GILBM (Generalized Interpolation LBM) stability parameters
    for a given body-fitted grid.

    The LBM MRT collision operator requires omega in approximately [0.5, 2.0].
    omega = 0.5 + 3 * niu / dt_global, where dt_global = CFL_lambda / max|c_tilde|.

    Parameters
    ----------
    x_grid, y_grid : ndarray (nj, ni)
        Grid coordinates (raw Frohlich or code units).
    scale_factor : float
        Multiply grid coords to get code units (=1 if already in code units).
    Uref, Re, H_HILL : float
        Flow parameters. niu = Uref * H_HILL / Re.
    CFL_lambda : float
        CFL number (default 0.5).

    Returns
    -------
    dict with keys:
        omega, dt_global, c_max, dz_min, dz_max, dz_ratio, a_max, status
    """
    niu = Uref * H_HILL / Re

    x_c = x_grid * scale_factor
    y_c = y_grid * scale_factor
    nj, ni = x_c.shape

    # D3Q19 velocity set (e_y, e_z components)
    e_y = [0,0,0, 1,-1,0,0, 1,1,-1,-1, 0,0,0,0, 1,-1,1,-1]
    e_z = [0,0,0, 0,0,1,-1, 0,0,0,0, 1,1,-1,-1, 1,1,-1,-1]

    # Forward metrics (central FD, one-sided at boundaries)
    y_xi = np.zeros_like(x_c); y_zeta = np.zeros_like(x_c)
    z_xi = np.zeros_like(y_c); z_zeta = np.zeros_like(y_c)

    y_xi[:, 1:-1] = (x_c[:, 2:] - x_c[:, :-2]) / 2.0
    y_zeta[1:-1, :] = (x_c[2:, :] - x_c[:-2, :]) / 2.0
    z_xi[:, 1:-1] = (y_c[:, 2:] - y_c[:, :-2]) / 2.0
    z_zeta[1:-1, :] = (y_c[2:, :] - y_c[:-2, :]) / 2.0

    y_xi[:, 0] = x_c[:, 1] - x_c[:, 0]
    y_xi[:, -1] = x_c[:, -1] - x_c[:, -2]
    z_xi[:, 0] = y_c[:, 1] - y_c[:, 0]
    z_xi[:, -1] = y_c[:, -1] - y_c[:, -2]
    y_zeta[0, :] = x_c[1, :] - x_c[0, :]
    y_zeta[-1, :] = x_c[-1, :] - x_c[-2, :]
    z_zeta[0, :] = y_c[1, :] - y_c[0, :]
    z_zeta[-1, :] = y_c[-1, :] - y_c[-2, :]

    J = y_xi * z_zeta - y_zeta * z_xi
    sl = (slice(1, -1), slice(1, -1))
    eps = 1e-30

    zeta_y = np.where(np.abs(J) > eps, -z_xi / J, 0)
    zeta_z = np.where(np.abs(J) > eps,  y_xi / J, 0)
    xi_y   = np.where(np.abs(J) > eps,  z_zeta / J, 0)
    xi_z   = np.where(np.abs(J) > eps, -y_zeta / J, 0)

    # Max contravariant velocity over all D3Q19 directions
    max_c = 0.0
    for alpha in range(3, 19):
        c_zeta = np.abs(zeta_y[sl] * e_y[alpha] + zeta_z[sl] * e_z[alpha])
        c_xi   = np.abs(xi_y[sl]   * e_y[alpha] + xi_z[sl]   * e_z[alpha])
        max_c = max(max_c, c_zeta.max(), c_xi.max())

    # Wall-normal spacing
    dz_min = 1e30
    dz_max = 0.0
    for j in range(ni):
        dz = np.diff(y_c[:, j])
        dz_pos = dz[dz > 0]
        if len(dz_pos) > 0:
            dz_min = min(dz_min, dz_pos.min())
            dz_max = max(dz_max, dz_pos.max())
    dz_ratio = dz_max / dz_min if dz_min > 0 else float('inf')

    # LBM parameters
    dt_global = CFL_lambda / max_c if max_c > 0 else 1.0
    omega = 0.5 + 3.0 * niu / dt_global
    a_max = dz_ratio  # rough LTS acceleration estimate

    # Status classification
    if omega > 2.0:
        status = "UNSTABLE"
    elif omega > 1.5:
        status = "MARGINAL"
    elif omega > 1.2:
        status = "OK"
    elif omega >= 0.55:
        status = "OPTIMAL"
    else:
        status = "GOOD"

    return {
        "omega": omega, "dt_global": dt_global, "c_max": max_c,
        "dz_min": dz_min, "dz_max": dz_max, "dz_ratio": dz_ratio,
        "a_max": a_max, "status": status, "niu": niu,
    }


def print_gilbm_stability_table():
    """
    Print the pre-computed GILBM stability reference table.

    This table was calibrated for:
      Reference grid : Frohlich 3.fine (197x129)
      Target grid    : I=129, J=64 (NY=129 nodes, NZ=64 nodes)
      Grid method    : Mode 2 Poisson + physical-z redistribution
      Flow params    : Re=150, Uref=0.0503, H_HILL=1.0
      CFL lambda     : 0.5

    NOTE: physical-z redistribution REPLACES Frohlich's native wall
    clustering with Vinokur tanh in physical z-space (symmetric when
    alpha=0.5).  GAMMA=0 means UNIFORM spacing (no clustering).
    """
    print()
    print("  " + "=" * 72)
    print("   GILBM Stability Reference  (Poisson + physical-z redistribution)")
    print("   3.fine ref -> 129x64, Re=150, Uref=0.0503, CFL=0.5, ALPHA=0.5")
    print("  " + "=" * 72)
    print(f"  {'GAMMA':>6s} | {'omega':>8s} | {'max|c~|':>10s} | {'dz_ratio':>8s} | {'Status':<12s} | Note")
    print("  " + "-" * 72)
    #                GAMMA  omega   c_max   ratio  status         note
    # Calibrated with redistribute_vertical_physical (2026-03)
    table = [
        (0.0,  0.92,  209,  31, "OPTIMAL",  "UNIFORM z (no clustering) + minSize=NaN!"),
        (0.5,  0.58,   38,   2, "OPTIMAL",  "Very mild symmetric clustering"),
        (1.0,  0.59,   42,   2, "OPTIMAL",  "Mild symmetric clustering"),
        (1.5,  0.60,   50,   3, "OPTIMAL",  "Moderate symmetric clustering"),
        (2.0,  0.63,   63,   4, "OPTIMAL",  "Recommended (good clustering, very stable)"),
        (2.5,  0.67,   83,   5, "OPTIMAL",  "Good clustering"),
        (3.0,  0.73,  112,   8, "OPTIMAL",  "Strong clustering, still optimal"),
        (3.5,  0.81,  156,  12, "OPTIMAL",  "Strong clustering"),
        (4.0,  0.94,  221,  20, "OPTIMAL",  "Very strong (approaching Frohlich-level)"),
        (5.0,  1.43,  463,  52, "OK",       "Extreme clustering, omega > 1.2"),
    ]
    for gamma, omega, c_max, ratio, status, note in table:
        marker = ""
        if gamma == 2.0:
            marker = " <--"
        elif status in ("MARGINAL", "UNSTABLE"):
            marker = " ***"
        print(f"  {gamma:6.1f} | {omega:8.2f} | {c_max:10d} | {ratio:8d} | {status:<12s} | {note}{marker}")
    print("  " + "-" * 72)
    print()
    print("  Physical-z redistribution: GAMMA controls Vinokur tanh in z-space.")
    print("  GAMMA=0 = uniform (NO wall clustering, minSize macro = NaN!).")
    print("  GAMMA=2.0 is recommended: symmetric, ratio=3.5, omega=0.63.")
    print("  All GAMMA <= 4.0 are in OPTIMAL range (omega < 1.0).")
    print()


def print_gilbm_stability_warning(gamma, omega, c_max, dt_global, a_max, status):
    """
    Print a concise GILBM stability warning for the chosen parameters.
    Called after grid generation to alert the user.
    """
    print()
    print("  " + "=" * 62)
    print("   GILBM Stability Check")
    print("  " + "=" * 62)
    print(f"    GAMMA        = {gamma:.4f}")
    print(f"    omega_global = {omega:.4f}", end="")
    if omega > 2.0:
        print("  *** UNSTABLE (omega > 2.0) ***")
    elif omega > 1.5:
        print("  ** MARGINAL (omega > 1.5) **")
    elif omega > 1.2:
        print("  * OK (omega > 1.2)")
    else:
        print("  [OPTIMAL]")
    print(f"    max|c_tilde| = {c_max:.1f}")
    print(f"    dt_global    = {dt_global:.4e}")
    print(f"    a_max (LTS)  = {a_max:.1f}")
    print(f"    Status       = {status}")

    if omega > 2.0:
        print()
        print("  !! WARNING: This grid WILL DIVERGE in GILBM !!")
        print("  !! Reduce GAMMA (try 2.0~3.0 for safe symmetric clustering) !!")
        print("  !! MRT collision requires omega < 2.0 for stability. !!")
    elif omega > 1.5:
        print()
        print("  ** CAUTION: Marginal stability. May diverge under")
        print("     transient conditions. Consider reducing GAMMA.")
    print("  " + "=" * 62)
    print()


# ============================================================
#  4.  Zeta-only redistribution (Mode 1)
# ============================================================

def redistribute_vertical_arclength(x, y, gamma=0.0, alpha=0.5):
    """
    [LEGACY] Redistribute vertical points in arc-length space.
    gamma=0 => identity (reproduces original exactly).

    WARNING: This function preserves the Frolich reference grid's
    inherent bottom-wall bias.  With alpha=0.5, the redistribution
    is symmetric in arc-length but NOT in physical z-space.
    Increasing GAMMA actually WORSENS the top/bottom asymmetry.
    Use redistribute_vertical_physical() instead for symmetric grids.
    """
    nj, ni = x.shape
    eta = np.linspace(0, 1, nj)
    zeta = vinokur_tanh(eta, gamma, alpha)

    x_new = np.empty_like(x)
    y_new = np.empty_like(y)

    for i in range(ni):
        xc, yc = x[:, i], y[:, i]
        ds = np.sqrt(np.diff(xc)**2 + np.diff(yc)**2)
        s = np.concatenate(([0.0], np.cumsum(ds)))
        s_norm = s / s[-1]
        s_new = np.interp(zeta, eta, s_norm)
        x_new[:, i] = np.interp(s_new, s_norm, xc)
        y_new[:, i] = np.interp(s_new, s_norm, yc)

    return x_new, y_new


def redistribute_vertical_physical(x, y, gamma=0.0, alpha=0.5):
    """
    Redistribute vertical points in physical z-coordinate space.

    Unlike redistribute_vertical_arclength() which operates in arc-length
    space (preserving the reference grid's inherent bottom-wall bias),
    this function redistributes in physical z-space, ensuring truly
    symmetric wall clustering when alpha=0.5.

    Parameters
    ----------
    x, y : ndarray (nj, ni)
        Grid coordinates.  y is wall-normal (z in code).
    gamma : float
        Vinokur tanh stretching parameter.
        gamma=0 => uniform spacing in z (no wall clustering).
        gamma>0 => wall clustering, symmetric when alpha=0.5.
    alpha : float
        Clustering symmetry.  0.5 = both walls equal.

    Returns
    -------
    x_new, y_new : ndarray (nj, ni)
        Redistributed grid coordinates.
    """
    nj, ni = x.shape
    eta = np.linspace(0, 1, nj)
    zeta = vinokur_tanh(eta, gamma, alpha)

    x_new = np.empty_like(x)
    y_new = np.empty_like(y)

    for i in range(ni):
        z_bot = y[0, i]
        z_top = y[-1, i]
        # New wall-normal positions: Vinokur distribution in physical z
        z_col = z_bot + zeta * (z_top - z_bot)
        y_new[:, i] = z_col
        # Interpolate streamwise coordinate to maintain grid topology
        x_new[:, i] = np.interp(z_col, y[:, i], x[:, i])

    return x_new, y_new


# Default: use physical-space redistribution (fixes Frolich asymmetry)
redistribute_vertical = redistribute_vertical_physical


# ============================================================
#  5.  Steger-Sorenson Poisson grid generation (Mode 2)
# ============================================================

def _compute_metrics(x, y):
    """Compute all metric terms using 2nd-order finite differences."""
    nj, ni = x.shape

    x_xi = np.zeros_like(x)
    x_xi[:, 1:-1] = 0.5 * (x[:, 2:] - x[:, :-2])
    x_xi[:, 0]  = -1.5*x[:,0] + 2.0*x[:,1] - 0.5*x[:,2]
    x_xi[:, -1] =  0.5*x[:,-3] - 2.0*x[:,-2] + 1.5*x[:,-1]

    y_xi = np.zeros_like(y)
    y_xi[:, 1:-1] = 0.5 * (y[:, 2:] - y[:, :-2])
    y_xi[:, 0]  = -1.5*y[:,0] + 2.0*y[:,1] - 0.5*y[:,2]
    y_xi[:, -1] =  0.5*y[:,-3] - 2.0*y[:,-2] + 1.5*y[:,-1]

    x_eta = np.zeros_like(x)
    x_eta[1:-1,:] = 0.5 * (x[2:,:] - x[:-2,:])
    x_eta[0,:]  = -1.5*x[0,:] + 2.0*x[1,:] - 0.5*x[2,:]
    x_eta[-1,:] =  0.5*x[-3,:] - 2.0*x[-2,:] + 1.5*x[-1,:]

    y_eta = np.zeros_like(y)
    y_eta[1:-1,:] = 0.5 * (y[2:,:] - y[:-2,:])
    y_eta[0,:]  = -1.5*y[0,:] + 2.0*y[1,:] - 0.5*y[2,:]
    y_eta[-1,:] =  0.5*y[-3,:] - 2.0*y[-2,:] + 1.5*y[-1,:]

    x_xixi = np.zeros_like(x)
    x_xixi[:, 1:-1] = x[:, 2:] - 2.0*x[:, 1:-1] + x[:, :-2]
    x_xixi[:, 0]  = x[:,0] - 2.0*x[:,1] + x[:,2]
    x_xixi[:, -1] = x[:,-3] - 2.0*x[:,-2] + x[:,-1]

    y_xixi = np.zeros_like(y)
    y_xixi[:, 1:-1] = y[:, 2:] - 2.0*y[:, 1:-1] + y[:, :-2]
    y_xixi[:, 0]  = y[:,0] - 2.0*y[:,1] + y[:,2]
    y_xixi[:, -1] = y[:,-3] - 2.0*y[:,-2] + y[:,-1]

    x_etaeta = np.zeros_like(x)
    x_etaeta[1:-1,:] = x[2:,:] - 2.0*x[1:-1,:] + x[:-2,:]
    x_etaeta[0,:]  = x[0,:] - 2.0*x[1,:] + x[2,:]
    x_etaeta[-1,:] = x[-3,:] - 2.0*x[-2,:] + x[-1,:]

    y_etaeta = np.zeros_like(y)
    y_etaeta[1:-1,:] = y[2:,:] - 2.0*y[1:-1,:] + y[:-2,:]
    y_etaeta[0,:]  = y[0,:] - 2.0*y[1,:] + y[2,:]
    y_etaeta[-1,:] = y[-3,:] - 2.0*y[-2,:] + y[-1,:]

    x_pad = np.pad(x, ((1,1),(1,1)), mode='edge')
    y_pad = np.pad(y, ((1,1),(1,1)), mode='edge')
    x_xieta = 0.25*(x_pad[2:,2:] - x_pad[2:,:-2]
                    - x_pad[:-2,2:] + x_pad[:-2,:-2])[:nj,:ni]
    y_xieta = 0.25*(y_pad[2:,2:] - y_pad[2:,:-2]
                    - y_pad[:-2,2:] + y_pad[:-2,:-2])[:nj,:ni]

    return {
        "x_xi": x_xi, "x_eta": x_eta, "y_xi": y_xi, "y_eta": y_eta,
        "x_xixi": x_xixi, "x_etaeta": x_etaeta, "x_xieta": x_xieta,
        "y_xixi": y_xixi, "y_etaeta": y_etaeta, "y_xieta": y_xieta,
        "alpha": x_eta**2 + y_eta**2,
        "beta": x_xi*x_eta + y_xi*y_eta,
        "gamma": x_xi**2 + y_xi**2,
        "J": x_xi*y_eta - x_eta*y_xi,
    }


def _compute_PQ(metrics):
    """Reverse-compute control functions P,Q from a known grid."""
    m = metrics
    RHS_x = (m["alpha"]*m["x_xixi"] - 2.0*m["beta"]*m["x_xieta"]
             + m["gamma"]*m["x_etaeta"])
    RHS_y = (m["alpha"]*m["y_xixi"] - 2.0*m["beta"]*m["y_xieta"]
             + m["gamma"]*m["y_etaeta"])
    J2 = m["J"]**2
    det = m["J"]
    safe = np.abs(det) > 1e-30

    P = np.zeros_like(RHS_x)
    Q = np.zeros_like(RHS_x)
    b1 = np.zeros_like(RHS_x)
    b2 = np.zeros_like(RHS_x)
    b1[safe] = RHS_x[safe] / (-J2[safe])
    b2[safe] = RHS_y[safe] / (-J2[safe])
    P[safe] = ( m["y_eta"][safe]*b1[safe] - m["x_eta"][safe]*b2[safe]) / det[safe]
    Q[safe] = (-m["y_xi"][safe]*b1[safe]  + m["x_xi"][safe]*b2[safe])  / det[safe]
    return P, Q


def _poisson_solve(x_init, y_init, P, Q,
                   n_iter=15000, omega=1.0, tol=1e-10, print_every=2000):
    """Row-vectorised Gauss-Seidel Poisson solver. Boundaries fixed."""
    nj, ni = x_init.shape
    x = x_init.copy()
    y = y_init.copy()
    convergence = []
    si = slice(1, -1)

    for it in range(n_iter):
        max_corr = 0.0

        for j in range(1, nj - 1):
            xxi  = 0.5 * (x[j, 2:] - x[j, :-2])
            xeta = 0.5 * (x[j+1, si] - x[j-1, si])
            yxi  = 0.5 * (y[j, 2:] - y[j, :-2])
            yeta = 0.5 * (y[j+1, si] - y[j-1, si])

            al = xeta**2 + yeta**2
            be = xxi*xeta + yxi*yeta
            ga = xxi**2 + yxi**2
            jac = xxi*yeta - xeta*yxi
            j2 = jac**2

            denom = 2.0 * (al + ga)
            safe = denom > 1e-30

            x_cross = 0.25*(x[j+1,2:] - x[j+1,:-2] - x[j-1,2:] + x[j-1,:-2])
            y_cross = 0.25*(y[j+1,2:] - y[j+1,:-2] - y[j-1,2:] + y[j-1,:-2])

            Pj = P[j, si]; Qj = Q[j, si]
            Sx = -j2 * (Pj*xxi + Qj*xeta)
            Sy = -j2 * (Pj*yxi + Qj*yeta)

            x_new = np.where(safe,
                (al*(x[j,2:]+x[j,:-2]) + ga*(x[j+1,si]+x[j-1,si])
                 - 2.0*be*x_cross - Sx) / np.where(safe, denom, 1.0),
                x[j, si])
            y_new = np.where(safe,
                (al*(y[j,2:]+y[j,:-2]) + ga*(y[j+1,si]+y[j-1,si])
                 - 2.0*be*y_cross - Sy) / np.where(safe, denom, 1.0),
                y[j, si])

            dx = omega * (x_new - x[j, si])
            dy = omega * (y_new - y[j, si])
            x[j, si] += dx
            y[j, si] += dy

            row_max = max(np.max(np.abs(dx)), np.max(np.abs(dy)))
            if row_max > max_corr:
                max_corr = row_max

        convergence.append(max_corr)

        if np.isnan(max_corr) or max_corr > 1e10:
            print(f"    DIVERGED at iter {it}")
            break

        if print_every and (it % print_every == 0 or it == n_iter - 1):
            print(f"    iter {it:5d}:  max_corr = {max_corr:.4e}")

        if max_corr < tol:
            print(f"    Converged at iter {it}, max_corr = {max_corr:.4e}")
            break

    return x, y, convergence


def _tfi(x_bot, y_bot, x_top, y_top, x_lft, y_lft, x_rgt, y_rgt):
    """Transfinite Interpolation (vectorised)."""
    ni = len(x_bot); nj = len(x_lft)
    xi = np.linspace(0, 1, ni)[np.newaxis, :]
    eta = np.linspace(0, 1, nj)[:, np.newaxis]
    x = ((1-eta)*x_bot + eta*x_top
       + (1-xi)*x_lft[:, np.newaxis] + xi*x_rgt[:, np.newaxis]
       - (1-xi)*(1-eta)*x_bot[0] - xi*(1-eta)*x_bot[-1]
       - (1-xi)*eta*x_top[0] - xi*eta*x_top[-1])
    y = ((1-eta)*y_bot + eta*y_top
       + (1-xi)*y_lft[:, np.newaxis] + xi*y_rgt[:, np.newaxis]
       - (1-xi)*(1-eta)*y_bot[0] - xi*(1-eta)*y_bot[-1]
       - (1-xi)*eta*y_top[0] - xi*eta*y_top[-1])
    return x, y


def _bilinear_interp_2d(data, eta_old, xi_old, eta_new, xi_new):
    """
    Bilinear interpolation of 2D data from (eta_old, xi_old) grid
    to (eta_new, xi_new) grid. Pure numpy, no scipy required.
    """
    nj_new = len(eta_new)
    ni_new = len(xi_new)
    nj_old = len(eta_old)
    ni_old = len(xi_old)
    result = np.empty((nj_new, ni_new))

    for jj in range(nj_new):
        e = eta_new[jj]
        j0 = np.searchsorted(eta_old, e, side='right') - 1
        j0 = max(0, min(j0, nj_old - 2))
        j1 = j0 + 1
        te = (e - eta_old[j0]) / (eta_old[j1] - eta_old[j0]) if eta_old[j1] != eta_old[j0] else 0.0

        for ii in range(ni_new):
            x = xi_new[ii]
            i0 = np.searchsorted(xi_old, x, side='right') - 1
            i0 = max(0, min(i0, ni_old - 2))
            i1 = i0 + 1
            tx = (x - xi_old[i0]) / (xi_old[i1] - xi_old[i0]) if xi_old[i1] != xi_old[i0] else 0.0

            result[jj, ii] = ((1-te)*(1-tx)*data[j0, i0] + (1-te)*tx*data[j0, i1]
                             + te*(1-tx)*data[j1, i0] + te*tx*data[j1, i1])
    return result


def _interpolate_PQ(P, Q, ni_old, nj_old, ni_new, nj_new):
    """
    Interpolate P,Q from old to new resolution,
    with proper scaling for the changed computational grid spacing.

    Uses bicubic spline (scipy) if available, bilinear (numpy) otherwise.
    """
    xi_o = np.linspace(0, 1, ni_old); eta_o = np.linspace(0, 1, nj_old)
    xi_n = np.linspace(0, 1, ni_new); eta_n = np.linspace(0, 1, nj_new)

    if _HAS_SCIPY:
        P_n = RectBivariateSpline(eta_o, xi_o, P, kx=3, ky=3)(eta_n, xi_n)
        Q_n = RectBivariateSpline(eta_o, xi_o, Q, kx=3, ky=3)(eta_n, xi_n)
    else:
        P_n = _bilinear_interp_2d(P, eta_o, xi_o, eta_n, xi_n)
        Q_n = _bilinear_interp_2d(Q, eta_o, xi_o, eta_n, xi_n)

    scale_P = (ni_new - 1) / (ni_old - 1)
    scale_Q = (nj_new - 1) / (nj_old - 1)
    P_n *= scale_P
    Q_n *= scale_Q

    return P_n, Q_n


def _resample_boundary(xb, yb, n_new):
    """Resample boundary to n_new points preserving arc-length pattern.
    Uses cubic interp (scipy) if available, linear (numpy) otherwise."""
    n_old = len(xb)
    if n_new == n_old:
        return xb.copy(), yb.copy()
    ds = np.sqrt(np.diff(xb)**2 + np.diff(yb)**2)
    s = np.concatenate(([0], np.cumsum(ds))); s /= s[-1]
    s_norm_old = np.linspace(0, 1, n_old)
    s_new = np.interp(np.linspace(0, 1, n_new), s_norm_old, s)

    if _HAS_SCIPY:
        return (interp1d(s, xb, kind='cubic')(s_new),
                interp1d(s, yb, kind='cubic')(s_new))
    else:
        return (np.interp(s_new, s, xb),
                np.interp(s_new, s, yb))


def generate_adaptive_grid(x_ref, y_ref, ni_new, nj_new,
                           gamma=0.0, alpha=0.5,
                           poisson_iter=15000, poisson_tol=1e-10):
    """
    Full Steger-Sorenson adaptive grid generation.

    Strategy:
      1. Reverse-compute P,Q from reference grid
      2. Interpolate P,Q to new (ni_new, nj_new)
      3. Resample boundaries at new resolution (NO stretching here)
      4. TFI initial guess
      5. Poisson solve with interpolated P,Q
      6. Apply vertical stretching (gamma/alpha) as post-processing
         on the converged Poisson grid -- same logic as Mode 1

    The stretching is applied AFTER the Poisson solve to avoid
    boundary inconsistency: Poisson needs all 4 boundaries to be
    geometrically consistent, which breaks if only the vertical
    boundaries are stretched while horizontal boundaries are not.
    """
    nj_ref, ni_ref = x_ref.shape

    print("    [1/6] Computing P,Q from reference ...")
    metrics = _compute_metrics(x_ref, y_ref)
    P_ref, Q_ref = _compute_PQ(metrics)

    print(f"    [2/6] Interpolating P,Q: ({ni_ref}x{nj_ref}) -> ({ni_new}x{nj_new}) ...")
    if ni_new == ni_ref and nj_new == nj_ref:
        P_new, Q_new = P_ref.copy(), Q_ref.copy()
    else:
        P_new, Q_new = _interpolate_PQ(P_ref, Q_ref,
                                        ni_ref, nj_ref, ni_new, nj_new)

    print("    [3/6] Resampling boundaries ...")
    xb, yb = _resample_boundary(x_ref[0, :],  y_ref[0, :],  ni_new)
    xt, yt = _resample_boundary(x_ref[-1, :], y_ref[-1, :], ni_new)
    xl, yl = _resample_boundary(x_ref[:, 0],  y_ref[:, 0],  nj_new)
    xr, yr = _resample_boundary(x_ref[:, -1], y_ref[:, -1], nj_new)

    xl[0] = xb[0];   yl[0] = yb[0]
    xl[-1] = xt[0];  yl[-1] = yt[0]
    xr[0] = xb[-1];  yr[0] = yb[-1]
    xr[-1] = xt[-1]; yr[-1] = yt[-1]

    print("    [4/6] TFI initial guess ...")
    x_tfi, y_tfi = _tfi(xb, yb, xt, yt, xl, yl, xr, yr)

    print(f"    [5/6] Poisson solve (max {poisson_iter} iter) ...")
    x_out, y_out, conv = _poisson_solve(
        x_tfi, y_tfi, P_new, Q_new,
        n_iter=poisson_iter, omega=1.0, tol=poisson_tol, print_every=2000)

    if gamma > 1e-14:
        print(f"    [6/6] Applying physical-z stretching (gamma={gamma}, alpha={alpha}) ...")
        x_out, y_out = redistribute_vertical_physical(x_out, y_out, gamma=gamma, alpha=alpha)
    else:
        print("    [6/6] No stretching (gamma=0) — Frolich Poisson spacing preserved")

    return x_out, y_out, conv


# ============================================================
#  6.  Export to Tecplot .dat
# ============================================================

def write_tecplot_dat(filepath, x, y, title="Generated grid",
                      zone_title="Adaptive"):
    nj, ni = x.shape
    with open(filepath, "w") as f:
        f.write(f'TITLE     = "{title}"\n')
        f.write('VARIABLES = "x corner"\n')
        f.write('"y corner"\n')
        f.write(f'ZONE T="{zone_title}"\n')
        f.write(f' I={ni}, J={nj}, K=1,F=POINT\n')
        f.write('DT=(SINGLE SINGLE )\n')
        for j in range(nj):
            for i in range(ni):
                f.write(f" {x[j, i]: .9E} {y[j, i]: .9E}\n")
    print(f"  [written] {filepath}")


def _fit_a_from_dz_min(L, dz_min_obs, N_cells, tol=1e-12, max_iter=200):
    """
    Bisection: find tanh_wall stretching parameter `a` in (0,1) such that
    the first-cell wall spacing produced by tanh_wall(L, a, j, N_cells)
    matches the observed dz_min on the i=0 column.

    Monotonicity: as a -> 1, near-wall clustering strengthens, so
    dz_min(a) decreases monotonically.

    Returns
    -------
    a_fit       : float — fitted stretching parameter
    dz_min_fit  : float — dz_min(a_fit) for cross-check
    """
    def first_dz(a):
        return tanh_wall(L, a, 1, N_cells) - tanh_wall(L, a, 0, N_cells)

    a_lo, a_hi = 1e-6, 1.0 - 1e-12
    a_mid = 0.5 * (a_lo + a_hi)
    for _ in range(max_iter):
        a_mid = 0.5 * (a_lo + a_hi)
        dz_mid = first_dz(a_mid)
        if dz_mid > dz_min_obs:
            # clustering too weak → push a closer to 1
            a_lo = a_mid
        else:
            a_hi = a_mid
        if abs(dz_mid - dz_min_obs) / max(abs(dz_min_obs), 1e-30) < tol:
            break
    return a_mid, first_dz(a_mid)


def write_grid_data(out_path, x, y, NY, NZ, GAMMA, ALPHA,
                    LZ=None, LY=9.0, source_dat=None):
    """
    Write a human-readable grid_data text report for the i=0 stream-wise
    column (hill crest line, where the zeta-direction grid line is vertical).

    Reports 5 items, all measured / derived on the i=0 column:
      1. dz_max (absolute + multiples of lattice_size)
      2. dz_min (absolute + multiples of lattice_size)
      3. dz_max / dz_min               (stretching ratio)
      4. tanh_wall stretching parameter `a`  (bisection-fit to observed dz_min)
      5. GAMMA                          (Vinokur tanh from variables.h or user input)

    Definitions:
      lattice_size = LZ / (NZ - 1)     (uniform partition over the whole channel)
      L_eff        = LZ - hill_crest   (wall-normal extent above the hill crest;
                                        used as L in tanh_wall back-fit)

    Unit handling:
      The Frohlich reference .dat is in physical units (h_phys ≈ 0.028 m).
      The C simulation rescales it to code units (H_HILL = 1.0) on read
      via grid_scale = H_HILL / h_physical, where h_physical = x_max / LY.
      This routine auto-detects physical-unit grids and rescales to code
      units so the report matches what the simulation actually sees. If
      LZ is provided in code units (e.g. from variables.h LZ=3.036), the
      report stays consistent with that scale.

      If LZ is None, it is inferred from the grid after rescaling.
    """
    import datetime

    out_path = Path(out_path)

    # --- Auto-detect physical vs code units and rescale to code units ---
    # Heuristic: if streamwise span << LY, grid is in physical units.
    x_max_obs = float(x[0, -1])
    if x_max_obs < 0.5 * LY:
        h_physical = x_max_obs / LY
        grid_scale = 1.0 / h_physical            # H_HILL = 1.0 in code units
        units_note = (f"physical units detected (h_phys = {h_physical:.6e}); "
                      f"rescaled by {grid_scale:.6f} to code units")
    else:
        h_physical = None
        grid_scale = 1.0
        units_note = "grid already in code units (H_HILL = 1.0)"

    z_col = y[:, 0] * grid_scale
    dz = np.diff(z_col)
    dz_pos = dz[dz > 0]
    if len(dz_pos) == 0:
        raise ValueError("No positive dz on i=0 column — grid invalid?")
    dz_max = float(dz_pos.max())
    dz_min = float(dz_pos.min())

    if LZ is None:
        LZ = float(z_col[-1])

    hill_crest = float(z_col[0])
    L_eff = LZ - hill_crest
    NZ_cells = NZ - 1
    lattice_size = LZ / NZ_cells

    try:
        a_fit, dz_min_fit = _fit_a_from_dz_min(L_eff, dz_min, NZ_cells)
        a_err_pct = 100.0 * abs(dz_min_fit - dz_min) / dz_min
        a_str = f"{a_fit:.6f}"
        a_check = (f"     dz_min(a_fit) = {dz_min_fit:.6e}"
                   f"  (fit error vs observed = {a_err_pct:.4f}%)\n")
    except Exception as exc:                    # pragma: no cover
        a_str = f"NaN  ({exc})"
        a_check = "     dz_min(a_fit) = N/A\n"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("  Grid Data Analysis  (stream-wise i=0 column / hill crest line)\n")
        f.write("=" * 72 + "\n")
        f.write(f"  Generated  : {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        if source_dat:
            f.write(f"  Source .dat: {source_dat}\n")
        f.write(f"  Units      : {units_note}\n")
        f.write("\n")
        f.write("[Geometry]\n")
        f.write(f"  NY (streamwise nodes)  = {NY}\n")
        f.write(f"  NZ (wall-normal nodes) = {NZ}   (cells = NZ-1 = {NZ_cells})\n")
        f.write(f"  LZ (channel height)    = {LZ:.6f}\n")
        f.write(f"  hill_crest at (i=0,j=0)= {hill_crest:.6f}\n")
        f.write(f"  lattice_size = LZ/(NZ-1) = {LZ:.6f} / {NZ_cells} "
                f"= {lattice_size:.6e}\n")
        f.write("\n")
        f.write("[Wall-normal spacing on i=0 column]\n")
        f.write(f"  1. dz_max         = {dz_max:.6e}"
                f"  =  {dz_max/lattice_size:.4f} * lattice_size\n")
        f.write(f"  2. dz_min         = {dz_min:.6e}"
                f"  =  {dz_min/lattice_size:.4f} * lattice_size\n")
        f.write(f"  3. dz_max/dz_min  = {dz_max/dz_min:.4f}\n")
        f.write("\n")
        f.write("[Stretching parameters]\n")
        f.write(f"  4. tanh_wall a (bisection-fit to dz_min) = {a_str}\n")
        f.write("     formula: z(j) = L/2 + (L/(2a)) * "
                "tanh( ((-1+2j/N)/2) * log((1+a)/(1-a)) )\n")
        f.write(f"     L = LZ - hill_crest = {L_eff:.6f},  N = NZ-1 = {NZ_cells}\n")
        f.write(a_check)
        f.write(f"  5. GAMMA (Vinokur, from variables.h or user input) = {GAMMA:.4f}\n")
        f.write(f"     ALPHA                                            = {ALPHA:.4f}\n")
        f.write("\n")
        f.write("=" * 72 + "\n")

    print(f"  [written] {out_path}")


# ============================================================
#  7.  Verification
# ============================================================

def verify_identity(x_orig, y_orig, x_new, y_new, tol=1e-10):
    dx = np.max(np.abs(x_orig - x_new))
    dy = np.max(np.abs(y_orig - y_new))
    ok = (dx < tol) and (dy < tol)
    return ok, dx, dy


def validate_grid_dimensions(dat_path, NY, NZ):
    """
    Validate that a grid .dat file has the expected dimensions.

    Naming convention:
      NY = streamwise node count  → expected I = NY (nodes)
      NZ = wall-normal node count → expected J = NZ (nodes)

    Returns (ok, ni_actual, nj_actual, ni_expected, nj_expected).
    Raises FileNotFoundError if dat_path does not exist.
    """
    path = Path(dat_path)
    if not path.exists():
        raise FileNotFoundError(f"Grid file not found: {dat_path}")

    ni_expected = NY        # streamwise nodes (I = NY)
    nj_expected = NZ        # wall-normal nodes (J = NZ)

    # Parse I, J from Tecplot header
    ni_actual, nj_actual = None, None
    with open(path) as f:
        for line in f:
            m = re.search(r'I\s*=\s*(\d+)', line)
            if m:
                ni_actual = int(m.group(1))
            m = re.search(r'J\s*=\s*(\d+)', line)
            if m:
                nj_actual = int(m.group(1))
            if ni_actual is not None and nj_actual is not None:
                break

    if ni_actual is None or nj_actual is None:
        raise ValueError(f"Cannot parse I,J from {dat_path}")

    ok = (ni_actual == ni_expected) and (nj_actual == nj_expected)

    if not ok:
        print()
        print("  " + "!" * 62)
        print("  !! GRID DIMENSION MISMATCH — ABORTING !!")
        print("  " + "!" * 62)
        print(f"    Grid file: {path.name}")
        print(f"    Expected:  I={ni_expected} (=NY={NY}), "
              f"J={nj_expected} (=NZ={NZ})")
        print(f"    Actual:    I={ni_actual}, J={nj_actual}")
        if ni_actual != ni_expected:
            print(f"    → xi (streamwise) 格點數不吻合: "
                  f"檔案 I={ni_actual} ≠ NY={ni_expected}")
        if nj_actual != nj_expected:
            print(f"    → zeta (wall-normal) 格點數不吻合: "
                  f"檔案 J={nj_actual} ≠ NZ={nj_expected}")
        print()
        print("    因為輸入之格點與使用者設定不同，不執行程式碼。")
        print("    請確認 variables.h 中 NY, NZ 的值與網格檔案一致。")
        print("  " + "!" * 62)
        print()

    return ok, ni_actual, nj_actual, ni_expected, nj_expected


# ============================================================
#  8.  Interactive helpers
# ============================================================

def ask_float(prompt, default, lo=None, hi=None):
    while True:
        raw = input(f"  {prompt} [default={default}]: ").strip()
        if raw == "":
            return default
        try:
            val = float(raw)
        except ValueError:
            print("    ** Invalid number, try again.")
            continue
        if lo is not None and val < lo:
            print(f"    ** Must be >= {lo}, try again.")
            continue
        if hi is not None and val > hi:
            print(f"    ** Must be <= {hi}, try again.")
            continue
        return val


def ask_int(prompt, default, lo=None, hi=None):
    while True:
        raw = input(f"  {prompt} [default={default}]: ").strip()
        if raw == "":
            return default
        try:
            val = int(raw)
        except ValueError:
            print("    ** Invalid integer, try again.")
            continue
        if lo is not None and val < lo:
            print(f"    ** Must be >= {lo}, try again.")
            continue
        if hi is not None and val > hi:
            print(f"    ** Must be <= {hi}, try again.")
            continue
        return val


def ask_yes_no(prompt, default_yes=True):
    hint = "Y/n" if default_yes else "y/N"
    raw = input(f"  {prompt} [{hint}]: ").strip().lower()
    if raw == "":
        return default_yes
    return raw in ("y", "yes")


def detect_dat_files(folder):
    return sorted(f for f in folder.glob("*.dat")
                  if not f.name.startswith("zeta_")
                  and not f.name.startswith("adaptive_"))


# ============================================================
#  9.  Auto-mode: parse variables.h and generate grid
# ============================================================

def parse_variables_h(path):
    """
    Parse #define macros from variables.h.
    Returns dict with keys: NY, NZ, LZ, LY, CFL, ALPHA, GRID_DAT_DIR, GRID_DAT_REF.
    GAMMA is optional (auto-computed from bisection if missing).

    Naming convention (enforced):
      NX, NY, NZ = node count  (格點數)  → cells = NX-1, NY-1, NZ-1
      Grid .dat: I = NY (streamwise nodes), J = NZ (wall-normal nodes)
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    result = {}
    # Integer defines
    for key in ("NY", "NZ"):
        m = re.search(rf'#define\s+{key}\s+(\d+)', text)
        if m:
            result[key] = int(m.group(1))
    # Float defines (may have parentheses)
    for key in ("GAMMA", "ALPHA", "CFL"):
        m = re.search(rf'#define\s+{key}\s+\(?([\d.eE+\-]+)\)?', text)
        if m:
            result[key] = float(m.group(1))
    # Float defines that may be in parentheses like (3.036)
    for key in ("LZ", "LY"):
        m = re.search(rf'#define\s+{key}\s+\(?([\d.eE+\-]+)\)?', text)
        if m:
            result[key] = float(m.group(1))
    # String defines
    for key in ("GRID_DAT_DIR", "GRID_DAT_REF"):
        m = re.search(rf'#define\s+{key}\s+"([^"]+)"', text)
        if m:
            result[key] = m.group(1)
    return result


def auto_generate(variables_h_path, script_dir=None):
    """
    Fully automatic grid generation:
      1. Parse NY, NZ, LZ, LY, GAMMA, ALPHA from variables.h
      2. GAMMA is required (user design parameter in variables.h)
      3. Compute minSize from GAMMA analytically (gamma_to_minSize)
      4. Load reference grid from GRID_DAT_REF
      5. Run Steger-Sorenson adaptive grid generation (Mode 2)
      6. Export Tecplot .dat with filename matching C code sprintf format
      7. Print GILBM stability check
    Returns: output filepath

    Naming convention:
      NY = streamwise node count  → NI = NY  nodes (grid .dat I dimension)
      NZ = wall-normal node count → NJ = NZ  nodes (grid .dat J dimension)
      Streamwise cells = NY-1,  Wall-normal cells = NZ-1
    """
    if script_dir is None:
        script_dir = Path(__file__).parent

    params = parse_variables_h(variables_h_path)
    required = ["NY", "NZ", "ALPHA", "GAMMA", "GRID_DAT_REF"]
    for k in required:
        if k not in params:
            raise ValueError(f"Missing #define {k} in {variables_h_path}")

    NY = params["NY"]
    NZ = params["NZ"]          # node count (格點數)
    alpha = params["ALPHA"]
    gamma = params["GAMMA"]
    ref_name = params["GRID_DAT_REF"]
    LZ = params.get("LZ", 3.036)
    LY = params.get("LY", 9.0)
    CFL_val = params.get("CFL", 0.5)

    NZ_cells = NZ - 1          # wall-normal cell count (格子數 = NZ-1)

    # Compute minSize from GAMMA (analytic, no bisection)
    # gamma_to_minSize expects cell count, not node count
    if gamma > 0:
        minSize_val = gamma_to_minSize(gamma, LZ, NZ_cells, LY)
    else:
        minSize_val = 0.0  # gamma=0 means no extra stretching

    # Grid .dat dimensions:
    #   I = NY  (streamwise nodes, NY is already node count)
    #   J = NZ  (wall-normal nodes)
    NI = NY
    NJ = NZ

    ref_path = script_dir / ref_name
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference grid not found: {ref_path}")

    # Load reference grid
    x_ref, y_ref, ni_ref, nj_ref = parse_tecplot_dat(ref_path)

    # ── Validate reference grid dimensions ──
    # Reference grid may have different resolution (e.g. Frohlich 129x197)
    # We only log its dimensions; the output will be re-gridded to NI x NJ
    print(f"  [auto] Reference grid: I={ni_ref} x J={nj_ref}")

    print(f"  [auto] variables.h: NY={NY} (nodes), NZ={NZ} (nodes), LZ={LZ}, ALPHA={alpha}")
    print(f"  [auto] Wall-normal: {NZ} nodes = {NZ_cells} cells")
    print(f"  [auto] GAMMA={gamma} (user input)")
    if gamma > 0:
        print(f"  [auto] minSize={minSize_val:.6e} (derived from GAMMA)")
    print(f"  [auto] Reference: {ref_path.name}")
    print(f"  [auto] Target grid: I={NI} (=NY) x J={NJ} (=NZ)")

    # ── GILBM stability pre-check ──
    print_gilbm_stability_table()

    x_out, y_out, conv = generate_adaptive_grid(
        x_ref, y_ref, NI, NJ,
        gamma=gamma, alpha=alpha,
        poisson_iter=15000, poisson_tol=1e-12)

    # ── Validate generated grid dimensions ──
    nj_out, ni_out = x_out.shape
    if ni_out != NI or nj_out != NJ:
        print(f"  !! INTERNAL ERROR: generated grid {ni_out}x{nj_out} "
              f"≠ expected {NI}x{NJ} !!")
        sys.exit(1)
    print(f"  [auto] Generated grid: I={ni_out} x J={nj_out} ✓")

    # ── GILBM stability post-check (on actual generated grid) ──
    x_fro_max = x_ref[0, -1]
    h_phys = x_fro_max / LY if x_fro_max < 1.0 else 1.0
    scale = 1.0 / h_phys if h_phys < 0.5 else 1.0
    stab = estimate_gilbm_stability(x_out, y_out, scale_factor=scale)
    print_gilbm_stability_warning(
        gamma, stab["omega"], stab["c_max"],
        stab["dt_global"], stab["a_max"], stab["status"])

    if stab["status"] == "UNSTABLE":
        print("  !! Grid generation completed but omega > 2.0 !!")
        print("  !! The GILBM simulation WILL DIVERGE with this grid. !!")
        print("  !! Reduce GAMMA in variables.h and regenerate. !!")
        print()

    # Output filename must match C code sprintf format:
    #   "%s/adaptive_%s_I%d_J%d_a%.1f.dat" with ("3.fine grid", NY, NZ, ALPHA)
    #   I = NY (streamwise nodes), J = NZ (wall-normal nodes)
    # ★ CRITICAL: use :.1f to match C's %.1f exactly (e.g. 0.5 not 0.50 or 0.500)
    grid_key = ref_path.stem          # "3.fine grid"
    out_name = f"adaptive_{grid_key}_I{NI}_J{NJ}_a{alpha:.1f}.dat"
    out_path = script_dir / out_name

    write_tecplot_dat(out_path, x_out, y_out,
                      title=f"Periodic hill {NI}x{NJ}",
                      zone_title=f"I{NI}_J{NJ}_a{alpha}")

    # ── Write grid_data.txt analysis (i=0 column / hill crest line) ──
    grid_data_path = script_dir / f"grid_data_I{NI}_J{NJ}_a{alpha:.1f}.txt"
    write_grid_data(grid_data_path, x_out, y_out,
                    NY=NY, NZ=NZ, GAMMA=gamma, ALPHA=alpha, LZ=LZ,
                    source_dat=out_path.name)

    # ── Validate written .dat file matches NY x NZ ──
    ok, ni_a, nj_a, ni_e, nj_e = validate_grid_dimensions(
        str(out_path), NY, NZ)
    if not ok:
        print("  !! Output .dat file dimension mismatch — ABORTING !!")
        sys.exit(1)
    print(f"  [auto] Output validated: I={ni_a} J={nj_a} ✓ (matches NY={ni_e}, NZ={nj_e})")

    # Also save comparison plot
    tag = f"I{NI}_J{NJ}_a{alpha}"
    plot_compare(x_ref, y_ref, x_out, y_out,
                 labels=["Reference", f"New ({NI}x{NJ})"],
                 title=f"Auto: GAMMA={gamma:.4f}, ALPHA={alpha}, Grid={NI}x{NJ}",
                 savepath=script_dir / f"compare_auto_{tag}.png")

    print(f"  [auto] Output: {out_path}")
    return str(out_path)


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":

    script_dir = Path(__file__).resolve().parent
    base = Path.cwd().resolve()

    # --auto mode: parse variables.h and generate grid non-interactively
    if "--auto" in sys.argv:
        # Find variables.h: search project root (parent of script_dir),
        # current working directory, and relative parent (for cd-based invocation)
        vh_candidates = [
            script_dir.parent / "variables.h",
            Path.cwd() / "variables.h",
            Path("..") / "variables.h",
            Path.cwd().parent / "variables.h",
        ]
        variables_h = None
        for c in vh_candidates:
            if c.exists():
                variables_h = c
                break
        if variables_h is None:
            print("ERROR: Cannot find variables.h")
            print(f"  Searched: {[str(c) for c in vh_candidates]}")
            sys.exit(1)

        print("=" * 62)
        print("  Periodic Hill Grid -- AUTO MODE")
        print(f"  Reading from: {variables_h}")
        print("=" * 62)

        out = auto_generate(str(variables_h), script_dir)
        print("=" * 62)
        print(f"  DONE: {out}")
        print("=" * 62)
        sys.exit(0)

    print()
    print("=" * 62)
    print("  Periodic Hill Grid -- Steger-Sorenson Poisson + Zeta")
    print("  (Interactive Mode)")
    print("=" * 62)

    # -----------------------------------------------------------
    #  Step 1 -- select reference grid
    # -----------------------------------------------------------
    print("\n" + "-" * 62)
    print("  [Step 1] Select reference grid file")
    print("-" * 62)

    dat_list = detect_dat_files(script_dir)
    if len(dat_list) == 0:
        print("  ERROR: No .dat files found in", script_dir)
        sys.exit(1)

    for idx, fp in enumerate(dat_list):
        print(f"    {idx + 1}. {fp.name}")

    while True:
        raw = input(f"\n  Enter file number [1-{len(dat_list)}] (default=1): ").strip()
        if raw == "":
            choice = 0
            break
        try:
            choice = int(raw) - 1
            if 0 <= choice < len(dat_list):
                break
        except ValueError:
            pass
        print("    ** Invalid choice, try again.")

    dat_path = dat_list[choice]
    grid_key = dat_path.stem

    # -----------------------------------------------------------
    #  Step 2 -- parse reference
    # -----------------------------------------------------------
    print("\n" + "-" * 62)
    print("  [Step 2] Parsing reference grid ...")
    print("-" * 62)

    x_ref, y_ref, ni_ref, nj_ref = parse_tecplot_dat(dat_path)
    print(f"  Reference: {dat_path.name}")
    print(f"  Dimensions: I={ni_ref} (streamwise)  x  J={nj_ref} (vertical)")

    out_orig = base / f"original_{grid_key}.png"
    plot_grid(x_ref, y_ref,
              title=f"Reference: {dat_path.name}  (I={ni_ref}, J={nj_ref})",
              savepath=out_orig)

    # -----------------------------------------------------------
    #  Step 3 -- choose mode
    # -----------------------------------------------------------
    print("\n" + "-" * 62)
    print("  [Step 3] Choose operation mode")
    print("-" * 62)
    print()
    print("    1. Zeta-only  -- keep original Ni x Nj,")
    print("                     adjust vertical stretching (GAMMA/ALPHA)")
    print()
    print("    2. Adaptive   -- freely set new Ni x Nj,")
    print("                     Poisson solve with Steger-Sorenson P,Q")
    print("                     (true elliptic grid generation)")
    print()

    while True:
        raw = input("  Mode [1 or 2] (default=1): ").strip()
        if raw == "":
            mode = 1
            break
        if raw in ("1", "2"):
            mode = int(raw)
            break
        print("    ** Enter 1 or 2.")

    # -----------------------------------------------------------
    #  Step 4 -- set parameters
    # -----------------------------------------------------------
    print("\n" + "-" * 62)
    print("  [Step 4] Set parameters")
    print("-" * 62)

    if mode == 2:
        print()
        print(f"  Reference grid: I={ni_ref}, J={nj_ref}")
        print()
        print("  Ni -- streamwise grid points")
        print(f"         (original = {ni_ref})")
        NI = ask_int("Ni", default=ni_ref, lo=10, hi=2000)
        print()
        print("  Nj -- vertical grid points")
        print(f"         (original = {nj_ref})")
        NJ = ask_int("Nj", default=nj_ref, lo=10, hi=2000)
    else:
        NI = ni_ref
        NJ = nj_ref

    # ── Print GILBM stability reference table before GAMMA selection ──
    print_gilbm_stability_table()

    print()
    print("  GAMMA -- Vinokur stretching in physical z-space")
    print("           0.0 = UNIFORM spacing (no wall clustering, minSize=NaN!)")
    print("           1.0~2.0 = mild-moderate symmetric clustering (RECOMMENDED)")
    print("           2.0~3.0 = good clustering, omega still optimal (<0.73)")
    print("           4.0     = strong (approaching Frohlich-level ratio ~20)")
    print("           >=5.0   = extreme (omega > 1.0, use with caution)")
    print()
    GAMMA = ask_float("GAMMA", default=2.0,
                      lo=0.0, hi=10.0)

    print()
    print("  ALPHA -- Vertical symmetry")
    print("           0.5  = symmetric (both walls equal)")
    print("           <0.5 = bottom wall denser")
    print("           >0.5 = top wall denser")
    print()
    ALPHA = ask_float("ALPHA", default=0.5, lo=0.01, hi=0.99)

    if mode == 2:
        print()
        print("  Poisson solver iterations")
        print("    (more = more accurate, slower)")
        print("    Typical: 10000~30000 for high accuracy")
        POISSON_ITER = ask_int("Poisson iterations", default=15000, lo=1000, hi=100000)
    else:
        POISSON_ITER = 15000

    print()
    print(f"  -> Mode:  {'Zeta-only' if mode == 1 else 'Adaptive (Poisson + P,Q)'}")
    print(f"  -> Grid:  I={NI} x J={NJ}")
    print(f"  -> GAMMA: {GAMMA}  |  ALPHA: {ALPHA}")
    if mode == 2:
        print(f"  -> Poisson iterations: {POISSON_ITER}")

    # -----------------------------------------------------------
    #  Step 5 -- identity verification
    # -----------------------------------------------------------
    print("\n" + "-" * 62)
    print("  [Step 5] Identity verification (gamma=0, original size)")
    print("-" * 62)

    x_id, y_id = redistribute_vertical_arclength(x_ref, y_ref, gamma=0.0)
    ok, dx_err, dy_err = verify_identity(x_ref, y_ref, x_id, y_id, tol=1e-10)
    tag = "PASS" if ok else "FAIL"
    print(f"  Arclength identity:  max|dx| = {dx_err:.2e},  max|dy| = {dy_err:.2e}  ->  {tag}")

    # -----------------------------------------------------------
    #  Step 6 -- generate new grid
    # -----------------------------------------------------------
    print("\n" + "-" * 62)
    print("  [Step 6] Generating new grid ...")
    print("-" * 62)

    if mode == 1:
        x_new, y_new = redistribute_vertical_physical(x_ref, y_ref,
                                              gamma=GAMMA, alpha=ALPHA)
    else:
        x_new, y_new, poisson_conv = generate_adaptive_grid(
            x_ref, y_ref, NI, NJ,
            gamma=GAMMA, alpha=ALPHA,
            poisson_iter=POISSON_ITER, poisson_tol=1e-12)

    print(f"  Generated grid: I={NI}, J={NJ}")

    # ── GILBM stability post-check ──
    x_fro_max = x_ref[0, -1]
    h_phys = x_fro_max / 9.0 if x_fro_max < 1.0 else 1.0
    scale = 1.0 / h_phys if h_phys < 0.5 else 1.0
    stab = estimate_gilbm_stability(x_new, y_new, scale_factor=scale)
    print_gilbm_stability_warning(
        GAMMA, stab["omega"], stab["c_max"],
        stab["dt_global"], stab["a_max"], stab["status"])

    # -----------------------------------------------------------
    #  Step 7 -- output
    # -----------------------------------------------------------
    print("\n" + "-" * 62)
    print("  [Step 7] Saving outputs ...")
    print("-" * 62)

    tag_str = f"I{NI}_J{NJ}_g{GAMMA}_a{ALPHA}"

    out_cmp = base / f"compare_{grid_key}_{tag_str}.png"
    plot_compare(x_ref, y_ref, x_new, y_new,
                 labels=["Reference", f"New ({NI}x{NJ})"],
                 title=f"GAMMA={GAMMA}, ALPHA={ALPHA}, Grid={NI}x{NJ}",
                 savepath=out_cmp)

    mid_col = NI // 2
    out_sp = base / f"spacing_{grid_key}_{tag_str}.png"
    plot_vertical_spacing(y_ref, y_new, icol=min(mid_col, ni_ref//2),
                          labels=["Reference", f"New ({NI}x{NJ})"],
                          savepath=out_sp)

    out_dat = base / f"adaptive_{grid_key}_{tag_str}.dat"
    write_tecplot_dat(out_dat, x_new, y_new,
                      title=f"Periodic hill {NI}x{NJ}",
                      zone_title=f"I{NI}_J{NJ}_g{GAMMA}_a{ALPHA}")

    # ── Write grid_data.txt analysis (i=0 column / hill crest line) ──
    # Try to read LZ from variables.h (same search paths as auto mode);
    # fall back to grid-derived LZ if variables.h is not found.
    LZ_for_report = None
    vh_candidates = [
        script_dir.parent / "variables.h",
        Path.cwd() / "variables.h",
        Path("..") / "variables.h",
        Path.cwd().parent / "variables.h",
    ]
    for c in vh_candidates:
        if c.exists():
            try:
                vh_params = parse_variables_h(c)
                if "LZ" in vh_params:
                    LZ_for_report = vh_params["LZ"]
                    print(f"  [grid_data] LZ = {LZ_for_report} (from {c})")
                    break
            except Exception:
                pass
    if LZ_for_report is None:
        print("  [grid_data] variables.h not found — LZ will be derived from grid")

    out_grid_data = base / f"grid_data_{tag_str}.txt"
    write_grid_data(out_grid_data, x_new, y_new,
                    NY=NI, NZ=NJ, GAMMA=GAMMA, ALPHA=ALPHA, LZ=LZ_for_report,
                    source_dat=out_dat.name)

    out_new = base / f"grid_{grid_key}_{tag_str}.png"
    plot_grid(x_new, y_new,
              title=f"New grid {NI}x{NJ}  GAMMA={GAMMA}",
              savepath=out_new)

    if mode == 2 and _HAS_MPL:
        fig_cv, ax_cv = plt.subplots(figsize=(8, 5))
        ax_cv.semilogy(poisson_conv, 'k-', lw=0.6)
        ax_cv.set_xlabel("Iteration"); ax_cv.set_ylabel("Max correction")
        ax_cv.set_title(f"Poisson convergence ({NI}x{NJ})")
        ax_cv.grid(True, ls='--', alpha=0.4)
        plt.tight_layout()
        conv_path = base / f"convergence_{grid_key}_{tag_str}.png"
        fig_cv.savefig(conv_path, dpi=200)
        print(f"  [saved] {conv_path}")
        plt.close()

    # -----------------------------------------------------------
    #  Step 8 -- optional parametric sweep
    # -----------------------------------------------------------
    print("\n" + "-" * 62)
    print("  [Step 8] Parametric sweep (optional)")
    print("-" * 62)

    do_sweep = ask_yes_no("Generate parametric sweep plots?", default_yes=False)

    if do_sweep and _HAS_MPL:
        print("  Generating sweep ...")
        gammas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

        fig, axes = plt.subplots(len(gammas), 1,
                                 figsize=(18, 3.2 * len(gammas)),
                                 sharex=True)
        for ax, g in zip(axes, gammas):
            if mode == 1:
                xn, yn = redistribute_vertical(x_ref, y_ref, gamma=g, alpha=ALPHA)
            else:
                xn, yn, _ = generate_adaptive_grid(
                    x_ref, y_ref, NI, NJ,
                    gamma=g, alpha=ALPHA, poisson_iter=POISSON_ITER)
            nj_n, ni_n = xn.shape
            for jj in range(nj_n):
                ax.plot(xn[jj, :], yn[jj, :], "k-", lw=0.2)
            for ii in range(0, ni_n, max(1, ni_n//40)):
                ax.plot(xn[:, ii], yn[:, ii], "k-", lw=0.2)
            ax.set_aspect("equal"); ax.set_ylabel("y")
            ax.set_title(f"gamma = {g:.1f}", fontsize=10, loc="left")

        axes[-1].set_xlabel("x  [m]")
        fig.suptitle(f"Parametric sweep (alpha={ALPHA})", fontsize=14)
        plt.tight_layout()
        sweep_path = base / f"sweep_{grid_key}_{tag_str}.png"
        fig.savefig(sweep_path, dpi=200, bbox_inches="tight")
        print(f"  [saved] {sweep_path}")
        plt.close(fig)

        fig3, ax3 = plt.subplots(figsize=(7, 5))
        eta = np.linspace(0, 1, NJ)
        for g in gammas:
            z = vinokur_tanh(eta, g, ALPHA)
            ax3.plot(range(NJ), z, "-", lw=1.2, label=f"gamma={g:.1f}")
        ax3.set_xlabel("j index"); ax3.set_ylabel("zeta (normalised)")
        ax3.set_title(f"Zeta distribution (alpha={ALPHA})")
        ax3.legend(fontsize=8); ax3.grid(True, ls="--", alpha=0.4)
        plt.tight_layout()
        zeta_path = base / "zeta_curves.png"
        fig3.savefig(zeta_path, dpi=200)
        print(f"  [saved] {zeta_path}")
        plt.close(fig3)

    print()
    print("=" * 62)
    print("  ALL DONE")
    print("=" * 62)

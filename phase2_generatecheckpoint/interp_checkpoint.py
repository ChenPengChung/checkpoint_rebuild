#!/usr/bin/env python3
"""
Generic LBM checkpoint rebuilder for fixed periodic-hill D3Q19 runs.

Fixed by design:
  - Domain: LX, LY, LZ, H_HILL
  - Ghost layer width: 3
  - Lattice model: D3Q19

Modes:
  Project mode:    auto-reads NEW NX/NY/NZ/jp/GAMMA/ALPHA from variables.h
  Standalone mode: prompts for missing grid parameters when variables.h is not found

Pipeline:
  1. Build OLD/NEW grid configs (auto-detect from metadata + variables.h, or interactive)
  2. Cross-validate grid .dat headers against NX/NY/NZ
  3. Cross-check the NEW interpolation grid against the exact solver runtime
     grid derived from variables.h GRID_DAT_DIR/GRID_DAT_REF (or
     --solver-grid-dat). Coordinate mismatch is fatal by default.
  4. Read old checkpoint, compute macros (rho, ux, uy, uz)
  5. Interpolate macros old -> new, then apply conservation corrections:
     5a. Interpolate rho, ux, uy, uz according to --interp-mode:
           phys (default): physical-space remap; correct when GAMMA changes.
             --interp-order 6 (default): 7-point Lagrange tensor product O(h^6).
                Near-wall stencils use cubic ghost extrapolation (solver-matched).
             --interp-order 2:          bilinear O(h^2) (legacy).
           comp:           legacy computational (j, k, i) remap for A/B tests.
     5b. Clamp wall velocity only: u=v=w=0 at k=3 and k=NZ6-4 (no-slip).
         Preserve wall rho so restart pressure stays consistent with the
         interpolated source field and with the runtime solver policy.
     5c. Global density correction: uniform additive offset on the full
         physical domain, matching the runtime mass-correction kernel.
     5d. Bulk velocity correction: scale interior streamwise velocity so
         Ub(NEW) = Ub(OLD); wall rows excluded from scaling (remain u=0).
     5e. Velocity projection (--project-velocity poisson, default):
         poisson:   approximate Helmholtz-Hodge correction.
         dg-exact:  direct solve of the exact CD2 D*G scalar projection.
         div-exact: direct minimum-norm velocity correction that zeroes the
                    final CD2 divergence diagnostic to roundoff.  Wall
                    velocity is constrained inside the projection, and no
                    later step modifies the interior velocity before f_eq.
  6. Reconstruct f_q for q = 0..18 from the corrected macroscopic quantities
     (rho, u, v, w) produced by steps 5a-5e. Mode --fneq-mode selects how the
     non-equilibrium component is handled:
       zero (default):           stability A/B test mode; write pure
                                 equilibrium f_q = f_eq, i.e. f_neq = 0.
       chapman-enskog:           f_eq and f_neq both built from corrected
                                 macros. f_neq reconstructed from NEW-grid
                                 velocity gradients via Chapman-Enskog.
                                 Wall rows use the solver-matched one-sided FD
                                 stencil for the wall CE formula.
       interp (legacy):          f_eq from corrected macros +
                                 scale * interp(f_neq_old) in computational
                                 space (loses gradient info across GAMMA changes).
  7. Preserve controller state (Force_integral, error_prev, ctrl_initialized,
     gehrke_activated) ONLY from origin metadata to avoid F* step on restart.
     FTT and accu_count are NOT preserved — they are reset to 0 because:
       - regrid is a fresh start on the new mesh (FTT=0 aligns new stats window);
       - accu_count > 0 would trigger fileIO.h:748 to load 36 stats binaries
         (sum_u_*.bin, ...) that this pipeline does NOT regenerate.
     Origin FTT / accu_count are written into metadata as `interp_origin_*`
     fields for audit only.
  8. Split into new ranks, write per-rank binary files + metadata.dat

Output written atomically:
  <output_root>/step_%08d.WRITING/ -> <output_root>/step_%08d/
  restart/grid_provenance records the session-level grid identity.

Usage:
  # Project auto mode:
  #   origin checkpoint: phase2_generatecheckpoint/step_*_origin* or oldcheckpoint_*
  #   OLD grid:          phase1_generategrid/oldgrid_*.dat
  #   NEW grid:          phase1_generategrid/newgrid_*.dat
  python3 phase2_generatecheckpoint/interp_checkpoint.py --auto --step 1

  # CLI override (skip prompts):
  python3 phase2_generatecheckpoint/interp_checkpoint.py --old-dir ./old_ckpt \\
      --old-gamma 2.0 --old-grid-dat old_grid.dat \\
      --new-nx 257 --new-ny 513 --new-nz 257 --new-jp 16 \\
      --new-gamma 3.0 --new-alpha 0.5 --new-grid-dat new_grid.dat \\
      --output-root restart/checkpoint --step 1 \\
      --interp-mode phys --fneq-mode zero

Expected folder structure:
  workspace/
  +-- variables.h                     (optional, project mode)
  +-- phase2_generatecheckpoint/interp_checkpoint.py
  +-- phase1_generategrid/
  |   +-- oldgrid_*_I{NY}_J{NZ}_g{G}_a{A}.dat    (OLD uniform gamma grid)
  |   +-- newgrid_*_I{NY}_J{NZ}_a{A}.dat         (NEW variable gamma grid)
  +-- phase2_generatecheckpoint/step_*_origin*/ or oldcheckpoint_*/  (source checkpoint)
      +-- metadata.dat
      +-- f00_0.bin ... f18_{jp-1}.bin
      +-- rho_0.bin ... rho_{jp-1}.bin
"""

import os
import sys
import math
import time
import argparse
import subprocess
import numpy as np

PROJECT_ROOT_FOR_IMPORTS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT_FOR_IMPORTS not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_FOR_IMPORTS)

from grid_params import read_grid_params_sha256

# ---------------------------------------------------------------
# Domain constants (must match variables.h)
# ---------------------------------------------------------------
LX = 4.5
LY = 9.0
LZ = 3.036
H_HILL = 1.0
BFR = 3

# ---------------------------------------------------------------
# Grid configurations
# ---------------------------------------------------------------
class GridConfig:
    def __init__(self, nx, ny, nz, jp, gamma, alpha, grid_dat):
        if (ny - 1) % jp != 0:
            raise ValueError('(NY-1)={} is not divisible by jp={}'.format(ny - 1, jp))
        self.NX = nx
        self.NY = ny
        self.NZ = nz
        self.JP = jp
        self.GAMMA = gamma
        self.ALPHA = alpha
        self.GRID_DAT = grid_dat
        self.NX6 = nx + 6
        self.NY6 = ny + 6
        self.NZ6 = nz + 6
        self.NYD6 = (ny - 1) // jp + 7
        self.CHUNK = self.NYD6 - 7  # = (NY-1)/jp


OLD = None  # set dynamically in main()
NEW = None

# ---------------------------------------------------------------
# Configuration helpers (dual-mode: project / standalone)
# ---------------------------------------------------------------
def parse_variables_h(path):
    """Parse selected numeric #define values from variables.h."""
    targets = {'NX', 'NY', 'NZ', 'jp', 'GAMMA', 'ALPHA', 'CFL',
               'LX', 'LY', 'LZ', 'H_HILL'}
    int_keys = {'NX', 'NY', 'NZ', 'jp'}
    defines = {}
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            stripped = line.strip()
            if not stripped.startswith('#define'):
                continue
            parts = stripped.split(None, 2)
            if len(parts) < 3:
                continue
            key = parts[1]
            if key not in targets:
                continue
            val_str = parts[2].split('//')[0].strip().strip('()')
            try:
                defines[key] = int(val_str) if key in int_keys else float(val_str)
            except ValueError:
                pass
    return defines


def parse_string_defines(path, keys=('GRID_DAT_DIR', 'GRID_DAT_REF')):
    """Parse #define KEY "value" string defines from variables.h."""
    import re
    result = {}
    with open(path, encoding='utf-8', errors='replace') as f:
        text = f.read()
    for key in keys:
        m = re.search(rf'#define\s+{key}\s+"([^"]+)"', text)
        if m:
            result[key] = m.group(1)
    return result


def parse_grid_dat_header(path):
    """Extract I=, J= from Tecplot .dat file header for cross-validation."""
    dims = {}
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            for token in line.replace(',', ' ').split():
                if token.startswith('I='):
                    try:
                        dims['I'] = int(token[2:])
                    except ValueError:
                        pass
                elif token.startswith('J='):
                    try:
                        dims['J'] = int(token[2:])
                    except ValueError:
                        pass
            if 'I' in dims and 'J' in dims:
                break
    return dims


def read_grid_dat_coords(path, expected_i=None, expected_j=None):
    """Read Tecplot POINT coordinates as raw (x, y) floats for identity checks."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    dims = parse_grid_dat_header(path)
    if expected_i is not None and dims.get('I') != expected_i:
        raise ValueError('{} I={} != expected {}'.format(path, dims.get('I'), expected_i))
    if expected_j is not None and dims.get('J') != expected_j:
        raise ValueError('{} J={} != expected {}'.format(path, dims.get('J'), expected_j))

    coords = []
    in_data = False
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if not in_data:
                if line.strip().startswith('DT='):
                    in_data = True
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    coords.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
    coords = np.asarray(coords, dtype=np.float64)
    if expected_i is not None and expected_j is not None:
        expected = expected_i * expected_j
        if coords.shape[0] != expected:
            raise ValueError('{} has {} coordinate rows, expected {}'.format(
                path, coords.shape[0], expected))
    return coords


def derive_solver_grid_dat(variables_h, cfg):
    """Return the exact external grid path main.cu will read, or None."""
    if not variables_h or not os.path.isfile(variables_h):
        return None
    str_defs = parse_string_defines(variables_h)
    grid_dir = str_defs.get('GRID_DAT_DIR')
    grid_ref = str_defs.get('GRID_DAT_REF')
    if not grid_dir or not grid_ref:
        return None
    ref_stem = os.path.splitext(grid_ref)[0]
    fname = 'adaptive_{}_I{}_J{}_g{:.2f}_a{:.1f}.dat'.format(
        ref_stem, cfg.NY, cfg.NZ, float(cfg.GAMMA), float(cfg.ALPHA))
    base = os.path.dirname(os.path.abspath(variables_h))
    if os.path.isabs(grid_dir):
        return os.path.abspath(os.path.join(grid_dir, fname))
    return os.path.abspath(os.path.join(base, grid_dir, fname))


def compare_grid_dat_coords(path_a, path_b, cfg):
    """Compare two Tecplot grid files in raw solver input coordinates."""
    a = read_grid_dat_coords(path_a, expected_i=cfg.NY, expected_j=cfg.NZ)
    b = read_grid_dat_coords(path_b, expected_i=cfg.NY, expected_j=cfg.NZ)
    if a.shape != b.shape:
        raise ValueError('coordinate row count differs: {} vs {}'.format(a.shape, b.shape))
    diff = np.abs(a - b)
    idx_flat = int(np.argmax(diff)) if diff.size else 0
    row, col = np.unravel_index(idx_flat, diff.shape) if diff.size else (0, 0)
    return {
        'count': int(a.shape[0]),
        'max_abs_x': float(diff[:, 0].max()) if diff.size else 0.0,
        'max_abs_y': float(diff[:, 1].max()) if diff.size else 0.0,
        'max_abs': float(diff.max()) if diff.size else 0.0,
        'max_row': int(row),
        'max_component': 'x' if col == 0 else 'y',
    }


def validate_solver_grid_match(new_grid_dat, solver_grid_dat, cfg, tol=0.0, fatal=True):
    """Fail fast when the interpolation grid differs from the solver runtime grid."""
    if not solver_grid_dat:
        print('  WARNING: solver grid path not available; cannot cross-check NEW grid')
        return None

    solver_grid_dat = os.path.abspath(os.path.normpath(solver_grid_dat))
    new_grid_dat = os.path.abspath(os.path.normpath(new_grid_dat))

    try:
        diff = compare_grid_dat_coords(new_grid_dat, solver_grid_dat, cfg)
        new_fp = read_grid_params_sha256(new_grid_dat)
        solver_fp = read_grid_params_sha256(solver_grid_dat)
        fp_ok = (new_fp == solver_fp) if (new_fp and solver_fp) else True
        ok = (diff['max_abs'] <= tol) and fp_ok
    except Exception as exc:
        msg = ('{}: cannot compare NEW grid against solver grid:\n'
               '        NEW grid:    {}\n'
               '        solver grid: {}\n'
               '        reason: {}').format(
                   'FATAL' if fatal else 'WARNING', new_grid_dat, solver_grid_dat, exc)
        if fatal:
            sys.exit(msg)
        print('  ' + msg)
        return {
            'path': solver_grid_dat,
            'ok': False,
            'error': str(exc),
        }

    status = {
        'path': solver_grid_dat,
        'ok': bool(ok),
        'new_grid_params_sha256': new_fp or '',
        'solver_grid_params_sha256': solver_fp or '',
        **diff,
    }
    if ok:
        print('  OK: NEW grid coordinates match solver runtime grid')
        print('      NEW grid:    {}'.format(new_grid_dat))
        print('      solver grid: {}'.format(solver_grid_dat))
        print('      compared {} points, max_abs_diff={:.3e}, tol={:.3e}'.format(
            diff['count'], diff['max_abs'], tol))
        if new_fp and solver_fp:
            print('      grid parameter fingerprint: {}'.format(new_fp))
        elif not (new_fp or solver_fp):
            print('      grid parameter fingerprint: not present (legacy .dat headers)')
        else:
            print('      WARNING: grid parameter fingerprint missing from one .dat header')
        return status

    if new_fp and solver_fp and new_fp != solver_fp:
        fp_msg = (
            '        parameter hash differs (NEW={} solver={})\n'
            '        This usually means a grid-generation setting such as '
            'Poisson iteration count, tolerance, interpolation backend, GAMMA, '
            'or ALPHA differs between paths.\n'
        ).format(new_fp, solver_fp)
    else:
        fp_msg = ''
    msg = (
        '{}: NEW grid does not match solver runtime grid\n'
        '        NEW grid:    {}\n'
        '        solver grid: {}\n'
        '        max_abs_diff={:.6e} (x={:.6e}, y={:.6e}), row={}, component={}, tol={:.3e}\n'
        '{}'
        '        Regenerate/sync phase1_generategrid/newgrid*.dat from the exact grid used by main.cu, '
        'or pass --solver-grid-dat to the correct runtime grid.'
    ).format(
        'FATAL' if fatal else 'WARNING', new_grid_dat, solver_grid_dat,
        diff['max_abs'], diff['max_abs_x'], diff['max_abs_y'],
        diff['max_row'], diff['max_component'], tol, fp_msg)
    if fatal:
        sys.exit(msg)
    print('  ' + msg)
    return status


def maybe_generate_solver_grid(solver_grid_dat, variables_h, enabled=True):
    """Create the solver runtime grid before comparing, using the same entry as main.cu."""
    if not solver_grid_dat:
        return solver_grid_dat, False
    solver_grid_dat = os.path.abspath(os.path.normpath(solver_grid_dat))
    if os.path.isfile(solver_grid_dat):
        return solver_grid_dat, False
    if not enabled:
        return solver_grid_dat, False
    if not variables_h or not os.path.isfile(variables_h):
        return solver_grid_dat, False

    str_defs = parse_string_defines(variables_h)
    grid_dir = str_defs.get('GRID_DAT_DIR')
    if not grid_dir:
        return solver_grid_dat, False

    root = os.path.dirname(os.path.abspath(variables_h))
    grid_dir_abs = grid_dir if os.path.isabs(grid_dir) else os.path.join(root, grid_dir)
    tool = os.path.join(grid_dir_abs, 'grid_zeta_tool.py')
    if not os.path.isfile(tool):
        return solver_grid_dat, False

    cmd = [sys.executable, tool, '--auto']
    print('  Solver runtime grid not found; generating it before comparison')
    print('      target: {}'.format(solver_grid_dat))
    print('      command: {}'.format(' '.join(cmd)))
    ret = subprocess.run(cmd, cwd=root)
    if ret.returncode != 0:
        sys.exit('FATAL: solver grid generation failed with exit code {}'.format(
            ret.returncode))
    if not os.path.isfile(solver_grid_dat):
        sys.exit('FATAL: solver grid generation completed but target is still missing: {}'.format(
            solver_grid_dat))
    return solver_grid_dat, True


def resolve_existing_file(path, label, base_dirs=()):
    """Resolve a user-supplied file path against cwd and optional base dirs."""
    tried = []
    if os.path.isabs(path):
        tried.append(path)
    else:
        tried.append(path)
        for base in base_dirs:
            if base:
                tried.append(os.path.join(base, path))
    seen = set()
    for candidate in tried:
        abs_candidate = os.path.abspath(candidate)
        if abs_candidate in seen:
            continue
        seen.add(abs_candidate)
        if os.path.isfile(abs_candidate):
            return abs_candidate
    sys.exit('FATAL: {} not found: {} (tried: {})'.format(
        label, path, ', '.join(sorted(seen))))


def infer_old_grid_params(path):
    """Infer old uniform-grid gamma/alpha from *_g{gamma}_a{alpha}.dat."""
    import re
    m = re.search(r'_g([0-9]+(?:\.[0-9]+)?)_a([0-9]+(?:\.[0-9]+)?)\.dat$',
                  os.path.basename(path))
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def infer_new_grid_alpha(path):
    """Infer variable-grid alpha from *_a{alpha}.dat when present."""
    import re
    m = re.search(r'_a([0-9]+(?:\.[0-9]+)?)\.dat$', os.path.basename(path))
    if not m:
        return None
    return float(m.group(1))


def project_root():
    """Return repository root inferred from this phase2 script location."""
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def _unique_paths(paths):
    """Preserve path order while removing duplicates after abs-normalization."""
    result = []
    seen = set()
    for p in paths:
        if not p:
            continue
        abs_p = os.path.abspath(os.path.normpath(p))
        if abs_p in seen:
            continue
        seen.add(abs_p)
        result.append(abs_p)
    return result


def find_variables_h():
    """Search for variables.h in standard project locations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        'variables.h',
        '../variables.h',
        os.path.join(script_dir, 'variables.h'),
        os.path.join(script_dir, '..', 'variables.h'),
    ]
    seen = set()
    for c in candidates:
        p = os.path.abspath(c)
        if p not in seen and os.path.isfile(p):
            return p
        seen.add(p)
    return None


def resolve_variables_h_arg(path):
    """Resolve and validate a variables.h path supplied by CLI or auto-detection."""
    if path is None:
        return None
    abs_path = os.path.abspath(os.path.normpath(path))
    if not os.path.isfile(abs_path):
        sys.exit('FATAL: variables.h not found: {}'.format(abs_path))
    return abs_path


def auto_detect_from_metadata(meta_path):
    """Extract NX/NY/NZ/JP from checkpoint metadata.dat grid_dims field."""
    meta = parse_metadata(meta_path)
    jp = int(meta.get('mpi_rank_count', 0))
    grid_dims = meta.get('grid_dims', '')
    parts = grid_dims.split(',')
    if len(parts) != 3 or jp == 0:
        return None
    nx6, nyd6, nz6 = int(parts[0]), int(parts[1]), int(parts[2])
    nx = nx6 - 6
    nz = nz6 - 6
    chunk = nyd6 - 7  # = (NY-1)/jp
    ny = chunk * jp + 1
    return {'NX': nx, 'NY': ny, 'NZ': nz, 'jp': jp}


def _grid_dat_search_dirs(grid_dat_dir=None):
    """
    GRID PIPELINE REGULATION:
      Phase 2 只認 phase1_generategrid/ 與 script_dir 自身。
      不再回退 J_Frohlich/ (main pipeline 的目錄, 不在 phase2 路徑上).
      grid_dat_dir 仍接受顯式 CLI 傳入 (explicit override)。
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, '..'))
    dirs = []
    if grid_dat_dir:
        dirs.append(grid_dat_dir)
        dirs.append(os.path.join(script_dir, '..', grid_dat_dir))
    dirs.extend([
        os.path.join(project_dir, 'phase1_generategrid'),
        os.path.join(script_dir, '..', 'phase1_generategrid'),
        '../phase1_generategrid',
    ])
    seen = set()
    result = []
    for d in dirs:
        p = os.path.abspath(d)
        if p not in seen:
            seen.add(p)
            result.append(d)
    return result


def try_find_grid_dat(ny, nz, gamma, alpha, search_dirs=None):
    """Try to find grid .dat file by naming convention.

    Searches for both formats:
      - I{NY}_J{NZ}_g{G}_a{A}.dat  (Mode 2, uniform gamma)
      - I{NY}_J{NZ}_a{A}.dat       (Mode 3, variable gamma)
    """
    if search_dirs is None:
        search_dirs = _grid_dat_search_dirs()
    candidates = set()
    for fmt in (str, lambda x: '{:g}'.format(x)):
        g_str = fmt(gamma)
        a_str = fmt(alpha)
        candidates.add('I{}_J{}_g{}_a{}'.format(ny, nz, g_str, a_str))
        candidates.add('I{}_J{}_a{}'.format(ny, nz, a_str))
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if not fname.endswith('.dat'):
                continue
            for pat in candidates:
                if pat in fname:
                    return os.path.join(d, fname)
    return None


def try_find_grid_dat_by_dims(ny, nz, search_dirs=None):
    """Find grid .dat by I{NY}_J{NZ} pattern; extract gamma/alpha from filename.

    Handles both formats:
      _g{G}_a{A}.dat  → gamma=G, alpha=A  (Mode 2)
      _a{A}.dat        → gamma=None, alpha=A (Mode 3)
    """
    import re
    if search_dirs is None:
        search_dirs = _grid_dat_search_dirs()
    pattern = 'I{}_J{}'.format(ny, nz)
    ga_re = re.compile(r'_g([\d.]+)_a([\d.]+)\.dat$')
    a_re = re.compile(r'_a([\d.]+)\.dat$')
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if not fname.endswith('.dat'):
                continue
            if pattern in fname:
                path = os.path.join(d, fname)
                m = ga_re.search(fname)
                if m:
                    return path, float(m.group(1)), float(m.group(2))
                m = a_re.search(fname)
                if m:
                    return path, None, float(m.group(1))
    return None, None, None


_AUTO_MODE = False

def ask_value(prompt_text, cast_fn=str, default=None):
    """Interactive prompt with optional default value. FATAL in --auto mode."""
    if _AUTO_MODE:
        if default is not None:
            return default
        sys.exit('FATAL: --auto mode requires all parameters but missing: {}'.format(prompt_text))
    if default is not None:
        full = '{} [{}]: '.format(prompt_text, default)
    else:
        full = '{}: '.format(prompt_text)
    while True:
        val = input(full).strip()
        if not val and default is not None:
            return default
        if not val:
            print('  (必須輸入值 / value required)')
            continue
        try:
            return cast_fn(val)
        except (ValueError, TypeError):
            print('  (格式錯誤, 請重新輸入 / invalid format)')


def origin_search_dirs(primary_dir=None):
    """Default direct parent folders that may contain step_*_origin* checkpoints."""
    root = project_root()
    return _unique_paths([
        primary_dir,
        os.path.join(root, 'phase2_generatecheckpoint'),
        os.path.join(root, 'restart'),
        os.path.join(root, 'restart', 'checkpoint'),
        'phase2_generatecheckpoint',
        'restart',
    ])


def _is_origin_dir_name(name):
    """Match origin checkpoint directory names.

    Accepted patterns:
      step_*_origin*            (canonical: step_24913001_origin_Re5600)
      oldcheckpoint_*           (manual copy: oldcheckpoint_Re5600_step_24913001)
    """
    if name.startswith('step_') and '_origin' in name:
        return True
    if name.startswith('oldcheckpoint_'):
        return True
    return False


def find_origin_checkpoint(search_dir=None):
    """Find origin checkpoint directories with valid metadata across phase layout.
    FATAL if multiple origins exist (ambiguous)."""
    candidates = []
    for parent in origin_search_dirs(search_dir):
        if not os.path.isdir(parent):
            continue
        for name in sorted(os.listdir(parent)):
            if _is_origin_dir_name(name):
                path = os.path.join(parent, name)
                if os.path.isfile(os.path.join(path, 'metadata.dat')):
                    candidates.append(os.path.abspath(path))
    if len(candidates) > 1:
        sys.exit('FATAL: multiple origin checkpoints found ({}): {}'.format(
            len(candidates), ', '.join(candidates)))
    return candidates[0] if candidates else None


def resolve_old_dir(old_dir):
    """Resolve source checkpoint directory, with a friendly fallback for local copies."""
    if old_dir is None:
        origin = find_origin_checkpoint()
        if origin:
            print('  Auto-detected origin checkpoint: {}'.format(origin))
            return origin
        sys.exit('FATAL: --old-dir not specified and no origin checkpoint found '
                 '(searched step_*_origin* and oldcheckpoint_* in phase2/restart)')

    old_dir = os.path.normpath(old_dir)
    meta_path = os.path.join(old_dir, 'metadata.dat')
    if os.path.isfile(meta_path):
        return old_dir

    candidates = []
    for parent in origin_search_dirs():
        if not os.path.isdir(parent):
            continue
        for name in sorted(os.listdir(parent)):
            path = os.path.join(parent, name)
            if name.startswith('step_') and os.path.isfile(os.path.join(path, 'metadata.dat')):
                candidates.append(path)

    if len(candidates) == 1:
        print('  OLD checkpoint default not found: {}'.format(old_dir))
        print('  Auto-selected only available checkpoint: {}'.format(candidates[0]))
        return candidates[0]

    if len(candidates) > 1 and sys.stdin.isatty():
        print('  OLD checkpoint default not found: {}'.format(old_dir))
        print('  Available restart/step_* checkpoints:')
        for i, path in enumerate(candidates, 1):
            print('    {}. {}'.format(i, path))
        idx = ask_value('  Select OLD checkpoint number', int, 1)
        if 1 <= idx <= len(candidates):
            return candidates[idx - 1]

    sys.exit('FATAL: {} not found. Use --old-dir <checkpoint_dir>.'.format(meta_path))


def resolve_output_dir(output_root, step, new_dir=None):
    """Return chain-compatible output directory for the requested step."""
    if new_dir:
        return new_dir
    return os.path.join(output_root, 'step_{:08d}'.format(step))


def cross_validate_grid_dat(cfg, label):
    """Validate grid .dat header I,J match cfg.NY (流向), cfg.NZ (法向)."""
    if not os.path.exists(cfg.GRID_DAT):
        print('  WARNING: {} grid .dat not found: {}'.format(label, cfg.GRID_DAT))
        return False
    dims = parse_grid_dat_header(cfg.GRID_DAT)
    ok = True
    if 'I' in dims and dims['I'] != cfg.NY:
        print('  FATAL: {} grid .dat I={} != NY={}'.format(label, dims['I'], cfg.NY))
        ok = False
    if 'J' in dims and dims['J'] != cfg.NZ:
        print('  FATAL: {} grid .dat J={} != NZ={}'.format(label, dims['J'], cfg.NZ))
        ok = False
    if ok and 'I' in dims:
        print('  OK: {} grid .dat validated: I={}=NY, J={}=NZ'.format(
            label, dims['I'], dims['J']))
    return ok


def build_old_config(args):
    """Build OLD GridConfig from metadata auto-detection + CLI args / interactive."""
    print('--- Configuring OLD grid ---')

    meta_path = os.path.join(args.old_dir, 'metadata.dat')
    detected = None
    if os.path.exists(meta_path):
        detected = auto_detect_from_metadata(meta_path)
        if detected:
            print('  Auto-detected from metadata: NX={} NY={} NZ={} jp={}'.format(
                detected['NX'], detected['NY'], detected['NZ'], detected['jp']))

    nx = args.old_nx if args.old_nx is not None else (detected and detected['NX']) or None
    ny = args.old_ny if args.old_ny is not None else (detected and detected['NY']) or None
    nz = args.old_nz if args.old_nz is not None else (detected and detected['NZ']) or None
    jp = args.old_jp if args.old_jp is not None else (detected and detected['jp']) or None
    gamma = args.old_gamma
    alpha = args.old_alpha
    grid_dat = args.old_grid_dat

    if any(v is None for v in (nx, ny, nz, jp)):
        print('  (metadata auto-detect incomplete — entering interactive mode)')
    if nx is None:
        nx = ask_value('  OLD NX (展向格點 / spanwise nodes)', int)
    if ny is None:
        ny = ask_value('  OLD NY (流向格點 / streamwise nodes)', int)
    if nz is None:
        nz = ask_value('  OLD NZ (法向格點 / wall-normal nodes)', int)
    if jp is None:
        jp = ask_value('  OLD jp (GPU/rank count)', int)

    if grid_dat is None and ny is not None and nz is not None:
        if gamma is not None and alpha is not None:
            grid_dat = try_find_grid_dat(ny, nz, gamma, alpha)
        if grid_dat is None:
            dat_path, dat_gamma, dat_alpha = try_find_grid_dat_by_dims(ny, nz)
            if dat_path:
                grid_dat = dat_path
                if gamma is None and dat_gamma is not None:
                    gamma = dat_gamma
                if alpha is None and dat_alpha is not None:
                    alpha = dat_alpha
                print('  Auto-found OLD grid .dat: {} (GAMMA={}, ALPHA={})'.format(
                    grid_dat, gamma, alpha))

    if gamma is None:
        gamma = ask_value('  OLD GAMMA (tanh stretching param)', float, 2.0)
    if alpha is None:
        alpha = ask_value('  OLD ALPHA (stretching center)', float, 0.5)

    if grid_dat is None:
        grid_dat = try_find_grid_dat(ny, nz, gamma, alpha)
        if grid_dat:
            print('  Auto-found OLD grid .dat: {}'.format(grid_dat))
        else:
            grid_dat = ask_value('  OLD grid .dat 路徑 (path to Tecplot grid file)', str)

    cfg = GridConfig(nx=nx, ny=ny, nz=nz, jp=jp,
                     gamma=gamma, alpha=alpha, grid_dat=grid_dat)
    if not cross_validate_grid_dat(cfg, 'OLD'):
        sys.exit('FATAL: OLD grid .dat cross-validation failed')
    print()
    return cfg


def build_new_config(args):
    """Build NEW GridConfig from variables.h (project mode) or interactive prompts."""
    print('--- Configuring NEW grid ---')

    vh_path = args.variables_h or find_variables_h()
    vh_defs = {}
    if vh_path and os.path.isfile(vh_path):
        vh_defs = parse_variables_h(vh_path)
        if vh_defs:
            print('  Project mode: reading from {}'.format(vh_path))
            for k in ('NX', 'NY', 'NZ', 'jp', 'GAMMA', 'ALPHA'):
                if k in vh_defs:
                    print('    {} = {}'.format(k, vh_defs[k]))
            global LX, LY, LZ, H_HILL
            if 'LX' in vh_defs:
                LX = vh_defs['LX']
            if 'LY' in vh_defs:
                LY = vh_defs['LY']
            if 'LZ' in vh_defs:
                LZ = vh_defs['LZ']
            if 'H_HILL' in vh_defs:
                H_HILL = vh_defs['H_HILL']
    else:
        print('  Standalone mode: variables.h not found')
        print('  Enter NEW grid parameters interactively.')

    nx = args.new_nx if args.new_nx is not None else vh_defs.get('NX')
    ny = args.new_ny if args.new_ny is not None else vh_defs.get('NY')
    nz = args.new_nz if args.new_nz is not None else vh_defs.get('NZ')
    jp = args.new_jp if args.new_jp is not None else vh_defs.get('jp')
    gamma = args.new_gamma if args.new_gamma is not None else vh_defs.get('GAMMA')
    alpha = args.new_alpha if args.new_alpha is not None else vh_defs.get('ALPHA')
    grid_dat = args.new_grid_dat

    if nx is None:
        nx = ask_value('  NEW NX (展向格點 / spanwise nodes)', int)
    if ny is None:
        ny = ask_value('  NEW NY (流向格點 / streamwise nodes)', int)
    if nz is None:
        nz = ask_value('  NEW NZ (法向格點 / wall-normal nodes)', int)
    if jp is None:
        jp = ask_value('  NEW jp (GPU/rank count)', int)
    if gamma is None:
        gamma = ask_value('  NEW GAMMA (tanh stretching param)', float)
    if alpha is None:
        alpha = ask_value('  NEW ALPHA (stretching center)', float, 0.5)

    if (ny - 1) % jp != 0:
        sys.exit('FATAL: (NY-1)={} 不能被 jp={} 整除 — 無法平均分割 MPI 子域'.format(
            ny - 1, jp))

    if grid_dat is None:
        grid_dat = try_find_grid_dat(ny, nz, gamma, alpha)
        if grid_dat:
            print('  Auto-found NEW grid .dat: {}'.format(grid_dat))
        else:
            grid_dat = ask_value('  NEW grid .dat 路徑 (path to Tecplot grid file)', str)

    cfg = GridConfig(nx=nx, ny=ny, nz=nz, jp=jp,
                     gamma=gamma, alpha=alpha, grid_dat=grid_dat)
    if not cross_validate_grid_dat(cfg, 'NEW'):
        sys.exit('FATAL: NEW grid .dat cross-validation failed')
    print()
    return cfg


# ---------------------------------------------------------------
# D3Q19 lattice (initialization.h:7-12)
# ---------------------------------------------------------------
E = np.array([
    [ 0, 0, 0],
    [ 1, 0, 0], [-1, 0, 0],
    [ 0, 1, 0], [ 0,-1, 0],
    [ 0, 0, 1], [ 0, 0,-1],
    [ 1, 1, 0], [-1, 1, 0], [ 1,-1, 0], [-1,-1, 0],
    [ 1, 0, 1], [-1, 0, 1], [ 1, 0,-1], [-1, 0,-1],
    [ 0, 1, 1], [ 0,-1, 1], [ 0, 1,-1], [ 0,-1,-1],
], dtype=np.float64)
W = np.array([1.0/3.0] + [1.0/18.0]*6 + [1.0/36.0]*12, dtype=np.float64)


# ---------------------------------------------------------------
# Metadata I/O
# ---------------------------------------------------------------
def parse_metadata(path):
    d = {}
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                k, v = line.split('=', 1)
                d[k.strip()] = v.strip()
    return d


def write_metadata(path, params):
    keys_order = [
        'checkpoint_version', 'mpi_rank_count', 'grid_dims',
        'step', 'FTT', 'accu_count', 'Force',
        'Force_integral', 'error_prev',
        'ctrl_initialized', 'gehrke_activated',
        'dt_global', 'gpu_time_ms', 'cv_count',
    ]
    with open(path, 'w', encoding='utf-8') as f:
        for k in keys_order:
            if k in params:
                f.write('{}={}\n'.format(k, params[k]))
        extra = sorted(set(params.keys()) - set(keys_order))
        if extra:
            f.write('# --- provenance ---\n')
            for k in extra:
                f.write('{}={}\n'.format(k, params[k]))


# ---------------------------------------------------------------
# Grid coordinate builder (mirrors initialization.h)
# ---------------------------------------------------------------
def build_grid_xyz(cfg):
    """Return x[NX6], y_2d[NY6, NZ6], z_2d[NY6, NZ6] in code (normalized) units.

    Mirrors initialization.h GenerateMesh_X + ReadExternalGrid_YZ:
    - X uniform: x[i] = (i - BFR) * LX / (NX - 1)
    - Read Tecplot POINT file -> rescale to H_HILL=1 -> map (file_x, file_y) -> (code_y, code_z)
    - K-direction (z) ghost: linear extrapolation
    - J-direction (y) ghost: periodic wrap with +/-LY shift on y
    """
    # X (spanwise, uniform)
    dx = LX / (cfg.NX - 1)
    x = (np.arange(cfg.NX6) - BFR) * dx

    if not os.path.exists(cfg.GRID_DAT):
        raise FileNotFoundError('Grid file not found: {}'.format(cfg.GRID_DAT))

    # Parse Tecplot POINT format: skip header until "DT=" line
    coords = []
    with open(cfg.GRID_DAT, encoding='utf-8', errors='replace') as f:
        in_data = False
        for line in f:
            if not in_data:
                if line.strip().startswith('DT='):
                    in_data = True
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    coords.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue

    coords = np.asarray(coords, dtype=np.float64)
    expected = cfg.NY * cfg.NZ
    if coords.shape[0] != expected:
        raise ValueError('Grid file {} has {} points, expected {} (NY*NZ = {}*{})'.format(
            cfg.GRID_DAT, coords.shape[0], expected, cfg.NY, cfg.NZ))

    # File is in physical units (h_phys ~ 0.028 m); rescale so H_HILL = 1
    # Reference: initialization.h:183-185
    #   x_fro_max = x_fro[NI-1]   (last point of J=0 row, max streamwise in physical)
    #   h_physical = x_fro_max / LY
    #   grid_scale = H_HILL / h_physical
    fro_x_max = coords[cfg.NY - 1, 0]
    h_physical = fro_x_max / LY
    grid_scale = H_HILL / h_physical
    coords *= grid_scale

    # Reshape to [J, I] (POINT format: I varies fastest)
    fro_x = coords[:, 0].reshape(cfg.NZ, cfg.NY)  # streamwise position
    fro_y = coords[:, 1].reshape(cfg.NZ, cfg.NY)  # wall-normal position

    # Allocate (NY6, NZ6) with code-coordinate indexing j, k
    y_2d = np.zeros((cfg.NY6, cfg.NZ6), dtype=np.float64)
    z_2d = np.zeros((cfg.NY6, cfg.NZ6), dtype=np.float64)

    # Map physical interior: code (j=BFR+jj, k=BFR+kk) <- file (J=kk, I=jj)
    # i.e., y_2d[BFR:BFR+NY, BFR:BFR+NZ] = fro_x.T
    y_2d[BFR:BFR+cfg.NY, BFR:BFR+cfg.NZ] = fro_x.T
    z_2d[BFR:BFR+cfg.NY, BFR:BFR+cfg.NZ] = fro_y.T

    # K-direction (z) ghost: linear extrapolation per j (initialization.h:236-256)
    nz6 = cfg.NZ6
    for j in range(BFR, BFR + cfg.NY):
        y_2d[j, 2] = 2.0 * y_2d[j, 3] - y_2d[j, 4]
        z_2d[j, 2] = 2.0 * z_2d[j, 3] - z_2d[j, 4]
        y_2d[j, 1] = 2.0 * y_2d[j, 2] - y_2d[j, 3]
        y_2d[j, 0] = 2.0 * y_2d[j, 1] - y_2d[j, 2]
        z_2d[j, 1] = 2.0 * z_2d[j, 2] - z_2d[j, 3]
        z_2d[j, 0] = 2.0 * z_2d[j, 1] - z_2d[j, 2]
        y_2d[j, nz6-3] = 2.0 * y_2d[j, nz6-4] - y_2d[j, nz6-5]
        z_2d[j, nz6-3] = 2.0 * z_2d[j, nz6-4] - z_2d[j, nz6-5]
        y_2d[j, nz6-2] = 2.0 * y_2d[j, nz6-3] - y_2d[j, nz6-4]
        y_2d[j, nz6-1] = 2.0 * y_2d[j, nz6-2] - y_2d[j, nz6-3]
        z_2d[j, nz6-2] = 2.0 * z_2d[j, nz6-3] - z_2d[j, nz6-4]
        z_2d[j, nz6-1] = 2.0 * z_2d[j, nz6-2] - z_2d[j, nz6-3]

    # J-direction (y) ghost: periodic wrap with +/-LY shift on y, no shift on z
    # initialization.h:270-288
    ny6 = cfg.NY6
    for k in range(nz6):
        y_2d[2, k] = y_2d[ny6-5, k] - LY
        y_2d[1, k] = y_2d[ny6-6, k] - LY
        y_2d[0, k] = y_2d[ny6-7, k] - LY
        z_2d[2, k] = z_2d[ny6-5, k]
        z_2d[1, k] = z_2d[ny6-6, k]
        z_2d[0, k] = z_2d[ny6-7, k]
        y_2d[ny6-3, k] = y_2d[4, k] + LY
        y_2d[ny6-2, k] = y_2d[5, k] + LY
        y_2d[ny6-1, k] = y_2d[6, k] + LY
        z_2d[ny6-3, k] = z_2d[4, k]
        z_2d[ny6-2, k] = z_2d[5, k]
        z_2d[ny6-1, k] = z_2d[6, k]

    return x, y_2d, z_2d


# ---------------------------------------------------------------
# Per-rank binary I/O + stitch / split
# ---------------------------------------------------------------
def read_rank_bin(path, cfg):
    """Read raw doubles, shape (NYD6, NZ6, NX6)."""
    expected = cfg.NYD6 * cfg.NZ6 * cfg.NX6 * 8
    sz = os.path.getsize(path)
    if sz != expected:
        raise ValueError('{}: size {} != expected {} (NYD6*NZ6*NX6*8 = {}*{}*{}*8)'.format(
            path, sz, expected, cfg.NYD6, cfg.NZ6, cfg.NX6))
    return np.fromfile(path, dtype=np.float64).reshape(cfg.NYD6, cfg.NZ6, cfg.NX6)


def stitch_y(per_rank_list, cfg):
    """Combine per-rank arrays into global (NY6, NZ6, NX6).

    Only each rank's unique physical rows are authoritative.  Checkpoint
    files also contain j-ghost rows and one overlap row, but those can be
    stale for post-collision f buffers.  Copying the whole slab lets a later
    rank's ghost rows overwrite the previous rank's interior rows, creating
    visible GPU-seam artifacts after interpolation.

    Unique mapping:
      local j = 3 .. 3+CHUNK-1  ->  global j = rank*CHUNK+3 .. +CHUNK-1

    The final physical j row is the periodic duplicate of the first physical
    row and is reconstructed explicitly.
    """
    g = np.zeros((cfg.NY6, cfg.NZ6, cfg.NX6), dtype=np.float64)
    for r in range(cfg.JP):
        j0 = r * cfg.CHUNK
        g[j0+BFR:j0+BFR+cfg.CHUNK, :, :] = per_rank_list[r][BFR:BFR+cfg.CHUNK, :, :]
    enforce_periodic_physical_duplicates(g, cfg)
    return g


def split_y(global_arr, cfg):
    """Split global (NY6, NZ6, NX6) into JP per-rank slices of (NYD6, NZ6, NX6)."""
    out = []
    for r in range(cfg.JP):
        j0 = r * cfg.CHUNK
        out.append(global_arr[j0:j0 + cfg.NYD6, :, :].copy())
    return out


def enforce_periodic_physical_duplicates(field, cfg):
    """Make physical periodic duplicate nodes bitwise identical.

    The solver's mass sum excludes the last physical node in periodic i/j, but
    the checkpoint and VTK surfaces still contain those duplicate nodes.  Keep
    them synchronized before ghost fill and rank splitting.
    """
    j0 = BFR
    jL = BFR + cfg.NY - 1
    i0 = BFR
    iL = BFR + cfg.NX - 1
    field[jL, :, :] = field[j0, :, :]
    field[:, :, iL] = field[:, :, i0]


# ---------------------------------------------------------------
# 3D trilinear interpolation in computational coordinates
# ---------------------------------------------------------------
def _interp_axis_linear(arr, old_n, new_n, axis):
    """Linearly interpolate arr along one computational axis."""
    if old_n == new_n:
        return arr.copy()

    coord = np.arange(new_n, dtype=np.float64) * (old_n - 1.0) / (new_n - 1.0)
    lo = np.floor(coord).astype(np.int64)
    lo = np.clip(lo, 0, old_n - 2)
    hi = lo + 1
    w = coord - lo

    a0 = np.take(arr, lo, axis=axis)
    a1 = np.take(arr, hi, axis=axis)
    shape = [1] * arr.ndim
    shape[axis] = new_n
    w = w.reshape(shape)
    return (1.0 - w) * a0 + w * a1


def interpolate_comp_3d(field_old, cfg_old, cfg_new):
    """Interpolate physical nodes in computational (j, k, i) space.

    The periodic-hill mesh is curvilinear: y(j,k) is not separable in j and k.
    The previous physical-space shortcut used the bottom-wall y(j,k=BFR) to
    bracket every wall-normal column, which misplaces data near the hill.
    For this refinement restart we preserve topology and map old/new nodes by
    normalized computational coordinates instead.
    """
    old_int = field_old[
        BFR:BFR + cfg_old.NY,
        BFR:BFR + cfg_old.NZ,
        BFR:BFR + cfg_old.NX,
    ]

    tmp = _interp_axis_linear(old_int, cfg_old.NY, cfg_new.NY, axis=0)
    tmp = _interp_axis_linear(tmp,     cfg_old.NZ, cfg_new.NZ, axis=1)
    tmp = _interp_axis_linear(tmp,     cfg_old.NX, cfg_new.NX, axis=2)

    field_new = np.zeros((cfg_new.NY6, cfg_new.NZ6, cfg_new.NX6), dtype=np.float64)
    field_new[
        BFR:BFR + cfg_new.NY,
        BFR:BFR + cfg_new.NZ,
        BFR:BFR + cfg_new.NX,
    ] = tmp
    return field_new


def fill_ghost(field, cfg):
    """Fill ghost cells of (NY6, NZ6, NX6) given physical interior is filled.

    Order: X periodic first, Z constant copy, Y periodic last
    (so Y/Z ghost cells inherit X-periodic values).
    """
    nx6 = cfg.NX6
    ny6 = cfg.NY6
    nz6 = cfg.NZ6

    # X (spanwise) periodic: i=2 <- NX+1 = NX6-5; i=NX+3 = NX6-3 <- 4; etc.
    field[:, :, 2] = field[:, :, nx6-5]
    field[:, :, 1] = field[:, :, nx6-6]
    field[:, :, 0] = field[:, :, nx6-7]
    field[:, :, nx6-3] = field[:, :, 4]
    field[:, :, nx6-2] = field[:, :, 5]
    field[:, :, nx6-1] = field[:, :, 6]

    # Z (wall-normal) constant copy from nearest wall
    # (BC kernel will overwrite ghost on first step; this is just a non-pathological seed)
    field[:, 2, :] = field[:, 3, :]
    field[:, 1, :] = field[:, 3, :]
    field[:, 0, :] = field[:, 3, :]
    field[:, nz6-3, :] = field[:, nz6-4, :]
    field[:, nz6-2, :] = field[:, nz6-4, :]
    field[:, nz6-1, :] = field[:, nz6-4, :]

    # Y (streamwise) periodic
    field[2, :, :] = field[ny6-5, :, :]
    field[1, :, :] = field[ny6-6, :, :]
    field[0, :, :] = field[ny6-7, :, :]
    field[ny6-3, :, :] = field[4, :, :]
    field[ny6-2, :, :] = field[5, :, :]
    field[ny6-1, :, :] = field[6, :, :]


# ---------------------------------------------------------------
# Phase C: Physical-space interpolation (replaces interpolate_comp_3d
# for GAMMA-changed regrids; computational-space remap places turbulent
# structure at wrong wall distance when GAMMA differs between OLD and NEW).
#
# Pipeline:
#   1. build_old_cell_search_index — per-cell bbox prefilter
#   2. find_containing_cell_2d — Newton 2x2 inverse + triangle fallback
#   3. precompute_phys_mapping_2d — build (j*, k*, xi, eta) cache once
#   4. interpolate_phys_3d_with_mapping — trilinear blend using cached mapping
# ---------------------------------------------------------------
class _DegenerateCellError(Exception):
    """Bilinear inverse failed (cell ill-conditioned or non-convex)."""


def build_old_cell_search_index(y_old, z_old):
    """Per-cell axis-aligned bounding boxes for fast point-in-cell prefilter.

    y_old, z_old : (NY, NZ) interior arrays (no ghost).
    Returns 4 arrays of shape (NY-1, NZ-1) — min/max y and z per cell.
    """
    cy = np.stack([y_old[:-1, :-1], y_old[1:, :-1],
                   y_old[:-1, 1:],  y_old[1:, 1:]], axis=-1)
    cz = np.stack([z_old[:-1, :-1], z_old[1:, :-1],
                   z_old[:-1, 1:],  z_old[1:, 1:]], axis=-1)
    return cy.min(-1), cy.max(-1), cz.min(-1), cz.max(-1)


def bilinear_inverse_newton(y_n, z_n, y_corners, z_corners,
                            max_iter=8, tol=1e-12):
    """Newton 2x2 solve for (xi, eta) in [0,1]^2 inside a bilinear cell.

    Bilinear:
      y(xi,eta) = (1-xi)(1-eta)*y_a + xi(1-eta)*y_b + (1-xi)*eta*y_c + xi*eta*y_d
      z(xi,eta) = same with z corners
    Corner index: a=(0,0), b=(1,0), c=(0,1), d=(1,1).

    Returns (xi, eta). Raises _DegenerateCellError on Jacobian collapse or
    non-convergence within max_iter.
    """
    y_a, y_b, y_c, y_d = y_corners
    z_a, z_b, z_c, z_d = z_corners
    xi, eta = 0.5, 0.5
    for _ in range(max_iter):
        one_xi = 1.0 - xi
        one_et = 1.0 - eta
        y_int = one_xi*one_et*y_a + xi*one_et*y_b + one_xi*eta*y_c + xi*eta*y_d
        z_int = one_xi*one_et*z_a + xi*one_et*z_b + one_xi*eta*z_c + xi*eta*z_d
        ry = y_int - y_n
        rz = z_int - z_n
        if abs(ry) < tol and abs(rz) < tol:
            return xi, eta
        dy_dxi  = -one_et*y_a + one_et*y_b - eta*y_c + eta*y_d
        dy_deta = -one_xi*y_a - xi*y_b + one_xi*y_c + xi*y_d
        dz_dxi  = -one_et*z_a + one_et*z_b - eta*z_c + eta*z_d
        dz_deta = -one_xi*z_a - xi*z_b + one_xi*z_c + xi*z_d
        det = dy_dxi*dz_deta - dy_deta*dz_dxi
        if abs(det) < 1e-30:
            raise _DegenerateCellError()
        inv = 1.0 / det
        xi  -= ( dz_deta*ry - dy_deta*rz) * inv
        eta -= (-dz_dxi *ry + dy_dxi *rz) * inv
    raise _DegenerateCellError()


def bilinear_inverse_triangle_fallback(y_n, z_n, y_corners, z_corners, eps=5e-5):
    """Triangle barycentric fallback when Newton fails or converges out-of-bounds.

    Splits cell (a, b, c, d) into 2 triangles:
      Triangle 1: a=(0,0), b=(1,0), d=(1,1)  -> covers xi >= eta region
      Triangle 2: a=(0,0), c=(0,1), d=(1,1)  -> covers eta >= xi region

    Solves barycentric per triangle; returns first one with all weights in [0,1].
    Raises _DegenerateCellError if neither triangle contains the point.
    """
    y_a, y_b, y_c, y_d = y_corners
    z_a, z_b, z_c, z_d = z_corners

    def _solve_tri(y0, z0, y1, z1, y2, z2):
        det = (y1-y0)*(z2-z0) - (z1-z0)*(y2-y0)
        if abs(det) < 1e-30:
            return None
        w1 = ((y_n-y0)*(z2-z0) - (z_n-z0)*(y2-y0)) / det
        w2 = ((y1-y0)*(z_n-z0) - (z1-z0)*(y_n-y0)) / det
        w0 = 1.0 - w1 - w2
        return w0, w1, w2

    # Triangle 1: a, b, d  ->  xi = w1 + w2, eta = w2
    w = _solve_tri(y_a, z_a, y_b, z_b, y_d, z_d)
    if w is not None and all(-eps <= wi <= 1 + eps for wi in w):
        return w[1] + w[2], w[2]

    # Triangle 2: a, c, d  ->  xi = w2, eta = w1 + w2
    w = _solve_tri(y_a, z_a, y_c, z_c, y_d, z_d)
    if w is not None and all(-eps <= wi <= 1 + eps for wi in w):
        return w[2], w[1] + w[2]

    raise _DegenerateCellError()


def find_containing_cell_2d(y_n, z_n, y_old, z_old, bboxes,
                            eps_phys=5e-6, eps_param=5e-5):
    """Locate OLD cell containing (y_n, z_n). Returns (j*, k*, xi, eta).

    eps_phys  — physical-space tolerance for bbox pre-filter
    eps_param — parametric-space tolerance for xi/eta in-bounds check

    Per-candidate strategy:
      1. Newton 2x2; accept if converged AND in [0,1]^2 (with eps_param).
      2. If Newton failed OR converged out-of-bounds -> triangle fallback.
      3. Both failed -> next candidate.
      4. All candidates exhausted -> ValueError.
    """
    bbox_y_min, bbox_y_max, bbox_z_min, bbox_z_max = bboxes
    candidates = ((bbox_y_min - eps_phys <= y_n) & (y_n <= bbox_y_max + eps_phys) &
                  (bbox_z_min - eps_phys <= z_n) & (z_n <= bbox_z_max + eps_phys))
    cand_jk = np.argwhere(candidates)
    if len(cand_jk) == 0:
        raise ValueError('No OLD cell brackets ({:.6e}, {:.6e})'.format(y_n, z_n))

    def _in_bounds(xi, eta):
        return -eps_param <= xi <= 1 + eps_param and -eps_param <= eta <= 1 + eps_param

    for j, k in cand_jk:
        y_corners = (y_old[j, k],   y_old[j+1, k],
                     y_old[j, k+1], y_old[j+1, k+1])
        z_corners = (z_old[j, k],   z_old[j+1, k],
                     z_old[j, k+1], z_old[j+1, k+1])
        xi, eta = None, None

        # Newton: accept only if converged AND in-bounds
        try:
            xi_n, eta_n = bilinear_inverse_newton(y_n, z_n, y_corners, z_corners)
            if _in_bounds(xi_n, eta_n):
                xi, eta = xi_n, eta_n
        except _DegenerateCellError:
            pass

        # Triangle fallback (Newton failed OR Newton out-of-bounds)
        if xi is None:
            try:
                xi_t, eta_t = bilinear_inverse_triangle_fallback(y_n, z_n,
                                                                 y_corners, z_corners, eps=eps_param)
                if _in_bounds(xi_t, eta_t):
                    xi, eta = xi_t, eta_t
            except _DegenerateCellError:
                pass

        if xi is not None:
            return int(j), int(k), float(np.clip(xi, 0, 1)), float(np.clip(eta, 0, 1))

    raise ValueError(
        'Point ({:.6e}, {:.6e}) not in any OLD cell after Newton+triangle'.format(y_n, z_n))


class PhysMapping2D:
    """Precomputed mapping from NEW (j_n, k_n) to OLD cell + bilinear weights.

    Built once per OLD/NEW grid pair; shared across all field interpolations
    (rho, ux, uy, uz). Cell search is the dominant cost; reusing it across
    4 fields gives ~3.3x speedup vs rebuilding for each field.
    """
    __slots__ = ('jstar', 'kstar', 'xistar', 'etastar',
                 'i_o_arr', 'xi_i_arr', 'cfg_old', 'cfg_new')

    def __init__(self, jstar, kstar, xistar, etastar, i_o_arr, xi_i_arr,
                 cfg_old, cfg_new):
        self.jstar = jstar
        self.kstar = kstar
        self.xistar = xistar
        self.etastar = etastar
        self.i_o_arr = i_o_arr
        self.xi_i_arr = xi_i_arr
        self.cfg_old = cfg_old
        self.cfg_new = cfg_new


def precompute_phys_mapping_2d(y2d_old, z2d_old, y2d_new, z2d_new,
                                cfg_old, cfg_new):
    """Build PhysMapping2D once for a given OLD/NEW grid pair.

    Cell search dominates Phase C runtime. Reusing this across rho/ux/uy/uz
    saves 4x cost on the dominant operation.
    """
    y_int_old = y2d_old[BFR:BFR+cfg_old.NY, BFR:BFR+cfg_old.NZ]
    z_int_old = z2d_old[BFR:BFR+cfg_old.NY, BFR:BFR+cfg_old.NZ]
    bboxes = build_old_cell_search_index(y_int_old, z_int_old)

    # Domain bounds for clamping: different .dat files may have FP noise at
    # shared boundaries (e.g., y=-3.4e-8 vs y=0.0).  Clamp NEW coords into
    # OLD domain so the bbox prefilter doesn't reject boundary points.
    y_old_min, y_old_max = float(y_int_old.min()), float(y_int_old.max())
    z_old_min, z_old_max = float(z_int_old.min()), float(z_int_old.max())

    jstar = np.empty((cfg_new.NY, cfg_new.NZ), dtype=np.int32)
    kstar = np.empty((cfg_new.NY, cfg_new.NZ), dtype=np.int32)
    xistar = np.empty((cfg_new.NY, cfg_new.NZ), dtype=np.float64)
    etastar = np.empty((cfg_new.NY, cfg_new.NZ), dtype=np.float64)
    n_clamped = 0
    for j_n in range(cfg_new.NY):
        for k_n in range(cfg_new.NZ):
            y_n = y2d_new[BFR + j_n, BFR + k_n]
            z_n = z2d_new[BFR + j_n, BFR + k_n]
            if y_n < y_old_min or y_n > y_old_max or z_n < z_old_min or z_n > z_old_max:
                y_n = max(y_old_min, min(y_old_max, y_n))
                z_n = max(z_old_min, min(z_old_max, z_n))
                n_clamped += 1
            j_o, k_o, xi, eta = find_containing_cell_2d(
                y_n, z_n, y_int_old, z_int_old, bboxes)
            jstar[j_n, k_n] = j_o
            kstar[j_n, k_n] = k_o
            xistar[j_n, k_n] = xi
            etastar[j_n, k_n] = eta

    # i mapping: uniform spanwise, periodic ghost handles wrap (no clamp)
    dx_old = LX / (cfg_old.NX - 1)
    dx_new = LX / (cfg_new.NX - 1)
    i_o_float_arr = (np.arange(cfg_new.NX, dtype=np.float64) * dx_new) / dx_old
    i_o_arr = np.floor(i_o_float_arr).astype(np.int64)
    xi_i_arr = i_o_float_arr - i_o_arr

    if n_clamped > 0:
        print('      domain-boundary clamp applied to {} of {} points (FP noise at shared boundary)'.format(
            n_clamped, cfg_new.NY * cfg_new.NZ))
    print('      Phys mapping cache built: {} cells located'.format(
        cfg_new.NY * cfg_new.NZ))
    return PhysMapping2D(jstar, kstar, xistar, etastar, i_o_arr, xi_i_arr,
                         cfg_old, cfg_new)


def interpolate_phys_3d_with_mapping(field_old, mapping):
    """Interpolate one (NY6_old, NZ6_old, NX6_old) field using cached mapping.

    No cell search — just trilinear blend with cached weights. Caller must
    ensure field_old has ghosts filled (real checkpoints already do; synthetic
    test fields require explicit fill_ghost(field_old, cfg_old)).
    """
    cfg_old, cfg_new = mapping.cfg_old, mapping.cfg_new
    field_new = np.zeros((cfg_new.NY6, cfg_new.NZ6, cfg_new.NX6), dtype=np.float64)

    i_o_arr = mapping.i_o_arr
    xi_i_arr = mapping.xi_i_arr
    ib0 = BFR + i_o_arr
    ib1 = BFR + i_o_arr + 1     # ghost wrap handles last-point case

    for j_n in range(cfg_new.NY):
        for k_n in range(cfg_new.NZ):
            j_o = int(mapping.jstar[j_n, k_n])
            k_o = int(mapping.kstar[j_n, k_n])
            xi  = float(mapping.xistar[j_n, k_n])
            eta = float(mapping.etastar[j_n, k_n])

            w_a = (1 - xi) * (1 - eta)
            w_b = xi       * (1 - eta)
            w_c = (1 - xi) * eta
            w_d = xi       * eta

            jb, kb = BFR + j_o, BFR + k_o
            a0 = field_old[jb,   kb,   ib0]; a1 = field_old[jb,   kb,   ib1]
            b0 = field_old[jb+1, kb,   ib0]; b1 = field_old[jb+1, kb,   ib1]
            c0 = field_old[jb,   kb+1, ib0]; c1 = field_old[jb,   kb+1, ib1]
            d0 = field_old[jb+1, kb+1, ib0]; d1 = field_old[jb+1, kb+1, ib1]

            v0 = w_a*a0 + w_b*b0 + w_c*c0 + w_d*d0
            v1 = w_a*a1 + w_b*b1 + w_c*c1 + w_d*d1
            field_new[BFR + j_n, BFR + k_n, BFR:BFR + cfg_new.NX] = (
                (1 - xi_i_arr) * v0 + xi_i_arr * v1)

    fill_ghost(field_new, cfg_new)
    return field_new


def interpolate_phys_3d(field_old, cfg_old, cfg_new,
                        y2d_old, z2d_old, y2d_new, z2d_new):
    """One-shot wrapper: build mapping + interp single field.

    For multi-field workflows (rho, ux, uy, uz), prefer:
        mapping = precompute_phys_mapping_2d(...)
        rho_new = interpolate_phys_3d_with_mapping(rho, mapping)
        ux_new  = interpolate_phys_3d_with_mapping(ux,  mapping)
    to avoid redundant cell search.
    """
    mapping = precompute_phys_mapping_2d(
        y2d_old, z2d_old, y2d_new, z2d_new, cfg_old, cfg_new)
    return interpolate_phys_3d_with_mapping(field_old, mapping)


# ---------------------------------------------------------------
# 6th-order 7-point Lagrange interpolation (O(h^6) tensor product)
# ---------------------------------------------------------------
# Uniform nodes {0,1,2,3,4,5,6}; fractional position t ∈ [0,1] within
# the central cell (nodes 3,4).  Stencil: 3 nodes left + 3 nodes right
# of the central pair.
#
# L_m(t) = ∏_{n=0,n≠m}^{6} (t+3-n) / (m-n)    for m=0..6
#
# where s = t + 3 maps the fraction into the stencil-local coordinate
# so that node m corresponds to s = m.
#
# Polynomial exactness: reproduces degree-6 polynomials exactly.

def lagrange7_weights(t):
    """Compute 7-point Lagrange basis weights for fractional position t ∈ [0,1].

    The stencil is centered on nodes 3 and 4: the interpolation point lies
    at stencil coordinate s = t + 3.  Returns array of 7 weights.
    """
    s = t + 3.0
    w = np.empty(7, dtype=np.float64)
    for m in range(7):
        val = 1.0
        for n in range(7):
            if n != m:
                val *= (s - n) / (m - n)
        w[m] = val
    return w


def lagrange7_weights_vectorized(t_arr):
    """Vectorized: compute 7-point Lagrange weights for an array of t values.

    t_arr : 1D array of fractional positions in [0,1].
    Returns (len(t_arr), 7) weight matrix.
    """
    s = t_arr[:, np.newaxis] + 3.0   # (N, 1) stencil coordinates
    nodes = np.arange(7, dtype=np.float64)[np.newaxis, :]  # (1, 7)

    denom = np.empty(7, dtype=np.float64)
    for m in range(7):
        d = 1.0
        for n in range(7):
            if n != m:
                d *= (m - n)
        denom[m] = d

    diff = s - nodes  # (N, 7): s_i - n for each node n
    prod_all = np.prod(diff, axis=1, keepdims=True)  # (N, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        weights = prod_all / (diff * denom[np.newaxis, :])
    mask = np.abs(diff) < 1e-15
    if np.any(mask):
        idx_i, idx_m = np.where(mask)
        weights[idx_i, :] = 0.0
        weights[idx_i, idx_m] = 1.0
    return weights


def extrapolate_wall_ghost_stencil_cubic(stencil_k, stencil_start, cfg):
    """Mirror solver cubic wall-ghost extrapolation for one 7-point k stencil.

    stencil_k has shape (7, ...), with axis 0 corresponding to consecutive
    wall-normal buffer indices [stencil_start, stencil_start + 6].  When that
    window crosses either wall, replace the ghost entries by direct cubic
    extrapolation from the nearest four in-domain stencil values before the
    Lagrange contraction.  This mirrors gilbm_ghost_zone_extrapolate() with
    GHOST_EXTRAP_ORDER=3 in gilbm/evolution_gilbm/1.algorithm1.h.
    """
    if stencil_k.shape[0] != 7:
        raise ValueError('extrapolate_wall_ghost_stencil_cubic expects axis-0 length 7')

    out = stencil_k.copy()
    fluid_lo = BFR
    fluid_hi = cfg.NZ6 - 1 - BFR

    n_ghost_bot = max(fluid_lo - stencil_start, 0)
    n_ghost_top = max(stencil_start + 6 - fluid_hi, 0)

    if n_ghost_bot > 0:
        p0 = n_ghost_bot
        p1 = n_ghost_bot + 1
        p2 = n_ghost_bot + 2
        p3 = n_ghost_bot + 3
        for g in range(n_ghost_bot - 1, -1, -1):
            d = float(p0 - g)
            d1, d2, d3 = d + 1.0, d + 2.0, d + 3.0
            c0 =  d1 * d2 * d3 / 6.0
            c1 = -d  * d2 * d3 / 2.0
            c2 =  d  * d1 * d3 / 2.0
            c3 = -d  * d1 * d2 / 6.0
            out[g] = c0 * out[p0] + c1 * out[p1] + c2 * out[p2] + c3 * out[p3]

    if n_ghost_top > 0:
        pN = 6 - n_ghost_top
        pN1 = pN - 1
        pN2 = pN - 2
        pN3 = pN - 3
        for g in range(pN + 1, 7):
            d = float(g - pN)
            d1, d2, d3 = d + 1.0, d + 2.0, d + 3.0
            c0 =  d1 * d2 * d3 / 6.0
            c1 = -d  * d2 * d3 / 2.0
            c2 =  d  * d1 * d3 / 2.0
            c3 = -d  * d1 * d2 / 6.0
            out[g] = c0 * out[pN] + c1 * out[pN1] + c2 * out[pN2] + c3 * out[pN3]

    return out


def interpolate_lagrange7_3d_with_mapping(field_old, mapping):
    """Interpolate one (NY6_old, NZ6_old, NX6_old) field using 7-point Lagrange.

    Uses the same PhysMapping2D (j*, k*, xi, eta) as the bilinear version,
    but applies 7x7 tensor product in the (j, k) plane and 7-point along i.
    Wall-normal stencils that cross either wall rebuild ghost entries by cubic
    extrapolation from in-domain values before the k contraction, mirroring
    the solver's GHOST_EXTRAP_ORDER=3 path.

    Stencil: for anchor cell (j*, k*), nodes j*-3..j*+3, k*-3..k*+3.
    Node m=3 at j* corresponds to xi=0; node m=4 at j*+1 to xi=1.
    """
    cfg_old, cfg_new = mapping.cfg_old, mapping.cfg_new
    field_new = np.zeros((cfg_new.NY6, cfg_new.NZ6, cfg_new.NX6), dtype=np.float64)

    i_o_arr = mapping.i_o_arr
    xi_i_arr = mapping.xi_i_arr
    NX_new = cfg_new.NX
    NY6_old = cfg_old.NY6
    NZ6_old = cfg_old.NZ6
    NX6_old = cfg_old.NX6

    i_weights = lagrange7_weights_vectorized(xi_i_arr)  # (NX, 7)

    # Precompute i-stencil buffer indices for all NX_new points: (NX_new, 7)
    # Stencil offsets: -3, -2, -1, 0, +1, +2, +3 relative to anchor i_o.
    # Node m=3 at (BFR + i_o) corresponds to xi_i=0; node m=4 to xi_i=1.
    # Ghost fill gives 3 valid layers on each side, so buf indices [0, NX6-1]
    # cover the stencil.  Clamp handles the rare boundary case (where
    # xi_i=0/1 makes the clamped-index weights exactly 0).
    i_stencil = np.empty((NX_new, 7), dtype=np.int64)
    for s in range(7):
        buf_idx = BFR + i_o_arr + (s - 3)
        i_stencil[:, s] = np.clip(buf_idx, 0, NX6_old - 1)

    for j_n in range(cfg_new.NY):
        for k_n in range(cfg_new.NZ):
            j_o = int(mapping.jstar[j_n, k_n])
            k_o = int(mapping.kstar[j_n, k_n])
            xi  = float(mapping.xistar[j_n, k_n])
            eta = float(mapping.etastar[j_n, k_n])

            wj = lagrange7_weights(xi)
            wk = lagrange7_weights(eta)

            jb = BFR + j_o
            kb = BFR + k_o

            j_indices = np.clip(np.arange(jb - 3, jb + 4), 0, NY6_old - 1)
            k_indices = np.clip(np.arange(kb - 3, kb + 4), 0, NZ6_old - 1)

            # wjk[mj, mk] = wj[mj] * wk[mk], shape (7, 7)
            wjk = np.outer(wj, wk)

            # Gather stencil slab: field_old[j_indices, k_indices, i_stencil]
            # → (7, 7, NX, 7) then contract with i_weights → (7, 7, NX)
            slab = field_old[np.ix_(j_indices, k_indices)]  # (7, 7, NX6_old)
            # For each (mj, mk) pair, gather the 7 i-stencil values for all NX points
            # slab_i shape: (7, 7, NX, 7) — last dim is the i-stencil
            slab_i = slab[:, :, i_stencil]  # (7, 7, NX, 7)
            # Contract i-stencil with i_weights: sum over last dim
            # i_weights shape (NX, 7) → broadcast
            val_jk = np.einsum('jkni,ni->jkn', slab_i, i_weights)  # (7, 7, NX)
            # Solver-style wall handling: rebuild any k-ghost entries from
            # in-domain stencil values before the wall-normal contraction.
            val_kjn = extrapolate_wall_ghost_stencil_cubic(
                np.moveaxis(val_jk, 1, 0), kb - 3, cfg_old)
            val_jk = np.moveaxis(val_kjn, 0, 1)
            # Contract (j, k) with wjk
            row = np.einsum('jk,jkn->n', wjk, val_jk)  # (NX,)

            field_new[BFR + j_n, BFR + k_n, BFR:BFR + NX_new] = row

    fill_ghost(field_new, cfg_new)
    return field_new


def clamp_wall_macros(rho, ux, uy, uz, cfg):
    """Clamp physical wall velocity before global conservation corrections."""
    kt = cfg.NZ6 - 1 - BFR
    wall_u_max_before = max(
        float(np.max(np.abs(ux[:, BFR, :]))), float(np.max(np.abs(ux[:, kt, :]))),
        float(np.max(np.abs(uy[:, BFR, :]))), float(np.max(np.abs(uy[:, kt, :]))),
        float(np.max(np.abs(uz[:, BFR, :]))), float(np.max(np.abs(uz[:, kt, :]))),
    )
    wall_rho_max_delta_before = max(
        float(np.max(np.abs(rho[:, BFR, :] - 1.0))),
        float(np.max(np.abs(rho[:, kt, :] - 1.0))),
    )

    for arr in (ux, uy, uz):
        arr[:, BFR, :] = 0.0
        arr[:, kt, :] = 0.0

    return wall_u_max_before, wall_rho_max_delta_before


def apply_rho_mass_correction(rho, cfg):
    """Restore full-domain mean rho=1 with the same uniform offset as runtime.

    The conserved domain matches the solver reduction domain:
      i∈[3, NX6-4), j∈[3, NY6-4), k∈[3, NZ6-3)
    including both physical wall rows and excluding periodic duplicates.

    The runtime GILBM kernel applies rho_modify to every physical k row,
    including both walls.  Rebuild checkpoints must use the same policy;
    otherwise the restart state introduces an artificial wall-pressure reset.
    """
    ni = cfg.NX6 - 7
    nj = cfg.NY6 - 7
    nk = cfg.NZ6 - 6
    N_full = ni * nj * nk

    full_domain = (slice(BFR, BFR + nj),
                   slice(BFR, BFR + nk),
                   slice(BFR, BFR + ni))

    rho_sum = float(np.sum(rho[full_domain]))
    rho_global_avg = rho_sum / float(N_full)
    rho_modify = 1.0 - rho_global_avg

    mean_before = rho_global_avg
    rho[full_domain] += rho_modify
    mean_after = float(np.sum(rho[full_domain])) / float(N_full)

    return rho_modify, mean_before, mean_after


def compute_Ub(uy, z_2d, cfg):
    """Compute bulk velocity Ub at j=BFR plane, matching evolution.h bilinear cell-average.

    Solver formula (evolution.h:544-555):
      for k in [3, NZ6-4):    # cell centres between walls
        for i in [3, NX6-4):
          v_cell  = avg of 4 corner nodes at (k,i),(k+1,i),(k,i+1),(k+1,i+1)
          dx_cell = x[i+1] - x[i]
          dz_cell = z_h[j=3, k+1] - z_h[j=3, k]   # physical z-spacing
          Ub += v_cell * dx_cell * dz_cell
          A  += dx_cell * dz_cell
      Ub /= A
    """
    j0 = BFR
    dx = LX / (cfg.NX - 1)
    x = (np.arange(cfg.NX6, dtype=np.float64) - BFR) * dx

    i_lo, i_hi = BFR, cfg.NX6 - 4   # i = 3 .. NX6-5
    k_lo, k_hi = BFR, cfg.NZ6 - 4   # k = 3 .. NZ6-5

    v_plane = uy[j0, :, :]  # [NZ6, NX6]
    v_cell = 0.25 * (v_plane[k_lo:k_hi, i_lo:i_hi]
                    + v_plane[k_lo+1:k_hi+1, i_lo:i_hi]
                    + v_plane[k_lo:k_hi, i_lo+1:i_hi+1]
                    + v_plane[k_lo+1:k_hi+1, i_lo+1:i_hi+1])

    dx_cell = x[i_lo+1:i_hi+1] - x[i_lo:i_hi]                       # [ni]
    dz_cell = z_2d[j0, k_lo+1:k_hi+1] - z_2d[j0, k_lo:k_hi]        # [nk]

    dA = dz_cell[:, np.newaxis] * dx_cell[np.newaxis, :]  # [nk, ni]
    return float(np.sum(v_cell * dA) / np.sum(dA))


def apply_Ub_correction(Ub_old, uy_new, z2d_new, cfg_new):
    """Scale streamwise velocity so Ub is conserved across interpolation.

    Modifies uy_new in-place, re-enforces periodic BCs and ghost cells.
    Returns (scale_factor, Ub_new_before, Ub_new_after).
    """
    Ub_new_before = compute_Ub(uy_new, z2d_new, cfg_new)
    print('      Ub correction: OLD Ub = {:.15e}'.format(Ub_old))
    print('      Ub correction: NEW Ub (before) = {:.15e}'.format(Ub_new_before))

    if abs(Ub_new_before) < 1e-30:
        print('      Ub correction: SKIP (Ub_new ≈ 0)')
        return 1.0, Ub_new_before, Ub_new_before

    scale = Ub_old / Ub_new_before
    interior = (slice(BFR, BFR + cfg_new.NY),
                slice(BFR + 1, BFR + cfg_new.NZ - 1),
                slice(BFR, BFR + cfg_new.NX))
    uy_new[interior] *= scale
    enforce_periodic_physical_duplicates(uy_new, cfg_new)
    fill_ghost(uy_new, cfg_new)

    Ub_new_after = compute_Ub(uy_new, z2d_new, cfg_new)
    print('      Ub correction: scale = {:.15e}'.format(scale))
    print('      Ub correction: NEW Ub (after)  = {:.15e}'.format(Ub_new_after))
    print('      Ub correction: residual = {:.3e}'.format(abs(Ub_new_after - Ub_old)))
    return scale, Ub_new_before, Ub_new_after


# ---------------------------------------------------------------
# Equilibrium reconstruction (initialization.h:36-42)
# ---------------------------------------------------------------
def compute_feq_q(rho, ux, uy, uz, q):
    udot = ux*ux + uy*uy + uz*uz
    if q == 0:
        return W[0] * rho * (1.0 - 1.5 * udot)
    eu = E[q, 0]*ux + E[q, 1]*uy + E[q, 2]*uz
    return W[q] * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*udot)


# ---------------------------------------------------------------
# New dt = minSize (variables.h:115-117)
# ---------------------------------------------------------------
def compute_minsize(cfg):
    a = cfg.GAMMA * (1.0/(cfg.NZ - 1) - cfg.ALPHA)
    b = cfg.GAMMA * cfg.ALPHA
    return (LZ - 1.0) * 0.5 * (1.0 + math.tanh(a) / math.tanh(b))


# ---------------------------------------------------------------
# Chapman-Enskog f_neq reconstruction (Direction A from review)
# ---------------------------------------------------------------
# Generalized from gilbm/boundary_conditions.h:13-21 (wall-only) to
# interior nodes by retaining all 9 partial derivatives ∂u_α/∂x_β.
#
#   f_neq_q = w_q * rho * ce_coeff * Σ_αβ (3·c_qα·c_qβ - δ_αβ) · ∂u_α/∂x_β
#
#   omega_new = 3*niu/dt_global_new + 0.5
#   ce_coeff = -omega_new * dt_global_new
#
# This replaces direct linear interpolation of f_neq, which destroys the
# velocity-gradient information encoded in f_neq via Chapman-Enskog and
# is the dominant divergence cause when GAMMA changes between grids.
#
# Conservation properties (analytically exact):
#   Σ_q f_neq_q          = 0     (since 3·Σ_q W_q·c_qα·c_qβ = δ_αβ)
#   Σ_q c_q · f_neq_q    = 0     (third-order moment vanishes for D3Q19)
def parse_niu_from_variables_h(vh_path):
    """Parse niu = Uref / Re from variables.h. Returns None if unavailable."""
    uref = None
    re_num = None
    try:
        with open(vh_path, encoding='utf-8', errors='replace') as f:
            for line in f:
                stripped = line.strip()
                if not stripped.startswith('#define'):
                    continue
                parts = stripped.split(None, 2)
                if len(parts) < 3:
                    continue
                key = parts[1]
                val_str = parts[2].split('//')[0].strip().strip('()')
                if key == 'Uref':
                    try: uref = float(val_str)
                    except ValueError: pass
                elif key == 'Re':
                    try: re_num = float(val_str)
                    except ValueError: pass
    except (IOError, OSError):
        return None
    if uref is None or re_num is None or re_num == 0.0:
        return None
    return uref / re_num


def compute_inverse_metric_2d(y_2d, z_2d):
    """Inverse Jacobian for curvilinear (j,k) → (y_phys, z_phys).

    Forward:  J = [[y_j, y_k], [z_j, z_k]]
    Inverse:  J^{-1} = (1/det) * [[z_k, -y_k], [-z_j, y_j]]
              det = y_j*z_k - y_k*z_j

    Centered FD on interior; ghost cells (already filled by build_grid_xyz
    with periodic ±LY in j and linear extrapolation in k) make boundary
    differencing 2nd-order. Returns four (NY6, NZ6) arrays.
    """
    y_j = np.empty_like(y_2d)
    z_j = np.empty_like(z_2d)
    y_j[1:-1, :] = (y_2d[2:, :] - y_2d[:-2, :]) / 2.0
    z_j[1:-1, :] = (z_2d[2:, :] - z_2d[:-2, :]) / 2.0
    y_j[0, :]  = y_2d[1, :] - y_2d[0, :]
    y_j[-1, :] = y_2d[-1, :] - y_2d[-2, :]
    z_j[0, :]  = z_2d[1, :] - z_2d[0, :]
    z_j[-1, :] = z_2d[-1, :] - z_2d[-2, :]

    y_k = np.empty_like(y_2d)
    z_k = np.empty_like(z_2d)
    y_k[:, 1:-1] = (y_2d[:, 2:] - y_2d[:, :-2]) / 2.0
    z_k[:, 1:-1] = (z_2d[:, 2:] - z_2d[:, :-2]) / 2.0
    y_k[:, 0]  = y_2d[:, 1] - y_2d[:, 0]
    y_k[:, -1] = y_2d[:, -1] - y_2d[:, -2]
    z_k[:, 0]  = z_2d[:, 1] - z_2d[:, 0]
    z_k[:, -1] = z_2d[:, -1] - z_2d[:, -2]

    det = y_j * z_k - y_k * z_j
    eps = 1e-30
    sing = int(np.sum(np.abs(det) <= eps))
    if sing > 0:
        print('      WARN: Jacobian near-singular at {} grid points'.format(sing))
    inv_det = 1.0 / np.where(np.abs(det) > eps, det, eps)

    dj_dy =  z_k * inv_det
    dj_dz = -y_k * inv_det
    dk_dy = -z_j * inv_det
    dk_dz =  y_j * inv_det
    return dj_dy, dj_dz, dk_dy, dk_dz


# ---------------------------------------------------------------
# Phase A: 6th-order Fornberg inverse metric (mirrors solver
# gilbm/metric_terms.h). Two stencils:
#   - j direction: pure 6th-order central, reads ±3 periodic ghost
#     (build_grid_xyz fills j-ghost with ±LY shift, valid 6th-order).
#   - k direction: 6th-order adaptive skew, NEVER reads k ghost
#     (k ghost is linear extrap, only 2nd-order valid; reading it
#     would degrade the entire stencil).
# ---------------------------------------------------------------
# 7-point Fornberg coefficients, 1st derivative, unit spacing.
# FD6_COEFF[p, m] : evaluation point at offset p in stencil window [s, s+6]
# Mirror gilbm/metric_terms.h:34-42. Divisor 60 absorbed into table.
FD6_COEFF = np.array([
    [-147.0,  360.0, -450.0,  400.0, -225.0,   72.0,  -10.0],   # p=0 forward
    [ -10.0,  -77.0,  150.0, -100.0,   50.0,  -15.0,    2.0],   # p=1
    [   2.0,  -24.0,  -35.0,   80.0,  -30.0,    8.0,   -1.0],   # p=2
    [  -1.0,    9.0,  -45.0,    0.0,   45.0,   -9.0,    1.0],   # p=3 central (6th-order)
    [   1.0,   -8.0,   30.0,  -80.0,   35.0,   24.0,   -2.0],   # p=4
    [  -2.0,   15.0,  -50.0,  100.0, -150.0,   77.0,   10.0],   # p=5
    [  10.0,  -72.0,  225.0, -400.0,  450.0, -360.0,  147.0],   # p=6 backward
], dtype=np.float64) / 60.0


def fd6_axis_central(arr, k_lo, k_hi, axis):
    """6th-order pure central FD using p=3 row of Fornberg table.

    For each evaluation point k in [k_lo, k_hi]:
      deriv[k] = sum FD6_COEFF[3, m] * arr[k + m - 3]   for m in [0, 6]

    REQUIRES: k_lo - 3 >= 0 and k_hi + 3 < arr.shape[axis]
    (3 ghost layers each side filled with valid 6th-order data; this holds for
    j-direction periodic ghost from build_grid_xyz).

    Mirrors gilbm/metric_terms.h:100-109 FD6_j_central exactly.
    """
    if axis not in (0, 1):
        raise ValueError('fd6_axis_central: axis must be 0 or 1')
    if axis == 0:
        arr_w = np.moveaxis(arr, 0, -1)
    else:
        arr_w = arr
    deriv = np.zeros_like(arr_w)
    coef = FD6_COEFF[3]
    for m in range(7):
        offset = m - 3
        deriv[..., k_lo:k_hi+1] += coef[m] * arr_w[..., k_lo+offset:k_hi+1+offset]
    if axis == 0:
        deriv = np.moveaxis(deriv, -1, 0)
    return deriv


def fd6_axis_adaptive(arr, k_lo, k_hi, axis):
    """6th-order Fornberg adaptive-skew derivative along one axis.

    For each evaluation point k in [k_lo, k_hi]:
      s = clip(k - 3, k_lo, k_hi - 6)        # stencil start, all 7 pts in [k_lo, k_hi]
      p = k - s                              # eval point's offset within stencil
      deriv[k] = sum FD6_COEFF[p, m] * arr[s + m]   for m in [0, 6]

    Outside [k_lo, k_hi]: returns 0 (caller should not use those values).
    Stencil never reads outside [k_lo, k_hi] -> safe even when ghosts are
    unreliable (e.g., k-direction with linear-extrapolated ghosts).

    Mirrors gilbm/metric_terms.h:71-95 k-direction adaptive Fornberg exactly.
    """
    if axis not in (0, 1):
        raise ValueError('fd6_axis_adaptive: axis must be 0 or 1')
    if axis == 0:
        arr_w = np.moveaxis(arr, 0, -1)
    else:
        arr_w = arr
    deriv = np.zeros_like(arr_w)
    if k_hi - k_lo < 6:
        # not enough fluid nodes for 7-point stencil; fall back to nothing
        # (caller must ensure fluid range >= 7 nodes for 6th-order)
        return np.moveaxis(deriv, -1, 0) if axis == 0 else deriv
    s_max = k_hi - 6
    for k in range(k_lo, k_hi + 1):
        s = k - 3
        if s < k_lo:
            s = k_lo
        elif s > s_max:
            s = s_max
        p = k - s
        for m in range(7):
            deriv[..., k] += FD6_COEFF[p, m] * arr_w[..., s + m]
    if axis == 0:
        deriv = np.moveaxis(deriv, -1, 0)
    return deriv


def compute_inverse_metric_2d_fornberg(y_2d, z_2d):
    """6th-order Fornberg version of compute_inverse_metric_2d.

    Mirrors solver: j-direction pure central (periodic ghost OK),
                    k-direction adaptive skew (wall ghost unreliable).

    j-direction range: j_lo=BFR, j_hi=NY6-1-BFR (fluid nodes; ghost reads OK)
    k-direction range: k_lo=BFR, k_hi=NZ6-1-BFR (fluid nodes; no ghost reads)

    Same return signature as compute_inverse_metric_2d: (dj_dy, dj_dz, dk_dy, dk_dz)
    each shape (NY6, NZ6). Ghost-row metric values are zeros (callers should
    only use the interior slice [BFR:BFR+NY, BFR:BFR+NZ]).
    """
    NY6, NZ6 = y_2d.shape
    j_lo, j_hi = BFR, NY6 - 1 - BFR
    k_lo, k_hi = BFR, NZ6 - 1 - BFR

    # j-direction: periodic ghost from build_grid_xyz is 6th-order valid
    y_j = fd6_axis_central(y_2d, j_lo, j_hi, axis=0)
    z_j = fd6_axis_central(z_2d, j_lo, j_hi, axis=0)

    # k-direction: wall ghost is linear extrap (only 2nd-order valid)
    # -> adaptive skew, never reads ghost
    y_k = fd6_axis_adaptive(y_2d, k_lo, k_hi, axis=1)
    z_k = fd6_axis_adaptive(z_2d, k_lo, k_hi, axis=1)

    det = y_j * z_k - y_k * z_j
    eps = 1e-30
    # Only count singularities inside the interior region (ghost rows are 0)
    interior_det = det[j_lo:j_hi+1, k_lo:k_hi+1]
    sing = int(np.sum(np.abs(interior_det) <= eps))
    if sing > 0:
        print('      WARN: Jacobian near-singular at {} interior grid points'.format(sing))
    inv_det = np.zeros_like(det)
    safe = np.abs(det) > eps
    inv_det[safe] = 1.0 / det[safe]

    dj_dy =  z_k * inv_det
    dj_dz = -y_k * inv_det
    dk_dy = -z_j * inv_det
    dk_dz =  y_j * inv_det
    return dj_dy, dj_dz, dk_dy, dk_dz


# ---------------------------------------------------------------
# Phase B: real dt_global computation (mirrors gilbm/precompute.h:78-115
# ComputeGlobalTimeStep). Replaces the dt_global=-1.0 placeholder that
# bypasses fileIO.h:658 Phase 5 drift check.
# ---------------------------------------------------------------
def compute_dt_global_gilbm(cfg, cfl, metric_order=6):
    """Mirror gilbm/precompute.h:78-115 ComputeGlobalTimeStep.

    dt_global = cfl / max|c~|, where max is over (eta, xi, zeta) and D3Q19 dirs:
      c~_eta(α)  = e_x[α] / dx           (spanwise, uniform)
                   max over α gives 1/dx (since |e_x| in {0, 1})
      c~_xi(α)   = xi_y · e_y[α] + xi_z · e_z[α]    (per α, per (j,k))
      c~_zeta(α) = zeta_y · e_y[α] + zeta_z · e_z[α] (per α, per (j,k))

    Returns (dt_global, max_component_label) where max_component_label is one
    of "eta", "xi (alpha=N)", or "zeta (alpha=N)" for audit.

    metric_order: 6 mirrors solver (recommended, drift check < 1e-6);
                  2 is legacy 2nd-order (drift may exceed 1e-6).
    """
    _, y_2d, z_2d = build_grid_xyz(cfg)
    if metric_order == 6:
        dj_dy, dj_dz, dk_dy, dk_dz = compute_inverse_metric_2d_fornberg(y_2d, z_2d)
    else:
        dj_dy, dj_dz, dk_dy, dk_dz = compute_inverse_metric_2d(y_2d, z_2d)

    sl = (slice(BFR, BFR + cfg.NY), slice(BFR, BFR + cfg.NZ))
    zeta_y, zeta_z = dk_dy[sl], dk_dz[sl]
    xi_y,   xi_z   = dj_dy[sl], dj_dz[sl]
    dx = LX / (cfg.NX - 1)

    # spanwise: c~_eta = max over α of |e_x[α]| / dx = 1/dx
    max_c = 1.0 / dx
    max_component = 'eta'

    # streamwise / wall-normal: scan D3Q19 non-zero shifts (α in [3, 18]).
    # α=0 is rest, α=1,2 are pure spanwise (handled by c_eta).
    for alpha in range(3, 19):
        ey, ez = E[alpha, 1], E[alpha, 2]
        c_xi_max   = float(np.abs(xi_y   * ey + xi_z   * ez).max())
        c_zeta_max = float(np.abs(zeta_y * ey + zeta_z * ez).max())
        if c_xi_max > max_c:
            max_c, max_component = c_xi_max, 'xi (alpha={})'.format(alpha)
        if c_zeta_max > max_c:
            max_c, max_component = c_zeta_max, 'zeta (alpha={})'.format(alpha)

    if max_c <= 0.0:
        raise ValueError('compute_dt_global_gilbm: max|c~|=0 (degenerate grid)')

    return cfl / max_c, max_component


def compute_velocity_gradient_3d(u, dx, dj_dy, dj_dz, dk_dy, dk_dz, cfg):
    """Compute (∂u/∂x, ∂u/∂y, ∂u/∂z) on interior for one velocity component.

    u : (NY6, NZ6, NX6) ghost-filled
    dx : LX/(NX-1) — uniform spanwise step
    dj_dy, ..., dk_dz : (NY6, NZ6) inverse metric

    Spanwise (i) is uniform → ∂u/∂x = (∂u/∂i)/dx
    Streamwise (y) and wall-normal (z) use chain rule:
      ∂u/∂y = (∂u/∂j)·(∂j/∂y) + (∂u/∂k)·(∂k/∂y)
      ∂u/∂z = (∂u/∂j)·(∂j/∂z) + (∂u/∂k)·(∂k/∂z)

    Returns three (NY, NZ, NX) interior arrays.
    """
    NZ6 = u.shape[1]
    du_di = np.zeros_like(u)
    du_dj = np.zeros_like(u)
    du_dk = np.zeros_like(u)

    # i-direction: 6th-order central (periodic ghost cells valid)
    du_di[:, :, 3:-3] = (
        -u[:, :, :-6] + 9.0*u[:, :, 1:-5] - 45.0*u[:, :, 2:-4]
        + 45.0*u[:, :, 4:-2] - 9.0*u[:, :, 5:-1] + u[:, :, 6:]
    ) / 60.0
    du_di[:, :, :3]  = du_di[:, :, 3:4]
    du_di[:, :, -3:] = du_di[:, :, -4:-3]

    # j-direction: 6th-order central (periodic ghost valid)
    du_dj[3:-3, :, :] = (
        -u[:-6, :, :] + 9.0*u[1:-5, :, :] - 45.0*u[2:-4, :, :]
        + 45.0*u[4:-2, :, :] - 9.0*u[5:-1, :, :] + u[6:, :, :]
    ) / 60.0
    du_dj[:3, :, :]  = du_dj[3:4, :, :]
    du_dj[-3:, :, :] = du_dj[-4:-3, :, :]

    # k-direction: 6th-order adaptive-skew Fornberg + 4th-order one-sided at walls
    kt = NZ6 - 1 - BFR

    # Central bulk (k = BFR+3 to kt-3): 6th-order central
    c_lo = BFR + 3
    c_hi = kt - 3
    du_dk[:, c_lo:c_hi+1, :] = (
        -u[:, c_lo-3:c_hi-2, :] + 9.0*u[:, c_lo-2:c_hi-1, :]
        - 45.0*u[:, c_lo-1:c_hi, :] + 45.0*u[:, c_lo+1:c_hi+2, :]
        - 9.0*u[:, c_lo+2:c_hi+3, :] + u[:, c_lo+3:c_hi+4, :]
    ) / 60.0

    # Near-bottom-wall: 6th-order skewed (Fornberg adaptive)
    for kk, p in [(BFR+1, 1), (BFR+2, 2)]:
        for m in range(7):
            du_dk[:, kk, :] += FD6_COEFF[p, m] * u[:, BFR + m, :]

    # Near-top-wall: 6th-order skewed (Fornberg adaptive)
    s_top = kt - 6
    for kk, p in [(kt-2, 4), (kt-1, 5)]:
        for m in range(7):
            du_dk[:, kk, :] += FD6_COEFF[p, m] * u[:, s_top + m, :]

    # Bottom wall (k=BFR): 4th-order forward one-sided (u_wall=0)
    du_dk[:, BFR, :] = (
        48.0 * u[:, BFR+1, :] - 36.0 * u[:, BFR+2, :]
       +16.0 * u[:, BFR+3, :] -  3.0 * u[:, BFR+4, :]
    ) / 12.0

    # Top wall (k=kt): 4th-order backward one-sided (u_wall=0)
    du_dk[:, kt, :] = -(
        48.0 * u[:, kt-1, :] - 36.0 * u[:, kt-2, :]
       +16.0 * u[:, kt-3, :] -  3.0 * u[:, kt-4, :]
    ) / 12.0

    # Ghost rows (not in interior crop)
    du_dk[:, :BFR, :]  = du_dk[:, BFR:BFR+1, :]
    du_dk[:, kt+1:, :] = du_dk[:, kt:kt+1, :]

    sl = (slice(BFR, BFR+cfg.NY), slice(BFR, BFR+cfg.NZ), slice(BFR, BFR+cfg.NX))
    du_di_int = du_di[sl]
    du_dj_int = du_dj[sl]
    du_dk_int = du_dk[sl]

    metric_sl = (slice(BFR, BFR+cfg.NY), slice(BFR, BFR+cfg.NZ))
    dj_dy_int = dj_dy[metric_sl][:, :, np.newaxis]
    dj_dz_int = dj_dz[metric_sl][:, :, np.newaxis]
    dk_dy_int = dk_dy[metric_sl][:, :, np.newaxis]
    dk_dz_int = dk_dz[metric_sl][:, :, np.newaxis]

    du_dx = du_di_int / dx
    du_dy = du_dj_int * dj_dy_int + du_dk_int * dk_dy_int
    du_dz = du_dj_int * dj_dz_int + du_dk_int * dk_dz_int
    return du_dx, du_dy, du_dz


def chapman_enskog_fneq_q(rho_int, grad, q, ce_coeff):
    """Reconstruct f_neq_q from velocity gradient tensor on interior nodes.

    grad : 9-tuple in (α, β) order (du_dx, du_dy, du_dz, dv_dx, ..., dw_dz),
           each (NY, NZ, NX). The tensor [3·c_qα·c_qβ − δ_αβ] is symmetric in
           αβ so the formula contracts symmetrized gradient pairs.
    """
    cqx = E[q, 0]; cqy = E[q, 1]; cqz = E[q, 2]
    Txx = 3.0*cqx*cqx - 1.0
    Tyy = 3.0*cqy*cqy - 1.0
    Tzz = 3.0*cqz*cqz - 1.0
    Txy = 3.0*cqx*cqy
    Txz = 3.0*cqx*cqz
    Tyz = 3.0*cqy*cqz
    dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz = grad
    contraction = (
        Txx * dudx + Tyy * dvdy + Tzz * dwdz
        + Txy * (dudy + dvdx)
        + Txz * (dudz + dwdx)
        + Tyz * (dvdz + dwdy)
    )
    return W[q] * rho_int * ce_coeff * contraction


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--auto', action='store_true',
                   help='fully automatic: detect phase2 origin checkpoint and phase1 oldgrid/newgrid files')
    p.add_argument('--old-dir', default=None,
                   help='old checkpoint directory (auto-detected from phase2_generatecheckpoint/step_*_origin* if omitted)')
    p.add_argument('--step', type=int, default=1,
                   help='new checkpoint step number written into metadata (default: 1; '
                        'must be > 0 for solver restart tripwire at main.cu:692)')
    p.add_argument('--output-root', default='restart/checkpoint',
                   help='root directory for chain-compatible output step_%%08d (default: %(default)s)')
    p.add_argument('--new-dir', default=None,
                   help='advanced override for output checkpoint directory; default is output-root/step_%%08d')
    p.add_argument('--fneq-scale', type=float, default=1.0,
                   help='scale factor applied to interpolated f_neq (legacy mode only; default: %(default)s)')
    p.add_argument('--fneq-mode', choices=['zero', 'interp', 'chapman-enskog'], default='chapman-enskog',
                   help='f_neq reconstruction strategy. '
                        '"chapman-enskog" = rebuild f_neq on NEW grid from velocity '
                        'gradients via CE expansion (default). '
                        '"zero" = pure equilibrium f=f_eq for stability A/B testing. '
                        '"interp" = legacy linear interp of f_neq (loses gradient info).')
    p.add_argument('--project-velocity', choices=['poisson', 'dg-exact', 'div-exact', 'none'],
                   default='poisson',
                   help='Velocity projection after interpolation. "poisson" = Helmholtz-Hodge '
                        'div(grad(phi))=div(u) to enforce solenoidal constraint (default). '
                        '"dg-exact" = direct solve of the exact CD2 D*G operator '
                        'used by the final divergence check. '
                        '"div-exact" = exact minimum-norm velocity correction '
                        'that directly zeroes the final CD2 divergence diagnostic. '
                        '"none" = skip projection (legacy/debug).')
    p.add_argument('--projection-max-outer', type=int, default=80,
                   help='maximum outer Richardson iterations for Poisson projection '
                        '(default: %(default)s; one-time preprocessing cost)')
    p.add_argument('--projection-div-tol', type=float, default=1e-6,
                   help='target RMS divergence tolerance for Poisson projection '
                        '(default: %(default)s)')
    p.add_argument('--interp-mode', choices=['comp', 'phys'], default='phys',
                   help='Macro field (rho, u) interpolation mode. "phys" = physical-space '
                        'with 2D cell search + bilinear inverse (default; correct for GAMMA changes). '
                        '"comp" = legacy computational-space (j,k,i) index remap '
                        '(causes physical mis-placement when GAMMA differs).')
    p.add_argument('--interp-order', type=int, choices=[2, 6], default=6,
                   help='Interpolation order for macro fields in physical-space mode. '
                        '6 = 7-point Lagrange tensor product O(h^6) (default). '
                        '2 = bilinear O(h^2) (legacy).')
    p.add_argument('--metric-order', type=int, choices=[2, 6], default=6,
                   help='Order of FD for inverse metric used in CE f_neq reconstruction. '
                        '6 mirrors solver (j-central + k-adaptive Fornberg, '
                        'gilbm/metric_terms.h). 2 is legacy 2nd-order central.')
    p.add_argument('--cfl', type=float, default=None,
                   help='CFL lambda override for dt_global computation. Production runs should '
                        'omit this and read CFL from --variables-h/variables.h.')
    p.add_argument('--skip-drift-check', action='store_true',
                   help='Write dt_global=-1.0 to bypass solver Phase 5 drift check '
                        '(fileIO.h:658). Debug only; production runs should leave this off.')
    p.add_argument('--niu', type=float, default=None,
                   help='kinematic viscosity for Chapman-Enskog mode (auto-read from variables.h '
                        'as Uref/Re when omitted)')
    p.add_argument('--solver-grid-dat', default=None,
                   help='path to the Tecplot grid .dat that main.cu will read at runtime. '
                        'When omitted with variables.h available, this is derived from '
                        'GRID_DAT_DIR/GRID_DAT_REF and NEW NY/NZ/GAMMA/ALPHA. The file is '
                        'used only for a strict preflight coordinate comparison.')
    p.add_argument('--no-generate-solver-grid', dest='generate_solver_grid',
                   action='store_false',
                   help='do not materialize the derived solver runtime grid before comparison. '
                        'By default, if the derived solver grid is missing, this script runs '
                        'the same J_Frohlich/grid_zeta_tool.py --auto entry that main.cu uses.')
    p.add_argument('--grid-match-tol', type=float, default=0.0,
                   help='absolute coordinate tolerance for --new-grid-dat vs solver grid '
                        '(default: 0.0, exact parsed float equality)')
    p.add_argument('--allow-solver-grid-mismatch', action='store_true',
                   help='warn instead of aborting when --new-grid-dat differs from the solver '
                        'runtime grid. Debug only; production should leave this off.')
    p.add_argument('--dry-run', action='store_true',
                   help='validate configuration and output path, then exit before reading/writing checkpoint data')
    p.set_defaults(generate_solver_grid=True)

    g_old = p.add_argument_group('OLD grid (auto-detected from metadata when possible)')
    g_old.add_argument('--old-nx', type=int, default=None)
    g_old.add_argument('--old-ny', type=int, default=None)
    g_old.add_argument('--old-nz', type=int, default=None)
    g_old.add_argument('--old-jp', type=int, default=None)
    g_old.add_argument('--old-gamma', type=float, default=None)
    g_old.add_argument('--old-alpha', type=float, default=None)
    g_old.add_argument('--old-grid-dat', default=None,
                       help='path to old Tecplot grid .dat file')

    g_new = p.add_argument_group('NEW grid (auto-read from variables.h in project mode)')
    g_new.add_argument('--new-nx', type=int, default=None)
    g_new.add_argument('--new-ny', type=int, default=None)
    g_new.add_argument('--new-nz', type=int, default=None)
    g_new.add_argument('--new-jp', type=int, default=None)
    g_new.add_argument('--new-gamma', type=float, default=None)
    g_new.add_argument('--new-alpha', type=float, default=None)
    g_new.add_argument('--new-grid-dat', default=None,
                       help='path to new Tecplot grid .dat file')
    g_new.add_argument('--variables-h', default=None,
                       help='path to variables.h used for NEW grid constants, CFL, and niu. '
                            'Auto-detected from project root or phase2_generatecheckpoint/variables.h.')

    args = p.parse_args()

    global OLD, NEW

    args.variables_h = resolve_variables_h_arg(args.variables_h)

    if args.auto:
        import re as _re
        global _AUTO_MODE
        _AUTO_MODE = True

        vh_path = args.variables_h or find_variables_h()
        if not vh_path:
            sys.exit('FATAL: --auto requires variables.h (not found)')
        vh_path = resolve_variables_h_arg(vh_path)
        args.variables_h = vh_path
        vh = parse_variables_h(vh_path)
        str_defs = parse_string_defines(vh_path)

        # ── GRID PIPELINE REGULATION ──────────────────────────────────
        # Phase 2 (initial-data-point checkpoint 生成) 的觸發條件：
        #   phase1_generategrid/ 內必須同時有 oldgrid_*.dat 與 newgrid_*.dat
        #   匹配 variables.h 的 NY/NZ/ALPHA。
        # 只有這條路徑會被走；不再 fallback 到 J_Frohlich/。
        # 若任一條件不符 → 印訊息退出，由使用者人工準備後再觸發。
        # ──────────────────────────────────────────────────────────────
        vh_dir = os.path.dirname(os.path.abspath(vh_path))
        phase1_grid_dir = os.path.join(vh_dir, 'phase1_generategrid')
        if not os.path.isdir(phase1_grid_dir):
            sys.exit(
                'FATAL: --auto: phase1_generategrid/ 不存在於 {}.\n'
                '  Phase 2 觸發條件: phase1_generategrid/ 內需有 oldgrid_*.dat + newgrid_*.dat.\n'
                '  J_Frohlich/ 屬於 main pipeline，不在 phase2 路徑上 (regulation).'
                .format(vh_dir))
        grid_dir = phase1_grid_dir

        NY_vh = int(vh['NY'])
        NZ_vh = int(vh['NZ'])
        ALPHA_vh = vh.get('ALPHA', 0.5)

        restart_dir = os.path.join(vh_dir, 'restart')
        phase2_dir = os.path.join(vh_dir, 'phase2_generatecheckpoint')
        ckpt_dir = os.path.join(restart_dir, 'checkpoint')
        has_normal = False
        if os.path.isdir(ckpt_dir):
            for name in sorted(os.listdir(ckpt_dir)):
                if name.startswith('step_') and '_origin' not in name and not _is_origin_dir_name(name):
                    if os.path.isfile(os.path.join(ckpt_dir, name, 'metadata.dat')):
                        has_normal = True
                        break
        if has_normal:
            print('[auto] Non-origin checkpoint in restart/checkpoint/ — interpolation not needed')
            sys.exit(0)

        origin = resolve_old_dir(args.old_dir) if args.old_dir else find_origin_checkpoint(phase2_dir)
        if not origin:
            sys.exit('FATAL: --auto: no origin checkpoint found '
                     '(searched step_*_origin* and oldcheckpoint_* in phase2/restart)')
        print('[auto] Origin checkpoint: {}'.format(origin))
        print('[auto] Grid directory: {}'.format(os.path.abspath(grid_dir)))

        dim_tag = '_I{}_J{}'.format(NY_vh, NZ_vh)
        old_grid = old_fname = old_gamma = old_alpha = None
        new_grid = new_fname = new_alpha = None
        resolve_bases = (grid_dir, vh_dir, project_root())

        if args.old_grid_dat:
            old_grid = resolve_existing_file(args.old_grid_dat, '--old-grid-dat',
                                             base_dirs=resolve_bases)
            old_fname = os.path.basename(old_grid)
            inferred_gamma, inferred_alpha = infer_old_grid_params(old_grid)
            old_gamma = args.old_gamma if args.old_gamma is not None else inferred_gamma
            old_alpha = args.old_alpha if args.old_alpha is not None else inferred_alpha
            if old_gamma is None or old_alpha is None:
                sys.exit('FATAL: --auto with explicit --old-grid-dat requires filename *_g{G}_a{A}.dat '
                         'or explicit --old-gamma/--old-alpha')
        if args.new_grid_dat:
            new_grid = resolve_existing_file(args.new_grid_dat, '--new-grid-dat',
                                             base_dirs=resolve_bases)
            new_fname = os.path.basename(new_grid)
            new_old_gamma, _ = infer_old_grid_params(new_grid)
            if new_old_gamma is not None and not new_fname.startswith('newgrid_'):
                sys.exit('FATAL: --new-grid-dat appears to be an OLD uniform-gamma grid: {}'.format(
                    new_fname))
            inferred_alpha = infer_new_grid_alpha(new_grid)
            new_alpha = ALPHA_vh
            if inferred_alpha is not None and abs(float(inferred_alpha) - float(ALPHA_vh)) > 1e-12:
                sys.exit('FATAL: --new-grid-dat alpha {} does not match variables.h ALPHA {}'.format(
                    inferred_alpha, ALPHA_vh))

        if not old_grid or not new_grid:
            # Scan grid files only for the side not explicitly supplied by run.sh.
            old_candidates = []
            new_candidates = []

            for f in sorted(os.listdir(grid_dir)):
                if not f.endswith('.dat') or dim_tag not in f:
                    continue
                full = os.path.join(grid_dir, f)
                if f.startswith('oldgrid_'):
                    gamma, alpha = infer_old_grid_params(f)
                    if gamma is None or alpha is None:
                        continue
                    if abs(float(alpha) - float(ALPHA_vh)) > 1e-12:
                        continue
                    old_candidates.append((full, f, gamma, alpha))
                elif f.startswith('newgrid_'):
                    alpha = infer_new_grid_alpha(f)
                    if alpha is not None and abs(float(alpha) - float(ALPHA_vh)) > 1e-12:
                        continue
                    new_candidates.append((full, f, ALPHA_vh if alpha is None else alpha))

            if not old_grid:
                if len(old_candidates) == 0:
                    sys.exit('FATAL: --auto: no OLD grid named oldgrid_*.dat containing {} '
                             'and *_g{{G}}_a{{A}}.dat (ALPHA={}) in {}'.format(
                                 dim_tag, ALPHA_vh, grid_dir))
                if len(old_candidates) > 1:
                    sys.exit('FATAL: --auto: ambiguous OLD grid candidates ({}): {}'.format(
                        len(old_candidates), ', '.join(c[1] for c in old_candidates)))
                old_grid, old_fname, old_gamma, old_alpha = old_candidates[0]
            if not new_grid:
                if len(new_candidates) == 0:
                    sys.exit('FATAL: --auto: no NEW grid named newgrid_*.dat containing {} '
                             '(ALPHA={}) in {}. Place newgrid_*.dat there or pass '
                             '--new-grid-dat.'.format(dim_tag, ALPHA_vh, grid_dir))
                if len(new_candidates) > 1:
                    sys.exit('FATAL: --auto: ambiguous NEW grid candidates ({}): {}'.format(
                        len(new_candidates), ', '.join(c[1] for c in new_candidates)))
                new_grid, new_fname, new_alpha = new_candidates[0]

        print('[auto] OLD grid (uniform gamma={}, alpha={}): {}'.format(old_gamma, old_alpha, old_fname))
        print('[auto] NEW grid (variable gamma, alpha={}): {}'.format(new_alpha, new_fname))

        args.old_dir = origin
        args.old_gamma = old_gamma
        args.old_alpha = old_alpha
        args.old_grid_dat = old_grid
        args.new_grid_dat = new_grid
        args.variables_h = vh_path

    if args.cfl is None:
        vh_for_cfl = args.variables_h or find_variables_h()
        if not vh_for_cfl:
            sys.exit('FATAL: CFL must come from variables.h or explicit --cfl. '
                     'Pass --variables-h /path/to/variables.h for production checkpoint rebuilds.')
        vh_for_cfl = resolve_variables_h_arg(vh_for_cfl)
        args.variables_h = vh_for_cfl
        cfl_from_vh = parse_variables_h(vh_for_cfl).get('CFL')
        if cfl_from_vh is None:
            sys.exit('FATAL: {} has no parseable #define CFL; pass --cfl only for controlled tests.'.format(
                vh_for_cfl))
        args.cfl = float(cfl_from_vh)
        args.cfl_source = 'variables.h'
    else:
        args.cfl_source = 'cli'
        if args.variables_h is None:
            detected_vh = find_variables_h()
            if detected_vh:
                args.variables_h = resolve_variables_h_arg(detected_vh)

    args.old_dir = resolve_old_dir(args.old_dir)

    print()
    OLD = build_old_config(args)
    NEW = build_new_config(args)

    t0 = time.time()
    print('=' * 72)
    print('LBM checkpoint interpolator: {}x{}x{} (jp={}) -> {}x{}x{} (jp={})'.format(
        OLD.NX, OLD.NY, OLD.NZ, OLD.JP, NEW.NX, NEW.NY, NEW.NZ, NEW.JP))
    print('=' * 72)
    print('OLD: NX={} NY={} NZ={} jp={} GAMMA={} grid={}'.format(
        OLD.NX, OLD.NY, OLD.NZ, OLD.JP, OLD.GAMMA, OLD.GRID_DAT))
    print('NEW: NX={} NY={} NZ={} jp={} GAMMA={} grid={}'.format(
        NEW.NX, NEW.NY, NEW.NZ, NEW.JP, NEW.GAMMA, NEW.GRID_DAT))
    print('Domain: LX={} LY={} LZ={} H_HILL={}'.format(LX, LY, LZ, H_HILL))
    print()

    solver_grid_dat = None
    if args.solver_grid_dat:
        solver_grid_dat = resolve_existing_file(
            args.solver_grid_dat, '--solver-grid-dat',
            base_dirs=(project_root(),
                       os.path.dirname(args.variables_h) if args.variables_h else None))
    else:
        solver_grid_dat = derive_solver_grid_dat(args.variables_h, NEW)
    solver_grid_dat, solver_grid_generated = maybe_generate_solver_grid(
        solver_grid_dat, args.variables_h, enabled=args.generate_solver_grid)
    args.solver_grid_dat = solver_grid_dat
    args.solver_grid_generated = solver_grid_generated

    print('--- Validating NEW grid against solver runtime grid ---')
    args.solver_grid_match_info = validate_solver_grid_match(
        NEW.GRID_DAT, solver_grid_dat, NEW,
        tol=args.grid_match_tol,
        fatal=not args.allow_solver_grid_mismatch)
    print()

    out_dir = resolve_output_dir(args.output_root, args.step, args.new_dir)
    writing_dir = out_dir + '.WRITING'
    if os.path.exists(out_dir):
        sys.exit('FATAL: {} already exists; refusing to overwrite'.format(out_dir))
    if os.path.exists(writing_dir):
        sys.exit('FATAL: {} already exists; remove it after verifying it is stale'.format(writing_dir))
    if args.new_dir is None:
        print('Output directory: {} (from --output-root + --step)'.format(out_dir))
    else:
        print('Output directory: {} (--new-dir override)'.format(out_dir))
    print()
    if args.dry_run:
        print('Dry run complete: configuration is valid and no output was written.')
        return

    # ---- Step 1: parse old metadata ----
    print('[1/8] Reading old metadata: {}/metadata.dat'.format(args.old_dir))
    meta_path = os.path.join(args.old_dir, 'metadata.dat')
    if not os.path.exists(meta_path):
        sys.exit('FATAL: {} not found'.format(meta_path))
    meta_old = parse_metadata(meta_path)
    expected_dims = '{},{},{}'.format(OLD.NX6, OLD.NYD6, OLD.NZ6)
    if meta_old.get('grid_dims') != expected_dims:
        sys.exit('FATAL: grid_dims mismatch: file={}, expected={}'.format(
            meta_old.get('grid_dims'), expected_dims))
    if int(meta_old.get('mpi_rank_count', 0)) != OLD.JP:
        sys.exit('FATAL: mpi_rank_count mismatch: file={}, expected={}'.format(
            meta_old.get('mpi_rank_count'), OLD.JP))
    Force_value = float(meta_old['Force'])
    dt_global_old = float(meta_old['dt_global'])
    half_Fdt_old = 0.5 * dt_global_old * Force_value
    print('      grid_dims={} mpi_rank_count={} step={} FTT={} Force={:.6e}'.format(
        meta_old['grid_dims'], meta_old['mpi_rank_count'],
        meta_old['step'], meta_old['FTT'], Force_value))
    print('      source dt_global={:.15e} half_Fdt={:.15e}'.format(
        dt_global_old, half_Fdt_old))

    # ---- Step 2: build OLD grid ----
    print('[2/8] Building OLD grid coordinates')
    _, y2d_old, z2d_old = build_grid_xyz(OLD)
    y_int = y2d_old[BFR:BFR+OLD.NY, BFR]
    z_int = z2d_old[BFR, BFR:BFR+OLD.NZ]
    print('      Y interior range [{:.4f}, {:.4f}] (expect [0, {:.1f}])'.format(
        y_int.min(), y_int.max(), LY))
    print('      Z interior range [{:.4f}, {:.4f}] (expect [hill, {:.3f}])'.format(
        z_int.min(), z_int.max(), LZ))

    # ---- Step 3: read checkpoint, compute macros ----
    print('[3/8] Reading {} f-files ({} ranks x 19 directions)'.format(OLD.JP*19, OLD.JP))
    rho_g = np.zeros((OLD.NY6, OLD.NZ6, OLD.NX6), dtype=np.float64)
    momx_g = np.zeros_like(rho_g)
    momy_g = np.zeros_like(rho_g)
    momz_g = np.zeros_like(rho_g)

    for q in range(19):
        per_rank = []
        for r in range(OLD.JP):
            path = os.path.join(args.old_dir, 'f{:02d}_{}.bin'.format(q, r))
            per_rank.append(read_rank_bin(path, OLD))
        f_g = stitch_y(per_rank, OLD)
        rho_g  += f_g
        if E[q, 0] != 0:
            momx_g += E[q, 0] * f_g
        if E[q, 1] != 0:
            momy_g += E[q, 1] * f_g
        if E[q, 2] != 0:
            momz_g += E[q, 2] * f_g
        print('      f{:02d}: stitched {} ranks'.format(q, OLD.JP), flush=True)

    rho_safe = np.where(rho_g > 1e-12, rho_g, 1.0)
    ux_g = momx_g / rho_safe
    # Match the solver macroscopic velocity definition under Guo forcing:
    # rho * code_v = sum_i(f_i * e_i,y) + 0.5 * dt * Force.
    uy_g = (momy_g + half_Fdt_old) / rho_safe
    uz_g = momz_g / rho_safe
    del momx_g, momy_g, momz_g, rho_safe

    interior_slice = (slice(BFR, BFR+OLD.NY), slice(BFR, BFR+OLD.NZ), slice(BFR, BFR+OLD.NX))
    print('      OLD interior rho = [{:.6f}, {:.6f}], mean = {:.6f}'.format(
        rho_g[interior_slice].min(), rho_g[interior_slice].max(),
        rho_g[interior_slice].mean()))
    print('      OLD interior max|u| = {:.6e}, max|v| = {:.6e}, max|w| = {:.6e}'.format(
        np.abs(ux_g[interior_slice]).max(),
        np.abs(uy_g[interior_slice]).max(),
        np.abs(uz_g[interior_slice]).max()))
    Ub_old = compute_Ub(uy_g, z2d_old, OLD)
    print('      OLD Ub (j=BFR cross-section) = {:.15e}'.format(Ub_old))

    # Cross-check stored rho against sum(f).  In a running LBM with mass
    # correction (checkrho.dat), rho is adjusted independently of f each step,
    # so rho_file != sum(f) by O(1e-4) is normal.  We use sum(f) as the
    # authoritative rho for feq/fneq computation (it's self-consistent with f).
    rho_file_g = stitch_y([
        read_rank_bin(os.path.join(args.old_dir, 'rho_{}.bin'.format(r)), OLD)
        for r in range(OLD.JP)
    ], OLD)
    rho_src_diff = float(np.max(np.abs(rho_file_g - rho_g)))
    print('      OLD max |rho_file - sum(f)| = {:.3e}'.format(rho_src_diff))
    if rho_src_diff > 1e-2:
        sys.exit('FATAL: source checkpoint rho vs sum(f) diff {:.3e} > 1e-2 (data corruption?)'.format(rho_src_diff))
    elif rho_src_diff > 1e-6:
        print('      WARN: rho_file != sum(f) by {:.3e} (expected from LBM mass correction)'.format(rho_src_diff))
        print('            Using sum(f) as authoritative rho for feq/fneq computation')
    del rho_file_g

    # ---- Step 4: build NEW grid ----
    print('[4/8] Building NEW grid coordinates')
    _, y2d_new, z2d_new = build_grid_xyz(NEW)
    y_int_new = y2d_new[BFR:BFR+NEW.NY, BFR]
    z_int_new = z2d_new[BFR, BFR:BFR+NEW.NZ]
    print('      Y interior range [{:.4f}, {:.4f}]'.format(y_int_new.min(), y_int_new.max()))
    print('      Z interior range [{:.4f}, {:.4f}]'.format(z_int_new.min(), z_int_new.max()))

    # ---- Step 5: interpolate macros ----
    # Ensure OLD ghosts are filled before interp reads them (interpolate_phys_3d
    # may read field_old[NX+3] for spanwise periodic wrap; real checkpoints
    # already have ghosts but defensive fill makes behavior identical to
    # synthetic test fields).
    fill_ghost(rho_g, OLD)
    fill_ghost(ux_g,  OLD)
    fill_ghost(uy_g,  OLD)
    fill_ghost(uz_g,  OLD)

    if args.interp_mode == 'phys':
        # OLD grid coords needed for physical-space inverse mapping
        _, y2d_old, z2d_old = build_grid_xyz(OLD)
        interp_order = args.interp_order
        interp_label = ('7-point Lagrange O(h^6) + cubic wall ghost extrapolation'
                        if interp_order == 6 else 'bilinear O(h^2)')
        print('[5/8] Interpolating macros (rho, ux, uy, uz) to NEW grid in PHYSICAL space')
        print('      interpolation order: {} ({})'.format(interp_order, interp_label))
        t = time.time()
        mapping = precompute_phys_mapping_2d(y2d_old, z2d_old, y2d_new, z2d_new, OLD, NEW)
        print('      mapping cache build: {:.1f}s'.format(time.time() - t))

        if interp_order == 6:
            interp_fn = interpolate_lagrange7_3d_with_mapping
        else:
            interp_fn = interpolate_phys_3d_with_mapping

        t = time.time(); rho_new = interp_fn(rho_g, mapping)
        print('      rho:  {:.1f}s'.format(time.time() - t))
        t = time.time(); ux_new  = interp_fn(ux_g,  mapping)
        print('      ux:   {:.1f}s'.format(time.time() - t))
        t = time.time(); uy_new  = interp_fn(uy_g,  mapping)
        print('      uy:   {:.1f}s'.format(time.time() - t))
        t = time.time(); uz_new  = interp_fn(uz_g,  mapping)
        print('      uz:   {:.1f}s'.format(time.time() - t))
    else:
        if OLD.GAMMA != NEW.GAMMA:
            print('      WARN: GAMMA differs (OLD={}, NEW={}) but --interp-mode=comp:'.format(
                OLD.GAMMA, NEW.GAMMA))
            print('            same (j,k,i) fraction maps to different physical z-heights.')
            print('            This causes turbulence structure mis-placement (use --interp-mode phys).')
        print('[5/8] Interpolating macros (rho, ux, uy, uz) to NEW grid in COMPUTATIONAL space (legacy)')
        t = time.time()
        rho_new = interpolate_comp_3d(rho_g, OLD, NEW)
        print('      rho:  {:.1f}s'.format(time.time() - t))
        t = time.time()
        ux_new = interpolate_comp_3d(ux_g, OLD, NEW)
        print('      ux:   {:.1f}s'.format(time.time() - t))
        t = time.time()
        uy_new = interpolate_comp_3d(uy_g, OLD, NEW)
        print('      uy:   {:.1f}s'.format(time.time() - t))
        t = time.time()
        uz_new = interpolate_comp_3d(uz_g, OLD, NEW)
        print('      uz:   {:.1f}s'.format(time.time() - t))

    print('      Applying wall velocity constraint: u=v=w=0 (preserve rho)')
    wall_residual_max, wall_rho_delta_max = clamp_wall_macros(
        rho_new, ux_new, uy_new, uz_new, NEW)
    print('      max |u_wall| before clamp = {:.3e}'.format(wall_residual_max))
    print('      max |rho_wall - 1| preserved = {:.3e}'.format(wall_rho_delta_max))

    rho_modify, mean_before, mean_after = apply_rho_mass_correction(rho_new, NEW)
    print('      rho mass correction (full domain): rho_modify = {:.6e}'.format(rho_modify))
    print('      rho full-domain mean: {:.15f} -> {:.15f}'.format(mean_before, mean_after))

    print('      Enforcing periodic duplicate nodes and filling ghost cells')
    for arr in (rho_new, ux_new, uy_new, uz_new):
        enforce_periodic_physical_duplicates(arr, NEW)
    fill_ghost(rho_new, NEW)
    fill_ghost(ux_new, NEW)
    fill_ghost(uy_new, NEW)
    fill_ghost(uz_new, NEW)

    new_int = (slice(BFR, BFR+NEW.NY), slice(BFR, BFR+NEW.NZ), slice(BFR, BFR+NEW.NX))
    print('      NEW interior rho = [{:.6f}, {:.6f}], mean = {:.6f}'.format(
        rho_new[new_int].min(), rho_new[new_int].max(), rho_new[new_int].mean()))
    print('      NEW interior max|u| = {:.6e}, max|v| = {:.6e}, max|w| = {:.6e}'.format(
        np.abs(ux_new[new_int]).max(),
        np.abs(uy_new[new_int]).max(),
        np.abs(uz_new[new_int]).max()))

    # Initialization correction order:
    #   wall-clamp -> rho-correct -> ghost-fill -> Ub-scale -> ghost-refill
    #   -> Poisson-with-wall-constraint -> final ghost-refill -> div-check -> f_eq
    # Poisson is the final operation that modifies the interior velocity.  This
    # prevents wall re-clamping or one-component Ub scaling from reintroducing
    # divergence after the solenoidal projection.

    # Ub conservation correction: scale streamwise velocity to match OLD Ub.
    # This is intentionally before the projection because uy-only scaling is
    # not a divergence-free operation on the curvilinear grid.
    ub_scale, Ub_new_before, Ub_new_after = apply_Ub_correction(
        Ub_old, uy_new, z2d_new, NEW)

    print('      Periodic duplicate enforcement and ghost-cell fill before projection')
    for arr in (rho_new, ux_new, uy_new, uz_new):
        enforce_periodic_physical_duplicates(arr, NEW)
    fill_ghost(rho_new, NEW)
    fill_ghost(ux_new, NEW)
    fill_ghost(uy_new, NEW)
    fill_ghost(uz_new, NEW)

    # ---- Poisson velocity projection (solenoidal correction) ----
    proj_info = None
    divergence_diagnostic_fn = None
    if args.project_velocity in ('poisson', 'dg-exact', 'div-exact'):
        print('      --- Velocity projection ---')
        from poisson_projection import (
            poisson_project, poisson_project_dg_exact,
            velocity_project_div_exact,
            PoissonProjectionError, divergence_diagnostic)
        divergence_diagnostic_fn = divergence_diagnostic
        try:
            if args.project_velocity == 'dg-exact':
                ux_new, uy_new, uz_new, proj_info = poisson_project_dg_exact(
                    ux_new, uy_new, uz_new, NEW, y2d_new, z2d_new,
                    max_outer=args.projection_max_outer,
                    div_tol=args.projection_div_tol)
            elif args.project_velocity == 'div-exact':
                ux_new, uy_new, uz_new, proj_info = velocity_project_div_exact(
                    ux_new, uy_new, uz_new, NEW, y2d_new, z2d_new,
                    max_outer=args.projection_max_outer,
                    div_tol=args.projection_div_tol)
            else:
                ux_new, uy_new, uz_new, proj_info = poisson_project(
                    ux_new, uy_new, uz_new, NEW, y2d_new, z2d_new,
                    max_outer=args.projection_max_outer,
                    div_tol=args.projection_div_tol)
        except PoissonProjectionError as e:
            sys.exit('FATAL: Poisson projection failed: {}\n'
                     '  Velocity field was NOT modified.\n'
                     '  Cannot produce a solenoidal checkpoint — aborting.'.format(e))
        print('      wall velocity constrained inside projection')
    else:
        print('      velocity projection: SKIPPED (--project-velocity none)')

    print('      Final periodic duplicate enforcement and ghost-cell fill')
    for arr in (rho_new, ux_new, uy_new, uz_new):
        enforce_periodic_physical_duplicates(arr, NEW)
    fill_ghost(rho_new, NEW)
    fill_ghost(ux_new, NEW)
    fill_ghost(uy_new, NEW)
    fill_ghost(uz_new, NEW)

    Ub_final = compute_Ub(uy_new, z2d_new, NEW)
    print('      Final Ub before f_eq = {:.15e} (target {:.15e}, residual {:.3e})'.format(
        Ub_final, Ub_old, abs(Ub_final - Ub_old)))

    # ---- Step 6: NEW-grid dt_global for CE coefficient and metadata ----
    print('[6/8] Computing NEW-grid dt_global')
    dt_real, dt_max_component = compute_dt_global_gilbm(
        NEW, cfl=args.cfl, metric_order=args.metric_order)
    dx_new = LX / (NEW.NX - 1)
    print('      dt_global_new = {:.6e}  (CFL={}, metric_order={}, dx={:.6e})'.format(
        dt_real, args.cfl, args.metric_order, dx_new))
    print('      dt limited by {} component'.format(dt_max_component))
    if dt_max_component == 'eta':
        print('      (eta dominates: spanwise dx is the tightest constraint; '
              'expected if y/z metric c~ < 1/dx; not an error)')
    if args.skip_drift_check:
        dt_for_meta = '-1.0'
        print('      WARN: --skip-drift-check; metadata dt_global=-1.0, '
              'but CE still uses dt_global_new above.')
    else:
        dt_for_meta = '{:.15e}'.format(dt_real)

    final_div_rms = None
    final_div_max = None
    if divergence_diagnostic_fn is None:
        try:
            from poisson_projection import divergence_diagnostic as divergence_diagnostic_fn
        except Exception:
            divergence_diagnostic_fn = None
    if divergence_diagnostic_fn is not None:
        div_rms, div_max = divergence_diagnostic_fn(
            ux_new, uy_new, uz_new, NEW, y2d_new, z2d_new)
        final_div_rms = div_rms
        final_div_max = div_max
        print('      divergence check: rms = {:.6e}, max|div(u)| = {:.6e}'.format(
            div_rms, div_max))
    else:
        j_mid = slice(BFR + 1, BFR + NEW.NY - 1)
        k_mid = slice(BFR + 1, BFR + NEW.NZ - 1)
        i_mid = slice(BFR + 1, BFR + NEW.NX - 1)
        dudx = (ux_new[j_mid, k_mid, BFR + 2:BFR + NEW.NX]
                - ux_new[j_mid, k_mid, BFR:BFR + NEW.NX - 2]) / (2.0 * dx_new)
        dy = (y2d_new[BFR + 2:BFR + NEW.NY, k_mid]
              - y2d_new[BFR:BFR + NEW.NY - 2, k_mid])[:, :, np.newaxis]
        dudy = ((uy_new[BFR + 2:BFR + NEW.NY, k_mid, i_mid]
                 - uy_new[BFR:BFR + NEW.NY - 2, k_mid, i_mid]) / dy)
        dz = (z2d_new[j_mid, BFR + 2:BFR + NEW.NZ]
              - z2d_new[j_mid, BFR:BFR + NEW.NZ - 2])[:, :, np.newaxis]
        dwdz = ((uz_new[j_mid, BFR + 2:BFR + NEW.NZ, i_mid]
                 - uz_new[j_mid, BFR:BFR + NEW.NZ - 2, i_mid]) / dz)
        div_max = float(np.max(np.abs(dudx + dudy + dwdz)))
        final_div_max = div_max
        print('      divergence check (finite-difference fallback): max|div(u)| = {:.6e}'.format(
            div_max))

    # ---- Step 7: f_eq + per-rank write ----
    print('[7/8] Reconstructing f_eq and writing per-rank files')
    parent_dir = os.path.dirname(writing_dir)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    os.makedirs(writing_dir)

    # Write rho per rank
    rho_pr = split_y(rho_new, NEW)
    for r in range(NEW.JP):
        rho_pr[r].tofile(os.path.join(writing_dir, 'rho_{}.bin'.format(r)))
    print('      wrote rho_0..rho_{}.bin'.format(NEW.JP - 1))

    # f_neq reconstruction. Three modes:
    #   zero (current stability A/B test): write pure equilibrium f = f_eq
    #     and therefore force f_neq = 0 on the rebuilt checkpoint.
    #   chapman-enskog: rebuild f_neq on NEW grid
    #     from velocity gradients via CE expansion. Drops the OLD f_q files
    #     after rho/u extraction; gradients are evaluated on the NEW grid so
    #     they are self-consistent with NEW spacing.
    #   interp (legacy): linearly interpolate f_neq from OLD computational
    #     space — destroys gradient information across GAMMA changes.
    rho_check = np.zeros_like(rho_new)
    min_f = float('inf')
    max_f = -float('inf')
    ce_omega_new = None
    ce_coeff_used = None

    if args.fneq_mode == 'zero':
        print('      mode = zero: writing pure equilibrium f=f_eq (f_neq forced to 0)')
        for q in range(19):
            f_new = compute_feq_q(rho_new, ux_new, uy_new, uz_new, q)
            enforce_periodic_physical_duplicates(f_new, NEW)
            fill_ghost(f_new, NEW)

            rho_check += f_new
            if np.any(np.isnan(f_new)) or np.any(np.isinf(f_new)):
                sys.exit('FATAL: f{:02d} contains NaN or Inf after equilibrium reconstruction'.format(q))
            min_f = min(min_f, float(np.min(f_new)))
            max_f = max(max_f, float(np.max(f_new)))

            pr = split_y(f_new, NEW)
            for r in range(NEW.JP):
                pr[r].tofile(os.path.join(writing_dir, 'f{:02d}_{}.bin'.format(q, r)))
            print('      wrote f{:02d}_0..f{:02d}_{} (equilibrium, f_neq=0)'.format(
                q, q, NEW.JP - 1), flush=True)
            del f_new, pr
        print('      max |f_neq / f_eq|  = 0.000e+00   (forced equilibrium test)')
    elif args.fneq_mode == 'chapman-enskog':
        # Resolve viscosity (variables.h: niu = Uref / Re).
        niu = args.niu
        if niu is None:
            vh_for_niu = getattr(args, 'variables_h', None) or find_variables_h()
            if vh_for_niu and os.path.isfile(vh_for_niu):
                niu = parse_niu_from_variables_h(vh_for_niu)
        if niu is None:
            sys.exit('FATAL: --fneq-mode chapman-enskog requires niu. '
                     'Pass --niu <value> or run from a project with variables.h.')

        # CE coefficient: -(omega_global) * dt_global
        #   omega_new = 3*niu/dt_global_new + 0.5   (main.cu:577)
        #   -omega_new * dt_global_new = -3*niu - 0.5*dt_global_new
        ce_omega_new = 3.0 * niu / dt_real + 0.5
        ce_coeff = -ce_omega_new * dt_real
        ce_coeff_used = ce_coeff
        print('      mode = chapman-enskog: niu = {:.6e}'.format(niu))
        print('      omega_new = 3*niu/dt_global_new + 0.5 = {:.12e}'.format(ce_omega_new))
        print('      ce_coeff = -(omega)*dt = {:.6e}  (= -3niu - 0.5dt = {:.6e})'.format(
            ce_coeff, -3.0 * niu - 0.5 * dt_real))

        # Inverse metric on NEW grid. Order 6 mirrors solver (gilbm/metric_terms.h);
        # order 2 is legacy 2nd-order central (kept for A/B comparison).
        if args.metric_order == 6:
            dj_dy, dj_dz, dk_dy, dk_dz = compute_inverse_metric_2d_fornberg(y2d_new, z2d_new)
        else:
            dj_dy, dj_dz, dk_dy, dk_dz = compute_inverse_metric_2d(y2d_new, z2d_new)
        print('      metric order = {} ({})'.format(
            args.metric_order,
            'Fornberg j-central + k-adaptive' if args.metric_order == 6 else '2nd-order central'))
        dx_phys = LX / (NEW.NX - 1)

        # 9-component velocity gradient tensor on NEW interior
        t = time.time()
        dudx, dudy, dudz = compute_velocity_gradient_3d(ux_new, dx_phys, dj_dy, dj_dz, dk_dy, dk_dz, NEW)
        dvdx, dvdy, dvdz = compute_velocity_gradient_3d(uy_new, dx_phys, dj_dy, dj_dz, dk_dy, dk_dz, NEW)
        dwdx, dwdy, dwdz = compute_velocity_gradient_3d(uz_new, dx_phys, dj_dy, dj_dz, dk_dy, dk_dz, NEW)
        print('      velocity gradient tensor: {:.1f}s'.format(time.time() - t))

        div_u = dudx + dvdy + dwdz
        max_div = float(np.max(np.abs(div_u)))
        max_strain = max(
            float(np.max(np.abs(dudx))), float(np.max(np.abs(dudy))), float(np.max(np.abs(dudz))),
            float(np.max(np.abs(dvdx))), float(np.max(np.abs(dvdy))), float(np.max(np.abs(dvdz))),
            float(np.max(np.abs(dwdx))), float(np.max(np.abs(dwdy))), float(np.max(np.abs(dwdz))),
        )
        print('      max |div(u)|  = {:.3e}   (incompressibility residual)'.format(max_div))
        print('      max |grad u|  = {:.3e}'.format(max_strain))
        del div_u

        rho_int = rho_new[BFR:BFR+NEW.NY, BFR:BFR+NEW.NZ, BFR:BFR+NEW.NX]
        grad = (dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz)

        max_fneq_ratio = 0.0
        for q in range(19):
            feq_full = compute_feq_q(rho_new, ux_new, uy_new, uz_new, q)
            fneq_int = chapman_enskog_fneq_q(rho_int, grad, q, ce_coeff)

            f_new = feq_full.copy()
            f_new[BFR:BFR+NEW.NY, BFR:BFR+NEW.NZ, BFR:BFR+NEW.NX] += fneq_int
            enforce_periodic_physical_duplicates(f_new, NEW)
            fill_ghost(f_new, NEW)
            del feq_full

            feq_int_for_diag = compute_feq_q(rho_int,
                ux_new[BFR:BFR+NEW.NY, BFR:BFR+NEW.NZ, BFR:BFR+NEW.NX],
                uy_new[BFR:BFR+NEW.NY, BFR:BFR+NEW.NZ, BFR:BFR+NEW.NX],
                uz_new[BFR:BFR+NEW.NY, BFR:BFR+NEW.NZ, BFR:BFR+NEW.NX], q)
            ratio = float(np.max(np.abs(fneq_int) / np.maximum(feq_int_for_diag, 1e-30)))
            max_fneq_ratio = max(max_fneq_ratio, ratio)
            del feq_int_for_diag, fneq_int

            rho_check += f_new
            if np.any(np.isnan(f_new)) or np.any(np.isinf(f_new)):
                sys.exit('FATAL: f{:02d} contains NaN or Inf after CE reconstruction'.format(q))
            min_f = min(min_f, float(np.min(f_new)))
            max_f = max(max_f, float(np.max(f_new)))

            pr = split_y(f_new, NEW)
            for r in range(NEW.JP):
                pr[r].tofile(os.path.join(writing_dir, 'f{:02d}_{}.bin'.format(q, r)))
            print('      wrote f{:02d}_0..f{:02d}_{} (CE)'.format(q, q, NEW.JP - 1), flush=True)
            del f_new, pr

        del dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz, grad
        del dj_dy, dj_dz, dk_dy, dk_dz
        print('      max |f_neq / f_eq|  = {:.3e}   (Knudsen-like; should be << 1)'.format(max_fneq_ratio))
    else:
        # Legacy: per-q interp f_neq, rebuild f = f_eq_new + scale*f_neq_new
        for q in range(19):
            per_rank = []
            for r in range(OLD.JP):
                path = os.path.join(args.old_dir, 'f{:02d}_{}.bin'.format(q, r))
                per_rank.append(read_rank_bin(path, OLD))
            f_old = stitch_y(per_rank, OLD)
            feq_old = compute_feq_q(rho_g, ux_g, uy_g, uz_g, q)
            fneq_old = f_old - feq_old
            del f_old, feq_old, per_rank

            fneq_new = interpolate_comp_3d(fneq_old, OLD, NEW)
            del fneq_old
            fill_ghost(fneq_new, NEW)

            feq = compute_feq_q(rho_new, ux_new, uy_new, uz_new, q)
            f_new = feq + args.fneq_scale * fneq_new
            del feq, fneq_new
            enforce_periodic_physical_duplicates(f_new, NEW)
            fill_ghost(f_new, NEW)

            rho_check += f_new
            if np.any(np.isnan(f_new)) or np.any(np.isinf(f_new)):
                sys.exit('FATAL: f{:02d} contains NaN or Inf after reconstruction'.format(q))
            min_f = min(min_f, float(np.min(f_new)))
            max_f = max(max_f, float(np.max(f_new)))

            pr = split_y(f_new, NEW)
            for r in range(NEW.JP):
                pr[r].tofile(os.path.join(writing_dir, 'f{:02d}_{}.bin'.format(q, r)))
            print('      wrote f{:02d}_0..f{:02d}_{} with f_neq scale {:.3f}'.format(
                q, q, NEW.JP - 1, args.fneq_scale), flush=True)
            del f_new, pr

    rho_diff = float(np.max(np.abs(rho_check - rho_new)))
    print('      f range after reconstruction = [{:.15e}, {:.15e}]'.format(min_f, max_f))
    print('      max |sum(f_new)-rho_new| = {:.3e}'.format(rho_diff))
    if min_f <= 0.0:
        sys.exit('FATAL: reconstructed f contains non-positive values (min_f={:.6e})'.format(min_f))
    if rho_diff > 1e-10:
        sys.exit('FATAL: reconstructed f is not conservative enough: max |sum(f)-rho| = {:.3e}'.format(rho_diff))

    # Free old arrays after f_neq reconstruction is complete.
    del rho_g, ux_g, uy_g, uz_g, rho_check

    # ---- Step 8: metadata + atomic rename ----
    print('[8/8] Writing new metadata.dat')
    # dt_global handling (Phase B):
    #   compute_dt_global_gilbm mirrors gilbm/precompute.h:78-115.
    #   CE reconstruction always uses dt_global_new computed above. Metadata
    #   writes the same value unless --skip-drift-check explicitly requests
    #   dt_global=-1.0 for solver drift-check debugging.
    naive_minsize = compute_minsize(NEW)
    origin_meta_path = os.path.join(args.old_dir, 'metadata.dat')
    # Preserve controller state from origin checkpoint to avoid the F* step
    # that occurs when a hot flow field is restarted with a cold PID integrator.
    # fileIO.h documents these fields as "required to avoid Force_integral /
    # error_prev reset on restart".
    #
    # IMPORTANT: accu_count and FTT are NOT preserved from origin.
    #   * accu_count > 0 triggers fileIO.h:748 to load 36 statistics binaries
    #     (sum_u_*.bin, sum_uu_*.bin, ...). This regrid pipeline only writes
    #     f00..f18 + rho + metadata.dat — the stats binaries are NOT regenerated
    #     on the new grid. Preserving accu_count > 0 would make the rebuilt
    #     checkpoint unloadable (result_readbin abort on missing stats).
    #     We set accu_count=0 so fileIO.h:748 skips stats loading entirely;
    #     the runtime will re-accumulate stats fresh from FTT_STATS_START.
    #   * FTT (flow-through-time clock) is reset to 0 because the regrid is
    #     a fresh start on the new mesh; statistics windows align to FTT.
    # Origin values are kept ONLY as provenance fields (interp_origin_*) for audit.
    controller_keys = ('Force_integral', 'error_prev',
                       'ctrl_initialized', 'gehrke_activated')
    controller_defaults = {
        'Force_integral':  '{:.15f}'.format(0.0),
        'error_prev':      '{:.15f}'.format(0.0),
        'ctrl_initialized': '0',
        'gehrke_activated': '0',
    }
    controller_preserved = {k: meta_old.get(k, controller_defaults[k]) for k in controller_keys}
    origin_ftt = meta_old.get('FTT', '0.0')
    origin_accu = meta_old.get('accu_count', '0')
    print('      Controller state preserved from origin: '
          'Force_integral={} error_prev={} ctrl_initialized={} gehrke_activated={}'.format(
              controller_preserved['Force_integral'],
              controller_preserved['error_prev'],
              controller_preserved['ctrl_initialized'],
              controller_preserved['gehrke_activated']))
    print('      accu_count=0, FTT=0 (regrid pipeline does NOT write stats binaries; '
          'preserving accu_count>0 would break checkpoint load)')
    print('      origin FTT={} accu_count={} kept as provenance only'.format(origin_ftt, origin_accu))

    new_meta = {
        'checkpoint_version': '2',
        'mpi_rank_count': str(NEW.JP),
        'grid_dims': '{},{},{}'.format(NEW.NX6, NEW.NYD6, NEW.NZ6),
        'step': str(args.step),
        'FTT': '{:.15f}'.format(0.0),
        'accu_count': '0',
        'Force': '{:.15f}'.format(Force_value),
        'Force_integral': controller_preserved['Force_integral'],
        'error_prev': controller_preserved['error_prev'],
        'ctrl_initialized': controller_preserved['ctrl_initialized'],
        'gehrke_activated': controller_preserved['gehrke_activated'],
        'dt_global': dt_for_meta,
        'gpu_time_ms': '0',
        'cv_count': '0',
        'interp_source': args.old_dir,
        'interp_old_grid': OLD.GRID_DAT,
        'interp_new_grid': NEW.GRID_DAT,
        'interp_old_gamma': str(OLD.GAMMA),
        'interp_new_gamma': str(NEW.GAMMA),
        'interp_fneq_mode': args.fneq_mode,
        'interp_macro_mode': args.interp_mode,
        'interp_macro_order': str(args.interp_order),
        'interp_metric_order': str(args.metric_order),
        'interp_cfl': str(args.cfl),
        'interp_cfl_source': args.cfl_source,
        'interp_dt_max_component': dt_max_component,
        'interp_origin_ftt': origin_ftt,
        'interp_origin_accu_count': origin_accu,
        'interp_Ub_old': '{:.15e}'.format(Ub_old),
        'interp_Ub_new_before': '{:.15e}'.format(Ub_new_before),
        'interp_Ub_new_after': '{:.15e}'.format(Ub_new_after),
        'interp_Ub_final_before_feq': '{:.15e}'.format(Ub_final),
        'interp_Ub_scale': '{:.15e}'.format(ub_scale),
        'interp_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'interp_project_velocity': args.project_velocity,
        'interp_projection_max_outer': str(args.projection_max_outer),
        'interp_projection_div_tol': '{:.6e}'.format(args.projection_div_tol),
    }
    if final_div_rms is not None:
        new_meta['interp_final_div_rms'] = '{:.6e}'.format(final_div_rms)
    if final_div_max is not None:
        new_meta['interp_final_div_max'] = '{:.6e}'.format(final_div_max)
    if proj_info is not None:
        new_meta['interp_proj_div_rms_before'] = '{:.6e}'.format(proj_info['div_rms_before'])
        new_meta['interp_proj_div_max_before'] = '{:.6e}'.format(proj_info['div_max_before'])
        new_meta['interp_proj_div_rms_after'] = '{:.6e}'.format(proj_info['div_rms_after'])
        new_meta['interp_proj_div_max_after'] = '{:.6e}'.format(proj_info['div_max_after'])
        new_meta['interp_proj_div_rms_interior'] = '{:.6e}'.format(proj_info['div_rms_interior'])
        new_meta['interp_proj_outer_iters'] = str(proj_info['outer_iters'])
        new_meta['interp_proj_solve_time_s'] = '{:.1f}'.format(proj_info['solve_time_s'])
        if 'method' in proj_info:
            new_meta['interp_proj_method'] = str(proj_info['method'])
        if 'rhs_mean' in proj_info:
            new_meta['interp_proj_rhs_mean'] = '{:.6e}'.format(proj_info['rhs_mean'])
        if 'true_residual_rms' in proj_info:
            new_meta['interp_proj_true_residual_rms'] = '{:.6e}'.format(
                proj_info['true_residual_rms'])
        if 'true_residual_max' in proj_info:
            new_meta['interp_proj_true_residual_max'] = '{:.6e}'.format(
                proj_info['true_residual_max'])
    if ce_omega_new is not None:
        new_meta['interp_ce_omega_global_new'] = '{:.15e}'.format(ce_omega_new)
        new_meta['interp_ce_coeff'] = '{:.15e}'.format(ce_coeff_used)
    # interp_fneq_scale only meaningful in legacy 'interp' mode; CE mode does not use it.
    if args.fneq_mode == 'interp':
        new_meta['interp_fneq_scale'] = str(args.fneq_scale)
    vh_for_prov = getattr(args, 'variables_h', None)
    if vh_for_prov and os.path.isfile(vh_for_prov):
        new_meta['interp_variables_h'] = os.path.abspath(vh_for_prov)
        new_meta['interp_variables_h_mtime'] = str(int(os.path.getmtime(vh_for_prov)))
    if NEW.GRID_DAT and os.path.isfile(NEW.GRID_DAT):
        new_meta['interp_new_grid_mtime'] = str(int(os.path.getmtime(NEW.GRID_DAT)))
        new_fp = read_grid_params_sha256(NEW.GRID_DAT)
        if new_fp:
            new_meta['interp_new_grid_params_sha256'] = new_fp
    if OLD.GRID_DAT and os.path.isfile(OLD.GRID_DAT):
        new_meta['interp_old_grid_mtime'] = str(int(os.path.getmtime(OLD.GRID_DAT)))
        old_fp = read_grid_params_sha256(OLD.GRID_DAT)
        if old_fp:
            new_meta['interp_old_grid_params_sha256'] = old_fp
    solver_match = getattr(args, 'solver_grid_match_info', None)
    if args.solver_grid_dat:
        new_meta['interp_solver_grid'] = os.path.abspath(args.solver_grid_dat)
        if os.path.isfile(args.solver_grid_dat):
            new_meta['interp_solver_grid_mtime'] = str(int(os.path.getmtime(args.solver_grid_dat)))
        new_meta['interp_solver_grid_generated'] = (
            '1' if getattr(args, 'solver_grid_generated', False) else '0')
    if solver_match:
        new_meta['interp_solver_grid_match'] = '1' if solver_match.get('ok') else '0'
        if 'max_abs' in solver_match:
            new_meta['interp_solver_grid_max_abs_diff'] = '{:.15e}'.format(
                solver_match['max_abs'])
            new_meta['interp_solver_grid_max_abs_x'] = '{:.15e}'.format(
                solver_match['max_abs_x'])
            new_meta['interp_solver_grid_max_abs_y'] = '{:.15e}'.format(
                solver_match['max_abs_y'])
        if solver_match.get('new_grid_params_sha256'):
            new_meta['interp_new_grid_params_sha256'] = solver_match['new_grid_params_sha256']
        if solver_match.get('solver_grid_params_sha256'):
            new_meta['interp_solver_grid_params_sha256'] = solver_match['solver_grid_params_sha256']
    if os.path.isfile(origin_meta_path):
        new_meta['interp_origin_metadata_mtime'] = str(int(os.path.getmtime(origin_meta_path)))
    write_metadata(os.path.join(writing_dir, 'metadata.dat'), new_meta)
    print('      Force={:.6e}  step={}  jp={}  grid_dims={}'.format(
        Force_value, args.step, NEW.JP, new_meta['grid_dims']))
    print('      dt_global written as {}'.format(dt_for_meta))
    if args.skip_drift_check:
        print('      Phase 5 drift check skipped by request; runtime computes its own dt')
    print('      (naive minSize for reference: {:.6e}; runtime Imamura dt typically ~0.4-0.5x of this)'.format(naive_minsize))

    print('      Atomic rename: {} -> {}'.format(writing_dir, out_dir))
    os.rename(writing_dir, out_dir)

    restart_root = os.path.dirname(os.path.abspath(args.output_root))
    prov_path = os.path.join(restart_root, 'grid_provenance')
    prov = {
        'new_grid': os.path.abspath(NEW.GRID_DAT),
        'old_grid': os.path.abspath(OLD.GRID_DAT),
        'origin': os.path.abspath(args.old_dir),
        'origin_metadata_mtime': str(int(os.path.getmtime(origin_meta_path))) if os.path.isfile(origin_meta_path) else '',
        'variables_h': os.path.abspath(vh_for_prov) if vh_for_prov else '',
        'variables_h_mtime': str(int(os.path.getmtime(vh_for_prov))) if vh_for_prov and os.path.isfile(vh_for_prov) else '',
        'new_grid_mtime': str(int(os.path.getmtime(NEW.GRID_DAT))) if NEW.GRID_DAT and os.path.isfile(NEW.GRID_DAT) else '',
        'old_grid_mtime': str(int(os.path.getmtime(OLD.GRID_DAT))) if OLD.GRID_DAT and os.path.isfile(OLD.GRID_DAT) else '',
        'new_grid_params_sha256': (read_grid_params_sha256(NEW.GRID_DAT) or '') if NEW.GRID_DAT and os.path.isfile(NEW.GRID_DAT) else '',
        'old_grid_params_sha256': (read_grid_params_sha256(OLD.GRID_DAT) or '') if OLD.GRID_DAT and os.path.isfile(OLD.GRID_DAT) else '',
        'solver_grid': os.path.abspath(args.solver_grid_dat) if args.solver_grid_dat else '',
        'solver_grid_mtime': str(int(os.path.getmtime(args.solver_grid_dat))) if args.solver_grid_dat and os.path.isfile(args.solver_grid_dat) else '',
        'solver_grid_params_sha256': (read_grid_params_sha256(args.solver_grid_dat) or '') if args.solver_grid_dat and os.path.isfile(args.solver_grid_dat) else '',
        'solver_grid_generated': ('1' if getattr(args, 'solver_grid_generated', False) else
                                  ('0' if args.solver_grid_dat else '')),
        'solver_grid_match': ('1' if solver_match and solver_match.get('ok') else
                              ('0' if solver_match else '')),
        'solver_grid_max_abs_diff': ('{:.15e}'.format(solver_match['max_abs'])
                                     if solver_match and 'max_abs' in solver_match else ''),
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    prov_tmp = prov_path + '.WRITING'
    with open(prov_tmp, 'w', encoding='utf-8') as fp:
        for k, v in prov.items():
            fp.write('{}={}\n'.format(k, v))
    os.rename(prov_tmp, prov_path)
    print('      grid_provenance written: {}'.format(prov_path))

    elapsed = time.time() - t0
    nf = 19 * NEW.JP + NEW.JP + 1
    print()
    print('Done in {:.1f}s. New checkpoint at: {}'.format(elapsed, out_dir))
    print('Total files: 19f x {} ranks + {} rho + 1 metadata = {}'.format(NEW.JP, NEW.JP, nf))


if __name__ == '__main__':
    main()

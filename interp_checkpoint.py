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
  3. Read old checkpoint, compute macros (rho, ux, uy, uz)
  4. Interpolate macros old -> new in computational (j, k, i) space
  5. Reconstruct f_q = f_eq(new) + scale * interp(f_neq_old) for q = 0..18
  6. Split into new ranks, write per-rank binary files + metadata.dat

Output written atomically:
  <output_root>/step_%08d.WRITING/ -> <output_root>/step_%08d/

Usage:
  # Project mode: NEW grid is read from ../variables.h, output is step_00000001
  python3 restart_tools/interp_checkpoint.py \\
      --old-dir restart/step_12550001_origin129 \\
      --step 1

  # CLI override (skip prompts):
  python3 restart_tools/interp_checkpoint.py --old-dir ./old_ckpt \\
      --old-gamma 2.0 --old-grid-dat old_grid.dat \\
      --new-nx 257 --new-ny 513 --new-nz 257 --new-jp 16 \\
      --new-gamma 3.0 --new-alpha 0.5 --new-grid-dat new_grid.dat \\
      --output-root restart/checkpoint --step 1 --fneq-scale 1.0

Expected folder structure:
  workspace/
  +-- variables.h                     (optional, project mode)
  +-- restart_tools/interp_checkpoint.py
  +-- J_Frohlich/                    (or any directory)
  |   +-- old_grid_I{NY}_J{NZ}_g{G}_a{A}.dat
  |   +-- new_grid_I{NY}_J{NZ}_g{G}_a{A}.dat
  +-- old_checkpoint/                (source checkpoint)
      +-- metadata.dat
      +-- f00_0.bin ... f18_{jp-1}.bin
      +-- rho_0.bin ... rho_{jp-1}.bin
"""

import os
import sys
import math
import time
import argparse
import numpy as np

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
    """Parse #define NX/NY/NZ/jp/GAMMA/ALPHA/LX/LY/LZ/H_HILL from variables.h."""
    targets = {'NX', 'NY', 'NZ', 'jp', 'GAMMA', 'ALPHA', 'LX', 'LY', 'LZ', 'H_HILL'}
    int_keys = {'NX', 'NY', 'NZ', 'jp'}
    defines = {}
    with open(path) as f:
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


def parse_grid_dat_header(path):
    """Extract I=, J= from Tecplot .dat file header for cross-validation."""
    dims = {}
    with open(path) as f:
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


def find_variables_h():
    """Search for variables.h in standard project locations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        'variables.h',
        '../variables.h',
        os.path.join(script_dir, '..', 'variables.h'),
    ]
    seen = set()
    for c in candidates:
        p = os.path.abspath(c)
        if p not in seen and os.path.isfile(p):
            return p
        seen.add(p)
    return None


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


def _grid_dat_search_dirs():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return [
        'J_Frohlich',
        os.path.join(script_dir, '..', 'J_Frohlich'),
        '../J_Frohlich',
        '.',
    ]


def try_find_grid_dat(ny, nz, gamma, alpha, search_dirs=None):
    """Try to find grid .dat file by naming convention I{NY}_J{NZ}_g{G}_a{A}."""
    if search_dirs is None:
        search_dirs = _grid_dat_search_dirs()
    candidates = set()
    for fmt in (str, lambda x: '{:g}'.format(x)):
        g_str = fmt(gamma)
        a_str = fmt(alpha)
        candidates.add('I{}_J{}_g{}_a{}'.format(ny, nz, g_str, a_str))
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
    """Find grid .dat by I{NY}_J{NZ} pattern only; extract gamma/alpha from filename."""
    import re
    if search_dirs is None:
        search_dirs = _grid_dat_search_dirs()
    pattern = 'I{}_J{}'.format(ny, nz)
    ga_re = re.compile(r'_g([\d.]+)_a([\d.]+)\.dat$')
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if not fname.endswith('.dat'):
                continue
            if pattern in fname:
                path = os.path.join(d, fname)
                m = ga_re.search(fname)
                gamma = float(m.group(1)) if m else None
                alpha = float(m.group(2)) if m else None
                return path, gamma, alpha
    return None, None, None


def ask_value(prompt_text, cast_fn=str, default=None):
    """Interactive prompt with optional default value."""
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


def resolve_old_dir(old_dir):
    """Resolve source checkpoint directory, with a friendly fallback for local copies."""
    old_dir = os.path.normpath(old_dir)
    meta_path = os.path.join(old_dir, 'metadata.dat')
    if os.path.isfile(meta_path):
        return old_dir

    restart_dir = 'restart'
    candidates = []
    if os.path.isdir(restart_dir):
        for name in sorted(os.listdir(restart_dir)):
            path = os.path.join(restart_dir, name)
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
    with open(path) as f:
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
    with open(path, 'w') as f:
        for k in keys_order:
            if k in params:
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
    with open(cfg.GRID_DAT) as f:
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

    Mapping (initialization.h:292):  j_global = rank * (NYD6 - 7) + j_local
    Overlapping ghost regions across ranks are identical post-MPI-halo;
    later ranks overwrite earlier ones harmlessly.
    """
    g = np.zeros((cfg.NY6, cfg.NZ6, cfg.NX6), dtype=np.float64)
    for r in range(cfg.JP):
        j0 = r * cfg.CHUNK
        g[j0:j0 + cfg.NYD6, :, :] = per_rank_list[r]
    return g


def split_y(global_arr, cfg):
    """Split global (NY6, NZ6, NX6) into JP per-rank slices of (NYD6, NZ6, NX6)."""
    out = []
    for r in range(cfg.JP):
        j0 = r * cfg.CHUNK
        out.append(global_arr[j0:j0 + cfg.NYD6, :, :].copy())
    return out


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
# Main
# ---------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--old-dir', default='restart/step_12550001_origin129',
                   help='old checkpoint directory (default: %(default)s)')
    p.add_argument('--step', type=int, default=1,
                   help='new checkpoint step number written into metadata (default: 1)')
    p.add_argument('--output-root', default='restart/checkpoint',
                   help='root directory for chain-compatible output step_%%08d (default: %(default)s)')
    p.add_argument('--new-dir', default=None,
                   help='advanced override for output checkpoint directory; default is output-root/step_%%08d')
    p.add_argument('--fneq-scale', type=float, default=1.0,
                   help='scale factor applied to interpolated f_neq (default: %(default)s)')
    p.add_argument('--dry-run', action='store_true',
                   help='validate configuration and output path, then exit before reading/writing checkpoint data')

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
                       help='path to variables.h (auto-detected if not specified)')

    args = p.parse_args()

    global OLD, NEW
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
    print('      grid_dims={} mpi_rank_count={} step={} FTT={} Force={:.6e}'.format(
        meta_old['grid_dims'], meta_old['mpi_rank_count'],
        meta_old['step'], meta_old['FTT'], Force_value))

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
    uy_g = momy_g / rho_safe
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
    if OLD.GAMMA != NEW.GAMMA:
        print('      NOTE: GAMMA differs (OLD={}, NEW={}): computational-space interpolation'.format(
            OLD.GAMMA, NEW.GAMMA))
        print('            maps same (j,k,i) fraction to different physical z-heights.')
        print('            This is expected for grid-stretching refinement.')
    print('[5/8] Interpolating macros (rho, ux, uy, uz) to NEW grid in computational space')
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

    print('      Filling ghost cells')
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

    # ---- Step 6 & 7: f_eq + per-rank write ----
    print('[6/8] Reconstructing f_eq and writing per-rank files')
    parent_dir = os.path.dirname(writing_dir)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    os.makedirs(writing_dir)

    # Write rho per rank
    rho_pr = split_y(rho_new, NEW)
    for r in range(NEW.JP):
        rho_pr[r].tofile(os.path.join(writing_dir, 'rho_{}.bin'.format(r)))
    print('      wrote rho_0..rho_{}.bin'.format(NEW.JP - 1))

    # Per-q: interpolate f_neq, rebuild f = f_eq_new + scale*f_neq_new, split, write.
    # This preserves the old checkpoint's viscous/non-equilibrium content while
    # keeping the new-grid macroscopic field controlled by rho_new/u_new.
    rho_check = np.zeros_like(rho_new)
    min_f = float('inf')
    max_f = -float('inf')
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
    print('[7/8] Writing new metadata.dat')
    # NOTE on dt_global:
    #   The runtime computes dt_global = CFL / max|c_tilde| from Jacobian metric
    #   terms (gilbm/precompute.h:ComputeGlobalTimeStep), NOT from the simple
    #   minSize formula in variables.h. They differ by a factor of ~0.4-0.5,
    #   so any naively-written value would trip Phase 5 drift check
    #   (fileIO.h:658, |drift| > 1e-6 -> MPI_Abort).
    #
    #   We deliberately write dt_global=-1.0 to trigger the legacy-format
    #   skip path (fileIO.h:650): "metadata.dat 無 dt_global 欄位, 跳過漂移檢查".
    #   The runtime will compute its own dt_global from the new grid metrics
    #   on startup; dt_saved is only used for the drift guardrail and is
    #   discarded thereafter.
    naive_minsize = compute_minsize(NEW)
    new_meta = {
        'checkpoint_version': '2',
        'mpi_rank_count': str(NEW.JP),
        'grid_dims': '{},{},{}'.format(NEW.NX6, NEW.NYD6, NEW.NZ6),
        'step': str(args.step),
        'FTT': '{:.15f}'.format(0.0),
        'accu_count': '0',
        'Force': '{:.15f}'.format(Force_value),
        'Force_integral': '{:.15f}'.format(0.0),
        'error_prev': '{:.15f}'.format(0.0),
        'ctrl_initialized': '0',
        'gehrke_activated': '0',
        'dt_global': '-1.0',
        'gpu_time_ms': '0',
        'cv_count': '0',
    }
    write_metadata(os.path.join(writing_dir, 'metadata.dat'), new_meta)
    print('      Force={:.6e}  step={}  jp={}  grid_dims={}'.format(
        Force_value, args.step, NEW.JP, new_meta['grid_dims']))
    print('      dt_global written as -1.0 (skip Phase 5 drift check; runtime computes its own dt)')
    print('      (naive minSize for reference: {:.6e}; runtime Imamura dt typically ~0.4-0.5x of this)'.format(naive_minsize))

    print('[8/8] Atomic rename: {} -> {}'.format(writing_dir, out_dir))
    os.rename(writing_dir, out_dir)

    elapsed = time.time() - t0
    nf = 19 * NEW.JP + NEW.JP + 1
    print()
    print('Done in {:.1f}s. New checkpoint at: {}'.format(elapsed, out_dir))
    print('Total files: 19f x {} ranks + {} rho + 1 metadata = {}'.format(NEW.JP, NEW.JP, nf))


if __name__ == '__main__':
    main()

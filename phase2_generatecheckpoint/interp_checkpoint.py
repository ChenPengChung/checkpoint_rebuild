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
  4. Interpolate macros old -> new according to --interp-mode:
       phys (default): physical-space remap; correct when GAMMA changes.
         --interp-order 6 (default): 7-point Lagrange tensor product O(h^6).
         --interp-order 2:          bilinear O(h^2) (legacy).
       comp:           legacy computational (j, k, i) remap for A/B tests.
  5. Reconstruct f_q for q = 0..18 according to --fneq-mode:
       chapman-enskog (default): f_q = f_eq(new) + f_neq_q reconstructed from
                                 NEW-grid velocity gradients via Chapman-Enskog.
                                 Wall rows use 6th-order one-sided FD to match
                                 the solver's wall CE formula.
       interp (legacy):          f_q = f_eq(new) + scale * interp(f_neq_old)
                                 (linear interp in computational space; loses
                                 gradient information across GAMMA changes).
  6. Preserve controller state (Force_integral, error_prev, ctrl_initialized,
     gehrke_activated) ONLY from origin metadata to avoid F* step on restart.
     FTT and accu_count are NOT preserved — they are reset to 0 because:
       - regrid is a fresh start on the new mesh (FTT=0 aligns new stats window);
       - accu_count > 0 would trigger fileIO.h:748 to load 36 stats binaries
         (sum_u_*.bin, ...) that this pipeline does NOT regenerate.
     Origin FTT / accu_count are written into metadata as `interp_origin_*`
     fields for audit only.
  7. Split into new ranks, write per-rank binary files + metadata.dat

Output written atomically:
  <output_root>/step_%08d.WRITING/ -> <output_root>/step_%08d/
  restart/grid_provenance records the session-level grid identity.

Usage:
  # Project auto mode:
  #   origin checkpoint: phase2_generatecheckpoint/step_*_origin*
  #   OLD grid:          phase1_generategrid/oldgrid_*.dat
  #   NEW grid:          phase1_generategrid/newgrid_*.dat
  python3 phase2_generatecheckpoint/interp_checkpoint.py --auto --step 1

  # CLI override (skip prompts):
  python3 phase2_generatecheckpoint/interp_checkpoint.py --old-dir ./old_ckpt \\
      --old-gamma 2.0 --old-grid-dat old_grid.dat \\
      --new-nx 257 --new-ny 513 --new-nz 257 --new-jp 16 \\
      --new-gamma 3.0 --new-alpha 0.5 --new-grid-dat new_grid.dat \\
      --output-root restart/checkpoint --step 1 \
      --interp-mode phys --fneq-mode chapman-enskog

Expected folder structure:
  workspace/
  +-- variables.h                     (optional, project mode)
  +-- phase2_generatecheckpoint/interp_checkpoint.py
  +-- phase1_generategrid/
  |   +-- oldgrid_*_I{NY}_J{NZ}_g{G}_a{A}.dat    (OLD uniform gamma grid)
  |   +-- newgrid_*_I{NY}_J{NZ}_a{A}.dat         (NEW variable gamma grid)
  +-- phase2_generatecheckpoint/step_*_origin*/  (source checkpoint)
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


def find_origin_checkpoint(search_dir=None):
    """Find step_*_origin* directories with valid metadata across phase layout.
    FATAL if multiple origins exist (ambiguous)."""
    candidates = []
    for parent in origin_search_dirs(search_dir):
        if not os.path.isdir(parent):
            continue
        for name in sorted(os.listdir(parent)):
            if name.startswith('step_') and '_origin' in name:
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
        sys.exit('FATAL: --old-dir not specified and no phase2_generatecheckpoint/step_*_origin* '
                 'or restart/step_*_origin* found')

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


def bilinear_inverse_triangle_fallback(y_n, z_n, y_corners, z_corners, eps=1e-9):
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


def find_containing_cell_2d(y_n, z_n, y_old, z_old, bboxes, eps=1e-9):
    """Locate OLD cell containing (y_n, z_n). Returns (j*, k*, xi, eta).

    Per-candidate strategy:
      1. Newton 2x2; accept if converged AND in [0,1]^2 (with eps tolerance).
      2. If Newton failed OR converged out-of-bounds -> triangle fallback.
      3. Both failed -> next candidate.
      4. All candidates exhausted -> ValueError.
    """
    bbox_y_min, bbox_y_max, bbox_z_min, bbox_z_max = bboxes
    candidates = ((bbox_y_min - eps <= y_n) & (y_n <= bbox_y_max + eps) &
                  (bbox_z_min - eps <= z_n) & (z_n <= bbox_z_max + eps))
    cand_jk = np.argwhere(candidates)
    if len(cand_jk) == 0:
        raise ValueError('No OLD cell brackets ({:.6e}, {:.6e})'.format(y_n, z_n))

    def _in_bounds(xi, eta):
        return -eps <= xi <= 1 + eps and -eps <= eta <= 1 + eps

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
                                                                 y_corners, z_corners)
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

    jstar = np.empty((cfg_new.NY, cfg_new.NZ), dtype=np.int32)
    kstar = np.empty((cfg_new.NY, cfg_new.NZ), dtype=np.int32)
    xistar = np.empty((cfg_new.NY, cfg_new.NZ), dtype=np.float64)
    etastar = np.empty((cfg_new.NY, cfg_new.NZ), dtype=np.float64)
    for j_n in range(cfg_new.NY):
        for k_n in range(cfg_new.NZ):
            y_n = y2d_new[BFR + j_n, BFR + k_n]
            z_n = z2d_new[BFR + j_n, BFR + k_n]
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


def interpolate_lagrange7_3d_with_mapping(field_old, mapping):
    """Interpolate one (NY6_old, NZ6_old, NX6_old) field using 7-point Lagrange.

    Uses the same PhysMapping2D (j*, k*, xi, eta) as the bilinear version,
    but applies 7x7 tensor product in the (j, k) plane and 7-point along i.

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
            # Contract (j, k) with wjk
            row = np.einsum('jk,jkn->n', wjk, val_jk)  # (NX,)

            field_new[BFR + j_n, BFR + k_n, BFR:BFR + NX_new] = row

    fill_ghost(field_new, cfg_new)
    return field_new


def apply_rho_mass_correction(rho, cfg):
    """Global additive mass correction matching solver's ReduceRhoSum_Kernel.

    Interior: i∈[3, NX6-4), j∈[3, NY6-4), k∈[3, NZ6-3)
      ni = NX6 - 7, nj = NY6 - 7, nk = NZ6 - 6
      N_interior = ni * nj * nk
      rho_modify = (N_interior * 1.0 - sum(rho[interior])) / N_interior
      rho[interior] += rho_modify
    """
    ni = cfg.NX6 - 7
    nj = cfg.NY6 - 7
    nk = cfg.NZ6 - 6
    N_interior = ni * nj * nk

    interior = (slice(BFR, BFR + nj),
                slice(BFR, BFR + nk),
                slice(BFR, BFR + ni))

    rho_sum = float(np.sum(rho[interior]))
    rho_target = float(N_interior) * 1.0
    rho_modify = (rho_target - rho_sum) / float(N_interior)

    mean_before = rho_sum / float(N_interior)
    rho[interior] += rho_modify
    mean_after = float(np.sum(rho[interior])) / float(N_interior)

    return rho_modify, mean_before, mean_after


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
def compute_dt_global_gilbm(cfg, cfl=0.5, metric_order=6):
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
    du_di = np.empty_like(u)
    du_dj = np.empty_like(u)
    du_dk = np.empty_like(u)

    du_di[:, :, 1:-1] = (u[:, :, 2:] - u[:, :, :-2]) / 2.0
    du_di[:, :, 0]  = du_di[:, :, 1]
    du_di[:, :, -1] = du_di[:, :, -2]

    du_dj[1:-1, :, :] = (u[2:, :, :] - u[:-2, :, :]) / 2.0
    du_dj[0, :, :]  = du_dj[1, :, :]
    du_dj[-1, :, :] = du_dj[-2, :, :]

    # k-derivative — centered on fluid interior, 6th-order one-sided AT walls.
    # Plain centered FD at k=BFR collapses to (u[BFR+1]-u[BFR])/2 because
    # fill_ghost copies u[BFR-1]=u[BFR], underestimating du/dk by ~50% in viscous
    # sublayer. Solver wall CE uses WALL_GRAD_ORDER=6; mirror that to keep
    # restart wall-stress consistent (gilbm/evolution_gilbm/1.algorithm1.h).
    du_dk[:, 1:-1, :] = (u[:, 2:, :] - u[:, :-2, :]) / 2.0

    # Bottom wall (k = BFR = 3):
    # du/dk = (360u1 - 450u2 + 400u3 - 225u4 + 72u5 - 10u6) / 60
    du_dk[:, BFR, :] = (
        360.0 * u[:, BFR+1, :]
       -450.0 * u[:, BFR+2, :]
       +400.0 * u[:, BFR+3, :]
       -225.0 * u[:, BFR+4, :]
       + 72.0 * u[:, BFR+5, :]
       - 10.0 * u[:, BFR+6, :]
    ) / 60.0

    # Top wall (k = NZ6-4): same coefficients below wall, reversed sign.
    kt = NZ6 - 1 - BFR  # = NZ6 - 4
    du_dk[:, kt, :] = -(
        360.0 * u[:, kt-1, :]
       -450.0 * u[:, kt-2, :]
       +400.0 * u[:, kt-3, :]
       -225.0 * u[:, kt-4, :]
       + 72.0 * u[:, kt-5, :]
       - 10.0 * u[:, kt-6, :]
    ) / 60.0

    # Ghost rows are not in the interior crop; fill non-pathologically.
    du_dk[:, 0, :]  = du_dk[:, BFR, :]
    du_dk[:, -1, :] = du_dk[:, kt, :]

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
    p.add_argument('--fneq-mode', choices=['interp', 'chapman-enskog'], default='chapman-enskog',
                   help='f_neq reconstruction strategy. "interp" = legacy linear interp of f_neq '
                        '(loses gradient info; default before fix). "chapman-enskog" = rebuild f_neq '
                        'on NEW grid from velocity gradients via CE expansion (recommended; default).')
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
                if name.startswith('step_') and '_origin' not in name:
                    if os.path.isfile(os.path.join(ckpt_dir, name, 'metadata.dat')):
                        has_normal = True
                        break
        if has_normal:
            print('[auto] Non-origin checkpoint in restart/checkpoint/ — interpolation not needed')
            sys.exit(0)

        origin = find_origin_checkpoint(phase2_dir)
        if not origin:
            sys.exit('FATAL: --auto: no phase2_generatecheckpoint/step_*_origin* '
                     'or restart/step_*_origin* found')
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
            if new_old_gamma is not None:
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
                    old_style_gamma, _ = infer_old_grid_params(f)
                    if old_style_gamma is not None:
                        continue
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
        interp_label = '7-point Lagrange O(h^6)' if interp_order == 6 else 'bilinear O(h^2)'
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

        # Mass correction for rho after Lagrange interpolation
        rho_modify, mean_before, mean_after = apply_rho_mass_correction(rho_new, NEW)
        print('      rho mass correction: rho_modify = {:.6e}'.format(rho_modify))
        print('      rho mean: {:.15f} -> {:.15f}'.format(mean_before, mean_after))
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

    # f_neq reconstruction. Two modes:
    #   chapman-enskog (default, fix for divergence): rebuild f_neq on NEW grid
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

    if args.fneq_mode == 'chapman-enskog':
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

        # No-slip wall clamp: physical wall is u=v=w=0. After macro interpolation,
        # u_new[BFR] may inherit OLD's wall value
        # which may carry small residual (~O(δz·∂u/∂z|_wall) plus FP noise) because
        # LBM CE wall BC produces approximately, not bitwise, zero. The 6th-order
        # one-sided FD formula in the solver explicitly assumes u_wall=0. Clamp here so CE wall stress
        # estimate matches the solver's wall BC kernel exactly.
        kt = NEW.NZ6 - 1 - BFR  # top wall row index
        wall_residual_max = max(
            float(np.max(np.abs(ux_new[:, BFR, :]))), float(np.max(np.abs(ux_new[:, kt, :]))),
            float(np.max(np.abs(uy_new[:, BFR, :]))), float(np.max(np.abs(uy_new[:, kt, :]))),
            float(np.max(np.abs(uz_new[:, BFR, :]))), float(np.max(np.abs(uz_new[:, kt, :]))),
        )
        print('      max |u_wall| before clamp = {:.3e}   (any value > 1e-10 indicates '
              'OLD checkpoint or interp introduced non-zero wall residual)'.format(wall_residual_max))
        for arr in (ux_new, uy_new, uz_new):
            arr[:, BFR, :] = 0.0
            arr[:, kt, :]  = 0.0

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
        'interp_time': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
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
    if OLD.GRID_DAT and os.path.isfile(OLD.GRID_DAT):
        new_meta['interp_old_grid_mtime'] = str(int(os.path.getmtime(OLD.GRID_DAT)))
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

# LBM Checkpoint Rebuilder (Periodic Hill D3Q19)

LBM (Lattice Boltzmann Method) checkpoint 插值重建工具，專用於 periodic-hill 流場模擬的網格轉換。  
將舊解析度的 checkpoint 插值到新解析度，保留流場物理狀態，並自動產生新的 per-rank 二進制檔案。

## 目錄結構

```
checkpoint_rebuild/
├── interp_checkpoint.py            # 主程式：checkpoint 讀取、插值、重建、寫出
├── J_Frohlich/
│   ├── grid_zeta_tool.py           # 網格生成工具 (Steger-Sorenson Poisson + tanh stretching)
│   ├── 2.medium grid.dat           # 原始中等網格 (Tecplot POINT 格式)
│   ├── 3.fine grid.dat             # 原始細網格
│   └── adaptive_*.dat              # 自適應生成的網格檔案
├── restart/
│   └── step_XXXXXXXX/              # checkpoint 資料夾 (每個 step 一個)
│       ├── metadata.dat            # 文字格式 metadata
│       ├── f00_0.bin ~ f18_{jp-1}.bin  # 分佈函數 (19 方向 x jp ranks)
│       └── rho_0.bin ~ rho_{jp-1}.bin  # 密度場 (jp ranks)
└── variables.h                     # (可選) 專案模式下自動讀取新網格參數
```

---

## 最低必備輸入資料

### 1. 舊 Checkpoint 目錄

一個完整的 LBM checkpoint 資料夾，必須包含：

| 檔案 | 數量 | 格式 | 說明 |
|------|------|------|------|
| `metadata.dat` | 1 | 文字 (key=value) | 記錄 grid_dims, mpi_rank_count, step, Force 等 |
| `f{qq}_{r}.bin` | 19 x jp | 二進制 (float64) | 分佈函數，qq=00~18, r=0~jp-1 |
| `rho_{r}.bin` | jp | 二進制 (float64) | 密度場，r=0~jp-1 |

**metadata.dat 必要欄位：**

| 欄位 | 型別 | 範例值 | 說明 |
|------|------|--------|------|
| `checkpoint_version` | int | `2` | 版本號 |
| `mpi_rank_count` | int | `8` | MPI rank (GPU) 數量 |
| `grid_dims` | string | `135,39,135` | `NX6,NYD6,NZ6`（含 ghost cells） |
| `step` | int | `12550001` | 當前時間步 |
| `Force` | float | `0.000003746299150` | 驅動力（PID 控制器輸出） |

**每個 binary 檔案的 shape：**
- `(NYD6, NZ6, NX6)`，其中 `NYD6 = (NY-1)/jp + 7`，`NZ6 = NZ+6`，`NX6 = NX+6`
- 以 C-order (row-major) float64 存儲，每個值 8 bytes

### 2. 網格檔案 (Tecplot POINT 格式 `.dat`)

舊網格和新網格各需要一份 2D Tecplot `.dat` 檔案。

| 項目 | 說明 |
|------|------|
| 格式 | Tecplot POINT format，`I=NY`, `J=NZ` |
| 座標 | 兩欄：(流向 x, 法向 y)，物理單位 |
| 資料點數 | `NY * NZ` 個座標點 |
| 命名慣例 | `*_I{NY}_J{NZ}_g{GAMMA}_a{ALPHA}.dat`（自動搜尋用） |

可以使用 `grid_zeta_tool.py` 產生自適應網格。

### 3. 網格參數

以下參數在 **Project mode** 下從 `variables.h` 自動讀取，或在 **Standalone mode** 下需手動指定：

| 參數 | 型別 | 說明 | 範例 |
|------|------|------|------|
| `NX` | int | 展向 (spanwise) 格點數 | 129, 257, 513 |
| `NY` | int | 流向 (streamwise) 格點數 | 129, 257, 513 |
| `NZ` | int | 法向 (wall-normal) 格點數 | 65, 129, 257 |
| `jp` | int | MPI rank / GPU 數量 | 8, 16 |
| `GAMMA` | float | tanh 壁面拉伸參數 | 2.0, 3.0 |
| `ALPHA` | float | 拉伸中心參數 | 0.5 |

**約束條件：** `(NY - 1) % jp == 0`（NY-1 必須能被 jp 整除）

### 4. 固定域常數

以下為 periodic-hill 幾何固定值（程式內建）：

| 常數 | 值 | 說明 |
|------|-----|------|
| `LX` | 4.5 | 展向域長度 |
| `LY` | 9.0 | 流向域長度 |
| `LZ` | 3.036 | 法向域高度 |
| `H_HILL` | 1.0 | 山丘高度 (歸一化) |
| `BFR` | 3 | Ghost cell 層數 (每側) |

---

## 預期輸出

### 輸出 Checkpoint 目錄

預設路徑：`restart/checkpoint/step_{step:08d}/`

| 檔案 | 數量 | 格式 | 說明 |
|------|------|------|------|
| `metadata.dat` | 1 | 文字 | 更新後的 metadata (新 grid_dims, jp, step) |
| `f{qq}_{r}.bin` | 19 x new_jp | 二進制 (float64) | 重建的分佈函數 |
| `rho_{r}.bin` | new_jp | 二進制 (float64) | 插值後的密度場 |

**寫入流程：** 先寫入 `.WRITING` 暫存目錄，完成後 atomic rename。

### 輸出 Metadata 欄位

| 欄位 | 值 | 說明 |
|------|-----|------|
| `checkpoint_version` | `2` | 固定 |
| `mpi_rank_count` | new jp | 新 rank 數 |
| `grid_dims` | `NX6,NYD6,NZ6` | 新網格維度 (含 ghost) |
| `step` | 指定值 (預設 1) | 新起始時間步 |
| `FTT` | `0.0` | 重置 |
| `Force` | 繼承舊值 | 保留 PID 驅動力 |
| `dt_global` | `-1.0` | 觸發 runtime 自行計算 |
| `accu_count` | `0` | 統計量累積重置 |

### 輸出資料特性

- 每個 binary 檔案 shape：`(new_NYD6, new_NZ6, new_NX6)`
- 重建公式：`f_new = f_eq(rho_new, u_new) + scale * interp(f_neq_old)`
- 確保 `sum(f_new) == rho_new` (誤差 < 1e-10)
- 確保所有 `f_new > 0` (正定性)

---

## 使用方式

### Project Mode（有 `variables.h`）

```bash
python interp_checkpoint.py \
    --old-dir restart/step_12550001 \
    --step 1
```

新網格參數自動從 `variables.h` 的 `#define` 讀取。

### Standalone Mode（CLI 全參數指定）

```bash
python interp_checkpoint.py \
    --old-dir ./old_checkpoint \
    --old-gamma 2.0 --old-grid-dat J_Frohlich/old_grid.dat \
    --new-nx 257 --new-ny 513 --new-nz 257 --new-jp 16 \
    --new-gamma 3.0 --new-alpha 0.5 --new-grid-dat J_Frohlich/new_grid.dat \
    --output-root restart/checkpoint --step 1 --fneq-scale 1.0
```

### Dry Run（僅驗證設定）

```bash
python interp_checkpoint.py --old-dir restart/step_12550001 --dry-run
```

### 生成自適應網格

```bash
python J_Frohlich/grid_zeta_tool.py
```

---

## 處理流程

```
[1/8] 讀取舊 metadata.dat
[2/8] 建構舊網格座標 (Tecplot .dat → 歸一化座標 + ghost cells)
[3/8] 讀取所有 f/rho binary 檔，拼接 MPI ranks，計算巨觀量 (ρ, ux, uy, uz)
[4/8] 建構新網格座標
[5/8] 在計算座標空間插值巨觀量 (trilinear interpolation)
[6/8] 重建分佈函數 f = f_eq(new) + scale * interp(f_neq_old)
[7/8] 寫出 metadata.dat
[8/8] Atomic rename (.WRITING → final)
```

---

## 相依套件

| 套件 | 必要性 | 用途 |
|------|--------|------|
| Python 3.x | 必要 | 執行環境 |
| NumPy | 必要 | 陣列運算、binary I/O、插值 |
| SciPy | 選用 | grid_zeta_tool.py 的高階插值 |
| Matplotlib | 選用 | grid_zeta_tool.py 的網格視覺化 |

---

## 典型轉換範例

| 項目 | 舊 (OLD) | 新 (NEW) |
|------|----------|----------|
| 網格 (NX × NY × NZ) | 129 × 129 × 65 | 257 × 513 × 257 |
| MPI ranks (jp) | 8 | 16 |
| GAMMA | 2.0 | 3.0 |
| 單一 rank 陣列 shape | (39, 71, 135) | (39, 263, 263) |
| f-files 數量 | 152 (19×8) | 304 (19×16) |
| rho-files 數量 | 8 | 16 |
| 預估 checkpoint 大小 | ~2.4 GB | ~6.4 GB |

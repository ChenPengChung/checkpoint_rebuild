# Regrid Checkpoint Pipeline — 網格重建資料點續跑機制

> Edit3_Re5600newmesh 專案
> 最後更新: 2026-05-03

---

## 1. 設計原則

**核心規則: 要從哪一格 checkpoint 續跑，是在進入 main.cu 前就決定好，而非由主程式決定。**

三層架構嚴格分離:

| 層 | 檔案 | 職責 | 不做的事 |
|----|------|------|----------|
| 決策層 | `./run` → `chain_code/run.sh` | 偵測狀態、觸發 grid 生成、觸發插值、驗證 provenance、提交 job | 不跑 solver |
| 轉換層 | `restart_tools/interp_checkpoint.py` | 讀取 origin checkpoint、插值到新 grid、寫入新 checkpoint + provenance | 不決定要不要跑、不提交 job |
| 執行層 | `main.cu` (solver) | 讀取指定 checkpoint、跑模擬、寫 solver checkpoint | 不選 checkpoint、不跑插值 |

---

## 2. 兩種 Chain 模式

### 2.1 普通 Chain (無 regrid)

```
./run → 偵測 checkpoint → 編譯 → sbatch
   → jobscript 選 latest valid checkpoint → ./a.out --restart=...
   → SIGUSR1 → checkpoint → exit 124 → sbatch 下一輪
```

特徵:
- 無 `restart/grid_provenance`
- 不依賴 origin; 即使 `restart/step_*_origin*/` 存在也不自動插值
- Preflight C 不觸發 (無 provenance → 跳過驗證)
- jobscript 自行倒序選最新有效 checkpoint

### 2.2 Regrid Chain (網格重建)

```
./run --regrid-from-origin --old-grid OLD.dat --new-grid NEW.dat
   → Preflight A: grid_zeta_tool.py --auto (確保 NEW grid 存在)
   → Preflight B: interp_checkpoint.py --auto (origin → 新 checkpoint)
   → Preflight C: provenance 驗證 (mtime 一致性)
   → 編譯 → sbatch
   → 後續每次 ./run 自動驗證 provenance
```

特徵:
- 有 `restart/grid_provenance` (session-level grid 身份紀錄)
- 有 `restart/step_*_origin*/` (前次模擬的 checkpoint)
- Preflight C **每次續跑都驗證** provenance
- Grid 或 variables.h 變更 → FATAL，強制重新插值

---

## 3. 完整 Preflight 流程

```
./run --regrid-from-origin --old-grid <OLD> --new-grid <NEW>
 │
 ├── 參數驗證
 │   --regrid-from-origin 與 --force-cold 互斥
 │   --force-regrid 必須搭配 --regrid-from-origin
 │   --old-grid 與 --new-grid 皆為必填
 │
 ├── Preflight A: 確保 NEW grid 存在
 │   python3 restart_tools/grid_zeta_tool.py --auto
 │   └── 冪等: grid 已存在且新鮮 → 秒回
 │
 ├── Preflight B: Checkpoint 插值
 │   │
 │   ├── 前置檢查
 │   │   ├── origin 必須唯一 (0 個 → FATAL, >1 個 → FATAL)
 │   │   ├── OLD grid 存在且非空
 │   │   ├── NEW grid 存在且非空
 │   │   ├── NEW grid header I/J == variables.h NY/NZ
 │   │   ├── OLD grid header I/J == origin metadata NY/NZ
 │   │   └── HAS_CKPT=1 時:
 │   │       無 --force-regrid → FATAL
 │   │       有 --force-regrid → 清除 checkpoint/ + provenance + chain state
 │   │
 │   ├── 執行
 │   │   python3 restart_tools/interp_checkpoint.py --auto --step 1 \
 │   │       --old-grid-dat <OLD> --new-grid-dat <NEW>
 │   │
 │   ├── 產物驗證 (全部存在且非空)
 │   │   ├── restart/checkpoint/step_00000001/metadata.dat
 │   │   ├── restart/checkpoint/step_00000001/f00_0.bin
 │   │   ├── restart/checkpoint/step_00000001/rho_0.bin
 │   │   └── restart/grid_provenance
 │   │
 │   └── 插值後動作
 │       HAS_CKPT=1
 │       清除舊 chain_count / chain_jobid → HAS_STATE=0
 │       → 進入 Scenario [2] (Round 2)
 │
 ├── Preflight C-0: Orphan provenance 偵測
 │   條件: HAS_CKPT=0 且 非 cold 且 非 regrid 且 restart/grid_provenance 存在
 │   → FATAL: provenance 存在但無有效 checkpoint (不一致狀態)
 │
 └── Preflight C: Provenance 一致性驗證
     條件: HAS_CKPT=1 且 restart/grid_provenance 存在
     驗證 4 個 mtime:
       ├── origin metadata.dat
       ├── variables.h
       ├── NEW grid .dat
       └── OLD grid .dat
     任一變更 → FATAL (阻止用錯 grid 的 checkpoint 續跑)
     provenance 不存在 → 跳過 (視為普通 chain)
```

---

## 4. interp_checkpoint.py 插值演算法

### 4.1 八步流程

```
Step 1: 讀取 origin metadata.dat
        驗證 grid_dims / mpi_rank_count 與 OLD config 一致

Step 2: 建立 OLD grid 座標
        讀取 OLD Tecplot .dat → y_2d[NY6,NZ6], z_2d[NY6,NZ6]

Step 3: 讀取 19×jp 個 f-files, 計算巨觀量
        rho = sum(f_q),  u = sum(e_q * f_q) / rho
        交叉驗證: |rho_file - sum(f)| < 1e-2

Step 4: 建立 NEW grid 座標
        讀取 NEW Tecplot .dat → 同上

Step 5: 計算空間插值
        預設在 physical space 以 cell search + Lagrange-7 做 3D 插值
        NEW 物理座標 → OLD cell → 取得 (xi, eta) → 插值
        注意: GAMMA 不同時, 同一個 (j,k,i) 對應不同物理高度 z

Step 6: 填充 ghost cells
        K-direction: 線性外推
        J-direction: 週期性 wrap (±LY)

Step 7: f_neq 保守重建
        對每個方向 q = 0..18:
          f_eq_old = feq(rho_old, u_old, q)
          f_neq_old = f_old - f_eq_old
          f_neq_new = interpolate(f_neq_old)
          f_eq_new = feq(rho_new, u_new, q)
          f_new = f_eq_new + fneq_scale × f_neq_new

        驗證:
          sum(f_new) == rho_new (保守性, |diff| < 1e-10)
          min(f_new) > 0 (正值性)

Step 8: 寫入
        metadata.dat (含 provenance 欄位)
        restart/grid_provenance (session-level)
        atomic rename: step_00000001.WRITING/ → step_00000001/
```

### 4.2 Metadata 欄位

checkpoint metadata.dat 包含兩類欄位:

**標準欄位** (solver 讀寫):
```
checkpoint_version=2
mpi_rank_count=16
grid_dims=135,22,135
step=1
FTT=0.000000000000000
Force=...
dt_global=-1.0          ← 刻意寫 -1.0, 跳過 drift check
```

**Provenance 欄位** (插值專用, solver 忽略):
```
# --- provenance ---
interp_source=restart/step_12758001_originRe5600
interp_old_grid=.../adaptive_3.fine grid_I257_J129_g2.0_a0.5.dat
interp_new_grid=.../adaptive_3.fine grid_I257_J129_a0.5.dat
interp_old_gamma=2.0
interp_new_gamma=4.3217
interp_fneq_scale=1.0
interp_time=2026-05-03 14:30:00
interp_variables_h_mtime=1746268800
interp_new_grid_mtime=1746268800
interp_old_grid_mtime=1746268800
interp_origin_metadata_mtime=1746268800
```

### 4.3 grid_provenance 欄位

`restart/grid_provenance` 是 session-level 檔案, 記錄整條 chain 的 grid 身份:

```
new_grid=/abs/path/to/adaptive_3.fine grid_I257_J129_a0.5.dat
old_grid=/abs/path/to/adaptive_3.fine grid_I257_J129_g2.0_a0.5.dat
origin=/abs/path/to/restart/step_12758001_originRe5600
origin_metadata_mtime=1746268800
variables_h=/abs/path/to/variables.h
variables_h_mtime=1746268800
new_grid_mtime=1746268800
old_grid_mtime=1746268800
created=2026-05-03 14:30:00
```

- 由 interp_checkpoint.py 在插值成功後寫入
- 由 `--force-cold` 隨 `rm -rf restart/` 一起刪除
- 由 `--force-regrid` 在重新插值前刪除
- 不存在時 = 普通 chain, Preflight C 跳過

---

## 5. 情境真值表

| HAS_CKPT | HAS_STATE | provenance | MODE | 結果 |
|----------|-----------|------------|------|------|
| 0 | 0 | - | `./run` | Scenario [1] 冷啟動 |
| 0 | 0 | - | `--regrid-from-origin` | Preflight B → 插值 → Scenario [2] |
| 1 | 0 | 有 | `./run` | Preflight C → 驗證通過 → Scenario [2] |
| 1 | 1 | 有 | `./run` | Preflight C → 驗證通過 → Scenario [3B] 接續 chain |
| 1 | 1 | 無 | `./run` | Preflight C 跳過 → Scenario [3B] 普通接續 |
| 1 | any | 有 | `--regrid-from-origin` | 無 `--force-regrid` → FATAL |
| 1 | any | any | `--regrid + --force-regrid` | 清除 → 重新插值 → Scenario [2] |
| 1 | any | 有 | `./run` (stale) | Preflight C → mtime 不符 → FATAL |
| any | any | any | `--force-cold` | 全清 → Scenario [1] |
| 0 | any | 有 | `./run` | **FATAL (C-0)**: provenance 存在但無 checkpoint → 不一致 |
| 0 | 0 | - | `./run` (origin 存在) | 印 advisory, 不插值 → Scenario [1] |

---

## 6. 安全機制

### 6.1 不自動觸發插值

Origin checkpoint 存在不會自動觸發插值。必須使用者明確下達:

```bash
./run --regrid-from-origin --old-grid <OLD.dat> --new-grid <NEW.dat>
```

設計理由: 自動觸發可能在使用者不知情時用錯 grid 配對, 造成靜默數據錯誤。

### 6.2 Origin 唯一性

`restart/step_*_origin*/` 只允許存在 0 或 1 個。多個 origin → FATAL。
run.sh 和 interp_checkpoint.py 都有此檢查。

### 6.3 交叉驗證 (run.sh Preflight B)

在呼叫 Python 之前, run.sh 在 bash 層進行:
- NEW grid header `I=`/`J=` vs `variables.h` `NY`/`NZ`
- OLD grid header `I=`/`J=` vs origin metadata `grid_dims` 反推的 NY/NZ

防止使用者指定錯誤的 grid 檔案。

### 6.4 Provenance 時戳驗證 (run.sh Preflight C)

每次續跑時 (HAS_CKPT=1 且 provenance 存在) 比對 4 個檔案的 mtime:

| 檔案 | 意義 |
|------|------|
| origin metadata.dat | origin 資料被替換 |
| variables.h | 網格參數被修改 |
| NEW grid .dat | 新網格被重新生成 |
| OLD grid .dat | 舊網格被替換 |

任一 mtime 不符 → FATAL, 必須 `rm -rf restart/checkpoint/ && rm -f restart/grid_provenance` 後重新插值。

### 6.5 Chain State 重設

插值成功後自動清除 `restart/chain_count` 和 `restart/chain_jobid`,
強制進入 Scenario [2] (chain_count=2)。
防止舊的 chain state 殘留, 導致走入 Scenario [3B] 接續錯誤的 chain。

### 6.6 dt_global = -1.0

插值產生的 checkpoint 刻意將 `dt_global` 寫為 -1.0。
Solver 的 Phase 5 drift check (fileIO.h) 遇到 dt_global=-1.0 時跳過漂移檢查,
由 runtime 從新 grid 的 Jacobian metric 重新計算 dt。
避免因新舊 grid 的 minSize 差異觸發 false positive FATAL。

### 6.7 f_neq 保守重建

不直接插值 f (分佈函數), 而是:
1. 計算 f_neq = f - f_eq (非平衡態部分)
2. 插值 f_neq 到新 grid
3. 用新 grid 的 rho/u 計算新的 f_eq
4. f_new = f_eq_new + scale * f_neq_interp

保證 sum(f_new) = rho_new (保守性) 和 min(f_new) > 0 (正值性)。

### 6.8 Atomic Write

插值產物先寫入 `step_00000001.WRITING/`, 全部完成後 rename。
防止中途失敗留下不完整 checkpoint。

---

## 7. 使用範例

### 7.1 首次從 origin 插值

```bash
./run --regrid-from-origin \
    --old-grid "J_Frohlich/adaptive_3.fine grid_I257_J129_g2.0_a0.5.dat" \
    --new-grid "J_Frohlich/adaptive_3.fine grid_I257_J129_a0.5.dat"
```

### 7.2 改了 grid 後強制重建

```bash
./run --regrid-from-origin \
    --old-grid "J_Frohlich/adaptive_3.fine grid_I257_J129_g2.0_a0.5.dat" \
    --new-grid "J_Frohlich/adaptive_3.fine grid_I257_J129_a0.5.dat" \
    --force-regrid
```

### 7.3 後續正常續跑

```bash
./run
```

Provenance 自動驗證, 通過即續跑。

### 7.4 完全重來

```bash
./run --force-cold
```

刪除所有 restart/, 包含 provenance, 回到冷啟動。

---

## 8. 檔案清單

### 8.1 程式碼

| 檔案 | 角色 |
|------|------|
| `run` | 根目錄 wrapper, 路由到 chain_code/run.sh |
| `chain_code/run.sh` | 決策層: preflight A/B/C + scenario dispatch + submit |
| `restart_tools/interp_checkpoint.py` | 轉換層: checkpoint 插值 + provenance 寫入 |
| `restart_tools/grid_zeta_tool.py` | 網格生成 (Mode 2/3, Vinokur tanh) |
| `chain_code/jobscript_chain.slurm.{GB200,H200}` | jobscript: 選 latest checkpoint + 啟動 solver |

### 8.2 產物

| 檔案 | 生命週期 |
|------|----------|
| `restart/checkpoint/step_00000001/` | 插值產物, solver 續跑後可被新 checkpoint 取代 |
| `restart/grid_provenance` | 整條 chain 存續期間, `--force-cold` 或 `--force-regrid` 時清除 |
| `restart/step_*_origin*/` | 使用者手動放置, 不被系統修改或刪除 |

### 8.3 設定

| 檔案 | 角色 |
|------|------|
| `variables.h` | 唯一設定來源: NX/NY/NZ/jp/GAMMA/ALPHA/GRID_DAT_REF/UTAU_* |

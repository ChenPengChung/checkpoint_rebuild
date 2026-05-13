#ifndef VARIABLES_FILE
#define VARIABLES_FILE

// ╔════════════════════════════════════════════════════════════════════╗
// ║     GILBM Periodic Hill — Edit3_Re5600newmesh Configuration        ║
// ║     (Fröhlich curvilinear grid, D3Q19, Algorithm1 GTS-only)       ║
// ║                                                                    ║
// ║  架構 (Architecture):                                              ║
// ║    演算法:  Algorithm1 (Split-Kernel) — 唯一保留                  ║
// ║    時間步:  GTS (Global Time Step) — 唯一保留                     ║
// ║    碰撞:    MRT / BGK (由 COLLISION_MODE 切換)                    ║
// ║    插值:    Lagrange-7 / WENO7-Z (由 USE_WENO7 切換)             ║
// ║                                                                    ║
// ║  主迴圈流程 (Launch_CollisionStreaming, 方案B 融合版):              ║
// ║    FusedKernel: f_post_read → 3D interp → register → collision    ║
// ║                 → f_post_write + u/v/w/rho                        ║
// ║    MPI:         f_post_write y-halo 交換 (19 方向)                ║
// ║    periodicSW:  f_post_write x-週期 BC                            ║
// ║    Swap:        f_post_read ↔ f_post_write                       ║
// ║                                                                    ║
// ║  記憶體佈局:                                                       ║
// ║    f_post_d  [19 × GRID_SIZE] — 碰後分佈 buffer A (雙緩衝)       ║
// ║    f_post_d2 [19 × GRID_SIZE] — 碰後分佈 buffer B (雙緩衝)       ║
// ║    [已移除] feq_d — collision 自算 feq，不需獨立陣列              ║
// ║    [方案B] f_new[19] 不再使用於主迴圈 — register 取代             ║
// ║    dt/omega/s_visc             — __constant__ scalar (非陣列)      ║
// ╚════════════════════════════════════════════════════════════════════╝


// ================================================================
//  §1. 求解器開關 (Solver Switches)
// ================================================================

// ── §1a. 碰撞算子 ──
//   0 = BGK/SRT (Single Relaxation Time)
//   1 = MRT     (Multi-Relaxation-Time, d'Humieres D3Q19)
#define     COLLISION_MODE      1

// ── §1b. 插值方案 (WENO7) ──
//   Semi-Lagrangian 3-pass dimensional splitting: η(展向) → ξ(流向) → ζ(法向)
//
//   USE_WENO7 = 0: 居中 Lagrange-7 + ghost 線性外推（穩定, 無 overhead）
//   USE_WENO7 = 1: 居中 WENO7-Z 非線性權重（smooth 區 = Lagrange-7, 不連續處自動降階）
//
//   各 pass 獨立開關:
//     PASS1 (η, 展向): 均勻網格 → Lagrange 足夠
//     PASS2 (ξ, 流向): 接近均勻 → Lagrange 足夠
//     PASS3 (ζ, 法向): tanh stretching 56.5x ratio → WENO7 可捕捉不連續
//
//   WENO7 對 collision 路徑完全透明 — 只替換 interpolation 函數
#define     USE_WENO7           0       // 1=WENO7-Z, 0=純居中 Lagrange-7
#define     USE_WENO7_PASS1     0       // η: 均勻 → Lagrange
#define     USE_WENO7_PASS2     0       // ξ: 接近均勻 → Lagrange
#define     USE_WENO7_PASS3     0       // ζ: 法向 WENO7 (需 USE_WENO7=1)
#define     USE_SMEM_INTERIOR   0       // 0=V100 高速版 (non-smem, L1 cache)
                                        // 1=P100 版 (shared memory cooperative load)
                                        // V100: 128KB L1 已吸收 η-row overlap → smem 無效益
                                        // P100: 24KB L1 不足 → smem ↓85% 3D DRAM reads

// ── §1c. 自動推導開關 (勿手動修改) ──
#define     USE_MRT      (COLLISION_MODE >= 1)


// ================================================================
//  §2. 物理域幾何
// ================================================================
#define     LX      (4.5)       // 展向 (spanwise) 長度
#define     LY      (9.0)       // 流向 (streamwise) 長度 = hill-to-hill 週期長度
#define     LZ      (3.036)     // 法向 (wall-normal) 長度
#define     H_HILL  (1.0)       // hill 高度 (Re_h 參考長度)


// ================================================================
//  §3. 網格設定
// ================================================================
// ┌──────────────────────────────────────────────────────────────┐
// │  NX, NY, NZ: 格點(node)數量, 格子(cell)數 = 格點 - 1       │
// │  NX = 展向, NY = 流向 (需 (NY-1) % jp == 0), NZ = 法向     │
// │  外部網格 .dat 格式: I = NY (流向), J = NZ (法向)           │
// └──────────────────────────────────────────────────────────────┘
#define     NX      129         // 展向格點
#define     NY      257         // 流向格點 (需 (NY-1)%jp==0; 原 139→138%8≠0, 改 145→144/8=18)
#define     NZ      129         // 法向格點
#define     jp      8           //   GPU 數量 (流向分割)

// 含 ghost zone 的陣列維度 (自動計算, 勿手動修改)
//   ghost 結構: [3 ghost | N nodes | 3 ghost]
#define     NX6     (NX+6)
#define     NYD6    ((NY-1)/jp+7)
#define     NY6     (NY+6)
#define     NZ6     (NZ+6)
#define     GRID_SIZE (NX6 * NYD6 * NZ6)

// ── 網格/GPU 相容性檢查 (編譯期) ──
// (NY-1) 必須能被 jp 整除，否則 MPI 分解的全域格點數 jp*(NYD6-7)
// 會少於實際物理域 (NY-1)，導致質量修正不匹配。
//
// 有效 NY 值 (常用 jp 值):
//   jp=1: 任意 NY
//   jp=2: NY 為奇數 (NY-1 為偶數)
//   jp=4: NY = 5,9,13,...,137,141,145,...
//   jp=8: NY = 9,17,25,...,129,137,145,153,...
//   通用: NY = lcm(1..8)*k + 1 = 840k+1 (如 841, 1681,...)
#if ((NY-1) % jp != 0)
#error "FATAL: (NY-1) must be divisible by jp! Fix: change NY or jp in variables.h so that (NY-1) % jp == 0. For jp=8, valid NY: 9,17,25,...,129,137,145,153,..."
#endif

// ── 非均勻網格拉伸 ──
//   本專案使用 variable gamma(y) 網格 (Mode 3, z+ < 1.0 everywhere)
//   GAMMA: 取 gamma(y) 場最大值作為 minSize 參考 (保守值)
//          實際網格由 J_Frohlich/grid_zeta_tool.py Mode 3 生成 (main 自動觸發)
//          gamma 範圍: [2.8493, 4.3217], mean=3.3230
//   ALPHA: 拉伸對稱中心 (0.5 = 上下壁等密)
//   minSize: 由 GAMMA 與 NZ 反推的最小壁面格距 (參考值; runtime 由 Jacobian 計算 dt_global)
#define     GAMMA               2.0
#define     ALPHA               0.5

#define     CFL                 0.5
#define     minSize             (                                              \
    (LZ-1.0) * 0.5 * (1.0 + tanh(GAMMA*(1.0/(double)(NZ-1) - ALPHA))              \
                            / tanh(GAMMA*ALPHA))                                   \
)

#define     Uniform_In_Xdir     1       // 1=均勻, 0=非均勻
#define     Uniform_In_Ydir     0       // y-z 平面非均勻 (外部網格)
#define     Uniform_In_Zdir     0       // 法向非均勻 (壁面加密)

// ── 展向映射參數 ──
#define     LXi     (10.0)

// ── 外部網格 (Fröhlich Periodic Hill grid) ──
#define     GRID_DAT_DIR        "J_Frohlich"
#define     GRID_DAT_REF        "3.fine grid.dat"
/*
// ── Variable gamma(y) 網格 (Mode 3, z+ < 1.0 everywhere) ──
//   定義 UTAU_BOT_DAT / UTAU_TOP_DAT 後, --auto 自動使用 Mode 3
//   不定義則 fallback 到 Mode 2 (Poisson + 均勻 GAMMA)
#define     UTAU_BOT_DAT        "29.Re5600_j257_zplus_bottom_normal_spanavg_2nd.dat"
#define     UTAU_TOP_DAT        "28.Re5600_j257_zplus_top_spanavg_2nd.dat"
#define     UTAU_RE             5600        // u_tau 資料來源的 Re
#define     ZP_TARGET           0.9         // z+ 設計目標 (< 1.0 含安全裕度)
*/

// ================================================================
//  §4. 物理參數
// ================================================================
#define     Re      5600       // Reynolds number (基於 H_HILL 和 Uref)
#define     Uref    0.015       // 參考速度 (bulk velocity)
                                // Re700:0.0583, Re1400/2800:0.0776
                                // Re5600:0.0464, Re10595:0.0878
                                // 限制: Uref <= cs = 0.1732 (Ma < 1)
#define     niu     (Uref/Re)   // 運動黏度

// 數學常數
#define     pi      3.14159265358979323846264338327950
#define     cs      (1.0/1.732050807568877)    // 1/sqrt(3), LBM 聲速

// 時間步長
//   直角坐標系: dt = minSize (lattice c=1)
//   曲線坐標系: dt_global (runtime 由 CFL 條件計算)
//   GTS: dt_global 為全場統一 → __constant__ GILBM_dt
#define     dt      minSize

// Flow-Through Time (FTT)
//   一個 FTT = LY/Uref 個 lattice time steps
//   第 n 步的 FTT = n * dt_global / flow_through_time
#define     flow_through_time   (LY / Uref)


// ================================================================
//  §5. 模擬控制
// ================================================================
#define     loop        50000000  // 最大時間步數
#define     NDTMIT      50        // 每 N 步輸出 monitor 資料
#define     NDTFRC      50        // 每 N 步更新外力項
#define     NDTBIN      1000   // 每 N 步輸出 binary checkpoint
#define     NDTVTK      1000      // 每 N 步輸出 VTK
#define     NDTCONV     1000      // 每 N 步輸出收斂進度
#define     NDTWENO     1000      // 每 N 步輸出 WENO 診斷 (USE_WENO7=1 時啟用)

// ── FTT 閾值與統計控制 ──
// Stage 0: FTT < FTT_STATS_START → 只跑瞬時場, 不累積統計量
// Stage 1: FTT >= FTT_STATS_START → 所有 33 個統計量同時累積
#define     FTT_STATS_START     40.0    // 統計量開始累積
#define     FTT_STOP            200.0   // 模擬結束

// VTK 輸出等級
// 0 = 基本 (13 SCALARS): 瞬時速度(3)+渦度(3)+U_mean+V_mean+RS(3)+k_TKE+P_mean
// 1 = 完整: Level 0 + W_mean(展向) + 展向RS + 平均渦度 + epsilon + Tturb + Pdiff + PP_RS
#define     VTK_OUTPUT_LEVEL    0


// ================================================================
//  §6. 收斂監控系統 (Convergence Monitoring)
// ================================================================

// ── 流態自動判定 ──
#define LAMINAR_RE_THRESHOLD  150     // Re <= 此值 → 層流模式
#define IS_LAMINAR            (Re <= LAMINAR_RE_THRESHOLD)

// ── 層流收斂 (場級迭代殘差 delta) ──
//    delta = max |u(n)-u(n-1)| / |u(n)| over all grid points
#define EPS_CONVERGED       1e-8    // delta < 此值 → CONVERGED
#define EPS_NEAR            1e-7    // delta < 此值 → NEAR CONVERGED
#define N_CONFIRM_LAMINAR   10      // 連續確認次數

// ── 紊流收斂 (統計 CV = sigma/|mu| x 100%) ──
#define CV_CONVERGED        0.0001  // CV < 0.01% → CONVERGED
#define CV_NEAR             1.0     // CV < 1% → NEAR CONVERGED
#define CV_WINDOW_FTT       10.0    // CV 計算視窗 (FTT 單位)
#define CV_WINDOW_SIZE      10000   // 環形緩衝區大小 (需涵蓋完整視窗)
#define N_CONFIRM_TURB      10      // 連續確認次數


// ================================================================
//  §7. 外力控制器 (Force Controller)
// ================================================================
// 0 = Simple Proportional (C.A. Lin)
//     Force += beta * (Uref - Ub) * Uref / LZ
//     Ref: DNS-PeriodicHill_C.A.Lin.pdf
//
// 1 = Hybrid Dual-Stage (PID + Gehrke multiplicative)
//     Phase 1 (PID):    |Re%| > SWITCH_THRESHOLD
//     Phase 2 (Gehrke): |Re%| <= SWITCH_THRESHOLD
//     Ref: Gehrke & Rung (2020), Int J Numer Meth Fluids
#define     FORCE_CTRL_MODE         0

// ── Mode 1 專用參數 (FORCE_CTRL_MODE==0 時不生效) ──
#define     FORCE_KP                2.0     // PID 比例增益
#define     FORCE_KI                0.3     // PID 積分增益
#define     FORCE_KD                0.5     // PID 微分增益
#define     FORCE_GEHRKE_GAIN       0.1     // Gehrke 乘法增益
#define     FORCE_GEHRKE_DEADZONE   1.5     // Gehrke 死區 (%)
#define     FORCE_GEHRKE_FLOOR      0.1     // Force 下限 = 10% * F_Poiseuille
#define     FORCE_SWITCH_THRESHOLD  5.0     // PID→Gehrke 切換閾值 (%)
#define     FORCE_CAP_MULT          70.0    // Force 上限 = 70 * F_Poiseuille
#define     MA_BRAKE_MULT_THRESHOLD 1.7     // Mach 安全制動: 開始衰減
#define     MA_BRAKE_MULT_CRITICAL  2.1     // Mach 安全制動: 緊急歸零
#define     MA_BRAKE_GROWTH_LIMIT   0.30    // 單步增長 >30% → 額外衰減


// ================================================================
//  §7.5 外力精度方案 (Forcing Scheme Accuracy)
// ================================================================
//   0 = Legacy (零階 body force; f += w_i·(c_i·F)/cs²·δt)
//       Chapman-Enskog: NS 動量方程有 O(ε²) 殘差項 ∝ (δt/τ)∇(uF)
//       時空精度: ~O(δx^1.5)
//
//   1 = Guo-Shi-Zheng (2002) + Premnath-Abraham (2007) MRT
//       F_i = w_i [(c_i−u)/cs² + (c_i·u)c_i/cs⁴]·F     (particle space)
//       m_k^post += δt·(1−s_k/2)·[M·F_particle]_k      (moment space)
//       ρu = Σ f_i·c_i + δt·F/2                         (半力宏觀修正)
//       時空精度: O(δx²) — Chapman-Enskog 嚴格一致
//
// Ref: Guo, Zheng & Shi, PRE 65 046308 (2002)
//      Premnath & Abraham, JCP 224 (2007) 539–559
// ================================================================
#ifndef USE_GUO_FORCING
#define     USE_GUO_FORCING     1   // 預設關, Poiseuille 驗證通過後翻 1
#endif


// ================================================================
//  §7.6 壁面速度梯度精度 (Wall Velocity Gradient Order)
// ================================================================
//   2 = 二階單邊差分: (4u₁ - u₂)/2,            O(h²), 需 2 個內部點
//   6 = 六階單邊差分: (360u₁-450u₂+400u₃-225u₄+72u₅-10u₆)/60, O(h⁶), 需 6 個內部點
#define     WALL_GRAD_ORDER     6

// ================================================================
//  §7.7 Ghost Zone 外推精度 (Ghost Zone Extrapolation Order)
// ================================================================
//   2 = 二次外推 (quadratic, 3-point), O(h³), Σ|c|_max=16 (d=3)
//   3 = 三次外推 (cubic, 4-point),     O(h⁴), Σ|c|_max=34 (d=3)
//   k=3 壁面點有 4 個 stencil 內部點 → cubic 為不擴展 stencil 的全域上限
#define     GHOST_EXTRAP_ORDER  3


// ================================================================
//  §8. 重啟 (Restart) 配置
// ================================================================
//   0 = 冷啟動 (zero velocity, rho=1, 2x Poiseuille Force)
//   1 = 從 per-rank binary 續跑 (legacy)
//   2 = [REMOVED in Phase 9] 原 merged VTK 續跑; 改用 --restart 指 atomic checkpoint
//   3 = 從 binary checkpoint 續跑 (精確: f + 統計量); Phase 8 後由 argv --restart=<dir> 覆寫
#define     INIT                (0)
#define     RESTART_BIN_DIR     "restart/checkpoint/step_4001"
#define     TBINIT              (0)     // INIT=1 時: 1=讀統計量, 0=不讀

// ── 初始擾動 (觸發 3D 湍流轉捩, 湍流建立後設為 0) ──
#define     PERTURB_INIT        0       // 1=注入隨機擾動, 0=不擾動
#define     PERTURB_PERCENT     5       // 擾動振幅 (% of Uref)


// ================================================================
//  §9. GPU 設定 & 計時系統
// ================================================================
#define     NT      128          // CUDA block size (x 方向 thread 數)

// ── 計時 ──
// USE_TIMING=1 啟用, TIMING_DETAIL=1 輸出 per-kernel 分解
//
// 計時區間對應 (Edit3_Re5600newmesh 流程):
//   ev_step1:  Step1+1.5 (interpolation + macro + feq)
//   ev_step2:  Step2 (collision, 逐點 MRT/BGK)
//   ev_mpi:    MPI(f_post) + periodicSW_fpost
//   ev_iter:   整個 time step
#define     USE_TIMING          1
#define     TIMING_INTERVAL     1000
#define     TIMING_DETAIL       1


// ================================================================
//  §10. 診斷開關 (Diagnostics)
// ================================================================

// ── WENO 診斷已一體化至 USE_WENO7 (§1b) ──
// Edit3_Re5600newmesh: 原 WENO_DIAG_SWITCH / WENO_VTK_SWITCH 已移除，
//         全部由 USE_WENO7 統一控制（診斷 log + VTK contour + 啟用統計）。
//         USE_WENO7=1 時自動啟用所有 WENO 診斷功能。

// ── 效能診斷 ──
// SKIP_MIDSTEP_MASSCORR: 跳過 even/odd sub-step 間的 mid-step mass correction
//   0 = 保留, 1 = 跳過 (減少 MPI barrier)
#define     SKIP_MIDSTEP_MASSCORR    0


// ================================================================
//  §11. Deprecated (下次清理時移除)
// ================================================================
#define     TBSWITCH    (1)     // 已被 FTT_STATS_START 取代, 部分舊 code 仍依賴

#endif // VARIABLES_FILE

/*
備註: 座標系對應關係 (與 fileIO.h L1257-1277 / L1655-1717 同步)

  ── 座標軸 (VTK DIMENSIONS 仍用 code 約定: NX × NY × NZ) ──

  Code 方向    物理方向          ERCOFTAC 符號
  ──────────────────────────────────────────────
  x (i)       展向 spanwise     W (ERCOFTAC)
  y (j)       流向 streamwise   U (ERCOFTAC)
  z (k)       法向 wall-normal  V (ERCOFTAC)

  ── VTK 純量場 (統一 ERCOFTAC 公約) ──

  時間平均速度:
    VTK U_mean <- code v_tavg / Uref / N    # 流向       (ERCOFTAC <U>/Ub)
    VTK V_mean <- code w_tavg / Uref / N    # 法向       (ERCOFTAC <V>/Ub)
    VTK W_mean <- code u_tavg / Uref / N    # 展向, Level 1 (※目前 fileIO.h 未實作, Level 0/1 都不輸出)

  瞬時速度 (Level 0 起):
    VTK u_inst <- code v / Uref             # 流向 instantaneous
    VTK v_inst <- code w / Uref             # 法向 instantaneous (注意已 ÷ Uref)
    VTK w_inst <- code u / Uref             # 展向 instantaneous

  Reynolds stress (÷ Uref², Level 0 起):
    VTK uu_RS  <- (Σvv/N − v̄²) / Uref²     # ERCOFTAC <u'u'>/Ub²  (stream × stream)
    VTK vv_RS  <- (Σww/N − w̄²) / Uref²     # ERCOFTAC <v'v'>/Ub²  (wallnorm × wallnorm)
    VTK uv_RS  <- (Σvw/N − v̄w̄) / Uref²     # ERCOFTAC <u'v'>/Ub²  (stream × wallnorm)
    VTK k_TKE  <- 0.5 × (uu_RS + vv_RS + 展向 RS)   # 完整 TKE 需 Level 1 ww_RS
    VTK P_mean <- 時均壓力 (lattice unit)

  瞬時渦度 (Level 0 起, ERCOFTAC frame):
    VTK omega_u <- code ω_y                  # 流向渦度 (繞 ERCOFTAC x = code y)
    VTK omega_v <- code ω_z                  # 法向渦度 (繞 ERCOFTAC y = code z)
    VTK omega_w <- code ω_x                  # 展向渦度 (繞 ERCOFTAC z = code x)

  ⚠ 注意:
    - VTK 座標軸 (X,Y,Z) 用 code 約定 = (展向, 流向, 法向);
      但 VTK 純量名 (U,V,W,uu,vv,uv,ω) 用 ERCOFTAC 公約.
      此「軸 code、純量 paper」混合是有意設計, 避免重新跑模擬.
    - W_mean 目前在 fileIO.h L1700-1717 未實作; Level 1 標記僅為設計意圖.
    - 過去 fileIO.h 曾以 W_mean 寫法向 (舊版), 已遷移為 V_mean (新版).
      Re-dir/MRT-dir/plot_yplus_vs_xH.py still reads W_mean, will KeyError on new VTK.

鬆弛時間 (GTS):
  omega_global = 3*niu/dt_global + 0.5   (__constant__ GILBM_omega_global)
  s_visc       = 1/omega_global           (__constant__ GILBM_s_visc_global)
  dt_global    = CFL * min(1/|e^eta|, 1/|e^xi|, 1/|e^zeta|)  (__constant__ GILBM_dt)

Removed items (prior versions):
  KERNEL_ALG      - Algorithm1 only, removed Algorithm2/3
  USE_LTS         - GTS only, removed LTS path
  USE_TWO_PASS    - Algorithm2/3 only, removed
  USE_ALG3_FUSED  - Algorithm3 only, removed
  dt_local_d      - LTS array, replaced by __constant__ dt_global
  omega_local_d   - LTS array, replaced by __constant__ omega_global
  omegadt_local_d - LTS array, removed (omega*dt computed on the fly)
*/
# FlyDSL gfx950 前向 192 / 128 调优续：QK / PV 改用 16x16x32 MFMA，同功耗换频率（TUNE 3）

2026-11-09 15:25（UTC+8），MI355X gfx950 单卡，BF16，smci355-ccs-aus-n02-25。接 [TUNE 2](./2026-11-09_1400_TUNE_flydsl_mla_192_fwd_xcd.md)
（round 15，1212.71 TFLOPS，对 aiter 1.0031）。做的是 TUNE 2 "下一步" 第 1 条：两个 GEMM 从 `v_mfma_f32_32x32x16_bf16` 换成
`v_mfma_f32_16x16x32_bf16`。round 16 换 QK，round 17 换 PV。只改 `megaattention_overlap_bench/flydsl_mla/` 里的副本，不碰
Megatron / Primus / 现有 attention kernel；未提交。

## 结论

- **前向几何平均 1237.68 TFLOPS，对 aiter 1.0260（同进程，2 次全集取均值）。** 比 round 16 同进程 +1.22%，round 16 比 round 15
  同进程 +0.69%。7 个 shape 只剩 B1 S8K 低于 aiter（0.998）；B1 S64K 1.053。两轮输出误差与 round 15 到 4 位小数相同
  （O 相对误差 0.0016–0.0020，LSE 最大绝对误差 0.0046–0.0051）。
- **收益全部来自频率，周期数反而变多。** B1 S16K 持续运行：round 15 1927–1933 MHz → round 16 2025–2040 MHz（round 16 那一批），
  round 16 2003–2048 MHz → round 17 2127–2141 MHz（round 17 这一批）；整卡功耗基本不变，都在 1.33–1.39 kW。同批次
  GRBM_GUI_ACTIVE round 16 +2.9%、round 17 +4.2%，`SQ_VALU_MFMA_BUSY_CYCLES` 两轮都没变（1.136e10）。功耗在 1400 W PWR_CAP
  之下，这个节点上限制频率的是什么没有查明（MFMA 功耗探针同样没顶到 1400 W）。
- **PV 直接吃 QK 的原始 16x16 输出，不做布局转换。** QK 用 A 操作数 kv 顺序 [0, 8, 4, 12] + r % 4 之后，QK^T 的 16x16 D 块
  正好是 O^T = V^T P^T 的 B 操作数。round 16 为了保留 32x32 的 S 布局，每 tile 要 16 条 `v_permlane16_swap`；round 17 去掉之后
  主循环每 2 tile 34 → 8 条（剩下的在很少走的 rebase 分支里）。
- **两条测量教训。** 同一个包背靠背跑两次 quick 能差约 1.5%，足以翻转 1% 的结论，所以本 note 的比较都是候选与基线在同一进程
  里交替跑（2–3 轮）。"上界" 假变体若读的是别处已读过的地址，编译器会合并重复 LDS 读（144 → 108 条 `ds_read_b128`），
  它承诺的 +2.7% 实际只兑现 +0.7%。

## 逐 shape 结果

DSV3 形状：Hq = Hkv = 128、d_qk 192、d_v 128、causal、sbhd。单位 TFLOPS，每列都是同进程交替 2 次全集的均值。

| shape | round 15 | round 16（r16 批） | round 16（r17 批） | round 17 | aiter（r17 批） | 比值 |
|---|---|---|---|---|---|---|
| B1 S4K | 1080.8 | 1079.3 | 1085.9 | 1090.8 | 1066.0 | 1.023 |
| B1 S8K | 1188.1 | 1187.8 | 1193.0 | 1202.4 | 1205.2 | 0.998 |
| B1 S16K | 1278.4 | 1284.3 | 1290.0 | 1311.3 | 1270.6 | 1.032 |
| B1 S32K | 1307.9 | 1320.3 | 1324.8 | 1345.3 | 1297.2 | 1.037 |
| B1 S64K | 1326.0 | 1353.7 | 1358.9 | 1376.7 | 1307.7 | 1.053 |
| B2 S4K | 1063.6 | 1063.3 | 1069.8 | 1081.3 | 1058.2 | 1.022 |
| B2 S16K | 1245.5 | 1264.7 | 1269.7 | 1291.6 | 1268.3 | 1.018 |
| 几何平均 | 1208.81 | 1217.15 | 1222.74 | 1237.68 | 1206.27 | 1.0260 |

round 15 与 round 16 第一列是 round 16 那一批（两者交替），后三列是 round 17 那一批。同一个 round 16 在两批之间差 0.46%，
即批次漂移的量级。round 16 的收益随 S 变大（S ≤ 8K 约 0，S64K +2.1%）；round 17 每个 shape 都涨（+0.46% 到 +1.73%）。

## 逐轮记录

| 轮 | 结果 | 改动 | 同批次增益 | 比值 |
|---|---|---|---|---|
| 探针 | — | 同 FLOPs 的 MFMA 功耗对比（下表） | — | — |
| 16 | 接受 | QK 换 16x16x32：swap 放在使用处、d 分块配对、K 中间块 DMA swizzle | +0.69% | 1.0127 |
| 17 | 接受 | PV 换 16x16x32，直接用原始 QK 布局；V 用 tr 读 + 奇数行 chunk ^ 2 | +1.22% | 1.0260 |

MFMA 功耗探针（`profiles/mfma_power/mfma_power.hip`，1024 WG × 8 wave，两种指令每次迭代 FLOPs 相同，持续 12 s）：

| 指令 | TFLOPS（2 次） | GFX 频率 (MHz) | 整卡功耗 (W) |
|---|---|---|---|
| 32x32x16 | 1774.7 / 1776.6 | 1868–1873 | 1242–1258 |
| 16x16x32 | 2013.9 / 2020.9 | 2074–2077 | 1278–1286 |

round 16 里的几个中间版本（B1 S16K）：

| K 读法 | spill | bank 冲突 | GRBM | quick（与 r15 交替） |
|---|---|---|---|---|
| swap 紧跟 QK MFMA | 51 | — | — | 约 690（单次） |
| swap 放到使用处，K 自然顺序 | 9 | 8.139e8 | 1.423e8 | — |
| + d 分块配对（block 0 / 2） | 9 | 2.712e8 | 1.356e8 | 1183.9 vs 1183.7 |
| + 中间块 DMA swizzle（终版） | 9 | 0 | 1.325e8 | 1192.0 vs 1183.7 |

- swap 紧跟 QK MFMA 时，LLVM 调度器把全部 K 包同时留在寄存器里（51 个 spill）；放到下一个 cluster 的使用处只剩 9 个，
  都在主循环外。
- 中间块拆成 `ds_read_b64` 无效：LLVM 把同基址的两条合成 `ds_read2_b64`，冲突反而变成 1.626e9，spill 多 25 个。
- QK 调度 hint 按 16x16 的 MFMA 时长重算后 -0.3%（3 轮交替），没有采用。

## 实测：频率、功耗、计数器（B1 S16K）

| | round 15 | round 16 | round 16 | round 17 |
|---|---|---|---|---|
| 批次 | r16 批 | r16 批 | r17 批 | r17 批 |
| GFX 频率 (MHz，2 × 14 s) | 1927–1933 | 2025–2040 | 2003–2048 | 2127–2141 |
| 整卡功耗 (W) | 1371–1386 | 1349–1367 | 1326–1363 | 1347–1363 |
| ms / call（持续） | 8.58 / 8.63 | 8.54 / 8.56 | 8.54 / 8.53 | 8.38 / 8.39 |
| GRBM_GUI_ACTIVE | 1.288e8 | 1.325e8 | 1.330e8 | 1.386e8 |
| SQ_LDS_BANK_CONFLICT | 0 | 0 | 0 | 0 |
| SQ_INSTS_LDS | 4.741e8 | 4.741e8 | 4.741e8 | 4.741e8 |
| SQ_VALU_MFMA_BUSY_CYCLES | 1.136e10 | — | 1.136e10 | 1.136e10 |

主循环（每次迭代 2 个 tile）ISA 差异：

| | round 15 → 16 | round 16 → 17 |
|---|---|---|
| `v_mfma_f32_16x16x32_bf16` | +96 | +64（104 → 168） |
| `v_mfma_f32_32x32x16_bf16` | -48 | -32（32 → 0） |
| `v_permlane16_swap` | +34 | -26（34 → 8） |
| 指令总数 | — | 835 → 892 |
| VGPR spill（全 kernel，均在主循环外） | 0 → 9 | 9 → 13 |

- round 17 多出的 30 条 `v_max` / `v_max3` 主要在 rebase 分支：d_qk 192 没有 score bound，每 tile 的触发条件仍是 32 个值的
  max 树，分支里按新布局再建两棵 16 值的树，没有共用。
- 推断（未测）：round 17 多出的周期来自两倍数量、更短的 PV MFMA 的发射与交错，以及 V 读的两个基址的地址计算。每 k-step 的
  8 条 PV MFMA 互不依赖，同一累加器隔两个 k-step 才复用，不是依赖链。

## 布局要点

16x16x32 bf16 MFMA 的操作数布局（lane l，e = 0..7 或 0..3）：A 的 lane 持有 A[l % 16][8 (l // 16) + e]，B 持有
B[8 (l // 16) + e][l % 16]，D 持有 D[4 (l // 16) + e][l % 16]。

- QK（round 16）：每 32 kv 一个 half，4 个 16x16 块（kv 块 × q 块），d_qk 192 分 6 个 k-step。K 行按 [0, 8, 4, 12] + r % 4
  排列，原始 half 的第 4 (2 kt + qt) + e 个元素是 kv = 32 h + 16 kt + [0, 8, 4, 12][l // 16] + e、q = l % 16 + 16 qt。
- PV（round 17）：上面每个 (half, qt) 取 8 个元素 [4 qt .. 4 qt + 3, 8 + 4 qt .. 8 + 4 qt + 3]，就是 B 操作数（kv × q）。
  V^T 作 A 操作数，用 `ds_read_b64_tr_b16` 从原来的 V LDS 行读出，每 lane 的 kv 偏移同样是 [0, 8, 4, 12][l // 16]。
  每 tile 4 个 k-step × 8 个 dv 块 × 2 个 q 块 = 32 条 PV MFMA（原来 32x32x16 是 16 条，FLOPs 相同）。
- 行标量（参考 max、l、LSE）仍是每 lane 一个 q（q = l % 32）。两个 q 块之间用一条 `permlane16_swap(x, x)` 配对；row max 是
  每个 q 块一棵 16 值的树，swap 一次后接原来的 permlane32 配对归约；row sum 是每个 q 块一条乘全 1 的 MFMA。
- O 写回：每个 dv 块（含两个 q 块）两条 `permlane16_swap`，然后每 lane 一次 16 字节写。

## gfx950 LDS 规则（本轮实测）

- `ds_read_b128` 按 16 lane 一个 phase 服务，每个 32 lane 的半边再按 quad 奇偶分（quad {0, 3, 5, 6} / {1, 2, 4, 7}），
  bank = dword 地址 % 64。
- `ds_read_b64` 的 phase 是 lane 0–31 / 32–63；`ds_read2_b64` 表现得像 16 字节读。
- 16x16x32 的 A 布局下，k chunk 组 0 / 1（以及 2 / 3）落在同一个 phase，必须命中相同的 d % 64：把相差 128 的 d chunk
  配对；没有配对对象的 d，在一半行上把 16 字节 chunk 下标 XOR 4，通过 direct-to-LDS DMA 的每 lane 全局地址实现，
  写侧不花代价（冲突 8.1e8 → 0）。V 的 tr 读同理，奇数行 chunk ^ 2，冲突为 0。
- `ds_read_b64_tr_b16` 的 phase 规则没有测；V 的 swizzle 按 32 lane phase 设计，若实际是 16 lane 也无害。

## 环境与方法

- 节点 smci355-ccs-aus-n02-25；容器 `xiaoming-dev`；ROCm 10.1.0、torch 2.14.0+rocm10.1.0；HIP 设备 0（`amd-smi` GPU 3）。
- 精度：每个 batch 256 个随机行加边界行做 fp32 参考，O 相对误差 ≤ 0.003、LSE 最大绝对误差 ≤ 0.01；全集 7 个 shape、quick 4 个
  shape 每次都检查。
- 计时：`quick_test_bench.py`，候选与基线同进程交替（`profiles/ab_rotate.sh`，quick 3 轮；全集 2 轮，含 aiter）。
- 功耗 / 频率：`profiles/clock_power.sh`，持续 14 s，第 6 s 起每 2 s 用 `amd-smi metric -g 3 -c -p` 采样，两包交替各 2 次。
- 计数器：`rocprofv3 --pmc`（`profiles/pmc_pass.sh`），同批次两包各跑一遍。

## 数据

campaign 目录 `Primus-Turbo/agent/workspace/flash_attn_fwd_mla_flydsl_gfx950_20261109/`：

- `rounds/round-16/`、`rounds/round-17/`：`summary.md`、`kernel_snapshot/`、`artifacts/`（全集交替的 csv / log、quick 交替、
  频率功耗、PMC、round 17 的主循环指令直方图）。
- `logs/optimize.md`（含 MFMA 功耗探针）、`logs/performance_trend.md`（含两批次的 round 15 / 16 重测行）。
- `profiles/mfma_power/`（功耗探针与 LDS 冲突微基准）、`profiles/variants/`（round 16 的各个 ISA 变体）、
  `profiles/pkg_r15/`、`profiles/pkg_r16/`（交替测量用的基线包）。

## 下一步

1. 回收周期而不丢频率：两轮累计周期多约 7%（同批次 GRBM 比值相乘），频率多约 10%。PV 的调度 hint 还是按 32x32x16
   （每 tile 16 条）写的，现在是 32 条更短的 MFMA，重调后实测（round 16 重调 QK hint 是 -0.3%）。
2. rebase 分支与每 tile 触发共用两棵 16 值 max 树，热路径多一条 `v_max`，分支变短。
3. B1 S8K 仍为 aiter 的 0.998；B = 2 的 batch 顺序（TUNE 2 下一步第 2 条）还没做。

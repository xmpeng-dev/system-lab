# M0-overlap-impl：Primus / Turbo 现有 ring attention 的 overlap 实测（MI355X，fwd + bwd）

日期 2026-11-08 09:15。接 [M0 overlap 基线](./2026-11-08_0830_BASELINE_m0_cp_ring_attn_overlap.md)：那篇测的是自写的 ring pipeline，
这篇测 Primus、Primus-Turbo **当前已支持**的 ring attention，原样调用，不做任何优化。

## 结论

- **Primus 训练路径里，只有 TE 这条有 ring overlap。** `use_turbo_attention: false`（默认）+ `cp_comm_type: p2p` 走
  TE `AttnFuncWithCPAndKVP2P`。`use_turbo_attention: true` 时 `PrimusTurboAttention` 把 `ring_group` 写死成单卡组
  （`primus/backends/megatron/core/extensions/primus_turbo.py:1007-1011`，`TODO (limou) enable ring attention`），
  数据切分也只接受 `cp_comm_type == "a2a"`，CP 只剩同步的 Ulysses all-to-all。Turbo 自己的 `usp/attention_ring.py`
  支持 ring（op 级单测覆盖，BF16），但 Primus 没接上。
- **TE overlap 相对不 overlap（单个 attention 层 fwd + bwd）：CP2 省 2.2–11.0%，CP4 省 15.1–32.2%，CP8 省 21.1–36.2%**
  （1.02–1.57×）。CP2 省得少是因为通信本身只有 3–12 ms，overlap 后通信几乎全部仍暴露。
- **overlap 效率 `(不 overlap − overlap) / T_comm`（fwd + bwd）：TE CP2 0.66–0.93、CP4 0.58–1.14、CP8 0.32–1.10；
  Turbo CP2 0.57–0.77、CP4 0.74–0.88、CP8 0.55–0.95。** 最差在 CP8 32K（TE causal 0.32）。
- **距 roofline（每个 phase max(T_comm, T_comp) 之和）：TE overlap 是 roofline 的 1.03–1.39×，Turbo 1.04–1.54×。**
  绝大多数格子是 comp-bound；TE 在 CP4 32K causal、CP8 32K、CP8 64K causal 是 comm-bound，roofline 等于 T_comm。
- **反向 overlap 比前向差，CP2 最明显**：TE 前向 0.89–1.33、反向 0.41–0.73。原因未定位（未 profile）。
- **TE 绝对耗时更短**：overlap 后 TE / Turbo = 0.48–0.96，CP2 / CP4 causal 下 0.48–0.66（TE 负载均衡切分；Turbo 连续切分，反向
  dK/dV 以 fp32 传，CP2 32K 反向通信 320 MiB 对 TE 128 MiB）。
- **TE 在 CP > 2 时 P2P 不走 batch**（`NVTE_BATCH_MHA_P2P_COMM` 默认 0，只有 CP == 2 才 batch），同样字节的前向纯通信
  TE 比 Turbo 慢 1.75–1.93×（CP4 32K 3.22 vs 1.74 ms，CP8 128K 14.53 vs 7.63 ms）。未测打开 batch 的效果。

## 环境与方法

8× MI355X（gfx950），ROCm 10.1.0，PyTorch 2.14.0+rocm10.1.0，RCCL 2.30.7，TransformerEngine 2.17.1+130099ce4
（FusedAttention sub-backend 1，即 CK；环境变量与 Primus `tools/installation/env.sh` 一致：`NVTE_FLASH_ATTN=0`、
`NVTE_FUSED_ATTN=1`、`NVTE_CK_USES_FWD_V3=1`、`NVTE_CK_USES_BWD_V3=1`、`NVTE_CK_IS_V3_ATOMIC_FP32=0`），
primus_turbo 0.6.1.dev4（与工作区源码一致，aiter 后端）。B = 1，32 Q / 8 KV heads，D = 128，BF16，bshd。
TE 输入用 Megatron 负载均衡切分（rank r 持有 2W 段中的第 r 与 2W−1−r 段）；Turbo 用连续切分，Ulysses 组大小 1。

四个变体都在 benchmark 里运行时包装实现自己的 P2P 函数（TE `flash_attn_p2p_communicate`、Turbo `RingComm.commit`），
库源码不改：

- overlap：原样。
- 不 overlap：每次 P2P 先等之前所有计算，P2P 完成后才开始后面的计算（同样的调用、同样的 buffer）。
- 纯计算：去掉 P2P（接收 buffer 内容无效，只计时）。
- 纯通信 T_comm：把真实 fwd / bwd 记录到的 P2P 序列逐个重放，不做 attention。

device 侧 spin 闸门排除 host 下发延迟，每点 100 次迭代（长配置按 12 s 预算，至少 30 次），取最慢 rank 的 median；
反向计时前的 forward 在计时窗口外。正确性 36/36 通过：输出 / dK / dV 对全序列 aiter 参考 rel-L2 ≤ 5.9e-3，输出 / dq 对
每 rank 128 行 fp32 公式精确值 ≤ 1.9e-2（Turbo dq 3.0e-3；TE dq 随 S 增大到 1.9e-2，推测是 CK v3 bwd 的 bf16 atomic，
未验证）。aiter 全序列参考自己的 causal dq 误差达 1.0–2.6e-2，所以 dq 以 fp32 精确行为准。

工程：`/perf_apps/xiaoming/megaattention_overlap_bench/ring_impl.py`（`scripts/run_ring_impl.sh`，汇总 `ring_impl_report.py`），
未 commit。数据：[data/2026-11-08-ring-impl/](./data/2026-11-08-ring-impl/)。

## fwd + bwd：通信 / 计算拆分与 roofline（单个 attention 层，ms）

各列都是 fwd、bwd 两个 phase 的 median 相加（写成「fwd + bwd = 合计」）：

- T_comm：纯通信变体，只重放真实 fwd / bwd 记录到的 P2P 序列，不做 attention。
- T_comp：纯计算变体，去掉 P2P，只做 attention。
- 瓶颈：每个 phase 里 T_comm 与 T_comp 谁大（comm / comp）。
- roofline = max(T_comm_fwd, T_comp_fwd) + max(T_comm_bwd, T_comp_bwd)，即每个 phase 里较短的一方被完全藏住时的耗时。
  这是下界：没算 ring 的依赖（比如前向最后一步没有通信可藏），也没算两者并发时互相拖慢，实际可达的下界比它高。
- 省 = 1 − overlap / 不 overlap；效率 = (不 overlap − overlap) / T_comm。

**TE**（`AttnFuncWithCPAndKVP2P`，负载均衡切分）

| CP | S | causal | T_comm fwd + bwd | T_comp fwd + bwd | 瓶颈 fwd/bwd | roofline | 不 overlap | overlap | 省 | overlap − roofline | overlap / roofline | 效率 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 32K | 0 | 1.07 + 2.16 = 3.24 | 7.56 + 20.17 = 27.73 | comp/comp | 27.73 | 32.66 | 30.21 | 7.5% | 2.48 | 1.09 | 0.76 |
| 2 | 32K | 1 | 1.08 + 2.16 = 3.24 | 4.07 + 10.51 = 14.58 | comp/comp | 14.58 | 19.30 | 17.18 | 11.0% | 2.59 | 1.18 | 0.66 |
| 2 | 64K | 0 | 2.10 + 4.21 = 6.30 | 28.59 + 76.61 = 105.20 | comp/comp | 105.20 | 115.84 | 110.78 | 4.4% | 5.58 | 1.05 | 0.80 |
| 2 | 64K | 1 | 2.10 + 4.20 = 6.30 | 14.91 + 38.64 = 53.56 | comp/comp | 53.56 | 64.96 | 60.06 | 7.6% | 6.50 | 1.12 | 0.78 |
| 2 | 128K | 0 | 4.15 + 8.29 = 12.44 | 112.93 + 299.96 = 412.89 | comp/comp | 412.89 | 434.72 | 425.16 | 2.2% | 12.27 | 1.03 | 0.77 |
| 2 | 128K | 1 | 4.15 + 8.29 = 12.44 | 57.33 + 153.64 = 210.97 | comp/comp | 210.97 | 231.77 | 220.15 | 5.0% | 9.18 | 1.04 | 0.93 |
| 4 | 32K | 0 | 3.22 + 6.30 = 9.52 | 4.02 + 10.74 = 14.76 | comp/comp | 14.76 | 25.87 | 18.06 | 30.2% | 3.29 | 1.22 | 0.82 |
| 4 | 32K | 1 | 3.22 + 6.30 = 9.52 | 2.24 + 5.93 = 8.17 | comm/comm | 9.52 | 18.81 | 13.26 | 29.5% | 3.74 | 1.39 | 0.58 |
| 4 | 64K | 0 | 6.23 + 12.29 = 18.53 | 15.13 + 39.67 = 54.80 | comp/comp | 54.80 | 81.18 | 60.80 | 25.1% | 6.00 | 1.11 | 1.10 |
| 4 | 64K | 1 | 6.23 + 12.30 = 18.53 | 8.11 + 21.37 = 29.48 | comp/comp | 29.48 | 52.98 | 35.93 | 32.2% | 6.45 | 1.22 | 0.92 |
| 4 | 128K | 0 | 12.21 + 24.26 = 36.47 | 57.17 + 140.50 = 197.67 | comp/comp | 197.67 | 262.59 | 223.00 | 15.1% | 25.34 | 1.13 | 1.09 |
| 4 | 128K | 1 | 12.21 + 24.26 = 36.47 | 30.21 + 78.86 = 109.07 | comp/comp | 109.07 | 160.06 | 118.33 | 26.1% | 9.26 | 1.08 | 1.14 |
| 8 | 32K | 0 | 4.04 + 7.61 = 11.65 | 2.25 + 6.19 = 8.43 | comm/comm | 11.65 | 21.42 | 14.95 | 30.2% | 3.30 | 1.28 | 0.56 |
| 8 | 32K | 1 | 4.04 + 7.61 = 11.65 | 1.32 + 3.39 = 4.71 | comm/comm | 11.65 | 17.94 | 14.16 | 21.1% | 2.51 | 1.22 | 0.32 |
| 8 | 64K | 0 | 7.53 + 14.61 = 22.13 | 7.78 + 21.47 = 29.25 | comp/comp | 29.25 | 55.30 | 36.52 | 34.0% | 7.27 | 1.25 | 0.85 |
| 8 | 64K | 1 | 7.52 + 14.61 = 22.13 | 4.35 + 11.83 = 16.18 | comm/comm | 22.13 | 40.89 | 28.39 | 30.6% | 6.26 | 1.28 | 0.56 |
| 8 | 128K | 0 | 14.53 + 28.55 = 43.08 | 30.06 + 80.55 = 110.60 | comp/comp | 110.60 | 169.51 | 121.95 | 28.1% | 11.35 | 1.10 | 1.10 |
| 8 | 128K | 1 | 14.53 + 28.55 = 43.08 | 15.47 + 42.39 = 57.86 | comp/comp | 57.86 | 112.50 | 71.81 | 36.2% | 13.95 | 1.24 | 0.94 |

**Turbo**（`usp/attention_ring.py`，连续切分）

| CP | S | causal | T_comm fwd + bwd | T_comp fwd + bwd | 瓶颈 fwd/bwd | roofline | 不 overlap | overlap | 省 | overlap − roofline | overlap / roofline | 效率 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 32K | 0 | 1.09 + 5.32 = 6.40 | 7.50 + 27.25 = 34.75 | comp/comp | 34.75 | 43.05 | 39.40 | 8.5% | 4.65 | 1.13 | 0.57 |
| 2 | 32K | 1 | 1.09 + 5.31 = 6.40 | 5.86 + 20.91 = 26.76 | comp/comp | 26.76 | 35.09 | 31.39 | 10.5% | 4.63 | 1.17 | 0.58 |
| 2 | 64K | 0 | 2.11 + 10.42 = 12.54 | 30.32 + 105.93 = 136.25 | comp/comp | 136.25 | 150.86 | 142.60 | 5.5% | 6.35 | 1.05 | 0.66 |
| 2 | 64K | 1 | 2.12 + 10.42 = 12.53 | 20.20 + 80.04 = 100.25 | comp/comp | 100.25 | 118.14 | 110.27 | 6.7% | 10.03 | 1.10 | 0.63 |
| 2 | 128K | 0 | 4.16 + 20.62 = 24.77 | 108.54 + 420.88 = 529.42 | comp/comp | 529.42 | 567.61 | 548.63 | 3.3% | 19.21 | 1.04 | 0.77 |
| 2 | 128K | 1 | 4.16 + 20.62 = 24.78 | 77.02 + 316.03 = 393.05 | comp/comp | 393.05 | 435.09 | 416.24 | 4.3% | 23.19 | 1.06 | 0.76 |
| 4 | 32K | 0 | 1.74 + 6.09 = 7.83 | 4.31 + 14.35 = 18.66 | comp/comp | 18.66 | 27.88 | 21.78 | 21.9% | 3.12 | 1.17 | 0.78 |
| 4 | 32K | 1 | 1.74 + 6.09 = 7.82 | 3.84 + 12.80 = 16.64 | comp/comp | 16.64 | 25.78 | 19.99 | 22.5% | 3.34 | 1.20 | 0.74 |
| 4 | 64K | 0 | 3.26 + 11.66 = 14.92 | 15.98 + 54.19 = 70.18 | comp/comp | 70.18 | 88.06 | 75.96 | 13.7% | 5.78 | 1.08 | 0.81 |
| 4 | 64K | 1 | 3.26 + 11.66 = 14.92 | 14.04 + 47.84 = 61.87 | comp/comp | 61.87 | 79.99 | 67.75 | 15.3% | 5.87 | 1.09 | 0.82 |
| 4 | 128K | 0 | 6.31 + 22.81 = 29.12 | 59.09 + 211.34 = 270.43 | comp/comp | 270.43 | 306.32 | 281.17 | 8.2% | 10.74 | 1.04 | 0.86 |
| 4 | 128K | 1 | 6.31 + 22.81 = 29.12 | 51.79 + 185.35 = 237.14 | comp/comp | 237.14 | 273.77 | 248.20 | 9.3% | 11.05 | 1.05 | 0.88 |
| 8 | 32K | 0 | 2.31 + 6.98 = 9.28 | 2.40 + 7.90 = 10.30 | comp/comp | 10.30 | 20.81 | 15.64 | 24.9% | 5.34 | 1.52 | 0.56 |
| 8 | 32K | 1 | 2.31 + 6.98 = 9.28 | 2.19 + 7.54 = 9.73 | comm/comp | 9.84 | 20.29 | 15.19 | 25.1% | 5.35 | 1.54 | 0.55 |
| 8 | 64K | 0 | 4.09 + 12.79 = 16.89 | 8.57 + 28.54 = 37.12 | comp/comp | 37.12 | 56.97 | 42.28 | 25.8% | 5.16 | 1.14 | 0.87 |
| 8 | 64K | 1 | 4.09 + 12.80 = 16.88 | 7.97 + 26.92 = 34.90 | comp/comp | 34.90 | 54.74 | 40.16 | 26.6% | 5.27 | 1.15 | 0.86 |
| 8 | 128K | 0 | 7.63 + 24.47 = 32.10 | 32.55 + 108.38 = 140.93 | comp/comp | 140.93 | 178.42 | 149.54 | 16.2% | 8.61 | 1.06 | 0.90 |
| 8 | 128K | 1 | 7.64 + 24.48 = 32.11 | 30.16 + 101.78 = 131.94 | comp/comp | 131.94 | 170.36 | 139.83 | 17.9% | 7.89 | 1.06 | 0.95 |

从这两张表直接读出的事实：

- 36 格全部 overlap ≥ roofline。TE overlap 是 roofline 的 1.03–1.39×（多 2.48–25.34 ms），Turbo 1.04–1.54×
  （多 3.12–23.19 ms）。
- 离 roofline 最远：TE CP4 32K causal 1.39×，Turbo CP8 32K 1.52–1.54×。最近：TE CP2 128K 非 causal 1.03×，
  Turbo CP2 / CP4 128K 非 causal 1.04×。
- Turbo 只有 CP8 32K causal 的前向是 comm-bound（2.31 vs 2.19 ms），其余全是 comp-bound。TE 有 4 格两个 phase 都是
  comm-bound：CP4 32K causal、CP8 32K（causal 与非 causal）、CP8 64K causal。这几格 roofline 就是 T_comm，计算再快也
  压不到它以下；TE 这几格前向 T_comm 是 Turbo 的 1.75–1.85×（CP > 2 不走 batch P2P，见结论）。
- 36 格的「不 overlap」都比 T_comm + T_comp 多 1.1–28.5 ms（每步等最慢的 peer），所以效率可以超过 1。

fwd / bwd 分表、P95、正确性明细见 `data/2026-11-08-ring-impl/tables.md` 与 `results.csv`。

## 事实与假设

- 事实（本次实测）：上表；反向 overlap 在 CP2 下效率 0.41–0.73（TE）/ 0.48–0.70（Turbo）；TE CP > 2 非 batch P2P
  纯通信比 Turbo 慢 1.75–1.93×；Turbo 反向 dK/dV 走 fp32；Primus Turbo 路径 ring 被禁用（代码位置见上）。
- 假设（未验证）：反向 overlap 差来自反向 kernel 与 RCCL 的 CU 争用或调度；TE dq 误差偏大来自 bf16 atomic。
- 整步训练（2026-11-08 10:10 补测，见 [端到端篇](./2026-11-08_1010_BASELINE_llama_e2e_te_ring_overlap.md)）：Llama3.1-8B
  整步省的时间 ≈ 32 × 上表 TE causal 单层省的时间（比值 0.93–1.12），每步省 CP2 4.4–5.2% / CP4 19.1–23.8% / CP8 15.3–31.6%。
- 适用范围：本篇只测了 GQA（32 Q / 8 KV heads，D = 128）。MLA 的通信 / 计算比按形状推算约是它的 4 倍，结论不能直接套用。
  MLA 实测（TE 3.3–4.3 倍，CP4 / CP8 全部 comm-bound）见 [MLA 篇](./2026-11-08_1220_BASELINE_dsv3_mla_ring_attn_overlap.md)。
- 与自写 pipeline 对比：CP2 32K 非 causal 前向 overlap，TE 7.97 ms，自写 pipeline 8.23 ms，Turbo 8.55 ms。

## 优化空间（只列观察，不在本篇做）

反向 overlap（尤其 CP2）、Primus Turbo 路径接上 ring、TE CP > 2 的 P2P batch、Turbo 反向 dK/dV 精度与切分方式。

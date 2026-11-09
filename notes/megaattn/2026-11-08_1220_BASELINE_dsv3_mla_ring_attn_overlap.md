# M0-overlap-MLA：DSV3 MLA 形状下现有 ring attention 的 comm / comp / roofline 与 overlap（MI355X，kernel 级）

日期 2026-11-08 12:20。接 [现有 ring 实现 overlap（GQA）](./2026-11-08_0915_BASELINE_te_turbo_ring_attn_overlap.md)：同一套
benchmark、同样四个变体，形状换成 DeepSeek-V3 MLA 在 core attention 看到的样子。只测 TE、Turbo **当前已支持**的实现，原样
调用，不做优化。

## 结论

- **MLA 的 ring 通信量是 GQA 的 20 倍，计算量（FLOPs）只有 5.0–5.2 倍（TE 实测纯计算耗时 4.3–5.7 倍），TE 的通信 / 计算比
  因此是 GQA 的 3.3–4.3 倍。** ring 传的是展开后的 K / V（128 heads × (192 + 128) × 2 B = 80 KiB / token），不是 latent；
  GQA 32 / 8 heads × D 128 是 4 KiB / token。
- **TE（Primus 默认 MLA CP 路径，`cp_comm_type: p2p`）在 CP4 / CP8 全部 comm-bound，roofline 就是 T_comm。** fwd + bwd 的
  T_comm 是 T_comp 的 1.22–8.49 倍（CP4 1.22–4.10、CP8 2.80–8.49）。overlap 后是 roofline（= T_comm）的 1.16–1.30 倍，相对
  不 overlap 省 CP4 17.4–35.1%、CP8 8.2–24.7%。计算再快也压不到 T_comm 以下。
- **TE CP2 除 32K 前向外都是 comp-bound，overlap 已接近 roofline**：overlap / roofline 1.04–1.28，省 18.6–36.3%。
- **TE 在 CP > 2 时 P2P 不走 batch，实测链路 33.5–33.7 GB/s；CP2（batch）65.6–65.8 GB/s；Turbo（总是 batch）65.5–66.0 GB/s。**
  按 AMD 规格每对 GPU 单向 72 GB/s 算，分别是 47% 与 91–92%。同样字节的前向纯通信 TE CP4 / CP8 是 Turbo 的 1.95–1.96 倍。
- **按规格算的理论下界，MLA 在 CP4 / CP8 也是 comm-bound**：理论 T_comm / 理论 T_comp = CP2 0.44–1.78、CP4 1.33–5.33、
  CP8 3.11–12.45。TE overlap 是理论 roofline 的 1.81–2.79 倍，差距主要来自 TE 非 batch P2P 只跑到规格一半、attention kernel
  只跑到 BF16 峰值的 30–51%。
- **Turbo ring（Primus 没接上）CP4 / CP8 前向更快，反向慢一个量级。** 靠 batch P2P，CP4 / CP8 前向 overlap 是 TE 的
  0.53–0.84 倍（CP2 两家都 batch，Turbo 反而慢 1.29–1.37 倍，前向纯计算 Turbo 是 TE 的 1.36–1.87 倍）；反向纯计算是 TE 的
  8.4–10.6 倍（BF16 峰值的 3–5%）。fwd + bwd overlap 后 TE / Turbo = 0.15–0.65，只有 CP8 32K Turbo 略快（TE / Turbo 1.05）。
  Turbo 反向发送字节是 TE 的 2.0–3.0 倍（CP2 3.0、CP4 2.2、CP8 2.0）。
- **正确性**：32K / 64K 两家全部通过。128K 下 aiter 全序列参考自己的反向失效（dq 对 fp32 精确行误差 0.86–1.0），只能用 fp32
  精确行检查 out / dq：TE 全部通过，**Turbo CP2 128K dq 对 fp32 精确行误差 0.33，不通过**；128K 的 dK / dV 没有独立参考，
  未验证。
- **对「tile 级 overlap 空间」的含义（MLA）**：CP4 / CP8 下 overlap 已经把计算几乎全部藏在通信下面，剩下的是
  overlap − T_comm（TE 31.8–216.7 ms，T_comm 的 16–30%）。融合 / tile 级 overlap 最多只能吃掉这部分；更大的量在通信本身
  （batch 与否差 2 倍、理论 T_comm 还要再低 1.1 倍左右）。这是由上面数据推出的，没有做实验验证。

## 形状与方法

DeepSeek-V3 MLA 训练时（不 absorb），core attention 的输入是 128 Q heads、128 KV heads（每个 head 独立的 K / V，即 MHA 形式），
d_qk = 192（128 nope + 64 rope），d_v = 128，causal。B = 1，BF16，bshd。CP 2 / 4 / 8 × S 32K / 64K / 128K，只测 causal。
softmax scale 用默认 1 / sqrt(192)；DSV3 实际是 yarn mscale² / sqrt(192) = 0.1352，不影响耗时。

环境、四个变体（overlap / 不 overlap / 纯计算 / 纯通信重放）、计时方式（device 侧 spin 闸门，最慢 rank 的 median）、切分方式
（TE 负载均衡切分、Turbo 连续切分）与 [GQA 篇](./2026-11-08_0915_BASELINE_te_turbo_ring_attn_overlap.md) 完全相同。
TE 用 `DotProductAttention(kv_channels=(192, 128))`，走 FusedAttention（CK）；Turbo 用 `usp/attention_ring.py`，Ulysses 组大小 1。

roofline 两种：

- 实测 roofline = Σ_phase max(T_comm, T_comp)，T_comm / T_comp 来自纯通信 / 纯计算变体。是下界，没算 ring 依赖和并发拖慢。
- 理论 roofline = Σ_phase max(理论 T_comm, 理论 T_comp)：
  - 理论 T_comp = FLOPs / 2.5166 PFLOPS。每 rank FLOPs = pairs × 128 × c / W，pairs = S(S + 1) / 2（causal），
    前向 c = 2(d_qk + d_v) = 640，反向 c = 2(3 d_qk + 2 d_v) = 1664。
  - 理论 T_comm = 实测每 rank 发送字节 / 72 GB/s。
  - 规格来源：BF16 稠密 2.5166 PFLOPS、7 × 144 GB/s Infinity Fabric 见
    [AMD MI355X brochure](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf)
    与 [AMD 产品页](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html)；
    [NEC datasheet](https://www.nec.com/en/global/solutions/hpc/lx/images/AMD/amd-instinct-mi355x-gpu-datasheet.pdf) 写 7 × 153.6 GB/s。
    72 GB/s 是把 144 GB/s 当作双向、取一半得到的单向值，规格文件没有明说是双向。实测 batch P2P 65.5–66.0 GB/s，与「单向 72」
    一致；如果 144 是单向，实测只有 46%。

工程：`/perf_apps/xiaoming/megaattention_overlap_bench/ring_impl.py`（新增 `--q-heads --kv-heads --d-qk --d-v`），
`OUT=results/ring_impl_mla CPS="2 4 8" scripts/run_ring_impl.sh --q-heads 128 --kv-heads 128 --d-qk 192 --d-v 128 --causal 1`，
汇总 `ring_impl_report.py results/ring_impl_mla`。未 commit。数据：[data/2026-11-08-ring-impl-mla/](./data/2026-11-08-ring-impl-mla/)。

## 现有 MLA CP 实现

| 实现 | 有无 comm–compute overlap | MLA（d_qk ≠ d_v）支持 | Primus 是否可用 | 本篇 |
|---|---|---|---|---|
| TE `p2p`（`AttnFuncWithCPAndKVP2P`） | 有：前向双 stream，反向单 stream ring | K / V 展平后拼接发送，无 padding | 是（默认） | 已测 |
| TE `a2a+p2p` | p2p 段有 | 同上 | 是 | 未测 |
| TE `all_gather` / `a2a` | 无（同步 all-gather + reduce-scatter / all-to-all） | 后端过滤里没有按 d_qk ≠ d_v 禁用 CP 的条件（TE `utils.py:1136-1205`），未实测 | 是 | 未测 |
| Turbo ring（`usp/attention_ring.py`） | 有：`batch_isend_irecv` | 支持，但反向把 V / out / dout pad 到 d_qk（`flash_attn_usp_interface.py:231-234`） | 否（`primus_turbo.py:1007-1011` ring_group 单卡） | 已测 |
| Turbo Ulysses（Primus `use_turbo_attention: true`） | 无 | 支持 | 是（`cp_comm_type: a2a`） | 未测 |

Megatron MLA 的 CP 只透传 `cp_comm_type` 给 TE；hybrid CP 不支持 MLA（`multi_latent_attention.py:532`）。

## fwd + bwd：通信 / 计算拆分与 roofline（单个 attention 层，ms）

列定义同 GQA 篇：各列是 fwd、bwd 两个 phase 的 median 相加；roofline = 每个 phase max(T_comm, T_comp) 之和；
省 = 1 − overlap / 不 overlap；效率 = (不 overlap − overlap) / T_comm。comm-bound 时效率上限是 T_comp / T_comm，不能和
comp-bound 格子直接比。

**TE**（`AttnFuncWithCPAndKVP2P`，负载均衡切分）

| CP | S | T_comm fwd + bwd | T_comp fwd + bwd | 瓶颈 fwd/bwd | roofline | 不 overlap | overlap | 省 | overlap − roofline | overlap / roofline | 效率 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 32K | 20.47 + 40.91 = 61.37 | 19.33 + 58.72 = 78.04 | comm/comp | 79.18 | 158.93 | 101.20 | 36.3% | 22.02 | 1.28 | 0.94 |
| 2 | 64K | 40.85 + 81.69 = 122.55 | 71.93 + 219.75 = 291.68 | comp/comp | 291.68 | 453.74 | 327.41 | 27.8% | 35.73 | 1.12 | 1.03 |
| 2 | 128K | 81.61 + 163.28 = 244.89 | 297.82 + 860.75 = 1158.57 | comp/comp | 1158.57 | 1485.45 | 1208.67 | 18.6% | 50.11 | 1.04 | 1.13 |
| 4 | 32K | 60.02 + 119.86 = 179.88 | 11.24 + 32.65 = 43.89 | comm/comm | 179.88 | 256.12 | 211.65 | 17.4% | 31.77 | 1.18 | 0.25 |
| 4 | 64K | 119.79 + 239.42 = 359.21 | 40.29 + 116.04 = 156.33 | comm/comm | 359.21 | 583.86 | 428.98 | 26.5% | 69.76 | 1.19 | 0.43 |
| 4 | 128K | 239.33 + 478.53 = 717.87 | 148.10 + 439.58 = 587.68 | comm/comm | 717.87 | 1439.72 | 934.56 | 35.1% | 216.69 | 1.30 | 0.70 |
| 8 | 32K | 70.27 + 139.91 = 210.18 | 5.64 + 19.13 = 24.77 | comm/comm | 210.18 | 265.30 | 243.48 | 8.2% | 33.30 | 1.16 | 0.10 |
| 8 | 64K | 139.94 + 279.12 = 419.06 | 19.22 + 63.94 = 83.16 | comm/comm | 419.06 | 577.83 | 487.18 | 15.7% | 68.12 | 1.16 | 0.22 |
| 8 | 128K | 279.29 + 557.58 = 836.88 | 68.62 + 230.54 = 299.16 | comm/comm | 836.88 | 1302.59 | 980.22 | 24.7% | 143.34 | 1.17 | 0.39 |

**Turbo**（`usp/attention_ring.py`，连续切分）

| CP | S | T_comm fwd + bwd | T_comp fwd + bwd | 瓶颈 fwd/bwd | roofline | 不 overlap | overlap | 省 | overlap − roofline | overlap / roofline | 效率 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 32K | 20.47 + 122.47 = 142.93 | 29.14 + 496.26 = 525.39 | comp/comp | 525.39 | 690.97 | 597.54 | 13.5% | 72.14 | 1.14 | 0.65 |
| 2 | 64K | 40.82 + 244.71 = 285.54 | 105.22 + 1928.63 = 2033.85 | comp/comp | 2033.85 | 2365.48 | 2162.68 | 8.6% | 128.83 | 1.06 | 0.71 |
| 2 | 128K | 81.55 + 489.48 = 571.03 | 405.14 + 7575.24 = 7980.38 | comp/comp | 7980.38 | 8652.91 | 8295.01 | 4.1% | 314.63 | 1.04 | 0.63 |
| 4 | 32K | 30.67 + 134.41 = 165.09 | 16.99 + 301.14 = 318.12 | comm/comp | 331.81 | 501.16 | 367.51 | 26.7% | 35.70 | 1.11 | 0.81 |
| 4 | 64K | 61.13 + 268.43 = 329.56 | 64.92 + 1150.64 = 1215.56 | comp/comp | 1215.56 | 1582.85 | 1285.73 | 18.8% | 70.17 | 1.06 | 0.90 |
| 4 | 128K | 122.08 + 536.37 = 658.45 | 228.00 + 4512.50 = 4740.50 | comp/comp | 4740.50 | 5492.35 | 4830.07 | 12.1% | 89.56 | 1.02 | 1.01 |
| 8 | 32K | 36.07 + 141.21 = 177.28 | 9.79 + 172.36 = 182.15 | comm/comp | 208.44 | 377.69 | 232.28 | 38.5% | 23.84 | 1.11 | 0.82 |
| 8 | 64K | 71.61 + 281.25 = 352.86 | 35.94 + 638.88 = 674.82 | comm/comp | 710.49 | 1069.95 | 751.51 | 29.8% | 41.02 | 1.06 | 0.90 |
| 8 | 128K | 142.63 + 561.59 = 704.22 | 123.16 + 2453.70 = 2576.85 | comm/comp | 2596.32 | 3395.69 | 2675.24 | 21.2% | 78.92 | 1.03 | 1.02 |

Turbo CP2 128K 的 dq 不正确（见正确性），这一格的耗时只作参考。

从这两张表直接读出的事实：

- 18 格全部 overlap ≥ roofline。TE 1.04–1.30 倍，Turbo 1.02–1.14 倍。
- TE CP4 / CP8 两个 phase 全是 comm-bound；CP2 只有 32K 前向是 comm-bound（20.47 vs 19.33 ms）。
- TE CP4 / CP8（comm-bound）6 格里，overlap 藏住的时间（不 overlap − overlap）是 T_comp 的 86–109%（不 overlap 比
  T_comm + T_comp 多出等待时间，所以可以超过 100%）。也就是计算基本全部藏住了，剩下的是 T_comm 本身。
- Turbo 反向通信基本被 8.4–10.6 倍慢的反向计算盖住（CP4 / CP8 反向效率 0.90–1.06，CP2 0.54–0.69），总耗时由反向计算决定。

## 实测 vs 规格理论下界（fwd + bwd）

| impl | CP | S | TFLOP / rank | T_comp 理论 / 实测 (ms) | 计算达成 (TFLOPS, 峰值占比) | MiB / rank 发送 | T_comm 理论 / 实测 (ms) | 链路达成 (GB/s) | roofline 理论 / 实测 (ms) | overlap (ms) | overlap / 理论 roofline |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TE | 2 | 32K | 79.2 | 31.46 / 78.04 | 1014 (40%) | 3840 | 55.92 / 61.37 | 65.6 | 55.92 / 79.18 | 101.20 | 1.81 |
| TE | 2 | 64K | 316.7 | 125.83 / 291.68 | 1086 (43%) | 7680 | 111.85 / 122.55 | 65.7 | 128.16 / 291.68 | 327.41 | 2.55 |
| TE | 2 | 128K | 1266.6 | 503.32 / 1158.57 | 1093 (43%) | 15360 | 223.70 / 244.89 | 65.8 | 503.32 / 1158.57 | 1208.67 | 2.40 |
| TE | 4 | 32K | 39.6 | 15.73 / 43.89 | 902 (36%) | 5760 | 83.89 / 179.88 | 33.6 | 83.89 / 179.88 | 211.65 | 2.52 |
| TE | 4 | 64K | 158.3 | 62.92 / 156.33 | 1013 (40%) | 11520 | 167.77 / 359.21 | 33.6 | 167.77 / 359.21 | 428.98 | 2.56 |
| TE | 4 | 128K | 633.3 | 251.66 / 587.68 | 1078 (43%) | 23040 | 335.54 / 717.87 | 33.7 | 335.54 / 717.87 | 934.56 | 2.79 |
| TE | 8 | 32K | 19.8 | 7.86 / 24.77 | 799 (32%) | 6720 | 97.87 / 210.18 | 33.5 | 97.87 / 210.18 | 243.48 | 2.49 |
| TE | 8 | 64K | 79.2 | 31.46 / 83.16 | 952 (38%) | 13440 | 195.73 / 419.06 | 33.6 | 195.73 / 419.06 | 487.18 | 2.49 |
| TE | 8 | 128K | 316.7 | 125.83 / 299.16 | 1058 (42%) | 26880 | 391.47 / 836.88 | 33.7 | 391.47 / 836.88 | 980.22 | 2.50 |
| Turbo | 2 | 32K | 79.2 | 31.46 / 525.39 | 151 (6%) | 8960 | 130.49 / 142.93 | 65.7 | 130.49 / 525.39 | 597.54 | 4.58 |
| Turbo | 4 | 32K | 39.6 | 15.73 / 318.12 | 124 (5%) | 10368 | 150.99 / 165.09 | 65.9 | 150.99 / 331.81 | 367.51 | 2.43 |
| Turbo | 8 | 32K | 19.8 | 7.86 / 182.15 | 109 (4%) | 11072 | 161.25 / 177.28 | 65.5 | 161.25 / 208.44 | 232.28 | 1.44 |
| Turbo | 8 | 128K | 316.7 | 125.83 / 2576.85 | 123 (5%) | 44288 | 644.99 / 704.22 | 65.9 | 644.99 / 2596.32 | 2675.24 | 4.15 |

Turbo 其余格子见 `data/2026-11-08-ring-impl-mla/tables.md`。分 phase 的计算达成（纯计算变体）：

- TE 前向 975–1282 TFLOPS（峰值 39–51%），反向 747–1063 TFLOPS（30–42%）。
- Turbo 前向 562–868 TFLOPS（22–35%），反向 83–121 TFLOPS（3–5%）。

## 与 GQA 对比（同 CP / S，causal，TE）

| 量 | MLA / GQA |
|---|---|
| 每 rank 发送字节（TE fwd / bwd） | 20.0 |
| 每 rank 发送字节（Turbo bwd） | 24.0 |
| 纯计算耗时（TE fwd） | 4.3–5.2 |
| 纯计算耗时（TE bwd） | 5.4–5.7 |
| T_comm / T_comp（TE fwd） | 3.7–4.3 |
| T_comm / T_comp（TE bwd） | 3.3–3.6 |

GQA causal 下 TE 只有 CP4 32K、CP8 32K、CP8 64K 三格 comm-bound；MLA 下 CP4 / CP8 全部 6 格、12 个 phase 都 comm-bound。
GQA 篇「MLA 通信 / 计算比约是 4 倍」的推算与本次实测 3.3–4.3 倍一致。

## 正确性

rel-L2，取所有 rank 最大值（明细见 `tables.md` 最后一张表）：

- 32K / 64K（12 格）全部通过：out / dK / dV 对 aiter 全序列参考 ≤ 4.8e-3；out / dq 对每 rank 128 行 fp32 精确值 ≤ 1.2e-2
  （TE dq 6.2e-3–1.2e-2，Turbo dq 3.0e-3，与 GQA 篇同样的现象）。
- 128K：aiter 全序列参考的反向自己就错了（参考 dq 对 fp32 精确行 0.86–1.0，参考 dK / dV 导致比较值出现 inf），所以改用 fp32
  精确行判定 out / dq：
  - TE CP2 / 4 / 8：out ≤ 4.2e-3，dq 9.5e-3–1.6e-2，通过。
  - Turbo CP4 / 8：out 2.9e-3，dq 3.0e-3，通过。
  - **Turbo CP2 128K：out 2.9e-3 通过，dq 3.3e-1 不通过。**
  - 128K 的 dK / dV 两家都没有独立参考，未验证。

## 事实与假设

- 事实（本次实测）：上面所有表；TE CP > 2 P2P 33.5–33.7 GB/s、batch P2P 65.5–66.0 GB/s；Turbo 反向纯计算是 TE 的
  8.4–10.6 倍；Turbo 反向发送字节是 TE 的 2.0–3.0 倍；128K 下 aiter 全序列参考反向失效；Turbo CP2 128K dq 错误。
- 假设（未验证）：
  - aiter 全序列参考 128K 失效是 32 位索引溢出：q 有 131072 × 128 × 192 ≈ 3.2e9 个元素，超过 2^31；GQA 128K 是 5.4e8，
    没出问题。Turbo CP2 128K 每 rank 64K，q 是 1.6e9 个元素，没超过 2^31，dq 错的原因不明。
  - Turbo 反向慢，是因为把 V pad 到 192 之后走了 192 / 192 的通用路径，而不是 aiter 里已有的 192 / 128 反向
    （aiter 有 `fmha_bwd_hd192_128_bf16_causal_a16_pssk` 一类 kernel）。未 profile。
    - 后续单卡实测（[TE vs Turbo attention](./2026-11-08_1530_BASELINE_te_vs_turbo_attn_kernel.md)）：V pad 到 192 后反向慢 5.4–6.1 倍，
      并且 S = 64K 时 dQ 相对误差 0.32，复现了这里 CP2 128K（每 rank 64K）的 dq 错误。ring 反向走这条路径时，
      耗时与 `PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32` 无关，所以这里没设该变量不影响 Turbo 的数。
  - Turbo 反向字节多来自 V pad 到 192、dK / dV 用 fp32 发送。
- 推算（未测）：Ulysses all-to-all 在 MLA CP8 128K 前向每 rank 发送约 2240 MiB
  （16384 token × 128 heads × (192 + 192 + 128 + 128) × 2 B × 7 / 8），是 TE ring 前向 8960 MiB 的 1 / 4，并且 all-to-all 同时
  用 7 条链路。按这个推算，MLA CP4 / CP8 下 a2a 可能比 ring overlap 快，需要实测 TE `a2a` 或 Turbo Ulysses 才能下结论。
- 适用范围：kernel 级单层，B = 1，causal；没有跑 DSV3 端到端。

## 优化空间（只列观察，不在本篇做）

TE CP > 2 的 P2P batch（`NVTE_BATCH_MHA_P2P_COMM`，未测）；ring 改传更少的字节；MLA CP4 / CP8 用 a2a 替代 ring；
Turbo 反向 192 / 128 kernel 路由与 dq 大尺寸正确性；Primus Turbo 路径接上 ring。

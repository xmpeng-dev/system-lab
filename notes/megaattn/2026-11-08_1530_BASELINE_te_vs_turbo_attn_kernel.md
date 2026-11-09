# TE attention vs Turbo attention：单卡 kernel 对比（BASELINE）

2026-11-08 15:30（UTC+8），MI355X gfx950 单卡，BF16。只测已有实现，不改 kernel、不调 knob。

## 结论

- **两家底层是同一系列 kernel。** TE 的 CK 融合 attention（`libck_fused_attn.so`）调用的是 aiter 的 `mha_fwd` / `mha_bwd` 接口，库里有
  `aiter::mha_fwd_args` / `aiter::mha_bwd_args` 符号，并链接自带的 `te_libmha_fwd.so` / `te_libmha_bwd.so`。把 Turbo 固定到
  AITER 后端，GQA / MHA（d = 128）的 dK / dV 和 TE 逐位相同。两家的差别来自 Turbo 的后端选择（FlyDSL vs AITER）、
  `PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32` 的默认值，以及 MLA 形状上两边打包的 aiter 版本不同。
- **按 Primus 默认设置（atomic fp32 = 0），长序列（B = 1、sbhd、4K–64K）：**

  | 形状 | Turbo 实际后端 | TE / Turbo 前向 | TE / Turbo 反向 | TE / Turbo fwd + bwd | 谁快 |
  |---|---|---|---|---|---|
  | GQA 32 / 8，d 128，non-causal | AITER | 1.00–1.10 | 1.00–1.02 | 1.00–1.03 | 持平 |
  | MHA 32 / 32，d 128，non-causal | AITER | 1.00–1.10 | 0.99–1.00 | 1.00–1.03 | 持平 |
  | GQA 32 / 8，d 128，causal | FlyDSL | 1.03–1.44 | 0.63–0.99 | 0.79–1.02（几何平均 0.96） | TE 略快，短序列更明显 |
  | MHA 32 / 32，d 128，causal | FlyDSL | 0.99–1.38 | 0.73–0.95 | 0.88–0.96（几何平均 0.92） | TE 快（Turbo 多 4–14%） |
  | MLA 128 / 128，d 192 / 128，causal | AITER | 1.04–1.10 | 0.98–0.99 | 1.00–1.01 | 持平 |
  | MLA 128 / 128，d 192 / 128，non-causal | AITER | 1.08–1.12 | 1.00 | 1.02–1.03 | Turbo 快 2–3% |

  比值 > 1 表示 Turbo 快。causal 下 FlyDSL 前向比 TE 快、反向比 TE 慢，S 越大差距越小（32K / 64K 时 fwd + bwd 差 0–5%）。
  把 Turbo 固定到 AITER 后，causal 也和 TE 持平（TE / Turbo AITER：反向 0.95–1.00，fwd + bwd 0.99–1.06）。
- **官方 bench（`bench_attention_{te,turbo}.py`，bshd，B 1 / 2 / 4，S 4K / 8K，96 个形状）：** 各组几何平均 fwd + bwd
  TE / Turbo 为 0.96–1.03。差距最大的是 head dim 64 causal：B ≥ 2 时 Turbo 走 AITER，反向慢 1.8–2.0 倍
  （fwd + bwd TE / Turbo 0.59–0.63）；B = 1 时 Turbo 走 FlyDSL，fwd + bwd 反而快 1.55–1.58 倍。
- **Turbo 有两个慢 1.3–6 倍的坑，TE 没有：**
  - `PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32` 在 Turbo 代码里默认是 1，Primus 设成 0。设 1 时 AITER 反向慢 1.29–1.42 倍（GQA）、
    1.71–1.90 倍（MLA），FlyDSL 不受影响（1.00–1.02）。代价换来的是 dQ 精度：设 1 时 dQ 误差与 S 无关，约 2.4e-3；
    设 0 时（TE 也一样）dQ 误差随 S 增长，MLA 64K non-causal 到 2.2e-2。
  - **V pad 到 d_qk（Turbo ring 反向的做法）：** MLA 反向慢 5.4–6.1 倍、前向慢 2.6–3.1 倍，并且 **S = 64K 时 dQ 错误**
    （相对误差 0.32，causal 和 non-causal 都错，32K 及以下正常）。这在单卡上复现了 Turbo ring CP2 128K（每 rank 64K）的 dQ 错误（0.328），
    也是 Turbo ring 反向慢的主因，见 [MLA 篇](./2026-11-08_1220_BASELINE_dsv3_mla_ring_attn_overlap.md)。

## 环境与方法

- 硬件 / 软件：8 × MI355X（gfx950），ROCm 10.1.0，torch 2.14.0+rocm10.1.0，TE 2.17.1+130099ce4，primus_turbo 0.6.1.dev4，
  aiter 0.1.24.post1。容器环境：`NVTE_FUSED_ATTN=1`、`NVTE_CK_USES_FWD_V3=1`、`NVTE_CK_USES_BWD_V3=1`、
  `NVTE_CK_IS_V3_ATOMIC_FP32=0`、`NVTE_CK_HOW_V3_BF16_CVT=2`。
- **长序列 sweep**（`megaattention_overlap_bench/attn_kernel_compare.py`）：
  - 形状：GQA 32 / 8 heads d 128（Llama3.1-8B）；MHA 32 / 32 d 128；MLA 128 / 128 d_qk 192 d_v 128（DSV3）。
    B = 1，sbhd 存储（Megatron 的布局），S 4K / 8K / 16K / 32K / 64K，causal 1 / 0。
  - 实现：
    - `te`：`DotProductAttention(qkv_format="sbhd")`，CK 融合 attention；
    - `turbo`：`pt.ops.flash_attn_func`，后端自动选择；
    - `turbo_aiter`：`GlobalBackendManager.set_attn_backend(AITER)` 固定 AITER；
    - `turbo_aiter_vpad`：AITER，V 先 pad 到 d_qk，只在 MLA 测。
  - 每组形状跑两遍：`PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=0`（Primus 默认，`runner/helpers/envs/base_env.sh:300`）
    和 `=1`（Turbo 代码默认；之前 ring 实验没设这个变量，所以也是 1）。MHA 只跑 0。TE 不读这个变量。
  - 计时：CUDA event，取中位数，10–100 次，单项约 3 秒。前向单独计时；反向每次先跑一遍不计时的前向，再用 event 只圈住
    `torch.autograd.grad`。
  - FLOPs：每个 (q, k) 对前向 2 (d_qk + d_v)、反向 2 (3 d_qk + 2 d_v)；causal 对数 S (S + 1) / 2。
  - 精度：随机 64 行 query 用 fp32 精确算 out / dQ，报相对 L2；dK / dV 和 TE 比。
- **官方 bench**：两个 bench 原样运行，bshd、`torch.utils.benchmark.Timer` 取 100 次均值、和 SDPA 比 SNR > 40。两个 bench
  都把 `PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32` 设成 0。TE 的 "Backward Time" 计的是前向加反向（stmt 里先调前向），
  Turbo 的只计反向，所以 TE 纯反向 = Backward − Forward；TE 原表的 "Backward TFLOPS" 也是用前向加反向的时间算的，偏低。
- 后端选择：gfx950 上 FlyDSL 只接受 causal、head dim 64 / 128、bf16、sbhd 存储（B = 1 时 bshd 也算），见
  `attention_impl.py:109-118, 175-183, 292-303`。所以长序列 sweep 里 GQA / MHA causal 走 FlyDSL，其余走 AITER；官方 bench
  里只有 B = 1 causal 的 d 64 / 128 走 FlyDSL（10 个形状）。MLA 一律走 AITER。

## 长序列 sweep（B = 1，sbhd，ms）

atomic fp32 = 0，比值 > 1 表示 Turbo 快。全表（含 TFLOPS、a32 = 1、V pad 两档）见
[`data/2026-11-08-attn-kernel/tables.md`](./data/2026-11-08-attn-kernel/tables.md)。

| 形状 | causal | S | TE 前向 | Turbo 前向 | TE / Turbo | TE 反向 | Turbo 反向 | TE / Turbo | Turbo AITER 反向 | Turbo 反向 a32 = 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| GQA | 1 | 4K | 0.222 | 0.155 | 1.44 | 0.391 | 0.624 | 0.63 | 0.402 | 0.632 |
| GQA | 1 | 8K | 0.616 | 0.507 | 1.22 | 1.381 | 1.447 | 0.95 | 1.387 | 1.464 |
| GQA | 1 | 16K | 2.146 | 1.862 | 1.15 | 5.231 | 5.443 | 0.96 | 5.362 | 5.438 |
| GQA | 1 | 32K | 7.297 | 6.978 | 1.05 | 19.797 | 19.939 | 0.99 | 20.049 | 20.070 |
| GQA | 1 | 64K | 28.438 | 27.647 | 1.03 | 76.861 | 77.824 | 0.99 | 78.124 | 79.188 |
| GQA | 0 | 4K | 0.345 | 0.315 | 1.10 | 0.663 | 0.666 | 1.00 | 0.667 | 0.933 |
| GQA | 0 | 64K | 56.745 | 56.735 | 1.00 | 148.093 | 148.050 | 1.00 | 147.843 | 209.790 |
| MHA | 1 | 4K | 0.219 | 0.159 | 1.38 | 0.370 | 0.506 | 0.73 | 0.389 | - |
| MHA | 1 | 16K | 2.090 | 1.920 | 1.09 | 5.095 | 6.014 | 0.85 | 5.191 | - |
| MHA | 1 | 64K | 28.623 | 28.972 | 0.99 | 76.287 | 80.472 | 0.95 | 77.503 | - |
| MHA | 0 | 64K | 56.643 | 56.605 | 1.00 | 148.558 | 148.546 | 1.00 | 148.419 | - |
| MLA | 1 | 4K | 0.747 | 0.680 | 1.10 | 2.156 | 2.208 | 0.98 | 2.197 | 3.829 |
| MLA | 1 | 16K | 9.226 | 8.907 | 1.04 | 29.218 | 29.564 | 0.99 | 29.614 | 52.786 |
| MLA | 1 | 64K | 143.248 | 132.688 | 1.08 | 442.926 | 445.638 | 0.99 | 445.532 | 824.378 |
| MLA | 0 | 4K | 1.298 | 1.169 | 1.11 | 3.975 | 3.988 | 1.00 | 3.978 | 6.975 |
| MLA | 0 | 64K | 291.392 | 263.030 | 1.11 | 863.760 | 864.612 | 1.00 | 865.364 | 1631.581 |

吞吐：64K 下 TE 反向 GQA causal 1144、non-causal 1188、MLA causal 1033、non-causal 1059 TFLOPS；Turbo 分别 1130 / 1188 / 1026 / 1058。
4K 下 FlyDSL 反向 GQA causal 只有 551 TFLOPS（TE 880）。

### MLA V pad（Turbo ring 反向的路径）

| causal | S | Turbo AITER 反向（192 / 128） | V pad 到 192 反向 | 倍数 | V pad dQ 相对误差 |
|---|---|---|---|---|---|
| 1 | 4K | 2.197 | 13.448 | 6.12 | 2.4e-3 |
| 1 | 8K | 8.068 | 43.459 | 5.39 | 2.4e-3 |
| 1 | 16K | 29.614 | 161.501 | 5.45 | 2.5e-3 |
| 1 | 32K | 113.163 | 626.764 | 5.54 | 2.4e-3 |
| 1 | 64K | 445.532 | 2472.515 | 5.55 | **3.2e-1** |
| 0 | 64K | 865.364 | 5062.516 | 5.85 | **3.2e-1** |

- V pad 后反向只有 133–185 TFLOPS，前向慢 2.6–3.1 倍。
- 反向耗时与 atomic fp32 设置无关（a32 = 1 / a32 = 0 为 0.996–1.008），dK / dV 正常（相对 TE ≤ 1.3e-3 / 3.9e-4）。
- 日志里 V pad 后加载的是 JIT 编译的 `mha_bwd_bf16_nbias_ndbias_mask_ndropout_ndeterministic`，没 pad 时这条 JIT 路径不出现。

## 官方 bench（bshd，B 1 / 2 / 4，S 4K / 8K）

几何平均 TE / Turbo（> 1 表示 Turbo 快）：

| head dims | causal | 形状数 | 前向 | 反向 | fwd + bwd | Turbo 后端 |
|---|---|---|---|---|---|---|
| d_qk ≠ d_v（192 / 128） | False | 9 | 1.06 | 1.02 | 1.03 | AITER |
| d_qk ≠ d_v（192 / 128） | True | 9 | 0.98 | 0.99 | 0.99 | AITER |
| d_qk = d_v（64 / 128） | False | 39 | 0.96 | 1.00 | 0.99 | AITER |
| d_qk = d_v（64 / 128） | True | 39 | 1.03 | 0.95 | 0.96 | AITER，FlyDSL |

- 走 FlyDSL 的 10 个形状（B = 1 causal）：几何平均前向 1.16、反向 1.14、fwd + bwd 1.14（Turbo 快）；其中 d 64 是 1.55–1.58，
  d 128 是 0.78–1.09。
- 走 AITER 的 86 个形状：几何平均 0.98 / 0.97 / 0.97（TE 略快），差距集中在 d 64 causal B ≥ 2（6 个形状）：Turbo 反向
  2.26–16.0 ms，TE 1.23–8.30 ms（1.79–1.95 倍），fwd + bwd TE / Turbo 0.59–0.63。
- 正确性：Turbo 96 / 96 PASS。TE 90 PASS、6 FAIL，6 个都是 d 64 GQA（32 / 4、64 / 4）8K non-causal，dQ SNR 39.5–39.6，
  差一点到 40；Turbo 同形状 dQ SNR 57.5。d 64 causal B = 2 时 dQ SNR TE 42.4、Turbo 58.2。

## 精度（长序列 sweep，相对 L2，所有 S 和 causal 中的最大值）

| 形状 | 实现 | a32 | out | dQ | dK vs TE | dV vs TE |
|---|---|---|---|---|---|---|
| GQA | TE | - | 2.3e-3 | 1.9e-2 | - | - |
| GQA | Turbo（FlyDSL / AITER） | 0 | 2.9e-3 | 1.9e-2 | 4.7e-3 | 4.2e-3 |
| GQA | Turbo AITER | 0 | 2.3e-3 | 1.9e-2 | 0 | 0 |
| GQA | Turbo AITER | 1 | 2.3e-3 | 2.5e-3 | 0 | 0 |
| MLA | TE | - | 2.3e-3 | 2.2e-2 | - | - |
| MLA | Turbo AITER | 0 | 2.4e-3 | 2.2e-2 | 1.4e-3 | 4.3e-4 |
| MLA | Turbo AITER | 1 | 2.4e-3 | 2.5e-3 | 1.4e-3 | 4.3e-4 |
| MLA | Turbo AITER V pad | 0 / 1 | 2.4e-3 | 3.2e-1（只 64K） | 1.3e-3 | 3.9e-4 |

dQ 误差在 a32 = 0 时随 S 增长：MLA causal 3.2e-3（4K）→ 1.1e-2（64K），non-causal 6.3e-3 → 2.2e-2；a32 = 1 时一直约 2.4e-3。
TE 与 a32 = 0 的 Turbo AITER 数值相同。

## TE vs Turbo FlyDSL（同形状、同字节，2026-11-08 16:00 补测）

Turbo 固定到 FlyDSL（`GlobalBackendManager.set_attn_backend(FLYDSL)`），和 TE 用同一份 sbhd 张量，causal，Primus 环境
（atomic fp32 = 0）。计时和精度方法同上。
- 形状取 `benchmark/ops/training/config.py` 里 head dim 64 / 128 的模型：原生序列长度 × B 1 / 2 / 4；
  再对去重后的 11 种 head 配置跑 B = 1、16K / 32K / 64K。
- 脚本：`run_flydsl_vs_te.sh`、`flydsl_report.py`。
- 全表（含 ms、TFLOPS、dQ 误差）见 [`data/2026-11-08-attn-kernel/flydsl/tables.md`](./data/2026-11-08-attn-kernel/flydsl/tables.md)。

FlyDSL 要求 G = Hq / Hkv 是 2 的幂，并且 Hq × D 是 128 的倍数（`attention_impl.py:88-97`）。所以 Llama-4（40 / 8）、
Mixtral-8x22B（48 / 8）、Qwen2.5-7B（28 / 4）固定 FlyDSL 时直接报错，自动选择时 Turbo 会走 AITER。

比值 = TE 耗时 / FlyDSL 耗时，> 1 表示 FlyDSL 快。

模型原生序列长度：

| 模型 | Hq / Hkv | G | d | S | 前向（B 1 / 2 / 4） | 反向（B 1 / 2 / 4） | 前向 + 反向（B 1 / 2 / 4） | B = 4 前向 + 反向 ms（TE / FlyDSL） |
|---|---|---|---|---|---|---|---|---|
| Llama-2-7B | 32 / 32 | 1 | 128 | 4K | 1.40 / 1.26 / 1.16 | 0.73 / 0.74 / 0.75 | 0.89 / 0.86 / 0.84 | 2.11 / 2.51 |
| Llama-3.1-8B | 32 / 8 | 4 | 128 | 8K | 1.22 / 1.17 / 1.14 | 0.96 / 0.97 / 0.99 | 1.03 / 1.02 / 1.02 | 7.88 / 7.69 |
| Mistral-7B / Mixtral-8x7B | 32 / 8 | 4 | 128 | 4K | 1.42 / 1.27 / 1.18 | 0.63 / 1.01 / 1.01 | 0.79 / 1.08 / 1.05 | 2.28 / 2.16 |
| Llama-2-70B / MoE-1T | 64 / 8 | 8 | 128 | 4K | 1.30 / 1.19 / 1.14 | 0.99 / 1.03 / 1.04 | 1.07 / 1.07 / 1.06 | 4.46 / 4.19 |
| Qwen2.5-72B / Grok-2 | 64 / 8 | 8 | 128 | 8K | 1.16 / 1.13 / 1.10 | 1.00 / 1.01 / 1.03 | 1.04 / 1.04 / 1.04 | 15.36 / 14.70 |
| MiniMax-M3 | 64 / 4 | 16 | 128 | 4K | 1.28 / 1.18 / 1.13 | 1.03 / 1.02 / 1.01 | 1.09 / 1.06 / 1.04 | 4.38 / 4.22 |
| Llama-3.1-405B | 128 / 8 | 16 | 128 | 8K | 1.11 / 1.08 / 1.04 | 1.03 / 1.02 / 1.04 | 1.05 / 1.04 / 1.04 | 30.58 / 29.40 |
| Qwen3-30B-A3B | 32 / 4 | 8 | 64 | 8K | 1.65 / 1.57 / 1.56 | 1.58 / 1.64 / 1.41 | 1.60 / 1.62 / 1.45 | 5.87 / 4.06 |
| GPT-OSS-20B | 64 / 8 | 8 | 64 | 4K | 1.73 / 1.60 / 1.55 | 1.59 / 1.60 / 1.44 | 1.63 / 1.60 / 1.47 | 3.17 / 2.16 |
| Qwen3-235B-A22B | 64 / 4 | 16 | 64 | 8K | 1.53 / 1.52 / 1.55 | 1.60 / 1.53 / 1.41 | 1.58 / 1.53 / 1.44 | 11.36 / 7.87 |
| Llama-4 / Mixtral-8x22B / Qwen2.5-7B | 40 / 8，48 / 8，28 / 4 | 5，6，7 | 128 | 4K / 8K | FlyDSL 不支持 | | | |

长序列（B = 1）：

| Hq / Hkv | G | d | 前向（16K / 32K / 64K） | 反向（16K / 32K / 64K） | 前向 + 反向（16K / 32K / 64K） | 64K 前向 + 反向 ms（TE / FlyDSL） |
|---|---|---|---|---|---|---|
| 32 / 32 | 1 | 128 | 1.09 / 1.01 / 1.02 | 0.86 / 0.94 / 0.94 | 0.92 / 0.95 / 0.96 | 107.67 / 111.97 |
| 32 / 8 | 4 | 128 | 1.13 / 1.05 / 1.04 | 0.98 / 1.00 / 0.97 | 1.02 / 1.01 / 0.99 | 106.33 / 107.80 |
| 64 / 8 | 8 | 128 | 1.11 / 1.04 / 1.03 | 1.00 / 1.00 / 0.98 | 1.03 / 1.01 / 0.99 | 210.82 / 212.40 |
| 64 / 4 | 16 | 128 | 1.09 / 1.05 / 1.00 | 1.00 / 0.99 / 0.99 | 1.03 / 1.01 / 0.99 | 212.91 / 214.84 |
| 128 / 8 | 16 | 128 | 1.06 / 1.04 / 1.04 | 0.99 / 0.98 / 0.97 | 1.01 / 1.00 / 0.99 | 422.64 / 425.96 |
| 32 / 4 | 8 | 64 | 1.52 / 1.51 / 1.69 | 1.67 / 1.50 / 1.52 | 1.63 / 1.50 / 1.57 | 81.74 / 52.11 |
| 64 / 8 | 8 | 64 | 1.50 / 1.70 / 1.74 | 1.53 / 1.55 / 1.51 | 1.52 / 1.59 / 1.57 | 170.05 / 108.09 |
| 64 / 4 | 16 | 64 | 1.53 / 1.62 / 1.67 | 1.56 / 1.51 / 1.51 | 1.55 / 1.54 / 1.56 | 163.96 / 105.30 |

汇总（几何平均，括号内为范围）：

| 部分 | d | 形状数 | 前向 | 反向 | 前向 + 反向 |
|---|---|---|---|---|---|
| 原生序列长度 | 128 | 21 | 1.19（1.04–1.42） | 0.95（0.63–1.04） | 1.01（0.79–1.09） |
| 原生序列长度 | 64 | 9 | 1.58（1.52–1.73） | 1.53（1.41–1.64） | 1.55（1.44–1.63） |
| 长序列 | 128 | 15 | 1.05（1.00–1.13） | 0.97（0.86–1.00） | 0.99（0.92–1.03） |
| 长序列 | 64 | 9 | 1.61（1.50–1.74） | 1.54（1.50–1.67） | 1.56（1.50–1.63） |

- **d 64：FlyDSL 全面快**，前向 + 反向快 1.44–1.63 倍。TE d 64 反向只有 577–768 TFLOPS，FlyDSL 870–1168。
  - 官方 bench 里 Turbo 自动选择时，B ≥ 2 的 bshd 走 AITER，比 TE 还慢 1.8–2.0 倍。所以 d 64 上三者排序是
    FlyDSL > TE > Turbo AITER，差距来自 Turbo 选了哪个后端。
- **d 128、G ≥ 4：基本持平**，前向 + 反向 0.99–1.09。
  - FlyDSL 前向快 0–42%（S 越短、B 越小越明显），反向 0.96–1.04。
  - 唯一例外是 32 / 8、B = 1、4K：反向 0.63，前向 + 反向 0.79；B = 2 时反向就到 1.01。
- **d 128、MHA（G = 1）：TE 快**。FlyDSL 反向慢 6–27%（0.73–0.94），前向 + 反向 0.84–0.96。
- 精度：FlyDSL dQ 相对误差 3.5e-3–1.0e-2，随 S 增长，与 TE 同量级；dK / dV 和 TE 差 4.3e-3–5.7e-3 / 3.8e-3–5.3e-3，不逐位相同。
- 假设（未验证）：
  - 32 / 8 B = 1 4K 反向慢，可能是 FlyDSL 反向的并行粒度在 B × Hkv × S 小时填不满 256 个 CU，G ≥ 8 或 B ≥ 2 时就消失了。
  - MHA 反向慢可能和 G = 1 时 dK / dV 阶段没有组内复用有关。
  - 都没有 profile。

### 哪些模型能走 FlyDSL（gfx950，按真实模型配置，2026-11-08 16:20 补）

`benchmark/ops/training/config.py` 把 Qwen3-30B-A3B / Qwen3-235B-A22B 的 head_dim 写成 64，但 HF 官方 config.json
（[Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json)、
[Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B/raw/main/config.json)）和 Primus 模型配置（`kv_channels: 128`）
都是 128。所以上表 "Qwen3 d 64" 两行不代表真实 Qwen3；真实形状补测如下（B 1 / 2 / 4，sbhd，causal，比值 = TE / FlyDSL）。

| 形状 | S | 前向 | 反向 | 前向 + 反向 |
|---|---|---|---|---|
| Qwen3-30B-A3B 32 / 4 d 128（G 8） | 4K | 1.42 / 1.28 / 1.19 | 0.94 / 1.02 / 1.00 | 1.07 / 1.09 / 1.05 |
| Qwen3-30B-A3B 32 / 4 d 128（G 8） | 8K | 1.20 / 1.16 / 1.12 | 0.94 / 0.98 / 0.97 | 1.00 / 1.02 / 1.00 |
| Qwen3-30B-A3B 32 / 4 d 128，B = 1 | 16K / 32K / 64K | 1.14 / 1.07 / 1.03 | 0.98 / 0.99 / 1.00 | 1.02 / 1.01 / 1.01 |
| Qwen3-235B-A22B 64 / 4 d 128（G 16） | 4K | 1.28 / 1.18 / 1.13 | 1.03 / 1.02 / 1.01 | 1.09 / 1.06 / 1.04 |
| Qwen3-235B-A22B 64 / 4 d 128（G 16） | 8K | 1.15 / 1.13 / 1.08 | 0.99 / 1.02 / 0.99 | 1.04 / 1.05 / 1.02 |
| Qwen3-235B-A22B 64 / 4 d 128，B = 1 | 16K / 32K / 64K | 1.09 / 1.05 / 1.00 | 1.00 / 0.99 / 0.99 | 1.03 / 1.01 / 0.99 |

235B 的 4K 和长序列行就是上表的 64 / 4 d 128（MiniMax-M3 / h64_4_d128）。

按 FlyDSL 规则（causal、d 64 / 128、G 为 2 的幂、bf16、softmax scale 1/√d、sink 为 fp32 [Hq]、左窗口 ≥ 64）判断：

| 能走 FlyDSL | 不能走 |
|---|---|
| Qwen3-8B（32 / 8 d 128）、Qwen3-32B（64 / 8 d 128）、Qwen3-30B-A3B、Qwen3-235B-A22B、Llama-2 / 3 / 3.1 全系、Mistral-7B、Mixtral-8x7B、Grok-2、MiniMax-M3、GPT-OSS-20B / 120B（64 / 8 d 64，sink + 128 滑窗） | DeepSeek-V2 / V3 / V4、Kimi-K2（d 192 / 128）；Qwen3-14B（40 / 8）、Llama-4（40 / 8）、Mixtral-8x22B（48 / 8）、Qwen2.5-7B（28 / 4）（G 不是 2 的幂）；Qwen3.5-35B-A3B（d 256） |

几点要注意：
- Primus 的 Qwen3-4B / 32B 配置没写 `kv_channels`，Megatron 会按 hidden / heads 推出 80，FlyDSL 不接受；HF 上是 128。
- Primus MI355X 的 Qwen3 配置默认 `use_turbo_attention: false`，GPT-OSS-20B / 120B 是 true。
- Primus 在 CP > 1 时会关掉 sink attention（`primus_turbo.py:959-967`）。
- Turbo 的 CP 路径（`flash_attn_usp_interface.py`，a2a 和 ring）调用的是 aiter CK / Triton，不走 FlyDSL；只有 CP = 1 的
  `flash_attn_func` 用得上 FlyDSL。

### gfx950 FlyDSL 192 / 128（DSV3 MLA）可行性（2026-11-08 17:50，代码阅读，未动手）

为什么考虑：
- 主流 MoE 的全 attention 层多是 192 / 128，例如 DeepSeek-V2 / V3、Kimi-K2；Kimi-K3 的全 attention 层也是。
- DSV3 在 CP4 / CP8 下是 comm-bound（MLA 篇）。
- 现在只有 aiter 的 asm kernel 能跑 192 / 128，改不了，做不了 tile 级融合。

| 项 | 现状（代码） | 192 / 128 需要的改动 |
|---|---|---|
| fwd（`flydsl/attention/flash_attn_fwd.py` 1081 行 + `utils/attn_helper.py`） | 入口硬断言 `head_dim in (64, 128)`；traits 已按用途拆开：`K_STEPS_QK = D / 16`（QK^T）、`D_CHUNKS = D / 32`（O 列）、`SMEM_D_RPT = D / 64`（LDS 行） | 拆成 D_QK / D_V。V / O 一侧 D_V = 128，和现有 D = 128 一样；只有 QK 一侧变：K_STEPS_QK 8 → 12、K 的 LDS 行 2 → 3、每 wave 的 Q 寄存器 32 → 48 VGPR（估算：32 行 × 192 × 2 B / 64 lane）、K 的 LDS tile 16 → 24 KB |
| bwd（`flash_attn_bwd.py` 6121 行：odo / dkdv / dq / 规约） | 大量 `head_dim == 128`、`D == 64` 专门调优的分支 | dK、dQ 192 列，dV 128 列；dkdv 累加器 256 → 320 列（1.25 倍）；每个 D 专用分支都要重新定 |
| 参考实现 | `gfx1250_mla/`（fwd 703 + bwd 1564 行）已经分开 D_QK = 192、D_V = 128 | 可以借鉴结构；但它是 wave32 + WMMA + TDM，gfx950 是 wave64 + MFMA，指令层不能直接搬 |
| kernel 内跨卡通信 | MegaMoE gfx950 已有 `flydsl/mega/bf16/symm_buffer.py`、`barrier.py`、`ep_intranode.py` | MegaAttn 在 kernel 里拉对端 KV 可以复用 |

要追的目标（本篇实测，B = 1，sbhd）：aiter 192 / 128 causal 前向 1039–1328 TFLOPS、反向 813–1027 TFLOPS（4K–64K）。

风险：DSV3 展开后是 G = 1。FlyDSL 现在在 G = 1（d 128，32 / 32）上反向比 TE 慢：
- 4K 时 682–698 vs 925–934 TFLOPS；
- 16K–64K 时 908–1064 vs 1052–1129 TFLOPS。

如果直接移植，192 版反向起步可能就落后 aiter（假设）。

## 事实与假设

- 事实（本次实测或代码可见）：
  - 上面所有表。
  - TE 融合 attention 库引用 aiter 的 `mha_fwd_args` / `mha_bwd_args`。
  - GQA / MHA 上 Turbo AITER 与 TE 的 dK / dV 逐位相同。
  - FlyDSL 的选择条件见 `attention_impl.py`。
  - V pad 路径在 64K 时 dQ 错误。
  - a32 = 1 对 AITER 反向的减速，以及对 dQ 精度的改善。
- 假设（未验证）：
  - MLA 上 TE 与 Turbo AITER 的 dK / dV 不逐位相同（1.4e-3），并且 Turbo 前向快 4–12%，原因可能是 TE 打包的 aiter 版本和
    pip 的 aiter 0.1.24.post1 不同，选到了不同的 192 / 128 前向 kernel。没有 profile kernel 名。
  - d 64 causal B ≥ 2 Turbo AITER 反向慢 1.8–2.0 倍、dQ SNR 却高很多，可能是两边 d 64 反向选了不同 kernel（Turbo 那条精度更高）。
    没有 profile。
  - V pad 后走的是 aiter 通用 CK 反向（JIT 加载记录支持这一点），64K dQ 错误可能是该路径的累加 buffer 索引溢出；
    q 只有 1.6e9 个元素，没超过 2^31，具体原因不明。
  - Primus 训练里 Turbo attention 拿到的是 Megatron 的 sbhd 张量，按代码推断 GQA causal d 128 会走 FlyDSL，
    没在训练里确认。
- 对之前结论的影响：
  - Turbo ring 反向慢 8.4–10.6 倍（MLA 篇）里，单 kernel V pad 就占 5.4–6.1 倍，剩下的可能来自 ring 每步的额外开销。
  - 之前 Turbo ring 没设 atomic 变量（等于 1），但 ring 反向走的是 V pad 路径，这条路径对该变量不敏感，所以那组数不受影响。
- 适用范围：单卡、BF16、B = 1（长序列）或 B ≤ 4（官方 bench），没测 FP8、varlen / THD、sliding window。

## 优化空间（只列观察，不在本篇做）

- FlyDSL causal 反向短序列比同 Turbo 的 AITER 慢：GQA 4K / 8K / 16K 慢 55% / 4% / 2%，MHA 慢 30% / 22% / 16%。
- Turbo ring 反向不 pad V，直接用 aiter 的 192 / 128 反向。
- V pad 路径 64K dQ 正确性。
- a32 = 0 的 dQ 误差随上下文增长，长上下文训练需要权衡。

# cco-alltoall：MiniMax / MiniMax-M3 / DeepSeek-V3 形状（2026-11-02 16:00）

4 × MI455X，`ctheliosr-rck-g02-e15-01`，容器 `amdprimus/amdprimus:tas-20261019`，RCCL 默认配置，300 ms 窗口 × 10 轮。
数据：`data/2026-11-02_cco_alltoall/results/models/<model>/`；单卡检查脚本：`data/2026-11-02_cco_alltoall/scripts/{kcheck,bwdcheck}.py`。
客户 README：`minimax-poc/examples-phase2/training/cco-alltoall/README_CN.md`（只有 M2 表）。

## 形状（EP4，每卡，路由均匀）

| model | E_local | rows/expert | gate+up K→N | down K→N | dense GEMM | attention | dispatch |
|---|---|---|---|---|---|---|---|
| minimax（opbench） | 72 | 912 | 6144→2048 | 1024→6144 | 8192×6144→24576 | B2 L4096 h64 d128，non-causal | 768 MiB |
| minimax_m3（T 4096） | 32 | 512 | 6144→6144 | 3072→6144 | 4096×6144→24576 | B1 GQA 64/4 d128，causal | 192 MiB |
| minimax_m3_t8k（T 8192） | 32 | 1024 | 6144→6144 | 3072→6144 | 8192×6144→24576 | B2 GQA 64/4 d128，causal | 384 MiB |
| dsv3（T 8192） | 64 | 1024 | 7168→4096 | 2048→7168 | 8192×7168→36864 | B2 MLA h128 qk192/v128，causal | 896 MiB |

M3 来自 Primus `minimax_m3.yaml`（128 expert，top-4，moe_ffn 3072，GQA 64/4）。M3 的 MoE 层实际是 MSA sparse attention，
dense attention 只是代理。DSv3 没有 attn_bwd：aiter gfx1250 asm bwd 只有 128/128。

## 正确性（单卡）

- 所有 kernel 和 FP32 参考对比：GEMM 1.7e-3，attention fwd 2.3e-3（GQA causal、MLA varlen 都对）。
- attn bwd 对 FP32 autograd（seq 0，kv head 0 及其 q 头组）：dq / dk / dv 2.3e-3 – 2.9e-3，GQA causal 正确。

## 单卡短测（约 40 ms，未到功耗墙）

| kernel | minimax | M3 T4k | M3 T8k | DSv3 |
|---|---|---|---|---|
| gmm gate+up | 727 µs（2274 TF） | 527（2349） | 982（2520） | 1531（2514） |
| gmm down | 490（1686） | 297（2083） | 529（2337） | 866（2222） |
| dense GEMM | 1037（2386） | 524（2362） | 1031（2401） | 1797（2409） |
| attn fwd | 466（2357） | 158（1743） | 282（1950） | 712（1932） |
| attn bwd | 1387（1981） | 452（1522） | 862（1595） | — |

4 卡持续跑时 kernel 慢 20–30%（功耗墙，见 1440 那篇）。

## M2：一次完整 dispatch 时的 E

| kernel | minimax 768 | M3 T4k 192 | M3 T8k 384 | DSv3 896 |
|---|---|---|---|---|
| gmm gate+up | 0.60 | 0.77 | 0.64 | 0.57 |
| gmm down | 0.83 | 0.75 | 0.64 | 0.61 |
| attn fwd | 0.62 | 0.73 | 0.65 | 0.52 |
| attn bwd | 0.59 | 0.63 | 0.62 | — |
| dense GEMM | 0.56 | 0.54 | 0.51 | 0.58 |

对外展示 E：M3 T4k 最好看（通信 0.2 ms，比所有 kernel 短）。绝对节省：minimax / DSv3 最多（大通信量）。

## M1：互相拖慢（单次，单独 → 同时）

| model / kernel | dispatch | kernel 变慢 | alltoall 变慢 | busbw |
|---|---|---|---|---|
| minimax gate+up | 768 | 1.36× | 1.36× | 742 → 543 |
| minimax down | 768 | 1.49× | **1.10×** | 736 → 668 |
| minimax attn fwd | 768 | 1.42× | 1.33× | 744 → 560 |
| minimax attn bwd | 768 | 1.37× | 1.39× | 740 → 531 |
| minimax dense GEMM | 768 | 1.41× | 1.37× | 746 → 544 |
| M3 T4k gate+up | 192 | 1.16× | 1.57× | 783 → 497 |
| M3 T4k down | 192 | 1.18× | 1.50× | 784 → 521 |
| M3 T4k attn bwd | 192 | 1.16× | 2.33× | 782 → 335 |
| M3 T4k dense GEMM | 192 | 1.17× | 2.46× | 780 → 317 |
| DSv3 gate+up | 896 | 1.38× | 1.40× | 753 → 538 |
| DSv3 down | 896 | 1.42× | **1.16×** | 754 → 650 |
| DSv3 attn fwd | 896 | 1.51× | 1.33× | 752 → 565 |
| DSv3 dense GEMM | 896 | 1.40× | 1.37× | 753 → 549 |

观察：

1. 大包（≥ 768 MiB）时两边各慢约 1.35–1.5×，带宽从 ~750 掉到 ~540 GB/s，对称地抢资源。
2. down GEMM 拖慢 alltoall 最少（1.10–1.17×），自己被拖慢最多（1.42–1.49×）。猜测：down 的 K 小（1024 / 2048），
   每个 tile 计算短、访存占比高，HBM / fabric 争用方式不同；或者 workgroup 数 / 调度让 RCCL 的 CU 更容易拿到。未验证，
   可用 rocprofv3 看 RCCL kernel 的 CU 占用和 HBM 带宽。
3. 小包（192 MiB）时 kernel 只慢 1.16×，alltoall 慢 1.5–2.5×：小包 alltoall 时延敏感，和 kernel 抢 CU 时排队。
   attn bwd / dense GEMM 对 alltoall 伤害最大（2.3–2.5×）。

## 卡死：校准循环的跨 rank 不一致（已修）

M3 T4k dense GEMM 第一次跑卡死（第二次过）。日志：rank 3 停在 ALLREDUCE（SeqNum 263780），rank 0–2 停在 ALLTOALL，
已多 enqueue 1800 个 alltoall。原因：`Case.plan()` 的校准循环 `while time.time() - t0 < warmup` 用各 rank 本地时钟，
边界附近 rank 3 少跑一轮校准，直接进下一个窗口，collective 序列错开。修复：继续与否由 rank 0 的时钟决定，
`_all_max` 广播。这是一个和 gate 无关的独立竞态，以前的 0 hang 只是没撞上。

## 下一步

- rocprofv3 看 down GEMM 为什么对 alltoall 友好（观察 2）。
- SDMA alltoall（等确认 SDMA P2P 在这台节点上安全）。

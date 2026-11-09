# M0-overlap：CP ring attention 前向的「RCCL P2P + attention」多 stream overlap 基线（MI355X）

日期 2026-11-08 08:30。设计文档 §7 备选 1（CP / attention-通信融合）的立项前基线：先量清楚不融合时 overlap 还差多少。

## 结论

长序列（32K–128K）前向，overlap 效率 `(不 overlap − overlap) / T_comm` 为 **0.73–1.08**（CP8 32K 例外：**0.42–0.47**），
多 stream overlap 相对不 overlap 省 **3.7–28.8%**，overlap 后已在「有依赖 ring 的理想下界」的
**1.01–1.14×** 以内；例外是 **CP8 32K（1.33×）** 和部分 causal（1.22×）。持久化融合在前向的可争取上限：CP2 1–8%、CP4 4–18%、
CP8 128K 5–12%、CP8 32K 约 25%，且拿到这部分的前提（避开 CU 争用）是假设、未证明。step 内切 chunk 的 tile 级 overlap 在非融合
实现里一律更慢（128K 下 chunk 1024 比整块慢 1.7–1.9×）。**前向不单独立项融合；下一步先测反向同一张表**。

## 环境与方法

8× MI355X（gfx950，xGMI 全互联 1 跳），ROCm 10.1.0，PyTorch 2.14.0+rocm10.1.0，RCCL 2.30.7，aiter 0.1.24.post1
（`fmha_fwd_hd128_bf16`）。B = 1，32 Q / 8 KV heads，D = 128，BF16，连续切分。torch.distributed `batch_isend_irecv`，
comm stream + compute stream + HIP event；device 侧 spin 闸门排除 host 下发延迟；每点 100 次迭代取最慢 rank 的 median。
pipeline 输出对 all-gather 参考 relL2 ≤ 3.2e-3（aiter 自身对 fp32 为 2.3e-3），payload 逐位一致；去掉依赖的对照会被抓到
（relL2 0.675）。

工程：`/perf_apps/xiaoming/megaattention_overlap_bench/`（独立，不改 Primus / Megatron / attention kernel，未 commit），
完整报告 `results/report/report.md`。数据：[data/2026-11-08-overlap/](./data/2026-11-08-overlap/)。

## 不 overlap vs overlap（整块发送，ms）

| CP | S | causal | T_comm | T_attn | 不 overlap | overlap | overlap 效率 | 省下 % | overlap / 理想 |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 32K | 0 / 1 | 1.27 | 7.87 / 5.96 | 9.19 / 7.31 | 8.23 / 6.29 | 0.76 / 0.81 | 10.5 / 14.0 | 1.07 / 1.09 |
| 2 | 64K | 0 / 1 | 2.47 | 29.11 / 22.60 | 32.18 / 25.00 | 29.87 / 22.97 | 0.94 / 0.82 | 7.2 / 8.1 | 1.04 / 1.06 |
| 2 | 128K | 0 / 1 | 4.86 | 115.60 / 87.02 | 120.60 / 92.80 | 116.19 / 87.93 | 0.91 / 1.00 | 3.7 / 5.2 | 1.01 / 1.02 |
| 4 | 32K | 0 / 1 | 1.90 | 4.31 / 3.73 | 6.20 / 5.64 | 4.64 / 4.25 | 0.82 / 0.73 | 25.1 / 24.6 | 1.14 / 1.22 |
| 4 | 64K | 0 / 1 | 3.64 | 16.24 / 14.04 | 19.58 / 17.68 | 16.36 / 14.81 | 0.89 / 0.79 | 16.4 / 16.2 | 1.04 / 1.10 |
| 4 | 128K | 0 / 1 | 7.09 | 59.46 / 52.32 | 68.02 / 61.28 | 60.96 / 53.65 | 1.00 / 1.08 | 10.4 / 12.4 | 1.04 / 1.04 |
| 8 | 32K | 0 / 1 | 2.42 | 2.35 / 2.18 | 4.73 / 4.59 | 3.58 / 3.57 | 0.47 / 0.42 | 24.3 / 22.2 | 1.33 / 1.34 |
| 8 | 64K | 0 / 1 | 4.42 | 8.52 / 7.59 | 12.97 / 12.24 | 9.25 / 8.72 | 0.84 / 0.80 | 28.7 / 28.8 | 1.14 / 1.22 |
| 8 | 128K | 0 / 1 | 8.44 | 32.49 / 28.90 | 40.74 / 38.53 | 33.45 / 30.99 | 0.86 / 0.89 | 17.9 / 19.6 | 1.06 / 1.14 |

overlap 效率 = `(不 overlap − overlap) / T_comm`，原始值不截断（> 1 是因为串行基线每个 step 要等最慢 peer，比
`T_comm + T_attn` 多 0.9–1.9 ms）。理想 = `max(T_attn, T_comm + T_attn / CP)`（整块 attention，不含 merge）。

## 关键事实

- 链路：每对 GPU 单向约 56 GB/s（256 MiB），不随 CP 变；RCCL channel 数调大无提升；单次交换 device 侧固定约 58 µs
  （RCCL kernel 本身约 10 µs，其余是跨 stream event 依赖），ring 连续交换每步约 44 µs。
- 争用：并发时 RCCL 慢 1.2–1.6×、attention 慢 1.0–1.2×（rocprof 确认 95% 的 RCCL kernel 时间与 attention 真并发）。
- 旋钮（32K）：comm stream 高优先级、`GPU_MAX_HW_QUEUES=8` 无效；`GPU_MAX_HW_QUEUES=2` 让 overlap 完全消失；
  `NCCL_MAX_P2P_NCHANNELS=2` 在 CP2 快 3.9%，在 CP8 慢 2.7×。
- 分 chunk 的三项固定开销：aiter 每次调用只看 C 个 key 时 TFLOPS 降到整块的 30%（C=128）/ 77%（C=1024）；
  每 chunk 一次 LSE merge 占 pipeline 27–30%（C=1024）；4 MiB 消息只有 29 GB/s。

## 未覆盖

反向（ring 反向回传 dK/dV，通信约翻倍）、zigzag causal、跨机、RCCL copy-engine 路径（`NCCL_P2P_USE_CUDA_MEMCPY=1`
在 4 MiB 卡住，300 s 超时）。

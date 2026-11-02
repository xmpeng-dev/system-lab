# 2026-10-29 13:30  M0 BASELINE — DSA decode 三独立 kernel 分段占比（MI355X）

> **When**: 2026-10-29 13:30 (Asia/Shanghai)
> **Where**: `smci355-ccs-aus-n06-25` / container `xiaoming-dev`，单卡 MI355X (gfx950)，ROCm 10.1 / torch 2.14.0 / triton 3.9.0 / amd-aiter 0.1.24.post1 (`/opt/aiter` `04896172e`) / flydsl 0.3.4.1
> **Context**: [v1 架构设计](./2026-08-05_1650_megaattn_v1_architecture_design.md) §5 M0 + §6 R1；[handoff](./2026-10-29_1255_handoff-prompt-for-turbo.md)
> **代码**: Primus-Turbo 分支 `feat/megaattn/m0-dsa-decode-baseline`（基于 `origin/main` `b4ad3bcb`，未 commit），新文件 `benchmark/ops/inference/bench_dsa_decode_breakdown_aiter.py`
> **数据**: [`data/2026-10-29-m0/`](./data/2026-10-29-m0/)（`m0-h128.csv` / `m0-h16.csv` / `max-split-sweep.csv` / `rocprof-counters.txt` / `parse-rocprof.py`）

## TL;DR

**判定：GO（有条件），下一轮可开 M1。** 「top-k + logits 往返」占三段 wall 的比例强依赖 batch，而不只是 N：

| 128 MLA heads (DP-attn) | N = 8K | N = 32K | N = 128K |
|---|---|---|---|
| B = 1 | **23.1%** | **32.1%** | **45.8%** |
| B = 4 | **22.3%** | **28.4%** | **33.8%** |
| B = 16 | 14.5% (边界) | **18.5%** | **21.5%** |
| B = 32 | 11.8% | 14.9% | 14.0% |
| B = 64 | 10.3% | 12.5% | 11.5% |

- 15% 门槛：B ≤ 4 全部 N 通过；B = 16 在 32K / 128K 通过；B ≥ 32 全部不过。B ≥ 32 时 32K / 128K 是 indexer 占大头（40–73% wall，即设计 §6 R1 的情形），8K 是 sparse MLA 占大头（68–73%）。TP8（16 MLA heads）占比更高，B ≤ 32 都过（见 Result 表 2）。
- logits **写回本身几乎不值钱**：每层 0.04–8.7 µs（≤ 2.4% wall，rocprof 实测写 1.0x logits）。可融合收益几乎全在 **top-k 的延迟**：k=2048 radix top-k 每层 9–11 µs @8K、15–20 µs @32K、29–38 µs @128K（128 heads 表），且几乎不随 batch 变。
- M3「@128K 降 ≥ 25%」只在 B ≤ 4 可达（天花板 46% / 34%）；B = 16 天花板 21.5%（加上把 indexer 做到 roofline 后约 28%，见 Interpretation）。
- 顺带发现 aiter 两个 bug（sparse MLA metadata 未初始化尾部导致 GPU fault；gfx950 缺 `top_k_per_row_decode_fast` hsaco 导致进程 abort），见 Result §5。

## Background

设计 §2.1 按字节估算，logits 往返只占单层访存 10–12%；§2.2 论证 wall 占比会更高，因为 top-k 是多趟、延迟主导的。§6 R1 是最大风险：如果 indexer 占绝大部分时间，融合天花板 < 15%，就该停下转 §7。M0 的任务就是在 gfx950 上用现成最快的三个独立 kernel 量出真实占比。

对比口径：这里比的是 **CDNA4 (MI355X) 上的三段独立 kernel 基线**，为「三段一核」提供分母；不是复现 LiteTopK / GVR 在 NVIDIA 上的数字。

## What I did

### 1. 基线组成（全部来自 aiter，取每格最快的变体）

| 段 | kernel | 形状 / 精度 |
|---|---|---|
| indexer | `deepgemm_fp8_paged_mqa_logits`（Gluon），扫 4 个变体：block 1 / block 64 preshuffle / block 64 preshuffle chunk 128 wpe 5 / block 16 preshuffle varctx | q fp8 e4m3fn `[B, 1, 64, 128]`；index-K fp8 + fp32 per-token scale（132 B/token）；weights fp32 `[B, 64]`；输出 logits fp32 `[B, N]` |
| top-k | `aiter.top_k_per_row_decode`（dispatch）/ `flydsl_top_k_per_row_decode`（multi-CTA radix）/ `flydsl_radix_topk_one_block_decode` / `_hip_top_k_per_row_decode` / `torch.topk`；`top_k_per_row_decode_fast` 在 gfx950 不可用 | k = 2048，输出 int32 `[B, 2048]` |
| sparse MLA | Triton req→global 索引转换 + `aiter.mla.mla_decode_fwd`（asm `mla_a8w8_qh32_qseqlen4_gqaratio32_ps` + `kn_mla_reduce_v1`），metadata `get_mla_metadata_v1(topk=2048, is_sparse)` | absorbed QK 576 / V 512，fp8 q/kv per-tensor scale 1，page_size 1，128 或 16 heads，max_split 32 |

- KV 物理布局：64-token page 用 `randperm` 打散，indexer 的 block table 与 MLA 的 token 索引指向同一份物理槽位。
- 输入随机（seed 0），`next_n = 1`，batch 内各序列等长。
- per-step metadata（MLA schedule 37–130 µs、indexer varctx schedule）每个 decode step 只算一次，不计入单层 wall，单独报告。

### 2. 计时方法

- 三段 capture 进 HIP graph，按前缀 `[indexer]`、`[indexer, top-k]`、`[indexer, top-k, MLA]` 分别 replay，CUDA event 计时，段时间 = 相邻前缀之差；首段减去 graph 入口开销（单 tiny kernel graph 的 wall − dependent-launch floor）。
- 每次 replay 前先跑一个 512 MB `add_` flush graph（不在计时窗口内；MI355X MALL 256 MB），所以 KV 是冷的，而 logits 在 indexer → top-k 之间自然是热的，与真实 decode 一致。
- 中位数，iters = 100；graph 内 dependent-launch floor 实测 1.49–1.51 µs / kernel。
- 不用 torch profiler 计时：它在 graph replay 中给每个依赖 dispatch 额外加约 2.4 µs（tiny kernel 3.92 µs vs event 1.49 µs），多 kernel 段会被严重高估。profiler 只用于拿每段 kernel 名单。
- 「logits 往返」估算 = top-k 段实测时间 + logits 写回估算；写回估算 = indexer 时间 × logits 字节 / (indexer 读字节 + logits 字节)。rocprofv3 `WRITE_SIZE` 证实 indexer 恰好写 1.0x logits。

### 3. 正确性（每格都跑）

| 检查 | 方法 | 结果（30 格） |
|---|---|---|
| indexer | 对 fp8 反量化输入的 fp32 参考，前 2 条序列 1 − cos | ≤ 1.5e-10 |
| top-k | 索引唯一且在范围内，min(选中) ≥ max(未选中) | 全部精确 |
| sparse MLA | 对 gather 后 fp32 attention 1 − cos；另验 `global_idx == tok[topk_idx]` | ≤ 1.25e-4（aiter fp8 门槛 3e-2） |

### 4. 复现

```bash
docker exec -it xiaoming-dev bash
cd /perf_apps/xiaoming/Primus-Turbo/benchmark/ops/inference    # 分支 feat/megaattn/m0-dsa-decode-baseline
HIP_VISIBLE_DEVICES=0 python -u bench_dsa_decode_breakdown_aiter.py \
    --batch 1 4 16 32 64 --ctx 8192 32768 131072 --mla-heads 128 --check --iters 100 -o m0-h128.csv
HIP_VISIBLE_DEVICES=1 python -u bench_dsa_decode_breakdown_aiter.py \
    --batch 1 4 16 32 64 --ctx 8192 32768 131072 --mla-heads 16 --check --iters 100 -o m0-h16.csv
# max_split 扫描：加 --max-split {4,8,16,32,64}
# 计数器（FETCH_SIZE 和 WRITE_SIZE 必须分两次采，合采超出硬件计数器容量）：
HIP_VISIBLE_DEVICES=2 rocprofv3 --pmc FETCH_SIZE --output-format csv -d $OUT -o B16_N131072_FETCH_SIZE -- \
    python -u bench_dsa_decode_breakdown_aiter.py --profile-run --batch 16 --ctx 131072 --iters 3 \
    --idx-variants p64s_c128w5 --topk-variants aiter
# 再分别用 --pmc WRITE_SIZE、--pmc TCC_HIT_sum TCC_MISS_sum TCP_TCC_READ_REQ_sum；data/ 下 parse-rocprof.py 汇总
```

## Result

### 1. 128 MLA heads（DP-attention，每卡 128 heads；主口径）

单位 µs。「logits 写」是写回估算，「融合占比」= (top-k + logits 写) / wall。kernel 数按 indexer / top-k / MLA。

| B | N | indexer | top-k | sparse MLA | wall | indexer % | top-k % | MLA % | logits 写 | 融合占比 | kernels |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 8K | 1.6 | 9.1 | 28.9 | 39.5 | 4.0 | 23.0 | 73.0 | 0.05 | **23.1%** | 1/1/3 |
| 4 | 8K | 1.6 | 9.6 | 32.2 | 43.5 | 3.8 | 22.2 | 74.0 | 0.05 | **22.3%** | 1/1/3 |
| 16 | 8K | 6.5 | 9.3 | 49.7 | 65.5 | 9.9 | 14.2 | 75.9 | 0.19 | 14.5% | 1/1/3 |
| 32 | 8K | 13.2 | 9.5 | 61.7 | 84.5 | 15.7 | 11.3 | 73.0 | 0.39 | 11.8% | 1/1/3 |
| 64 | 8K | 25.6 | 10.8 | 76.0 | 112.4 | 22.8 | 9.6 | 67.6 | 0.75 | 10.3% | 1/1/3 |
| 1 | 32K | 1.8 | 14.7 | 29.6 | 46.1 | 3.9 | 32.0 | 64.1 | 0.05 | **32.1%** | 1/1/3 |
| 4 | 32K | 6.2 | 15.3 | 33.0 | 54.6 | 11.4 | 28.1 | 60.5 | 0.18 | **28.4%** | 1/1/3 |
| 16 | 32K | 25.6 | 16.5 | 51.2 | 93.3 | 27.4 | 17.7 | 54.9 | 0.75 | **18.5%** | 1/1/3 |
| 32 | 32K | 51.2 | 17.5 | 58.7 | 127.4 | 40.2 | 13.7 | 46.1 | 1.51 | 14.9% | 1/1/3 |
| 64 | 32K | 95.6 | 20.3 | 69.1 | 185.0 | 51.7 | 11.0 | 37.3 | 2.81 | 12.5% | 1/1/3 |
| 1 | 128K | 6.2 | 29.0 | 28.4 | 63.6 | 9.7 | 45.6 | 44.7 | 0.18 | **45.8%** | 1/7/3 |
| 4 | 128K | 25.6 | 28.8 | 32.8 | 87.2 | 29.3 | 33.0 | 37.6 | 0.75 | **33.8%** | 1/7/3 |
| 16 | 128K | 94.5 | 34.8 | 45.2 | 174.5 | 54.2 | 20.0 | 25.9 | 2.78 | **21.5%** | 1/7/3 |
| 32 | 128K | 166.8 | 30.8 | 57.0 | 254.7 | 65.5 | 12.1 | 22.4 | 4.91 | 14.0% | 1/7/3 |
| 64 | 128K | 296.1 | 37.6 | 70.0 | 403.8 | 73.3 | 9.3 | 17.3 | 8.71 | 11.5% | 1/1/3 |

- 最快 indexer 几乎都是 block 64 preshuffle chunk 128 wpe 5（B = 64 @128K 是 block 16 varctx，差 < 1%）。实测 indexer 带宽 2.7–3.9 TB/s（B = 1 @8K 是 launch-bound，0.7 TB/s）。
- top-k：N ≤ 32K 时无论 aiter dispatch 还是直接调用，跑的都是 one-block radix（`radix_topk_one_block_decode_mixed_k2048_b1024_w6`，1 kernel）；N = 128K、B ≤ 32 时最快的是 multi-CTA radix（3x histogram + 3x reduce_select + gather，共 7 kernel）；B = 64 @128K aiter dispatch 回到 one-block。B = 1 @128K 各变体：aiter 29.0 / flydsl 29.2 / one-block 35.2 / hip 47.8 / `torch.topk` 68.0 µs。
- 复测（最终版脚本，B = 1 / 16 @128K）：45.4% / 21.6%，与上表差 ≤ 0.4 个百分点。

### 2. 16 MLA heads（TP8，indexer 仍复制 64 heads）

融合占比（%），括号内为 sparse MLA µs：

| B | N = 8K | N = 32K | N = 128K |
|---|---|---|---|
| 1 | **31.7** (17.3) | **41.7** (18.7) | **55.7** (17.0) |
| 4 | **30.1** (19.4) | **37.2** (19.9) | **38.8** (21.2) |
| 16 | **22.2** (25.3) | **25.1** (25.9) | **22.6** (21.7) |
| 32 | **16.7** (32.7) | **17.0** (32.9) | **16.4** (26.0) |
| 64 | 12.4 (46.1) | 13.2 (35.1) | 12.9 (35.2) |

indexer / top-k 时间与 128 heads 一致（同一 kernel、同一形状），只是分母里的 MLA 更小。完整数据见 `data/2026-10-29-m0/m0-h16.csv`。

### 3. rocprofv3 计数器（128 heads，最后一次迭代，L2 之外的读 / 写）

FETCH_SIZE / WRITE_SIZE 是 L2 与 MALL/HBM 之间的流量。「x logits」= 该段读量 / logits 张量字节。

| 格 | logits | indexer 读 / 写 | top-k 读 / 写 | top-k 读 x logits | sparse MLA 读 / 写 |
|---|---|---|---|---|---|
| B = 1, 8K | 0.03 MB | 1.19 / 0.03 MB | 0.04 / 0.01 MB | 1.35x | 9.33 / 6.97 MB |
| B = 1, 128K | 0.52 MB | 17.47 / 0.52 MB | 2.35 / 0.17 MB | 4.49x | 9.43 / 6.97 MB |
| B = 16, 128K | 8.39 MB | 277.43 / 8.61 MB | 36.33 / 2.77 MB | 4.33x | 98.43 / 65.35 MB |
| B = 64, 128K | 33.55 MB | 1112.13 / 36.25 MB | 34.36 / 0.52 MB | 1.02x（L2 请求 1.96x，L2 hit 55%） | 185.71 / 77.19 MB |

- indexer 读量与设计 §2.1 估算一致（128K 单序列 16.8 MB + 0.5 MB 估算 vs 17.47 / 0.52 MB 实测）；写回恰好 1.0x logits。
- multi-CTA top-k 每层读 4.3–4.5x logits（设计估算 R ≈ 3–4 趟，1.5–2.0 MB @128K 单序列；实测 2.35 MB），L2 hit 只有 9–11%：刚写的 logits 并没有留在 L2。
- B = 1 时 sparse MLA 为读 1.18 MB 的 KV 搬了约 16 MB：split-KV 的 fp32 partial（约 26 split x 128 heads x 512 x 4 B）写出再读回，是这一段 28 µs 下限的主要原因。

### 4. sparse MLA max_split 扫描（µs）

128 heads（单独一轮扫描，ms 32 一列与主表同配置、不同次运行）：

| B | N | ms 4 | ms 8 | ms 16 | ms 32（主表配置） | ms 64 |
|---|---|---|---|---|---|---|
| 1 | 8K | 41.0 | 34.4 | 31.0 | 29.7 | 28.7 |
| 4 | 8K | 41.4 | 35.5 | 33.6 | 31.8 | 37.1 |
| 16 | 8K | 46.6 | 45.8 | 48.8 | 48.9 | 49.1 |
| 64 | 8K | 75.9 | 75.7 | 75.3 | 74.7 | 76.2 |
| 1 | 128K | 42.4 | 34.2 | 30.5 | 28.4 | 28.7 |
| 4 | 128K | 40.0 | 36.0 | 33.6 | 32.7 | 35.9 |
| 16 | 128K | 44.1 | 43.2 | 44.6 | 44.7 | 44.3 |
| 64 | 128K | 69.9 | 69.6 | 70.1 | 69.4 | 71.0 |

max_split = 32 离逐格最优最多 3.2 µs（128 heads）/ 1.6 µs（16 heads）。最差的恰好是边界格 B = 16 @8K：用 ms 8 时融合占比从 14.5% 变为约 15.2%，即正好压线，判定为「边界」不变。完整扫描（含 16 heads）见 `data/2026-10-29-m0/max-split-sweep.csv`。

### 5. aiter 问题（应上报）

| # | 问题 | 复现 / 证据 | 规避 |
|---|---|---|---|
| A1 | `get_mla_metadata_v1(topk=2048, is_sparse)` 只写 `work_indptr[0..num_cu]`（257 / 305 项），B = 1 时 `work_info_set` 2224 / 2432 项、`reduce_partial_map` 278 / 304 项未写；asm kernel `mla_a8w8_qh32_qseqlen4_gqaratio32_ps` 按分配大小发 304 个 WG，读到未初始化内容 → HSA memory fault | aiter 自己的 `op_tests/test_mla_sparse.py -n 128,1` 和单独的最小复现都能过（推测是缓冲拿到的新内存恰好为 0）；harness 里先分配过 indexer / top-k 张量后用 `torch.empty` 分配即 fault；已排除池大小、默认设备、indexer / top-k 越界写 | metadata 缓冲用 `torch.zeros` 分配（harness 已这么做，之后 1 − cos = 9e-5） |
| A2 | `aiter.top_k_per_row_decode_fast` 在 gfx950 直接 abort 进程（不可 catch） | `aiter_meta/hsa/gfx950/` 下没有 `topk_per_row_decode/`，只有 gfx942 有 | harness 调用前检查 hsaco 目录，缺失则跳过 |

## Interpretation

1. **可融合的不是字节，是 top-k 的延迟。** logits 写回在所有格都 < 9 µs、≤ 2.4% wall；top-k 段自身占 9–46% wall（128 heads）。top-k 段时间几乎与 batch 无关（@128K：B = 1 29 µs，B = 32 31 µs），说明它是延迟/串行趟数主导：multi-CTA radix 是 7 个依赖 kernel，每个 kernel 至少约 1.5 µs launch floor，有效带宽只有约 1 TB/s 量级（B = 16 @128K：36 MB / 34.8 µs）；one-block 路径读量 1.0x，但每行一个 workgroup 串行扫 128K，B = 64 时也要 37.6 µs。这正是设计 §2.2 的论点，且比 §2.1 的字节估算（10–12%）高得多。
2. **batch 是比 N 更强的开关。** indexer 随 B x N 线性增长，sparse MLA 随 B 增长，top-k 基本不变，所以占比随 B 快速下降。B ≥ 32 时：32K / 128K 下 indexer 占 40–73%（R1 的情形），8K 下 sparse MLA 占 68–73%。DSA decode 的部署 batch 决定这个项目值不值：小 batch / 长上下文 / TP8 是它的主场。
3. **indexer 本身有约 1.5x 余量，而且会抬高融合占比。** 同卡 torch `add_` 512 MB 实测 5.93 TB/s、`copy_` 1 GB 5.29 TB/s（读+写），aiter indexer 在大格只有 2.7–3.9 TB/s。推算（非实测）：若 M1 的 indexer 达到 5.9 TB/s、其余两段不变，128K 下 B = 16 融合占比 21.5% → 28.3%，B = 32 14.0% → 18.3%，B = 64 11.5% → 14.4%。所以 M1 的 FP8 indexer 是 M3 在 B = 16 达到 25% 的前提条件，不是附带工作。
4. **sparse MLA 在小 batch 有一个与 N 无关的约 28 µs（128 heads）/ 约 17 µs（16 heads）下限**，主要来自 split-KV partial 往返（Result §3），max_split 扫描去不掉。它在 B ≤ 4 占 wall 38–74%（128 heads），但不在 top-k 融合的射程内：MLA 必须等最终 top-k。M3 的 role-split 流水最多把它和下一层的 indexer 重叠，这一点 M3 再评估。
5. **对 M2/M3 的含义。** M2 的真实对手不是「logits 写回」，而是「一个更好的独立 top-k」。如果一个单 kernel / persistent 的独立 top-k（例如用上一 step 阈值的 GVR 式初值）就能把 128K 的 29 µs 压到 10 µs 以下，那么融合的边际收益在 B = 16 会掉到 15% 以下。所以 M1 必须先回答「独立 top-k 能做到多快」，再决定 M2 的融合是否还值。
6. 与 LiteTopK / GVR 的关系：我们比的是 CDNA4 + 三段独立 kernel（aiter）与未来三段一核的差距。这里实测的 top-k 结构性问题（多趟、低 L2 命中、延迟主导、128K 时成为单层第一或第二大项）与它们的定性结论一致，但数字不可直接比较（不同硬件、不同基线实现）。

### 局限

- 随机输入：radix 趟数与 one-block 的提前退出依赖数据分布，真实 DSv3.2 分数分布可能更尖锐或更平坦（R2）。
- 只测 `next_n = 1`（无 MTP）、batch 内等长序列。
- fp8 MLA 用的是 576 B/token + per-tensor scale，不是 V3.2 的 656 B `fp8_ds_mla` 布局（`mla_decode_fwd_ds32` 未测）。
- 单卡，graph 模式；小格（< 50 µs）误差约 ±0.5–1 µs（同格 top-k 变体扫描与主测之间最多差约 3 µs @B = 64 128K）。
- req→global 索引转换 kernel 计入 sparse MLA 段，没有算进可融合部分（保守口径）。
- logits 写回用字节份额估算，rocprof 证实写量为 1.0x logits。

## Next

| 优先级 | 事项 | 验收 |
|---|---|---|
| P0（下一轮） | 开 M1：FlyDSL FP8 paged MQA logits indexer（DeepGEMM 语义逐位对拍） | 大格带宽 ≥ 5 TB/s（aiter 2.7–3.9 TB/s），不差于 aiter |
| P0（下一轮） | M1：k = 2048 精确 radix top-k，单 kernel / persistent，含「上一 step 阈值」初值实验 | 精确 top-k；@128K 优于 aiter 29 µs；并给出「独立 top-k 极限」作为 M2 的新门槛 |
| P1 | M2 开工前重算门槛：用 M1 的 indexer + top-k 替换本表对应列，B = 16 @128K 融合占比仍 ≥ 15% 才做 M2 | 更新本表 |
| P1 | M3 目标改写为「@128K、B ≤ 4（128 heads）或 B ≤ 16（16 heads）降 ≥ 25%」；B = 16 / 128 heads 的 25% 依赖 M1 indexer 达到 roofline | 设计文档 §5 更新 |
| P2 | 补测：真实 DSv3.2 logits 分布、MTP `next_n = 2`、变长 batch、`fp8_ds_mla` 656 B 布局 | 判定是否稳健 |
| P2 | 向 aiter 上报 A1（sparse MLA metadata 尾部未初始化）和 A2（gfx950 缺 fast top-k hsaco） | issue 链接 |

§7 备选方向（CP 通信融合 / Mega-MLA / FMHA bwd）本轮不触发。本轮未写任何 M1 代码，也未改 FlyDSL 稀疏接口。

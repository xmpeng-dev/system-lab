# 2026-10-29 14:00  M0 BASELINE（训练）— CP 下 DSA / CSA attention 分段占比（MI355X）

> **When**: 2026-10-29 14:00 (Asia/Shanghai)
> **Where**: `smci355-ccs-aus-n06-25` / container `xiaoming-dev`，单卡 MI355X (gfx950)，同 [decode M0](./2026-10-29_1330_BASELINE_m0_dsa_decode_three_kernel_breakdown.md) 的软件栈；primus_turbo 0.6.1.dev4（容器内安装版，gfx950 sparse MLA 路径与 Primus-Turbo `origin/main` `b4ad3bcb` 一致）
> **Context**: 用户要求看训练场景：开 CP、不考虑 TP。生产配置参照 Primus `examples/megatron/models/deepseek-v4/gfx942/run_full_128k_thd_multinode.sh`（V4-Flash SFT，seq 131072，TP 1 / PP 3 / CP 8 / EP 8，MBS 1）
> **代码**: Primus-Turbo 分支 `feat/megaattn/m0-dsa-decode-baseline`（未 commit），新文件 `benchmark/ops/training/bench_dsa_train_breakdown.py`
> **数据**: [`data/2026-10-29-m0-train/train-breakdown.csv`](./data/2026-10-29-m0-train/train-breakdown.csv)

## TL;DR

**判定：训练场景 NO-GO。** 「top-k + logits 写回」只占一个 CSA 层 attention 前向 + 反向的 1–5%（128K 及以下），远低于 15% 门槛：

| S（CP） | V4-Flash | V4-Pro | V3.2 |
|---|---|---|---|
| 4K（CP 1） | 0.7 (0.6) | 0.2 (0.1) | 0.3 (0.2) |
| 32K（CP 8） | 2.1 (1.5) | 1.0 (0.7) | 1.6 (1.0) |
| 64K（CP 8） | 3.6 (2.4) | 1.6 (1.0) | 2.8 (1.4) |
| **128K（CP 8，生产配置）** | **5.3 (2.9)** | **2.5 (1.4)** | **4.6 (2.2)** |
| 256K（CP 8） | 8.1 (3.9) | 4.5 (2.2) | 8.4 (4.0) |
| 512K（CP 16） | 11.9 (5.5) | 7.3 (3.4) | — |
| 1M（CP 32） | **16.2** (7.7) | 11.4 (5.5) | — |

单位 %。括号外是上界：top-k + logits 写回，写回按满带宽单独计；括号内只算 top-k。

- **mbs 不影响结论。** 训练里每一段都按「行数 x 可见 key」或「行数 x top-k」线性增长，mbs 和 CP 只改变行数。128K 下 CP 4 / 8 / 16（行数 32K / 16K / 8K）耗时近似按 2 倍缩放，占比不变：V4-Flash 5.3 / 5.3 / 5.2%。唯一的开关是上下文长度 S。
- **放到整个 attention 栈更小。** 128K、CP 8 下 V4-Flash 的 HCA 层（20.1 ms）比 CSA 层（15.2 ms）还贵，可融合部分只占全部 attention 层 fwd+bwd 的 2.3%（V4-Pro 1.3%）。再加上 MoE 等其余部分，占整个训练 step 的比例更低。
- **训练里时间花在 sparse MLA 反向上。** 128K 时占 CSA 层 66–73%。上下文 ≥ 256K 后，indexer 占比明显上升：V4-Flash / V3.2 为 24–51%，V4-Pro 为 14–36%。
- 与 decode 不同：decode 的 top-k 是与 batch 无关的固定延迟（每层约 30 µs @128K）；训练每个 rank 有几千到几万行，top-k 在 8K 行以上时单趟读带宽已达 2.2–3.5 TB/s，不是瓶颈。

## Background

[decode M0](./2026-10-29_1330_BASELINE_m0_dsa_decode_three_kernel_breakdown.md) 的判定是「小 batch 有条件 GO」。训练形状完全不同：每个 token 都是 query，CP rank 持有 `mbs x S / CP` 行，每行的因果窗口平均为全局 key 数的一半。设计 §6 R1 的问题（融合收益是否 < 15%）需要在训练形状下重新回答。

模型与层类型（Primus `primus/configs/models/megatron/deepseek_v4_{flash,pro}.yaml`）：

| 模型 | attention heads | index_topk | CSA / HCA / 纯滑窗 层数（含 MTP） | indexer |
|---|---|---|---|---|
| V4-Flash | 64 | 512 | 21 / 20 / 3 | 64 heads x 128，压缩比 4 的 pool（S/4 项） |
| V4-Pro | 128 | 1024 | 30 / 31 / 1 | 同上 |
| V3.2（对照） | 128 | 2048 | 全部为 DSA 层 | 64 heads x 128，全部 token |

只有 CSA 层有 indexer + top-k；HCA 层看全部可见的 S/128 pool，纯滑窗层只看 128 token 窗口。

## What I did

### 1. 每段 kernel（全部是当前最快的独立实现）

| 段 | kernel | 说明 |
|---|---|---|
| indexer | aiter `flydsl_fp8_mqa_logits`（clean / 不 clean 两种）与 `fp8_mqa_logits`（gfx950 上是 Gluon），逐格取最快 | fp8 q / k + per-token scale，fp32 logits `[rows, n_keys]` |
| top-k | aiter `top_k_per_row_prefill`（按每行 `[start, end)` 选） | k = 512 / 1024 / 2048 |
| sparse MLA fwd / bwd | Primus-Turbo `sparse_mla_func`，固定用 FlyDSL backend | bf16，QK 576（kernel 只算 512 latent，见局限）/ V 512；V4 带 128 窗口 + sink |

- CP 切分：Megatron 负载均衡切法，rank r 持有第 r 块和第 2·CP−1−r 块（每块 S / (2·CP) token），key 为 all-gather 后的全局集合。测的是 rank 0；各 rank 的可见长度之和相同。
- V4 CSA：pool 第 p 行从 token `4(p+1)−1` 起可见；sparse MLA 的索引 = 128 窗口 + top-k 选中的 pool 行。V3.2：索引直接是 top-k 选中的 token。
- `--stack`：用同一个 sparse MLA kernel 额外测 HCA（窗口 + 全部可见的 S/128 pool 行）和纯滑窗层的 fwd+bwd，按层数加权得到整个 attention 栈。
- 计时：eager，每段单独用 CUDA event，前置 512 MB flush，取 10 次中位数。所有段都是 ms 级，launch 开销可忽略。
- logits 写回上界 = logits 字节 / 同进程实测 `add_` 带宽（5.95 TB/s 读+写），假设写回完全没被 indexer 的计算掩盖。

### 2. 正确性（25 格全部通过）

| 检查 | 结果 |
|---|---|
| indexer：抽 8 行，对 fp8 反量化输入的 fp32 参考 | 1 − cos ≤ 1.6e-10 |
| top-k：抽 16 行，索引唯一、在窗口内、选中的最小值 ≥ 未选中的最大值 | 全部精确 |
| sparse MLA：首尾各 256 行，FlyDSL 对 Triton，o / dq / dkv / dsink 取最小 SNR | ≥ 50.2 dB |

### 3. 复现

```bash
docker exec -it xiaoming-dev bash
cd /perf_apps/xiaoming/Primus-Turbo/benchmark/ops/training
HIP_VISIBLE_DEVICES=0 python -u bench_dsa_train_breakdown.py --models v4flash v4pro v32 \
    --seqlen 32768 65536 131072 --cp 8 --check --stack --iters 10 -o train-cp8.csv
# 4K: --seqlen 4096 --cp 1；线性检验: --seqlen 131072 --cp 4 16；
# 长上下文: --seqlen 262144 --cp 8 / --seqlen 524288 --cp 16 / --seqlen 1048576 --cp 32（只跑 v4）
```

## Result

### 1. 128K、CP 8（生产配置）单个 CSA 层，单 rank，fwd + bwd

| 模型 | rows | indexer ms | top-k ms | MLA fwd ms | MLA bwd ms | 合计 ms | logits GB | indexer PFLOPS | MLA fwd / bwd TFLOPS | top-k 单趟读 TB/s |
|---|---|---|---|---|---|---|---|---|---|---|
| V4-Flash | 16384 | 2.22 | 0.44 | 2.15 | 10.33 | 15.15 | 2.15 | 1.98 | 608 / 316 | 2.43 |
| V4-Pro | 16384 | 2.38 | 0.44 | 5.96 | 23.18 | 31.96 | 2.15 | 1.85 | 738 / 474 | 2.42 |
| V3.2 | 16384 | 8.96 | 1.29 | 9.94 | 39.05 | 59.24 | 8.59 | 1.96 | 830 / 528 | 3.32 |

各段占合计的比例：indexer 15 / 7 / 15%，top-k 2.9 / 1.4 / 2.2%，sparse MLA 前向 + 反向 82 / 91 / 83%（V4-Flash / V4-Pro / V3.2）；其中反向单独占 68 / 73 / 66%。

TFLOPS 按 Primus-Turbo bench 的口径：前向 `2 x H x 有效槽位 x 1024`，反向按 2.5 倍计。

### 2. mbs / CP 线性检验（128K）

| 模型 | CP 4（32K 行） | CP 8（16K 行） | CP 16（8K 行） |
|---|---|---|---|
| V4-Flash fwd+bwd ms（占比上界） | 29.53 (5.3%) | 15.15 (5.3%) | 8.07 (5.2%) |
| V4-Pro | 57.29 (2.8%) | 31.96 (2.5%) | 17.21 (2.4%) |
| V3.2 | 108.31 (5.1%) | 59.24 (4.6%) | 29.90 (4.7%) |

行数翻倍，耗时约翻倍，占比在 0.5 个百分点内不变。mbs 增大等价于行数按同样的可见长度分布翻倍，所以结论同样适用于 mbs。

### 3. 整个 attention 栈（128K、CP 8，每层 fwd+bwd ms）

| 模型 | CSA 层 | HCA 层 | 纯滑窗层 | 层数 CSA / HCA / 滑窗 | 可融合占 attention 栈（上界） |
|---|---|---|---|---|---|
| V4-Flash | 15.15 | 20.07 | 3.93 | 21 / 20 / 3 | 2.3% |
| V4-Pro | 31.96 | 27.58 | 7.26 | 30 / 31 / 1 | 1.3% |

这只算 sparse attention kernel 本身；不含 q/kv 投影、compressor、CP 通信和 MoE。

### 4. 上下文长度趋势

完整数值见 TL;DR 表与 CSV。从 128K 到 1M，V4-Flash 的各段占比变化：
- indexer：15% → 51%（FP8 稳定在约 2.0–2.1 PFLOPS）
- top-k：2.9% → 7.7%（单趟读带宽 2.4–3.5 TB/s）
- sparse MLA：82% → 41%

## Interpretation

1. **为什么训练和 decode 结论相反。** decode 的 top-k 每层只有几行到几十行，是串行多趟、延迟主导的固定成本（约 30 µs @128K，与 batch 无关），所以小 batch 下占比可以到 46%。训练每个 rank 至少几千行，top-k 跑满带宽，成本与 indexer 一样都按「行数 x 可见 key」增长；而 indexer 是 2 PFLOPS 的 FP8 计算，top-k 只是它的 14–20%。
2. **能融合的那部分在训练里有上限。** 不管 S 多大，top-k + 写回最多也只相当于「indexer + top-k」的约 25–30%（写回本来就包含在 indexer 时间里）。剩下的是 sparse MLA，它按「行数 x (k + 窗口)」增长、与 S 无关。所以只有当 S 大到 indexer 压过 sparse MLA 时，占比才接近 15%：V4-Flash 在 1M 时上界 16.2%（只算 top-k 为 7.7%），V4-Pro 在 1M 时也只有 11.4%。
3. **mbs 不是开关。** 见 Result §2。「mbs 取几值得做」的答案是：训练场景下任何 mbs 都不值得；值不值得只看上下文长度，而且要到约 1M 才开始接近门槛。
4. **logits 的显存是另一个问题，不是速度问题。** 每层每 rank 的 logits 是 `rows x n_keys x 4 B`：128K / CP 8 时 V4 为 2.15 GB、V3.2 为 8.59 GB，1M / CP 32 时 V4 为 34 GB。Primus `Indexer` 已经按 pool 分块并维护流式 top-k 来规避（`primus/backends/megatron/core/transformer/indexer.py`），所以这不构成融合的理由。
5. **训练侧真正的大头**（只是指出方向，本轮没做）：
   - sparse MLA 反向：128K 时占 CSA 层 66–73%。V4-Flash（64 heads）反向只有 316 TFLOPS，前向是 608 TFLOPS。
   - V4 的 HCA 层：同一个 sparse MLA kernel 处理「窗口 + 全部可见 pool」，128K 时 V4-Flash 一层 20.1 ms，比 CSA 层还贵。
   - 上下文 ≥ 256K 时的 indexer（V4-Flash 在 1M 时占 51%）。
   - 这三项与设计 §7 的备选方向 3（反向）最接近。备选方向 1（CP 通信融合）本轮没有测量：harness 不含 all-gather。

### 局限

- 单 rank（rank 0），不含 CP 通信、投影、compressor、RoPE、indexer distill loss 及其反向。生产默认 `PRIMUS_V4_INDEXER_DISTILL_LOSS_COEFF=0.0`，indexer 冻结、没有反向；从头预训练要打开，那样分母更大，占比只会更低。
- indexer 用的是 aiter FP8。生产默认 `USE_V4_FP8_INDEXER=False`（bf16 indexer，走 Primus `Indexer`），本轮没测；生产路径的 indexer + top-k 可能明显更慢，但那是「换更快的独立 kernel」能解决的，不需要融合。
- 生产默认 `PRIMUS_RECOMPUTE_LAYERS=0`，不开 recompute。若开 full recompute，前向付两次，例如 V4-Flash 128K 的上界约 8%（(0.44 + 0.36) x 2 / (4.81 x 2 + 10.33)），仍低于 15%。
- Primus-Turbo sparse MLA 按 V4 single-latent 语义只算 512 维 QK，V3.2 真实的 64 维 rope 被跳过。V3.2 的 sparse MLA 时间因此约偏小 6%，占比相应略偏高。
- 输入随机；radix top-k 的趟数与数据分布有关。
- logits 写回按满带宽、不与计算重叠取上界，真实值在「只算 top-k」与上界之间。

## Next

| 优先级 | 事项 |
|---|---|
| — | MegaAttn v1（indexer + top-k 融合）的适用范围定为 decode 小 batch（见 decode M0）。训练场景不立项，除非目标上下文到约 1M。 |
| 待定（需用户决定） | 训练侧若要做 attention，候选是 sparse MLA 反向（64 heads 时只有前向吞吐的一半左右）、V4 HCA 层路径、≥ 256K 时的 indexer |
| 待定 | 量一次生产 Primus 的 bf16 `Indexer` + top-k 路径，与 aiter FP8 kernel 对比，判断是否存在「换 kernel」的快速收益 |
| 待定 | CP 通信（pool / KV all-gather、窗口边界交换）占比，用于评估设计 §7 备选方向 1 |

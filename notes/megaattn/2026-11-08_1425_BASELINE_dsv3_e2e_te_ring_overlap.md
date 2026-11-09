# M0-overlap-e2e-MLA：TE ring attention overlap 的端到端收益（DeepSeek-V3 4 层代理，MI355X）

日期 2026-11-08 14:25。接 [MLA kernel 级](./2026-11-08_1220_BASELINE_dsv3_mla_ring_attn_overlap.md)：那篇是单个 MLA attention
层，这篇是 Primus 整步训练。方法与 [Llama 端到端篇](./2026-11-08_1010_BASELINE_llama_e2e_te_ring_overlap.md) 相同。只测
Primus 当前已支持的路径，不做任何优化。

## 结论

- **DSV3 4 层代理 BF16 pretrain，TE ring overlap 相对不 overlap 每步省：CP2 18.2–20.0%，CP4 13.7–30.3%，CP8 6.5–22.5%**
  （1.07–1.43×；CP8 32K 两次运行 7.6% / 6.5%，其余重跑格子差 ≤ 0.2 个百分点）。CP2 128K 两种模式都 OOM。
- **端到端省的时间 ≈ 5 层 × 单层 MLA attention 省的时间**：比值 0.94–1.12（含重跑；4 个主层 + 1 个 MTP 层）。kernel 级结论可以直接
  换算到整步，和 Llama（0.93–1.12）一样。
- **attention 占 overlap 后步时约 44–88%**（5 × 单层 overlap 耗时 / 整步，估算）；CP4 / CP8 是 73–88%。
- **把 overlap 推到单层 roofline 的上限，整步最多还能再省 6.4–18.6%**（roofline 是乐观下界）。CP4 / CP8 下 roofline 就是
  T_comm，这部分只能靠通信本身变快（见 MLA kernel 篇）。
- **和 Llama 比**：CP2 省得多（MLA 18.2–20.0% vs GQA 4.4–5.2%，MLA 通信大且在 CP2 还能藏住）；CP8 省得少（MLA
  6.5–22.5% vs GQA 15.3–31.6%，MLA CP8 comm-bound，overlap 只能藏住更小的计算）。
- **同一 S 下 CP 加大，每卡吞吐掉得比 token 数减少更快**：32K overlap 后 CP2 / 4 / 8 每卡 14233 / 5646 / 2770 tokens/s，
  步时 1151 / 1451 / 1479 ms，每 rank 处理的 token 是 16384 / 8192 / 4096。原因和 kernel 级一致：TE 单层 MLA attention
  CP2 101 ms、CP4 212 ms、CP8 243 ms（CP > 2 时 P2P 不 batch，comm-bound）。
- **MoE 路径没有按配置生效**（不影响 attention 与两种模式的差值，影响步时与 attention 占比）：Primus 当前代码要
  `primus_turbo.triton.elementwise` 等模块，已安装的 primus_turbo 0.6.1.dev4 与工作区源码都没有，5 个 patch import 失败
  （见方法），实际走 Megatron `alltoall` 分发与 TE 实现，不是配置里的 Turbo DeepEP / Turbo RMSNorm。

## 环境与方法

- 环境同 Llama 端到端篇：8× MI355X，ROCm 10.1.0，PyTorch 2.14.0+rocm10.1.0，RCCL 2.30.7，TE 2.17.1+130099ce4（CK fused
  attention），Primus `c3a221b7b`（工作区有别人的未提交改动），Megatron-LM `d3528a213`。
- 配置：`examples/megatron/configs/MI355X/deepseek_v3-BF16-pretrain.yaml`，按 `Primus/run_recompute.sh` 的方式切到单机：
  `num_layers 4`、`moe_layer_freq 1`（4 层都是 MoE；真实 DSV3 前 3 层是 dense）、PP 1、EP 8、无重计算、
  `moe_router_force_load_balancing_type uniform`。另外覆盖 `seq_length = max_position_embeddings = S`、`context_parallel_size = CP`、
  `cp_comm_type: p2p`、TP 1、`micro_batch_size 1`、`global_batch_size = DP = 8 / CP`、`train_iters 15`、`mock_data`、
  `create_attention_mask_in_dataloader: false`。其余照配置：MLA（128 heads，q_lora 1536，kv_lora 512，d_qk 192，d_v 128），
  `mtp_num_layers 1`，256 experts top-8，`use_turbo_attention: false`（走 TE `AttnFuncWithCPAndKVP2P`），
  `apply_rope_fusion`，precision-aware optimizer。
- 未生效的 patch（rank 0 日志 `[Patch] ✗ ... failed due to missing dependency`）：`megatron.moe.primus_topk_router`、
  `megatron.fused_pad_routing_map`、`megatron.moe.permute_fusion`、`megatron.turbo.moe_dispatcher`、`megatron.turbo.rms_norm`。
  生效参数里 `moe_token_dispatcher_type = alltoall`。Llama 那批只有 MoE 专用 patch 失败，对 dense 模型无影响。
- 不 overlap：同 Llama 篇，`sitecustomize.py` 运行时把 TE `flash_attn_p2p_communicate` 串行化，不改源码。校验：8 个 rank 退出
  时的调用次数都等于 15 步 × 5 层 × (前向 CP−1 + 反向 CP)，即 CP2 225、CP4 525、CP8 1125；这也确认了每步 5 个 attention 层。
- 步时：Primus 打印的 `elapsed time per iteration`（rank 7），去掉前 3 步，取 12 步的中位数。少数格子有单步离群（如 CP2 64K
  overlap 12 步里 1 步 5635 ms，其余 2793–2797 ms），中位数不受影响。

## 每步耗时（ms）

| CP | S | 不 overlap | overlap | 省 | 加速比 | 重跑（不 overlap → overlap） | 每卡 tokens/s（overlap） | 峰值显存 (GB) |
|---|---|---|---|---|---|---|---|---|
| 2 | 32K | 1438.8 | 1151.2 | 20.0% | 1.250× | 1438.8 → 1150.7（20.0%） | 14233 | 169 |
| 2 | 64K | 3415.3 | 2795.4 | 18.2% | 1.222× | | 11722 | 218 |
| 2 | 128K | OOM | OOM | - | - | | - | - |
| 4 | 32K | 1680.8 | 1450.8 | 13.7% | 1.158× | | 5646 | 145 |
| 4 | 64K | 3558.9 | 2791.7 | 21.6% | 1.275× | | 5869 | 168 |
| 4 | 128K | 8345.2 | 5818.9 | 30.3% | 1.434× | 8322.1 → 5819.9（30.1%） | 5631 | 225 |
| 8 | 32K | 1600.8 | 1478.8 | 7.6% | 1.082× | 1580.7 → 1478.3（6.5%） | 2770 | 127 |
| 8 | 64K | 3267.1 | 2828.5 | 13.4% | 1.155× | | 2896 | 150 |
| 8 | 128K | 7157.5 | 5547.8 | 22.5% | 1.290× | | 2953 | 176 |

重跑是另起的 3 次独立运行（不 overlap 在前）。CP8 32K 两次的差别在不 overlap 一侧（1600.8 vs 1580.7 ms），overlap 两次一致。

CP2 128K：每卡 64K token、无重计算，两种模式都在训练开始后申请 10 GiB 时 OOM（剩余约 3 GiB / 288 GiB），没有打出任何一步。

## 与 5 × 单层的对照

单层数据来自 [MLA kernel 篇](./2026-11-08_1220_BASELINE_dsv3_mla_ring_attn_overlap.md) 的 TE causal fwd + bwd。

| CP | S | e2e 省 (ms) | 5 × 单层省 (ms) | e2e / 预测 | 5 × 单层 overlap (ms) | 占 overlap 步时 | 5 × (单层 overlap − roofline) (ms) | 占 overlap 步时 |
|---|---|---|---|---|---|---|---|---|
| 2 | 32K | 287.7 | 288.7 | 1.00 | 506.0 | 44% | 110.1 | 9.6% |
| 2 | 64K | 619.9 | 631.7 | 0.98 | 1637.0 | 59% | 178.7 | 6.4% |
| 4 | 32K | 230.0 | 222.3 | 1.03 | 1058.2 | 73% | 158.8 | 10.9% |
| 4 | 64K | 767.2 | 774.4 | 0.99 | 2144.9 | 77% | 348.8 | 12.5% |
| 4 | 128K | 2526.4 | 2525.8 | 1.00 | 4672.8 | 80% | 1083.5 | 18.6% |
| 8 | 32K | 122.0 | 109.1 | 1.12 | 1217.4 | 82% | 166.5 | 11.3% |
| 8 | 64K | 438.6 | 453.2 | 0.97 | 2435.9 | 86% | 340.6 | 12.0% |
| 8 | 128K | 1609.8 | 1611.9 | 1.00 | 4901.1 | 88% | 716.7 | 12.9% |

CP8 32K 用重跑数据时 e2e 省 102.4 ms，比值 0.94。

## 事实与假设

- 事实（本次实测）：上面两张表；hook 调用次数校验；5 个 Turbo MoE / RMSNorm patch 因缺模块未生效。
- 估算：attention 占比与 roofline 余量用的是单层 bench 时间乘 5，不是整步里实测的 attention 时间。
- 假设（未验证）：
  - MoE 路径按配置生效（Turbo DeepEP 等）后 MoE 变快，attention 占比会更高，overlap 省的比例也会更高；省的毫秒数不变。
  - 真实 61 层 DSV3（3 dense + 58 MoE + MTP，PP 切分）每层 attention 形状相同，整步省的时间按层数线性放大；PP 气泡与
    重计算会改变比例，未测。
- 适用范围：B = 1、uniform 路由、无重计算、单机 8 卡。

## 文件

- 脚本：`/perf_apps/xiaoming/megaattention_overlap_bench/e2e/`（`run_e2e.sh dsv3 ...`、`sweep.sh`、`parse.py --layer-csv
  results/ring_impl_mla/results.csv --layers 5`），未 commit。
- 数据：[data/2026-11-08-e2e/dsv3_grid.csv](./data/2026-11-08-e2e/dsv3_grid.csv)、[dsv3_grid.md](./data/2026-11-08-e2e/dsv3_grid.md)、
  重跑 [dsv3_repeat.csv](./data/2026-11-08-e2e/dsv3_repeat.csv) / [dsv3_repeat.md](./data/2026-11-08-e2e/dsv3_repeat.md)。

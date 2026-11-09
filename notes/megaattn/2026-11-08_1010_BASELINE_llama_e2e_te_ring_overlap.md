# M0-overlap-e2e：TE ring attention overlap 的端到端收益（Llama3.1-8B，MI355X）

日期 2026-11-08 10:10。接 [现有 ring 实现 overlap](./2026-11-08_0915_BASELINE_te_turbo_ring_attn_overlap.md)：那篇是单个
attention 层，这篇是 Primus 整步训练。只测 Primus 当前已支持的路径，不做任何优化。

## 结论

- **Llama3.1-8B BF16 pretrain，TE ring overlap 相对不 overlap 每步省：CP2 4.4–5.2%，CP4 19.1–23.8%，CP8 15.3–31.6%**
  （CP8 32K 取 6 次运行的众数 15.3%，跨运行 10.7–17.6%，见下）。CP2 128K 两种模式都 OOM（每卡 64K token、无重计算）。
- **端到端省的时间 ≈ 32 层 × 单层 attention 省的时间**：比值 0.93–1.12。单层 bench 的结论可以直接换算到整步。
- **attention 占 overlap 后步时约 44–78%**（32 × 单层 overlap 耗时 / 整步，估算），随 S 增大、随 CP 增大而升高。
- **把 overlap 推到单层 roofline 的上限，整步最多还能再省 5.9–15.2%**（roofline 是乐观下界，实际可得更少）。
- **只适用于 GQA**（32 Q / 8 KV heads，D = 128）。MLA 的通信 / 计算比按形状推算约是它的 4 倍，结论不能套用。
  MLA 实测见 [DSV3 端到端篇](./2026-11-08_1425_BASELINE_dsv3_e2e_te_ring_overlap.md)。
- **测量陷阱**：Primus 默认 `create_attention_mask_in_dataloader: true`（`primus/configs/modules/megatron/trainer_base.yaml:436`），
  每个样本在 dataloader 里生成 S × S bool mask（32K 每样本 1 GiB，128K 每样本 16 GiB）。dataloader 跟不上，两种模式的步时
  都在 0.77–2.1 s 之间跳。设成 `false` 后步内波动 ≤ 0.3%。TE `DotProductAttention` 只在 `attn_mask_type` 含 padding /
  arbitrary 时才用 `attention_mask`（TE docstring），这里是 causal，所以不影响 attention 的计算和通信。Primus 自己的长序列
  配置（如 `deepseek_v4_flash_4layer-BF16-sft.yaml`）也设成 `false`。本篇全部数据都设了 `false`。

## 环境与方法

- 环境与单层篇相同：8× MI355X，ROCm 10.1.0，PyTorch 2.14.0+rocm10.1.0，RCCL 2.30.7，TE 2.17.1+130099ce4（CK fused
  attention）。Primus `c3a221b7b`（工作区有别人的未提交改动，与本篇无关），Megatron-LM `d3528a213`。
- 配置：`examples/megatron/configs/MI355X/llama3.1_8B-BF16-pretrain.yaml`，原样，只覆盖：
  `seq_length = max_position_embeddings = S`、`context_parallel_size = CP`、`cp_comm_type: p2p`、TP 1、`micro_batch_size 1`、
  `global_batch_size = DP = 8 / CP`（每 rank 每步 1 个 micro batch）、`train_iters 15`、`mock_data`、
  `create_attention_mask_in_dataloader: false`。其余照配置：32 层，无重计算，distributed optimizer，
  overlap_grad_reduce / overlap_param_gather，`use_turbo_attention: false`（走 Megatron `TEDotProductAttention` → TE
  `AttnFuncWithCPAndKVP2P`）。
- 不 overlap：`PYTHONPATH` 上的 `sitecustomize.py` 在运行时包装 TE `flash_attn_p2p_communicate`，和单层 bench 的串行变体
  语义相同：每次 P2P 先等之前所有计算，P2P 完成后才开始后面的计算，调用和 buffer 不变。没有改 Megatron、Primus、TE 源码。
- 补丁生效的校验：每个 rank 退出时写调用次数。8 个 rank 都等于 iters × 32 层 × (前向 CP−1 + 反向 CP) 次
  （CP8 15 步 = 7200）；overlap 运行没有补丁。
- 步时：Primus 打印的 `elapsed time per iteration`（rank 7），去掉前 3 步，取 12 步的中位数。

## 每步耗时（ms）

| CP | S | 不 overlap | overlap | 省 | 加速比 | 重跑（不 overlap → overlap） |
|---|---|---|---|---|---|---|
| 2 | 32K | 1330.3 | 1261.6 | 5.2% | 1.055× | 1327.8 → 1261.8（5.0%） |
| 2 | 64K | 3363.6 | 3216.0 | 4.4% | 1.046× | |
| 2 | 128K | OOM | OOM | - | - | |
| 4 | 32K | 1035.9 | 837.4 | 19.2% | 1.237× | 1035.9 → 838.4（19.1%） |
| 4 | 64K | 2391.6 | 1822.7 | 23.8% | 1.312× | |
| 4 | 128K | 6350.6 | 5048.9 | 20.5% | 1.258× | |
| 8 | 32K | 878.8 | 744.2 | 15.3% | 1.181× | 6 次运行，见下 |
| 8 | 64K | 1744.0 | 1372.0 | 21.3% | 1.271× | |
| 8 | 128K | 4284.5 | 2931.6 | 31.6% | 1.461× | 4277.6 → 2931.9（31.5%） |

CP8 32K 跑了 6 次，两种模式都是双峰：不 overlap 5 次 878.1–879.2、1 次 903.8；overlap 4 次 744.0–744.5、2 次
784.9–785.1。表里取众数。其他重跑过的格子两次之差 ≤ 0.3%。双峰原因未查（猜测是每次启动时通信初始化的状态不同，未验证）。

## 与单层 attention 的对照

单层数字取单层篇 TE causal 的 fwd + bwd（形状与 Llama3.1-8B 的 attention 完全相同）。

| CP | S | 端到端省 (ms) | 32 × 单层省 (ms) | 端到端 / 预测 | 32 × 单层 overlap (ms) | 占 overlap 步时 | 推到 roofline 还能省（占步时） |
|---|---|---|---|---|---|---|---|
| 2 | 32K | 68.7 | 67.9 | 1.01 | 549.6 | 44% | 6.6% |
| 2 | 64K | 147.6 | 156.9 | 0.94 | 1921.8 | 60% | 6.5% |
| 4 | 32K | 198.5 | 177.4 | 1.12 | 424.4 | 51% | 14.3% |
| 4 | 64K | 568.9 | 545.6 | 1.04 | 1149.7 | 63% | 11.3% |
| 4 | 128K | 1301.7 | 1335.4 | 0.97 | 3786.4 | 75% | 5.9% |
| 8 | 32K | 134.6 | 121.0 | 1.11 | 453.0 | 61% | 10.8% |
| 8 | 64K | 372.0 | 400.1 | 0.93 | 908.4 | 66% | 14.6% |
| 8 | 128K | 1352.9 | 1302.1 | 1.04 | 2298.0 | 78% | 15.2% |

最后一列 = 32 × (单层 overlap − 单层 roofline) / overlap 步时。roofline 的定义见单层篇，是每个 phase 的
max(T_comm, T_comp) 之和。

## 事实与假设

- 事实（本次实测）：上面两张表；dataloader mask 打开时 CP8 32K 步时 overlap 767–2149 ms、不 overlap 908–2061 ms，
  慢的那几步所有 GPU 同时掉到约 300 W（正常 400–840 W），GPU 时钟 2.1–2.4 GHz、无降频标志；关掉后两种模式步内波动 ≤ 1 ms。
- 推导：端到端 / 预测比值接近 1，说明 overlap 的收益几乎全部来自 attention 层本身，GEMM、优化器、梯度通信没有把它吃掉或放大。
- 假设（未验证）：
  - 「推到 roofline 还能省」是上限：roofline 不含 ring 步间依赖和并发争用，tile 级 overlap 实际能拿到多少需要原型才知道。
  - GQA 的通信 / 计算比偏低。MLA（DSV3，TE 收到的是 up-projection 后的 128 头 K 192 / V 128）按形状推算每 token 通信是
    Llama 的 20 倍、每对 token 计算是 5 倍，比值约 4 倍；CP4 / CP8 很可能变成通信瓶颈。

## 文件

工程：`/perf_apps/xiaoming/megaattention_overlap_bench/e2e/`（`run_e2e.sh` 单次运行、`sweep.sh` 网格、`repeat.sh` 重跑、
`parse.py` 汇总、`hook/sitecustomize.py` 串行补丁），未 commit。数据：[data/2026-11-08-e2e/](./data/2026-11-08-e2e/)
（`llama_grid.*` 主网格、`llama_repeat.*` 与 `llama_cp8_32k_*` 重跑、`*_with_dataloader_mask.*` mask 打开时的步时与 GPU
功耗 / 时钟采样）。

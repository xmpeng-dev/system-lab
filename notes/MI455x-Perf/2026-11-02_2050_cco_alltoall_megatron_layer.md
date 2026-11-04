# cco-alltoall：按 Megatron overlap_moe_expert_parallel_comm 的真实配对（2026-11-02 20:50）

4 × MI455X，`ctheliosr-rck-g02-e15-01`，容器 `amdprimus/amdprimus:tas-20261019`，RCCL 默认配置，300 ms 窗口 × 10 轮。
数据：`data/2026-11-02_cco_alltoall/results/megatron_layer/`（M3 一层 4 段），`results/dsv3_flydsl_attn/`（DSv3 FlyDSL MLA）；
单卡检查：`scripts/lcheck.py`、`scripts/bwdcheck.py`。客户 README：`minimax-poc/examples-phase2/training/cco-alltoall/`。

## 为什么改

之前的测例假设 "TBO 稳态"，单个 kernel × 5 个通信大小。查 Primus / Megatron 后：

- **没有 TBO 这个配置**。TBO 是 SGLang 推理的说法。训练里对应的是 Megatron `overlap_moe_expert_parallel_comm`
  （`trainer_base.yaml` 默认 false，M3 示例配置没开）：不切 micro-batch，而是 micro-batch i + 1 的前向和 i 的反向逐层交替
  （`pipeline_parallel/combined_1f1b.py`，`combined_1f1b_schedule_for_no_pipelining`）。
- **通信量是定的**：mbs 1 × seq 4096、top-4、hidden 6144、Megatron `alltoall` dispatcher（不考虑 DeepEP），每次 alltoall =
  4096 × 4 × 6144 × 2 B = 192 MiB；dispatch / combine / 两个反向都是这个大小。96 MiB（"TBO 一半"）和更小的不是真实场景。
  （Primus M3 示例配置实际开了 `use_turbo_deepep`，按用户要求不考虑。）
- **配对的计算是整段模块**，不是单 kernel。`models/common/model_chunk_schedule_plan.py`，`TransformerLayerSchedulePlan.run`：

  ```text
  comm_stream: combine_bwd            | dispatch_fwd->dispatch_bwd  | combine_fwd
  comp_stream: attn_fwd->post_attn_fwd| mlp_bwd->mlp_bwd_dw->mlp_fwd| post_attn_bwd->attn_bwd
  ```

  加上数据依赖（dispatch_bwd 要 mlp_bwd 的梯度，mlp_fwd 要 dispatch_fwd 的数据，attn_bwd 要 dispatch_bwd 的结果），
  实际是 4 对 "1 次 192 MiB alltoall ‖ 1 段无依赖计算"：

  | 段 | 配对 alltoall | 发起顺序 | 计算 |
  |---|---|---|---|
  | `layer_attn_fwd` | combine bwd (i) | 通信先 | QKV + attention + O proj fwd |
  | `layer_mlp_bwd` | dispatch fwd (i+1) | 计算先 | routed + shared expert dgrad + wgrad |
  | `layer_mlp_fwd` | dispatch bwd (i) | 通信先 | shared expert + routed expert fwd |
  | `layer_attn_bwd` | combine fwd (i+1) | 通信先 | O proj + attention + QKV dgrad + wgrad |

  shared expert：这版 Megatron 在 `submodule_moe_forward` 里调 `shared_experts_compute`，所以在 mlp 段。
  `delay_wgrad_compute` 默认 false，wgrad 跟 dgrad 一起在 mlp_bwd / attn_bwd 里（`mlp_bwd_dw` 是空操作）。
  没计入：norm、router、SwiGLU、permute / unpermute（访存型，可能和 RCCL 抢 HBM，之后可以补）。

## 实现（`common/cco_bench.py`）

- `_Layer`：M3 每卡形状。QKV 6144 → 9216（64 × 128 + 2 × 4 × 128），O 8192 → 6144；shared expert 3072；routed 32 × 512 行。
- grouped GEMM：FlyDSL gfx1250 NT（fwd）、NN（dgrad，复用 `m_tiles`）、variable-K（wgrad，复用 `build_expert_table`）。
  dense：hipBLASLt data-parallel grid。attention：aiter asm（GQA causal）。
- `Episodes.run(..., comm_first)`：同一时刻发起的计算 / 通信按 Megatron 顺序排。
- 单卡检查（`lcheck.py`）：4 段都对 FP32 1.7e-3，`torch.cuda.set_sync_debug_mode` 下 0 次 host 同步。

| 段 | 单卡短测 | TF | 4 卡持续 |
|---|---|---|---|
| attn fwd | 0.550 ms | 2092 | 0.635 |
| mlp bwd | 2.732 | 1698 | 3.089 |
| mlp fwd | 1.043 | 2223 | 1.308 |
| attn bwd | 1.360 | 1794 | 1.550 |

4 卡持续比单卡慢 13–25%（功耗墙，见 1440 那篇）。mlp bwd 的 1698 TF 偏低：NN / variable-K 比 NT 慢，值得单独看。

## 结果（M2，N = 1）

| 段 | 纯计算 | 纯通信 | overlap | 加速比（理想） | E | overlap 中计算 / 通信 |
|---|---|---|---|---|---|---|
| attn fwd | 0.635 | 0.202 | 0.705 | 1.19×（1.32×） | 0.65 | 0.696 / 0.519 |
| mlp bwd | 3.089 | 0.201 | 3.128 | 1.05×（1.07×） | 0.81 | 3.118 / 2.968 |
| mlp fwd | 1.308 | 0.202 | 1.387 | 1.09×（1.15×） | 0.61 | 1.376 / 1.380 |
| attn bwd | 1.550 | 0.201 | 1.633 | 1.07×（1.13×） | 0.59 | 1.623 / 0.619 |
| 整层 | 6.582 | 0.806 | 6.852 | 1.08×（1.12×） | 0.67 | |

## M1（单独 → 同时）

| 段 | 计算变慢 | alltoall 变慢 | busbw |
|---|---|---|---|
| attn fwd | 1.22× | 2.10× | 780 → 371 GB/s |
| mlp bwd | 1.12× | 1.53× | 782 → 510 |
| mlp fwd | 1.16× | 1.72× | 780 → 454 |
| attn bwd | 1.16× | 2.21× | 781 → 353 |

观察：

1. 计算远长于通信（6.6 vs 0.8 ms），通信理论上全能藏住；实际每段多出 0.04–0.12 ms，来自计算被拖慢 12–22%。
   上限 1.12×，做到 1.08×。
2. **alltoall 被拖慢、被推迟**：attention 段旁边 alltoall 慢 2.1–2.2×；mlp bwd 段里 alltoall 单独 0.2 ms，overlap 中到
   2.97 ms 才结束（计算 3.12 ms 结束），几乎被饿到最后。现在测例里没有后续依赖所以看不出代价；真实 1F1B 里 dispatch fwd
   结束晚了会推迟 mlp fwd 的开始（mlp fwd 要它的数据）。下一步可以把 "dispatch fwd → mlp fwd" 的依赖串起来测。
3. 不同段 alltoall 受影响程度不同：attention（aiter asm + hipBLASLt 投影）伤害最大，grouped GEMM（FlyDSL）最小，
   和 1600 那篇 "down GEMM 拖慢 alltoall 最少" 一致。

## DSv3 attention：Primus-Turbo FlyDSL gfx1250 MLA

- Turbo 0.6.1.dev2 的 FlyDSL attention：`flydsl/attention/flash_attn_{fwd,bwd}.py` 只支持 gfx950（硬 assert，用
  `ds_read_tr16_b64`）；gfx1250 只有 `flydsl/attention/gfx1250_mla/`（qk 192 / v 128，MHA，causal）。M3（GQA d128）用不了。
- 接进 `cco_bench.py`：`attn_fwd_flydsl` / `attn_bwd_flydsl`（`attention_flydsl_gfx1250_mla_impl`）。
  单卡：fwd 727 µs（aiter 728），bwd 3286 µs（1087 TF）；bwd 对 FP32 autograd dq / dk / dv 2.3e-3 – 2.5e-3。

| kernel | alltoall | 纯计算 | 纯通信 | overlap | 加速比 | E | alltoall 变慢（M1） |
|---|---|---|---|---|---|---|---|
| fwd aiter | 896 MiB | 0.930 | 0.942 | 1.386 | 1.35× | 0.52 | 1.33× |
| fwd FlyDSL | 896 MiB | 0.863 | 0.942 | 1.214 | 1.49× | 0.68 | 1.19× |
| fwd aiter | 256 MiB | 0.914 | 0.258 | 1.093 | 1.07× | 0.30 | 1.95× |
| fwd FlyDSL | 256 MiB | 0.866 | 0.250 | 0.934 | 1.19× | 0.73 | 1.35× |
| bwd FlyDSL | 896 MiB | 3.544 | 0.942 | 4.117 | 1.09× | 0.39 | 4.19×（178 GB/s） |
| bwd FlyDSL | 256 MiB | 3.571 | 0.259 | 3.623 | 1.06× | 0.80 | 1.28× |

FlyDSL fwd 对 alltoall 明显更友好。FlyDSL bwd 在 448 / 896 MiB 时把 alltoall 压到 178 GB/s（自己只慢 1.14×）。
猜测：bwd 占满 CU，大包 alltoall 时 RCCL 要更多 channel（CU）拿不到；小包要的少。未验证（rocprofv3 看 RCCL kernel 的 CU 占用）。

## 下一步

- 把 1F1B 的依赖串起来测（dispatch fwd → mlp fwd，dispatch bwd → attn bwd），看 alltoall 被推迟的真实代价。
- 补 permute / unpermute、SwiGLU、norm 这些访存型 kernel。
- FlyDSL NN / variable-K 在 M3 形状上的效率（mlp bwd 1698 TF）。
- rocprofv3：FlyDSL MLA bwd 和大包 alltoall。

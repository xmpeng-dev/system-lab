# [ROADMAP DELTA] Primus 2026-10-12 — 对 2026-09-03 roadmap 的执行复盘与修订

> 这是一份 **delta note**，不替代 [`2026-09-03_primus-roadmap-2026q4-2027h2.md`](./2026-09-03_primus-roadmap-2026q4-2027h2.md)。
> 原 roadmap 的分期（C0 可信度 → C1 插件化 → C2 存量性能 → C3 规模化可靠性 → C4 执行层收拢）与 13 条反证护栏仍然有效，本文只记录 7 周后的实际执行结果、被证伪的前提，以及据此修订的排序。
>
> 核查基线：Primus `6e2b114b`（2026-08-30，原 roadmap 基线）→ `ad3614d6`（2026-10-09，分支 `dev/xmpeng/qwen3-fp8-cheap-fence-inv`），区间 **106 commits**。
> Primus-Turbo 同期一并核查（无对应基线，引用其自身 PR 号）。
> 核查方式：三路只读 subagent 全仓审计，所有结论带 `path:line` 或 commit。
>
> ⚠ **业界侧的后续扫描见 [`2026-10-28_0546_industry-scan-update.md`](./2026-10-28_0546_industry-scan-update.md)**，其中更正了本文 **C2-8 的判定**（见 §1.4 表下方），并给 R1/R3/R4 补了新证据、新增 R6（相关性专家放置）与 R7（借用 TorchTitan 的可信度设施）。

---

## 0. 执行摘要

三句话版本：

1. **原 roadmap 的排序前提被证伪了。** 它假设"先修可信度、再修结构、再收性能红利"。实际 7 周：**C2（存量性能）跑完一大半，C0（可信度）几乎零位移，C1（插件化）七条全部未动，C4（执行层收拢）倒退**。106 commits / +79,556 −16,286 行里，真正的 kernel 性能工作只占约 20%。
2. **格式缝已经有人在隔壁施工。** `eb7c31da`（#1208）把 RMSNorm→MXFP4 量化融进 norm epilogue 并顺带 fuse QK RoPE —— 这正是 quant-GEMM seam 的同一模式，只是落在 attention QKV 而非 MoE colwise requant。模式已在树内证明可行，目标缝仍空着，窗口在收窄。
3. **C0-4 的性价比彻底变了。** #1229 已经把 convergence harness 建起来（7.4K 行），同时本地出现了一个活的同类 bug（MegaMoE mxfp8 静默错梯度）。C0-4 从"新建一套 CI"降级成"在已有 harness 上加两节点 + 一条臂"。

修订后的排序原则：**不再要求组织先做一个卫生阶段。每一条 C0 必须搭着一个有即时收益的交付一起走。**

---

## 1. 逐条状态 delta

### 1.1 总览

| 阶段 | 7 周后结果 |
|---|---|
| **C0 可信度** | C0-1 部分、C0-4 部分，**C0-2 / C0-3 / C0-5 / C0-6 / C0-7 零位移**；另有一处反向 |
| **C1 插件化** | **七条全部未动** |
| **C2 存量性能** | C2-1 / C2-4 / C2-5 / C2-8 **完成**；C2-2 / C2-3 / C2-6 / C2-7 部分 |
| **C3 规模化可靠性** | 未系统推进，但 C3-4（copy-engine 集合通信）被提前跳过式落地 |
| **C4 执行层收拢** | **C4-3 倒退**，C4-2 零 commit，C4-4 四棵 deprecated 树全在 |

### 1.2 C0 — 可信度

| 条目 | 状态 | 证据 |
|---|---|---|
| C0-1 度量可判定性 | 部分 | `tools/perf`（#1173, `a80ac298`, 4,628 行）拿到了环境指纹：镜像 digest / submodule pin / GPU 型号 / ROCm·Torch 版本 / world size，密钥脱敏（`tools/perf/README.md:380-410`）。但 `NUM_REPS` 默认 **1**（`:216`），`extract_results.py` 报的是单次 run 内 post-warmup step 的调和平均（`:416`）而非 run-to-run 分布；无 clock pin、无 `perf_level`、无 PMC |
| C0-2 patch 层 fail-loud | **零位移** | `primus/core/runtime/train_runtime.py:210-214` 仍把版本探测异常吞成 `None`；`primus/core/patches/patch.py:70-71` 把 `None` 转成 `False` → **静默跳过**；`patch_runner.py:137-140` 只记日志；`patch_registry.py:58-59` 重复 patch id 只 warning 后覆盖。核心 patch 文件区间内 **0 commit** |
| C0-3 patch 层机器可读 | **零位移** | `targets=` **0 处**；`backend_versions=` 从 1 → **2**（`te_spec_provider_patches.py:43`、`gptoss_router_bias_patches.py:51`）；同期 `@register_patch` **172 → 194**（tracked 非测试）。无 lock-bump 影响报告 CI |
| C0-4 多机数值 CI | 部分 | `tools/convergence_test`（#1229, `ba3530c2`, 7,412 行）落地：真语料训练、loss 漂移以 exit code 判定（0 pass / 3 漂移超 `--tolerance` 默认 0.05 / 4 迭代不足）、Megatron + MaxText 双后端、bf16 与 fp8 均有。但：**单节点 8 卡**（`README:453`）、**未接入任何 workflow**（`.github/workflows/` grep 不到）、**不跑 DeepEP 开关矩阵**（其自身建议是关掉 DeepEP 以求收敛稳定，`README:357,626`）。数值探针 `probe_rowmap.py` / `probe_graddump.py` / `cmp_moestage_ranks.py` 仍只在未 track 的 `bk/` 下 |
| C0-5 恢复被注释的测试 | **零位移** | `tests/trainer/test_megatron_trainer.py:499` 仍是 `# def test_zero_bubble_pipeline_parallelism`。ZBV 格式三例（`:824/:836/:853`）是另一套。**MXFP4 / MXFP8 无任何训练级 E2E**，只有 unit test |
| C0-6 统一 Turbo pin | **零位移** | `.github/workflows/ci.yaml:45` = `e2d9f1d7…`；`.github/workflows/benchmark.yaml:14` = `a04a233c…`。#1195 为 MegaMoE FlyDSL 只 bump 了 CI 一侧。`requirements*.txt` / `pyproject.toml` 无 Turbo pin；`zbpp_gemm.py:74` 与 `mega_moe.py:50` 只查 import，**无运行时最低版本断言** |
| C0-7 对外表述准确性 | **零位移** | `README.md:7` 仍称 "reinforcement learning workflows"、`:22` 仍称 "Built-in fault tolerance"。`primus/` 内无 RL 实现；容错只在一个 MLPerf bridge recipe 里经 `torchft` 出现，其余指向外部 Primus-SaFE（`README.md:242`）。`docs/backend-gap/` 不存在 |

**一处反向**：#1182（`66cd3131`）把 GPU 单测挂到 `ci:gpu` PR label（`ci.yaml:279,702,885`）。未打标签的 PR 直接跳过 GPU 测试 —— 默认覆盖率是降的。

**仍然没有吞吐回归门禁**：`tools/daily/daily_report.py:85,129,158` 解析并平均 `Tokens/s/GPU`，但无阈值、无基线比对、无 assert。唯一的 loss 门禁是手工跑的 `tools/convergence_test` exit code。

### 1.3 C1 — 插件化与上游对齐

七条全部未动。两条值得单独记：

**C1-1 `BackendSpecProvider` —— 名义存在，实质仍是 monkey-patch。**
`class PrimusTurboSpecProvider(BackendSpecProvider)` 在 `primus/backends/megatron/core/extensions/transformer_engine_spec_provider.py:120`，实现了完整接口面，Turbo-vs-fallthrough 按 `use_turbo_*` 逐方法门控：

| 方法 | Turbo 实现（flag） | 回退 |
|---|---|---|
| `linear` / `column_parallel_linear` / `row_parallel_linear`（`:130,:138,:146`） | `PrimusTurbo*Linear`（`use_turbo_gemm`） | `TE*Linear` |
| `column_parallel_layer_norm_linear`（`:195`） | `PrimusTurboLayerNormColumnParallelLinear`（`use_turbo_gemm`）/ turbo-norm 工厂（`use_turbo_norm_te_linear`） | `TELayerNormColumnParallelLinear` |
| `core_attention`（`:223`） | `PrimusTurboAttention`（`use_turbo_attention`） | `TEDotProductAttention` / `DotProductAttention` |
| `grouped_mlp_modules`（`:233`） | `PrimusGroupedMLP` + `PrimusTurbo*GroupedLinear`（`use_turbo_grouped_gemm`） | `TEGroupedMLP` / `SequentialMLP` |
| `layer_norm`（`:214`）、`activation_func`（`:287`） | 无，恒为 TE/Apex | — |
| `column_parallel_linear_with_gather_output`（`:153`）、`row_parallel_linear_with_scatter_input`（`:171`） | 无，恒为上游 | — |

但安装方式仍是 `register_patch(..., backend_versions=["<0.17"], condition=_needs_primus_spec_provider)` 覆写 4 个 Megatron 模块的 `TESpecProvider`（`:80`），**不是上游注入点**。子类 `DeepSeekV4SpecProvider` / `Glm5NextSpecProvider` / `KimiK3SpecProvider`（`:291,:445,:461`）都挂在这条 patch 上。上游 `LocalSpecProvider.grouped_mlp_modules` 仍然忽略入参（`third_party/Megatron-LM/megatron/core/models/backends.py:129-135`）。

**C1-4 后端 pin —— 冻住了，而且在变成定时炸弹。**

| submodule | pin | describe | HEAD 日期 |
|---|---|---|---|
| Megatron-LM | `d3528a21` | `core_v0.15.0rc7-1042-gd3528a213` | **2026-03-06** |
| torchtitan | `73a0e697` | tag `v0.2.2` | **2026-02-20** |
| Megatron-Bridge | `9577b128` | `v0.2.0rc6-497` | 2026-01-18 |

pin 未移向任何 `core_rX.Y.0` 发布分支。两个主后端分别冻了 7 个月和 8 个月，期间 patch 172 → 194，版本门 1 → 2。

其余：C1-2 `torchtitan/overrides/` 目录不存在、`@override` 0 处（Turbo 侧有 27 个 `custom_op` + 48 个 `register_fake`，但 **0 个 `register_autograd`**）；C1-3 `HybridModel` / `gpt_checkpoint_interop` 均 0 命中，`gpt_layer_specs` 仍被引用 23 次；C1-5 无上游 PR 痕迹；C1-6 `pyproject.toml` 无 `[tool.importlinter]`，4 个反向 `import megatron` 文件全在（`primus/backends/transformer_engine/` 下的 `transformer_engine_torch/gemm.py`、`transformer_engine_torch/comm_overlap.py`、`pytorch/cpp_extensions/gemm.py`、`pytorch/module/base.py`）；C1-7 能力契约仍是 `primus/` 内 **263 处** `gfx950`/`gfx942` 字符串判断。

### 1.4 C2 — 存量性能红利（本期唯一真正推进的一格）

| 条目 | 状态 | 证据 |
|---|---|---|
| C2-1 DataLoader | **完成** | `dataloader_mp_context` yaml 参数（`primus/configs/modules/megatron/primus_megatron_module.yaml:101`）+ `dataloader_patch.py:79`；`num_workers: 8`（`trainer_base.yaml:294`）；`fp8_param_gather` 进 28 个 recipe |
| C2-2 `PrimusFP8GroupedMLP` 开销 | 部分 | 融合路径 `_forward_with_turbo_fused_grouped_mlp`（`experts.py:263-292`，#1165 `4ab55b36`）落地；但 `tokens_per_expert.tolist()` 的 D2H 同步**仍在** `experts.py:303,343` |
| C2-3 SwiGLU 融进 GEMM | 部分（仅 MoE grouped） | Turbo `grouped_gemm/grouped_gemm_fp8_glu_kernel.py`；`activation="silu"` epilogue 在 `grouped_gemm_fp8_impl.py` / `grouped_gemm_fp4_impl.py`。dense 路径无 |
| C2-4 autotune 作用域 | **完成** | 已改为逐 op 字典 `backend.py:101-105` + 逐 op opt-in `set_*_backend(auto_tune=True)`（`:95-97`）+ 逐 entry `autotune=False`（`attention_impl.py:437`）。DeepEP dispatch/combine 不再被全局 autotune 拖累 |
| C2-5 permute/unpermute | **完成**（残留同步） | `moe_permute_impl.py` → `triton_permutation.{make_row_id_map,permute_with_mask_map,unpermute_with_mask_map}`；残留 `.item()` D2H 在 `moe_permute_impl.py:171,310` |
| C2-6 quant↔GEMM 格式缝 | 部分（仅 QKV） | 见 §2 |
| C2-7 attention 缺口 | 部分 | `resolve_flash_attn_backend`（`attention_impl.py:629`，默认 FLYDSL 否则 AITER）；dense backend 已有 5 个（FLYDSL / AITER / HIPKITTENS / GLUON / TRITON，`_DENSE_FWD_BACKENDS` `:433`）。head-dim：FlyDSL 仍只 64/128（`:109`），Gluon 可到 `head_dim<=256`（`>256` 在 `:403` 拒绝）。两个 "unified attention kernel dispatcher" TODO **仍在** `attention_aiter_impl.py:373,478` |
| C2-8 ROCm FP8 grouped-GEMM API 缺口 | ~~完成~~ → **判错，实为部分** | 见本表下方更正 |

> **⚠ 2026-10-28 更正 C2-8。** 上面写的「完成」在实质上是错的。`HipblasltGroupedGemm::run()`（`third_party/Turbo/csrc/kernels/grouped_gemm/hipblaslt_grouped_gemm.cu`）是把 `num_gemms` 个 group **逐个**调普通 `hipblasLtMatmul`、扇到 `kMaxNumStreams`（1 条调用方 stream + 3 条内部）与 8 个预分配 handle 上，再用 event 汇合——**没有调用任何 grouped-GEMM 入口**，注释里还记着"句柄预分配不足时 BF16 多流 grouped GEMM 在 MI355X 上会间歇性停顿"。所以 roadmap 承诺的收益（「一旦可用，单次 API 调用消掉全部 dispatch 开销」）**并没有拿到**，拿到的是多流批处理。另外 9-11 那条「ROCm 7.1 穷举后只有 BF16 grouped GEMM」作为笼统结论也站不住——[`hipBLASLt#1316`](https://github.com/ROCm/hipBLASLt/pull/1316)「FP8 TN Compute for Grouped GEMMs」2024-11-07 就合了；gfx950 上具体支持哪些 dtype/layout 组合仍未知。详见 [`2026-10-28_0546_industry-scan-update.md`](./2026-10-28_0546_industry-scan-update.md) §1.2 与 §7.1。

### 1.5 C4 — 执行层收拢（倒退）

| 条目 | 状态 | 证据 |
|---|---|---|
| C4-2 三套流水线调度栈 → 一套 | **零 commit** | `primus/core/pipeline_parallel/` 2,280 行；`primus/backends/megatron/core/pipeline_parallel/zerobubble/` 8,217 行；torchtitan `pipelining_schedule_patches.py` 79 行。双 IR 完整保留：`scheduler/scheduler_node.py:12-23` 的 `FuncType.R = "RELOAD"`（含 `O = "OFFLOAD"`）vs `zerobubble/offload.py` 自带的 recompute IR |
| C4-3 SDMA 重复实现 | **倒退** | #1128（`a525a906`）**新增**了 distributed-optimizer 这一族，而非合并。现为 **7 个文件 / 1,655 行**，原先那两个 `sdma_symm_mem_collectives*` 仍并存 |
| C4-4 / X-2 废弃代码 | 零位移 | 四棵树全在，共 ~14.9K 行：`moe/deprecated_20251209/` 2,660；`moe/deprecated_2caa681a1/` 485；`v4_attention_kernels/_flydsl_v0_deprecated/` **10,620**；`_triton_v0_deprecated/` 1,156。`NotImplementedError` 147 处 |

SDMA 现状明细：

| 文件 | 行数 |
|---|---|
| `backends/megatron/core/distributed/sdma_param_gather.py` | 243 |
| `backends/megatron/core/distributed/rccl_sdma_param_gather.py` | 204 |
| `backends/megatron/patches/parallelism/sdma_param_all_gather_patches.py` | 220 |
| `backends/megatron/patches/parallelism/rccl_sdma_grad_reduce_scatter_patches.py` | 332 |
| `backends/megatron/patches/parallelism/rccl_sdma_param_all_gather_patches.py` | 351 |
| `backends/megatron/patches/sdma_symm_mem_collectives_patches.py` | 142 |
| `backends/torchtitan/patches/sdma_symm_mem_collectives.py` | 163 |

**许可证与版本不一致已修复**：`LICENSE` 与 `pyproject.toml:24` 均为 MIT；`primus/__init__.py` `__version__ = "26.7"` 被 `[tool.hatch.version]`（`pyproject.toml:97-99`）静态读取，无冲突。

---

## 2. 格式缝：先例已出现，但不是目标那条缝

这是本次核查对 [`career-strategy/2026-08-12_1505_the-quant-gemm-seam.md`](../career-strategy/2026-08-12_1505_the-quant-gemm-seam.md) 影响最大的一条。

- `eb7c31da`（#1208，+585 −7）在 **attention QKV 路径**把 RMSNorm→MXFP4 量化做成 norm 的 epilogue，由融合 norm 直接发射量化后的 GEMM 输入，并顺带 fuse 了 QK RoPE。涉及 `extensions/fused_residual_rmsnorm.py`、`primus_turbo.py`、新建 `patches/turbo/qk_rmsnorm_rope_patches.py`（283 行）。**这就是格式缝塌缩，模式已在树内成立。**
- `ea35b348`（#1174，+215 −6）是 **fused MXFP4 weight-gradient accumulation**，紧贴 dW GEMM，但属于 wgrad 累加融合，**不是** quant↔GEMM 契约塌缩 —— 是另一类融合。
- `db434340`（#1156）在 Llama-3.1-8B 上做了 FP4 dense MLP + residual RMSNorm + skip-y（新建 `patches/turbo/dense_mlp_fp4_patches.py` 241 行）。
- 除 MoE grouped-GLU kernel 外，**dense 路径仍无"GEMM prologue 吃未量化输入 / epilogue 发射下游 scale 格式"**。

结论：目标缝（MoE / FP8 wgrad 路径上的 colwise requant）仍然空着，但论证它可行的先例已经由别人写进树里了。这同时降低了立项阻力和独占性。

---

## 3. 一个活的 C0-4 证据

本地 `Primus/HANDOFF_mxfp8_topk_grad.md`（未 track）记录的 MegaMoE mxfp8 bug，正是 C0-4 当初引用的那一类静默错梯度：

- DeepSeek-V2/V3 4 层 proxy，EP8 / TP1 / PP1，mxfp8 专家臂
- 触发条件两个且必须同时成立：`moe_router_topk` 非 2 的幂（5/6/7 爆，4/8 正常）、每步 >1 micro-batch
- it1 grad norm 从 ~1.3 跳到 ~1300–1550，被 `clip_grad=1.0` 压成约 1/1300，loss 几乎不降，**全程无 NaN**
- 前向正确（it1 loss 与 bf16 臂一致）、dx 正确（‖dx‖/‖dy‖ 与 bf16 差 1–2%），坏的是 dW
- 当前最可信假设：多层 MoE 共享同一块 fp8 symmetric memory（`mega/fp8/symm_buffer.py`）跨 µbatch / 跨层复用时污染 dW 路径

这比抽象论证更有说服力：单节点、单算子、bf16 臂对照正常，现有单测全过（`test_fused_mega_moe_fp8.py` topk=6 单次 fwd+bwd SNR≈22 dB 通过），**只有跑到多 µbatch 的训练级才暴露**。这正是 C0-4 要覆盖的那层。

---

## 4. 原 roadmap 未涵盖的新战线

| 战线 | 规模 | 判断 |
|---|---|---|
| **SpecForge** 投机解码 draft 模型训练（#1139 `372b0fa1` / #1191 `43e1c9be` / #1206 `12dac83f`） | `primus/backends/specforge/` 2,147 行 + ~1,200 行测试 | **不在任何生产 recipe**（megatron / mlperf / customer_package 下 grep 为空），只有一个模型 `qwen3.5-4b-dflash`，单测因 overlay 镜像未进 CI 被 skip（#1163）。两个旋钮：`specforge_mode`（`train`/`capture`）、`specforge_train_mode`（`online`/`offline`）。⚠ **它不是 RL**，不能用来兑现 `README.md:7` |
| **Projection 多后端** TorchTitan（#1202 `1c99481b`）、MaxText/JAX（#1232 `81886337`，#1203 曾被 #1220 revert 后重落）、DLRM/HSTU 显式反向 GEMM（#1224 `0f996490`） | ~5.7K 行净增 | 真资产，上游对标物为空（TorchTitan 已删自己的 estimator）。但精度仍无测试 |
| **RCCL-SDMA copy-engine 集合通信已上生产** #1128 / #1168 `857c4e1c` / #1186 `34069293` / #1207 `8476bbc4` | 1,655 行 | 原 roadmap 把它放在 2027 的 C3-4。实际已 default-on，但**只对 gpt-oss 一个模型**：`MEGATRON_PARAM_GATHER_BACKEND=rccl_sdma` 仅在 `examples/mlperf/gpt_oss_20b/config_MI355X_1x8x1.sh:75` 默认开启（配 `MEGATRON_RCCL_SDMA_EAGER_PARAM_BYTES:-40978350080`）。CE-grad 那条是独立 flag `MEGATRON_GRAD_REDUCE_BACKEND=rccl_sdma`，仍 opt-in（`rccl_sdma_grad_reduce_scatter_patches.py:270`） |
| **optimizer 流量融合** fused Adam+clip 手写 HIP kernel（#1219 `c60f3a6b`，`te_patches/fused_adam_clip_kernel.cu` 274 行，AMD 2026 © ，bf16x8 ILP）、fused / grid-matched MXFP4 de-osc（#1184 `a48595c8`、#1222 `78287080`） | ~2.1K 行 | AMD 独有。原 C2-3 只提 SwiGLU epilogue，未覆盖 optimizer 流量这一类 |
| **qk_clip（MuonClip）** 支持 hybrid 模型 + 分布式优化器（#1164 `248fc7c6`）、FLA 后端暴露 per-head max logit（#1162 `e10f8469`） | ~345 行 | 原 roadmap §1.2 把 Muon 类判为"停手（上游在吸收）"；hybrid / dist-opt 这块是本地净增工作。`qk_clip_hybrid_patches.py:35-36` 自门控，无新 flag |
| **GLM-5.3-Flash**（#1230 `325643c4`，+2,773）、**MiniMax-M3**（#1197 `bc8fd627`，+7,886） | ~10.6K 行 | 纯模型使能。原 X-4 列的是 DSv4 / Kimi-K3 / Gemma4，未含这两个 |

**mHC 现状**（上一轮审计的遗留问题）：DSv4 那条 `hyper_connection.py:359` 的 `post = 2.0 * torch.sigmoid(post_logit)` 与 Triton 镜像 `_triton_common/hc_glue.py:21` **区间内零 commit，未变，无退火**。GLM-5.3-Flash 没有复用它，而是从 SGLang 转写了一套独立 mHC（`glm5_next/mhc.py:67`），把倍数提成配置项 `hc_post_mult`（`:55`，接线于 `glm5_next_block.py:179`），默认仍为 2.0（`glm5_next_transformer_config.py:85`）—— 可配但仍是常数，仍无退火。

**DeepEP cheap fence 已闭环**：上游修复（Turbo #539）+ 本地消费完毕。`b003dff5` 把 `PRIMUS_TURBO_COMMIT` 推到 `e2d9f1d7…`（`ci.yaml:42-44` 注明 intranode cheap fence 已 default-on），`ad3614d6` 把 #1218 塞进 13 个 MoE recipe 的 `PRIMUS_TURBO_DEEPEP_DISABLE_CHEAP_FENCE: "0"` 强制块全部删除（MI325X/MI355X 的 `deepseek_v2_lite` / `mixtral_8x7B` / `mixtral_8x22B` / `qwen3_30B_A3B` BF16+FP8）。全仓仅剩 `ci.yaml:43` 一句注释。

**其他值得记的修复**：#962（`dc6182e9`，+1,201）Triton buffer-store 误编译 + ROCm compiled-training NaN，新建 `core/patches/triton_bufops_war_patches.py`（216 行）；#1158（`92355a12`）停止把 V4 TFLOPs 乘以 `hc_mult` —— 之前报出去的 V4 算力数字偏高。

**#999（`1b083f87`）retire examples launchers → primus-cli**：删掉 7 个 `examples/run_*pretrain*.sh` 与 `examples/scripts/`、`examples/*/prepare.py`，新增 `runner/helpers/launch/slurm_pretrain.sh`。**recipe YAML 面未变**，只是启动管线搬家；现在规范入口是根目录 `./primus-cli`（→ `runner/primus-cli`）。唯一 flag 变化是数据集 split 种子改名 `--split_seed`，让 Megatron 的 `--seed` 专职训练 RNG。

---

## 5. 7 周实际投入分布

按实质新增代码（扣除 release-docs 的生成式 JSON 快照）粗估：

| 方向 | 占比 |
|---|---|
| 模型使能 | ~30% |
| 工具 / 基础设施 | ~25% |
| 性能 / kernel | ~20% |
| Projection | ~12% |
| SpecForge | ~6% |
| 通信 | ~5% |
| optimizer / bugfix | ~2% |

变更集中区（`git diff --stat` 首层目录）：`examples/*/configs` 77 个文件、`backends/megatron` 39、`core/*` 60+、`tools/release_docs` 22。

一句话概括：主投入是 **GPT-OSS / MXFP4 MLPerf 路径上的 Turbo 融合**，配一条 **copy-engine 集合通信线**；同时两条大体量加性工作（新模型族、工具链）抬高了行数；另有两条原 roadmap 未立项的新能力线（SpecForge、多后端 Projection）。

---

## 6. 修订后的排序

原 roadmap 的 C0→C4 线性分期在组织层面没有被执行。不要再发一版同构的。改成**让每条 C0 搭着一个有即时收益的交付走**：

**R1. 把 MegaMoE mxfp8 bug 变成 C0-4 的入场券。**
先定位修复（`HANDOFF_mxfp8_topk_grad.md` 已钉死触发条件，下一步是逐层 A/B 确认 symm buffer 跨层/跨 µbatch 复用）。修好后把"必须能复现 BLOWN"的回归测试直接加进 `tools/convergence_test` 的 MegaMoE mxfp8 臂，顺势把 harness 扩到 2 节点 + DeepEP on/off + 挂 nightly workflow。用一个真实的、已经消耗了工程时间的静默错梯度去论证，远强于抽象论证。成本量级已被 #1229 拉低。

**R2. 统一 Turbo pin + 运行时最低版本断言（C0-6）。**
改动几行，但它决定报出去的性能数字和测过的正确性是不是同一套二进制。当前 `ci.yaml` 与 `benchmark.yaml` 跑的是不同 Turbo commit，且无任何版本断言兜底。

**R3. `targets=` + fail-loud（C0-2 / C0-3），对外叙事改成"解锁 Megatron 版本推进"。**
三个数字放一起即是论证：194 个 patch / 2 个版本门 / 主后端 pin 冻结 7 个月。C1-4 事实上被 C0-2 与 C0-3 锁死 —— 没有 `targets=` 就不知道一次 bump 会撞掉什么，没有 fail-loud 就连撞掉了都没有信号。这把锁每过一周更难开。

**R4. 格式缝，趁 #1208 的先例还新。**
QKV 路径已经证明 norm epilogue 发射量化输入可行。剩余目标：dense MLP 路径、MoE colwise requant（wgrad 侧）、以及 `experts.py:303,343` 的 `tolist()` D2H 同步。这是 career note 里的 P0，现在它同时被验证和被逼近。

**R5. SDMA 重新包装成 feature，而非删重复代码。**
现状 7 条路径 / 1,655 行 / 只给一个模型 default-on。把 C4-3（去重）+ C1-7（能力查询契约，取代 263 处散落的 `gfx950`/`gfx942` 字符串判断）+ C3-4（copy-engine 集合通信对外化）合成一件事：收敛成一个带能力查询的 copy-engine 集合通信后端，之后才谈得上全模型 default-on。这样它能作为 feature 排进计划，而不是以 hygiene 身份被无限后置。

---

## 7. 需要继续盯的风险

1. **patch 债的增速超过版本门的增速**（+22 patch / +1 版本门）。增长主要来自模型使能（7 个新 patch 文件）与 perf/Turbo（4 个），正是原 roadmap C0-2/C0-3 预警的方向。
2. **GPU 测试默认关闭**（`ci:gpu` label）叠加**无吞吐门禁**叠加**Turbo pin 分叉**，意味着当前既测不住正确性回归，也测不住性能回归，而报出的数字来自第三套二进制。
3. **`_flydsl_v0_deprecated/` 单棵 10,620 行**仍在树内，与 5 个活跃 dense attention backend 并存；`NotImplementedError` 147 处。X-2 的清理压力只会随 backend 数量继续上升。
4. **SpecForge 与多后端 Projection 共约 7.8K 行新代码没有生产路径或精度测试**。需要尽早给出"收编 / 维持实验 / 停"的判断，否则会在下一个周期变成第二批 deprecated 树。

---

## 附：核查溯源

三路只读 subagent 全仓审计，2026-10-12：

- C0 可信度逐条核查 —— [`3d667fa8-a0ea-460d-a963-2dbe05c78b2e`](3d667fa8-a0ea-460d-a963-2dbe05c78b2e)
- 106 commit 内容盘点与分桶 —— [`13ab5191-ac6e-4356-9d01-8f50487639b7`](13ab5191-ac6e-4356-9d01-8f50487639b7)
- C1 / C2 / C4 逐条核查（含 Primus-Turbo）—— [`0ed6182a-e0bc-495d-a104-5384f00754fb`](0ed6182a-e0bc-495d-a104-5384f00754fb)

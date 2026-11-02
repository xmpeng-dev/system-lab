# Primus-Turbo Roadmap 2026Q4 – 2027H1(接任 owner 版)

> **When**: 2026-10-28 16:30 UTC+8
> **Where**: 桌面只读代码审计,无上机实验。审计对象 `/perf_apps/xiaoming/Primus-Turbo` @ `3ba4cbe8`(分支 `feat/mega_moe/gfx1250-bf16`,领先 `main` 59 个 commit、落后 0 个;`main` 停在 `20606a32` 2026-10-11)
> **Context**: 我接任 Primus-Turbo owner。五路只读 subagent 并行深读(算子面 / 六栈碎片化 / MegaMoE+DeepEP / CI 与基建 / 量化栈),关键结论由我手工复核。业界侧基线取 [`../primus-moe/2026-10-28_0546_industry-scan-update.md`](../primus-moe/2026-10-28_0546_industry-scan-update.md)

## TL;DR

1. **最严重的问题不在 kernel,在契约。** Turbo 的唯一消费者 Primus 有 66 行顶层 `from primus_turbo` import、只有 24 行带守卫,其中**至少 3 个 import 指向在 Turbo 任何分支上都不存在的模块**(我遍历了全部远程分支)。`primus/core/odc/fsdp/fsdp2.py:20-25` 是裸 import,直接 ImportError。Turbo 没有被任何地方做版本约束。
2. **CI 不测团队正在做的硬件。** CI 只构建 `gfx942;gfx950`(`ci.yaml:181`),而主力分支做的是 **gfx1250**,后者在 `tests/conftest.py:94-118` 里是**默认全跳**、只有 12 条 allow-list 放行。且所有 GPU job 在 PR 上要靠 `ci:gpu` label 才跑(`ci.yaml:52-55`)。
3. **Turbo 对"静默数值回归"的现行答案是 vendoring,不是测试。** `flydsl/mega/bf16/__init__.py:14-24` 白纸黑字:`ed8d7af4`(#486)重建共享 FlyDSL BF16 grouped GEMM,**静默打爆 MegaMoE BF16(it1 grad norm 从 ~1.45 跳到 ~1.88e6)**,处置方式是把 tile 钉在 `f6d5ab68` 快照里私有化。fp8 包同样"vendor 以免共享改动波及"。这是会复利的债 —— 现在有三套 mega 栈、三套 symm_buffer。
4. **真正的资产有两个,且都被低估:** MegaMoE 的量化融合深度(fp8 pool 直接 in-register colwise requant,不走 bf16 往返)已经领先业界公开实现;MXFP4 是 Turbo 对 dist_moe 的**唯一明确领先项**(对方 README 明写 MXFP4 不支持)。但格式缝的成果**困在 MoE 路径里** —— 通用 `gemm_fp8`/`gemm_fp4` 的量化全是独立 launch。
5. **没有任何性能门。** nightly benchmark 是 `continue-on-error`,只出 CSV 给人看;根目录那两个 `*_MI355X.csv` 没有任何代码读。

---

## 0. 现状盘点

### 0.1 规模

| 区域 | 文件 | 行数 | 最后改动 |
|---|---|---|---|
| `primus_turbo/`(Python,含 DSL) | 286 | 141,144 | 2026-10-27 |
| `csrc/`(HIP C++ 手写) | 198 | 55,074 | 2026-10-12 |
| `tests/` | 88 | 23,618 | — |
| `benchmark/` | 44 | 15,793 | — |
| `agent/`(git 跟踪部分) | 22 | 3,519 | — |

公开算子面约 45 个 op + 5 个 module + 5 个 core 导出。近 90 天 20 位贡献者、118 个 commit,`xiaoming-amd`/`Xiaoming-AMD` 合计 63 个居首。

### 0.2 六套 kernel 编写栈

| 栈 | 文件/行数 | 实现什么 | 依赖 | 最后改动 | 活性 |
|---|---|---|---|---|---|
| `flydsl/` | 94 / 84,965 | attention、GEMM、grouped GEMM、量化、RoPE、**MegaMoE 全部** | 外部 pip `flydsl==0.3.4.1`(`setup.py:536`) | 10-27 | **主力** |
| `csrc/` | 198 / 55,074 | CK/hipBLASLt/Turbo-MFMA GEMM、HK attention、量化、DeepEP、ODC | CK 子模块 `78ae3835`、hipkittens 子模块 `a288366e` | 10-12 | 活 |
| `triton/` | 43 / 15,039 | attention、GEMM、grouped GEMM、activation、RMSNorm、MoE permute | `triton>=3.7.0`(`setup.py:540`,**gfx1250 上不装**) | 10-27 | **主力** |
| `gluon/` | 4 / 4,457 | **只有** gfx950 f16 FA forward | `triton.experimental.gluon` | 08-27 | 休眠 |
| `jax/` | 27 / 3,900 | 不写新 kernel,只是 csrc 的 FFI 前端 | jax/jaxlib;**gfx1250 上是硬错误**(`setup.py:226-228`) | 08-20 | 休眠 |
| `hipkittens/` | 3 / 360 | 不写新 kernel,只是 csrc HK kernel 的 Python 壳 | hipkittens 子模块 | 08-19 | 休眠 |

**重复实现**:attention **5 份**(flydsl / triton / gluon / hipkittens-C++ / aiter);FP8 GEMM **5 个后端**;grouped GEMM 4 份;量化在 csrc / triton / flydsl 三处;SwiGLU 在 triton 与 flydsl 四处。

### 0.3 两个真实资产

**资产一:MegaMoE 的融合深度。** 对照 [业界扫描 §5.3](../primus-moe/2026-10-28_0546_industry-scan-update.md),"把量化融进前一个 kernel 的 epilogue"这件事 Meta LP-FA4 / TE 2.14 / Olmo-core 3 都在做,但 Turbo 的 MoE 路径有一条**公开实现里没见过**的:

```
colwise_requant_mxfp8_grouped_fp8in_flydsl  (flydsl/mega/fp8/quant.py:608)
  fp8 rowwise pool → in-register dequant → colwise-transpose requant
  不经 bf16 HBM 往返
```

加上 F6–F11 这一串(SwiGLU+mxfp8 量化同 kernel、grouped fc2 GEMM + mxfp8 epilogue + fp8 combine PUSH + bf16 reduce 同 kernel、SwiGLUᵀ 一读两向出 row/col 双量化使 `grad_l1` 根本不落 HBM)。**这是 Turbo 最值钱的东西,而且没有对外说清楚过。**

**资产二:MXFP4。** dist_moe 的 "Not supported in this release" 明写 MXFP4 不支持,Turbo 这边有完整的 fwd+bwd(gfx950):`gemm_fp4`/`mlp_fp4`/`grouped_mlp_fp4`/`grouped_gemm_fp4`、**融合的 mxfp4 wgrad beta=1 累加**(`gemm_fp4.py:71`)、融合 RMSNorm→双向 mxfp4(`rmsnorm_mxfp4_fusion.py:382`)、随机舍入 + 16 点 RHT、以及 `deoscillation` 这套 BF16 master-weight 振荡检测/吸附(gfx950 原生 `v_cvt_scalef32_pk_fp4_f32`,全程不落 packed fp4/scale/转置/dequant 临时量)。配上 AMD 自己的 [ALTO](https://arxiv.org/abs/2610.32229) 配方,这是最可防守的差异化。

---

## 1. 五个结构性问题(按风险排序)

### 1.1 【P0】与唯一消费者的 API 契约已经断了

Primus 对 Turbo 的依赖面不是"公开算子",而是**深入实现层**:

| Primus 引用的 Turbo 路径 | 层级 |
|---|---|
| `turbo.pytorch.ops.*` | 公开算子面(正常) |
| `turbo.pytorch.kernels.gemm.gemm_fp8_impl`、`...grouped_gemm_impl`、`...quantization_impl` | **impl 层** |
| `turbo.pytorch.core.backend` | **调度器内部** |
| `turbo.pytorch._C`、`turbo.pytorch._C.odc_rocshmem_host` | **裸 C 扩展** |

更要紧的是**指向不存在模块的 import**。我遍历了 Turbo 全部远程分支,下面四个路径在**任何分支上都不存在**:

| Primus 侧 | 形态 | Turbo 侧 |
|---|---|---|
| `primus/core/odc/fsdp/fsdp2.py:20-25` → `primus_turbo.pytorch.dist.odc.primitives.{gather,scatter_accumulate,utils}` | **裸顶层 import** | 零分支存在。Turbo 只有 `primus_turbo/pytorch/dist/{fp8_alltoall.py,__init__.py}` |
| `.../kimi_k3/moe/k3_quantile_balancing.py:112` → `primus_turbo.pytorch.ops.moe.quantile_balancing` | **裸顶层 import** | 零分支存在 |
| `.../minimax_m3/attention.py:138`、`indexer.py:123` → `primus_turbo.flydsl.attention.minimax_m3.*` | 函数内惰性 import,且 `configs/models/megatron/minimax_m3.yaml:84` 指向它 | 零分支存在 |
| `primus/backends/.../quantization/float8.py:12` → `primus_turbo.pytorch.core.float8` | **try/except 兜底到 `low_precision`** | 已改名,**这是唯一做对的那个** |

ODC 这一条尤其说明问题:**ODC 被劈成三半** —— C++ 在 Turbo(`csrc/kernels/odc_rocshmem/`,且只被一个"确认 pybind 符号存在"的 smoke test 碰过,没有任何 op 消费)、FSDP 集成在 Primus(`primus/core/odc/`)、而桥接两者的 Python primitives 层**哪边都没有**。`primus/core/odc/README.md:30` 还在声称"The primitives live in Primus-Turbo as `primus_turbo.pytorch.dist.odc.primitives`"。

**为什么这是 P0:** Turbo 的价值完全经由 Primus 兑现。契约无版本、无守卫、已断三处,意味着任何一次 Turbo 重构都可能在 Primus 侧变成运行时 ImportError,而且没有任何 CI 会发现 —— 两个仓库的 CI 互不交叉。

### 1.2 【P0】CI 不测团队正在做的硬件

| 事实 | 证据 |
|---|---|
| CI 只构建 gfx942+gfx950 | `ci.yaml:181`、`:287`、`release.yml:48,86` |
| nightly benchmark 只构建 gfx942 | `benchmark.yaml:50` |
| 主力分支做 gfx1250 | HEAD 59 个 commit 全是 `mega gfx1250:` |
| gfx1250 上默认全跳,12 条 allow-list 放行 | `tests/conftest.py:34-62` + `:94-118`,注释自陈 "The default on gfx1250 is still skip" |
| PR 上 GPU job 要 label | `ci.yaml:52-55` `contains(...labels.*.name, 'ci:gpu')` |
| gfx1250 与其他 arch 混编会**整体关掉** turbo+CK | `setup.py:230-238`(仅 `warnings.warn`) |
| 上次 release validate 是红的 | `3_validate.txt` 末行 `29 failed, 82435 passed, 143348 skipped`,29 个全是 `test_flash_attn_varlen` 的 SNR too low |

也就是说:**正在投入最多人力的那条线(gfx1250 Mega),在 CI 里既不编译也不测试**;而 CI 测的那两个 arch,发版门当前是红的。

### 1.3 【P0】对静默数值回归的现行答案是 vendoring

`primus_turbo/flydsl/mega/bf16/__init__.py:14-24` 原文:

> ``ed8d7af4`` (#486) rebuilt the *shared* FlyDSL BF16 grouped GEMM around ``Mfma16x16x32`` and new S2R loaders. That change is fine for the host-side grouped-GEMM path, but it **silently broke MegaMoE BF16 (it1 grad norm jumped from ~1.45 to ~1.88e6)**. The tiles are therefore vendored here

fp8 包同理(`flydsl/mega/fp8/__init__.py:16-25`:vendor 自己的 symmetric-memory 底座,"so shared `flydsl/utils` edits cannot reach this path")。

**这正是业界扫描里那个"静默错梯度"家族的第四例**,前三例是 [Broken Symmetry](https://arxiv.org/abs/2609.34272)(梯度范数涨 1000× 零 NaN)、`Primus/HANDOFF_mxfp8_topk_grad.md`(topk 非 2 幂 + ≥2 µbatch ⇒ dW 爆,无 NaN)、dist_moe 把 in-place WGRAD 累加定义成"需要显式契约的危险操作"。四例的共同签名完全一致:**前向对、无 NaN、dW 错、只靠 grad norm 才看得出来**。

Turbo 的处置是 fork 一份快照。代价已经可见:

| 重复 | 份数 |
|---|---|
| mega 栈(bf16 / fp8 / gfx1250) | 3 |
| `symm_buffer.py` 实现 | 3 |
| mega 侧 `barrier.py` | 3 |

而且 fp8 那份**没有 gfx1250 门**:我核过 `is_gfx1250` 在 `fused_mega_moe_{forward,backward,stage1,stage2}_impl.py` 里出现 11 次,在 `*_fp8_impl.py` 与 `ops/moe/fused_mega_moe_fp8.py` 里出现 **0 次**。gfx1250 上没有 fp8 mega kernel(没有 `gfx1250/fp8/`),所以在 MI455X 上调 fp8 mega 会**直接去跑 gfx950 kernel**,而不是报错。

附带一个死守卫:`SymmBuffer.assert_capacity()`(`flydsl/mega/fp8/symm_buffer.py:427`)写了最可操作的溢出报错("raise pool_mult"),**全树零调用**。真正生效的是 `dispatch_grouped_gemm_mxfp8_kernel.py:858` 那个 readback —— 那个写得很好(所有 rank 读同一张表因而一起 raise 而不是部分挂死),但好东西和死代码并存本身就是信号。

### 1.4 【P1】六套栈无治理政策

两种调度制度并存:

- **运行时注册表**(`core/backend.py:622` `dispatch` / `:681` `resolve`,优先级 user > autotune > default > fallback):attention、GEMM、grouped GEMM、sparse MLA、moe permute、EP 后端
- **import 期硬接线**:RMSNorm→Triton、RoPE→FlyDSL、量化→FlyDSL+Triton 混、**MegaMoE→FlyDSL**、`mlp_fp4`、`cross_entropy`、`deoscillation`

外加 8 种 ad-hoc 机制并行:逐 Function 硬编码 `default_backend`、`is_gfx942/950/1250` 分支、裸 `gcnArchName` 字符串解析(`attention_impl.py:194`、`deoscillation.cpp:73`)、`check_*_support` 能力断言、仅 env 的旋钮、try/except import 探测、impl 内逐 arch kernel 组合分支。

后果是**同一个"选后端"问题有九种答案**,新人改任何一条都不知道会不会漏掉另外八条。另外三处具体债:`primus_turbo/flydsl/deep_ep/` 目录里**只有 `__pycache__`、没有任何 `.py` 源文件**;`tokens_per_expert_to_mask` 调的是 `.forward(...)` 而不是 `.apply(...)`(`ops/moe/tokens_per_expert_to_mask.py:28`),**因此根本不是 autograd 节点**;`Format.HYBRID` 与 `ScalingStrategy.DELAYED_SCALING` 声明了但无实现(后者在 `low_precision.py:150` 被注释掉)。

### 1.5 【P1】没有任何性能门

| 事实 | 证据 |
|---|---|
| nightly benchmark 不能让流程失败 | `benchmark.yaml:61-64` `continue-on-error: true` |
| `run_suite.py` 只看进程返回码,不读基线 | `run_suite.py:430` |
| 汇总脚本做的是 **AMD vs NVIDIA H200 横向对比**,不是 vs 上一次 | `summarize_results.py:14-18` |
| 逐算子 bench 只有 SNR 正确性断言,无 TFLOPS/延迟断言 | `bench_attention_fa.py:77-78`、`bench_grouped_gemm_turbo.py:69` |
| 根目录两个 `*_MI355X.csv` 无任何代码读取 | 全仓 grep 文件名零命中 |

对一个 **kernel 加速库**来说,没有性能回归门等于没有产品门。

### 1.6 工作树卫生(顺手但别拖)

`gpucore.5652.gpu` **42 GB** 核心转储、`3_validate.txt` 151 MB、`logs/` 162 MB、`build/` 290 MB、`libprimus_turbo_kernels.so` 53 MB。都是 untracked/gitignored,不污染历史,但占着 ~700 GB 量级之外的实打实 0.7 TB 不到的盘,且 42 GB 那个会让任何 `rsync`/容器 build context 变成灾难。

---

## 2. 业界趋势 → Turbo 的四个映射

依据 [2026-10-28 业界扫描](../primus-moe/2026-10-28_0546_industry-scan-update.md)。

| 趋势 | 对 Turbo 的含义 | 动作落点 |
|---|---|---|
| **TorchTitan 为 dist_moe 定义了厂商可插拔的 routed-expert 契约**(`DistMoeTransform` 替换 config 但保留参数布局与 FSDP/checkpoint 所有权;可选 runtime 字段;守卫 import;"base requirements intentionally do not install this Blackwell-only backend") | 这是 Turbo↔Primus **应该长成的样子**,且证明"外挂 kernel 包"这条路上游认可。我们的差距不在能力,在契约 | **T0** |
| **格式缝已被生产关闭**(LP-FA4 融进前一 kernel epilogue;TE 2.14 fused GEMM+SwiGLU grouped MLP;Olmo-core 3 融进 NVSHMEM PUT;dist_moe 融进 **FSDP post-all-gather hook**) | Turbo 在 MoE 路径上**已经做到甚至更深**(fp8-in colwise requant 不走 bf16),但通用 `gemm_fp8`/`gemm_fp4` 全是 U1–U12 的独立 launch。**资产存在但没外溢** | **T3-1** |
| **MXFP4 是空位**:dist_moe 明写不支持;ALTO 是 AMD 自己的配方且替我们排除了三个死胡同;DeepEP V2.5 门槛抬到 CUDA 13.1/Hopper+ | Turbo 唯一能说"我有你没有"的格子 | **T3-2** |
| **通信层基底变了**:RCCL 2.30.7 带 GIN device API(ROCm 10.0.0),卡在 `nccl_device.h` 编译卫生;mori 把 `ccoSdma` put/get 暴露给 **FlyDSL 和 Triton**(#792, 10-20)、EP v2 AsyncLL 跑在 cco 上并支持跨节点(#768, 10-27) | Turbo 的 DeepEP 是 gfx942 时代的 ROCm 内部 fork(README 原文 "only Supported MI300 (gfx942)"),`csrc/kernels/deep_ep/` 最后改动 10-12。**mori 已经把 put/get 送到 FlyDSL 门口,而 MegaMoE 正是 FlyDSL 写的** | **T4** |

第四条值得展开一句:mori #792 把 cco put/get 暴露给 FlyDSL,意味着 MegaMoE 的 EP 原语有了一条**不需要自己维护 fork 的** SDMA/copy-engine 路径。这是本窗口对 Turbo 最直接的外部利好,但也意味着继续喂那份 DeepEP fork 的性价比在下降。

---

## 3. Roadmap

### T0 — 契约与止血(第 1–3 周)

目标:**让 Turbo 变成一个有边界的包**。这一阶段不碰任何 kernel。

| 项 | 内容 | 验收 |
|---|---|---|
| **T0-1** | 定义并在代码里固化**公开 API 面**:`primus_turbo.pytorch.{ops,modules,core}` 为公开,`kernels/`、`flydsl/`、`triton/`、`_C` 为私有。给私有层加 `__all__` 收紧 + 导入期 `DeprecationWarning` | 一份 `API.md` 列出公开符号;私有层被外部 import 时有告警 |
| **T0-2** | 修 §1.1 的三处断裂:ODC Python primitives 层**要么补齐在 Turbo、要么整体移回 Primus**(我倾向后者 —— 它在 Turbo 侧零消费者);`quantile_balancing` 与 `minimax_m3` 同样二选一 | Primus `main` 能在干净环境里 `import primus` 不报 ImportError |
| **T0-3** | Primus 侧对 Turbo 下**版本约束**,Turbo 侧暴露 `primus_turbo.__version__` 与一张能力表(`has_mxfp4()`、`has_mega_fp8()`、`supported_archs()`) | Primus 的 66 行 import 里,跨越公开面的那些全部带守卫或版本门 |
| **T0-4** | 删死代码:接上 `assert_capacity()` 或删掉;`tokens_per_expert_to_mask` 改 `.apply()` 或明确文档化为非可导;清掉 `flydsl/deep_ep/` 的孤儿 `__pycache__`;删 `Format.HYBRID` 等无实现枚举 | TODO 从 72 降一档;无"声明未实现"的公开枚举 |
| **T0-5** | 工作树卫生:删 42 GB core dump 与 CI 日志;把 `*_MI355X.csv` 要么接进 T1-3 的基线、要么删 | 仓库目录 < 2 GB |

**工期估**:3 人周。T0-2 需要和 Primus 侧对齐归属,是唯一的跨仓协调项。

### T1 — 可信度(第 2–7 周,与 T0 部分并行)

目标:**让 CI 测我们真正在做的东西,并且能挡住 #486 那类回归**。

| 项 | 内容 | 验收 |
|---|---|---|
| **T1-1** | **把 gfx1250 加进 CI 构建矩阵**。先只跑那 12 条 allow-list,但必须真编译真跑 | `ci.yaml` 的 `GPU_ARCHS` 含 gfx1250;MI455X runner 上绿 |
| **T1-2** | **修红的发版门**。29 个 `test_flash_attn_varlen` SNR 失败要么修、要么连同判据一起重新论证(当前 tolerance 本身带着 `# TODO: Need to check again whether these values are reasonable.`,`test_utils.py:22`) | `release.yml` 的 validate 全绿,且 tolerance 有书面依据 |
| **T1-3** | **性能回归门**:把 nightly 的 `continue-on-error` 去掉,给核心算子集(attention fwd/bwd、grouped GEMM bf16/fp8/fp4、MegaMoE fwd/bwd)存基线 CSV,回归超阈值即失败 | 人为把某 kernel 改慢 10% 能让 nightly 变红 |
| **T1-4** | **梯度范数回归门(直接针对 §1.3)**:一个固定 seed、固定 shape 的小 MoE 训练 step,断言 it1 grad norm 落在区间内。#486 那次 1.45→1.88e6,任何合理区间都会抓到 | 把 mega bf16 的 tile 换回 `ed8d7af4` 版本,这个门必须红 |
| **T1-5** | 把 GPU job 从 `ci:gpu` label 门改成默认跑(或至少:改 `primus_turbo/flydsl/mega/**`、`csrc/kernels/deep_ep/**` 的 PR 强制跑) | 触碰关键路径的 PR 无法绕过 GPU 测试 |

**工期估**:5 人周。T1-4 是性价比最高的一项 —— 它把"静默错梯度"这个四次重复出现的家族变成一个会响的门,成本只是一个小训练 step。

**T1-4 的论据不止 #486。** 它同时覆盖 `Primus/HANDOFF_mxfp8_topk_grad.md` 那个未解 bug 的签名。而且 Broken Symmetry 那篇给了一个零成本诊断可以直接抄进来:**把两层的 backward 用 FP32 重算一遍,看多出来的梯度是否消失。**

### T2 — 收敛栈(第 6–14 周)

目标:**六套栈变成有政策的三套**。

| 项 | 内容 | 验收 |
|---|---|---|
| **T2-1** | 写下**栈选型政策**并执行:FlyDSL = 融合超级 kernel 与 gfx950/gfx1250 新 arch;Triton = 可移植基线与快速原型;csrc = 需要 ISA 级控制或厂商库(CK/hipBLASLt)的东西。Gluon / Python-HipKittens / JAX 三者明确表态 | 一份 `docs/kernel-stack-policy.md`;三个休眠栈各有明确归宿 |
| **T2-2** | **Gluon 决断**:4,457 行只换来一个 gfx950 f16 FA forward,且 08-27 后未动。要么证明它比 FlyDSL 那条快到值得留,要么删 | 有数据的删除或保留决定 |
| **T2-3** | **JAX 决断**:3,900 行纯 FFI 前端,08-20 后未动,gfx1250 上是硬错误。判断它是否还有客户 | 同上 |
| **T2-4** | **统一调度入口**:把 §1.4 的九种机制收敛到 `AutoKernelDispatcher` 一种。MegaMoE、`mlp_fp4`、`cross_entropy`、量化这几条绕过调度器的要接进来 | 新增一个后端只需改注册表一处 |
| **T2-5** | **de-vendor 三套 mega 栈**。前提是 T1-4 的门先立起来 —— 有了门才敢合并 tile。目标是 bf16/fp8/gfx1250 共享一份 tile 契约 + 一份 `symm_buffer` 抽象 | 三份 `symm_buffer.py` 合为一份且 T1-4 仍绿 |

**工期估**:8 人周。**T2-5 必须排在 T1-4 之后** —— 没有梯度门就去合并 tile,就是在重演 #486。

### T3 — 差异化(第 10–20 周)

目标:**把两个资产变成别人引用得到的东西**。

| 项 | 内容 | 验收 |
|---|---|---|
| **T3-1** | **把格式缝从 MoE 路径外溢到通用算子**。具体是 U1–U6:给 `gemm_fp8`/`gemm_fp4` 的 A/B/grad_out 量化找到可融的宿主 kernel(前一个 norm/activation 的 epilogue)。`[32,32]` 2D block 已经在用于权重(`gemm_fp8.py:468`),LP-FA4 那条"方块 scaling 让量化对转置不变、反向只量化一次"的论证可以直接套到激活侧 | 通用 fp8/fp4 路径的独立量化 launch 数下降,且端到端 step 有可测收益 |
| **T3-2** | **MXFP4 做成可引用的成果**。现在是 gfx950-only、分散在四个 op 里。要:补 gfx1250、把 `deoscillation` 的算法写成 knowledge/paper、和 ALTO 配方对齐出一份"AMD MXFP4 训练配方"| 一份可外发的 MXFP4 训练结果 + `slab/knowledge/kernels/` 下的设计笔记 |
| **T3-3** | **MegaMoE 融合深度对外说清楚**。`colwise_requant_mxfp8_grouped_fp8in_flydsl` 这类设计目前只存在于代码注释里。至少一份技术文档,最好一篇投稿 | 外部可读的 MegaMoE 设计文档 |
| **T3-4** | **gfx1250 / Helios 的 EP 成本结构**。Helios 是 72 卡单 scale-up 域(UALoE,260 TB/s),而业界扫描 §5.4 指出**至今无人写过 AMD 侧的 scale-up 域 MoE 成本模型**。gfx1250 Mega 的工作正好在这条线上 | 一份 UALoE 域下 EP 成本模型 + MegaMoE 在其上的设计含义 |

**工期估**:12 人周。T3-2 与 T3-4 是对外署名价值最高的两项。

### T4 — 通信层决断(第 14 周起,单独立项)

| 项 | 内容 |
|---|---|
| **T4-1** | **DeepEP fork 的去留**。现状:ROCm 内部 fork、README 自称 gfx942-only、10-12 后未动、上游 DeepSeek V2.5 已删 V1 且门槛抬到 Hopper+。选项是(a)继续自养,(b)改接 mori 的 `ccoSdma` put/get(#792 已暴露给 FlyDSL,与 MegaMoE 同栈),(c)等 RCCL GIN(能力已在 ROCm 10.0.0,卡 `nccl_device.h` 编译卫生,见 `ROCm/rccl#2190`) |
| **T4-2** | ODC 的归宿(与 T0-2 同一件事的下半截):C++ GDA 后端在 Turbo 里零消费者,要么接进 MegaMoE/DeepEP,要么和 Primus 侧一起退役 |
| **T4-3** | 低精度 EP 通信。dist_moe 明写 communication quantization **不支持**,Motif-3 指的那个"EP dispatch 之前做 MXFP8"的位置官方栈是空的。Turbo 的 `dist/fp8_alltoall.py` 已经在这条线上 |

---

## 4. 排序与工期

```
周次   1   2   3   4   5   6   7   8   9  10  11  12  14  16  18  20
T0  ████████████
T1      ████████████████████
T2                  ████████████████████████████████
T3                                  ████████████████████████████████
T4                                              ████████████████████
```

**硬依赖只有一条:T2-5(de-vendor mega)必须在 T1-4(梯度范数门)之后。** 其余可按人力并行。

**如果只能做三件事**,按边际收益排:

1. **T1-4 梯度范数回归门** —— 成本最低(一个小训练 step),挡住的是已经四次重复出现、每次都靠人肉 grad norm 发现的失效家族
2. **T0-2 修复断裂的 import** —— 不修的话 Turbo 的价值在 Primus 侧随时可能整段失效,且没有任何自动化会告诉我们
3. **T1-1 gfx1250 进 CI** —— 团队最大的人力投入目前在 CI 覆盖之外

---

## 5. 明确不做

| 不做 | 理由 |
|---|---|
| 搬 dist_moe 的 kernel | CUTLASS DSL 无 ROCm 对等物且 DSL 不可 hipify;SM100/Linux-ARM64 硬门槛、无 CPU fallback,**连行为基线都建不起来**。偷执行模型(SyncFree、三级容量预算、`[32,32]`)不搬代码 |
| 追 DeepEP V2.5 | 门槛已到 CUDA 13.1 / NCCL 2.32.3 / Hopper+,V1 与 NVSHMEM 基座已被上游删除,非 NV 支持另开 `DeepEP-Ascend` 仓 |
| 本季度排 MoonEP | 虽已重写成完整训练库,设备列表仍只有 NVIDIA + 真武 PPU |
| 新增第七套 kernel 栈 | 在 T2-1 政策写下来之前,任何新栈都是净负债 |
| 在 T1-4 之前合并 mega tile | 就是 #486 的重演 |

---

## 6. 风险

| 风险 | 说明 | 缓解 |
|---|---|---|
| **T0-2 需要跨仓裁决** | ODC / quantile_balancing / minimax_m3 的归属不是 Turbo 单方面能定的,涉及 Primus 侧 owner | 先把事实摆清(本文 §1.1 的表),再谈归属;归属未定前至少给裸 import 加守卫 |
| **gfx1250 runner 可得性** | T1-1 依赖 MI455X CI runner;Helios 本身"是参考设计不是在售产品",量产预期 2H 2026 | 先用 squeue 上的 MI455X 节点跑 self-hosted,不等正式 runner |
| **de-vendor 可能根本做不成** | 三套 mega 栈的分叉可能已经深到合并代价大于收益 | T2-5 先做一次只读的 diff 评估再决定,不预设结论 |
| **性能门会制造噪声** | MI 系列 run-to-run 抖动 + 功耗封顶(HEAD 里就有一条 "EP4 round 4 — step is power-capped")会让阈值要么太松要么天天误报 | 阈值按实测分布定,不拍脑袋;先观察两周只告警不失败 |
| **本文基于只读审计** | 所有结论来自静态读码,未上机验证任何性能或行为声明 | T0/T1 的每一项验收都以实跑为准 |

---

## 相关文件

- 业界基线:[`../primus-moe/2026-10-28_0546_industry-scan-update.md`](../primus-moe/2026-10-28_0546_industry-scan-update.md)
- Primus 侧 roadmap 与复盘:[`../primus-moe/2026-09-03_primus-roadmap-2026q4-2027h2.md`](../primus-moe/2026-09-03_primus-roadmap-2026q4-2027h2.md) · [`../primus-moe/2026-10-12_0101_primus-roadmap-update.md`](../primus-moe/2026-10-12_0101_primus-roadmap-update.md)
- 格式缝:[`../career-strategy/2026-08-12_1505_the-quant-gemm-seam.md`](../career-strategy/2026-08-12_1505_the-quant-gemm-seam.md)
- scale-up 域成本结构:[`../../papers/mixture-of-kittens.md`](../../papers/mixture-of-kittens.md)
- 活跃 bug(T1-4 的直接论据):`/perf_apps/xiaoming/Primus/HANDOFF_mxfp8_topk_grad.md`

## 核查溯源

五路只读 subagent,2026-10-28。**由我手工复核的**:`#486` 静默打爆 MegaMoE 的注释原文、fp8 mega 路径无 gfx1250 门(`is_gfx1250` 出现次数 11 vs 0)、`assert_capacity` 零调用、`flydsl/deep_ep/` 只有 `__pycache__`、CI 的 `GPU_ARCHS`、`conftest.py` 的 gfx1250 全跳 + allow-list、工作树体积、DeepEP README 的 gfx942-only、`3_validate.txt` 的 29 个失败明细、分支距离、**Primus 侧四处跨仓 import 的性质(裸/惰性/带守卫)与这四个模块在 Turbo 全部远程分支上的不存在**。

- 算子面与后端调度审计 —— [`dde2f42e-ff8d-45ad-9e93-30d4fc2af9ba`](dde2f42e-ff8d-45ad-9e93-30d4fc2af9ba)
- 六栈碎片化审计 —— [`61a5d672-16da-4c5b-8055-ff071fedde7e`](61a5d672-16da-4c5b-8055-ff071fedde7e)
- MegaMoE + DeepEP + ODC 深读 —— [`dbdcebe9-17a7-4612-9ead-2fda5ea95f90`](dbdcebe9-17a7-4612-9ead-2fda5ea95f90)
- CI / 测试 / 性能门 / 构建审计 —— [`57c5ca3e-a047-4b23-a916-c25708734cb5`](57c5ca3e-a047-4b23-a916-c25708734cb5)
- 量化栈与格式缝审计 —— [`94ccaddc-34cb-4ff7-a17c-285aa8556312`](94ccaddc-34cb-4ff7-a17c-285aa8556312)

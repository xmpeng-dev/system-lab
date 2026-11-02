# MoonEP vs UltraEP：EP 负载均衡的移植选型

> **时间：** 2026-09-07 09:00 (UTC+8) / 2026-09-07 01:00 UTC
> **背景：** 排查「2026 年业界开源系统里有什么值得移植进 Primus」时，筛出两个 MegaMoE 量级的候选，都在解 EP 负载不均衡。本文定路线。
> **上游语境：** [`2026-09-03_primus-roadmap-2026q4-2027h2.md`](./2026-09-03_primus-roadmap-2026q4-2027h2.md) 的停手/追赶/**独占**三分类。EP 负载均衡属于"独占"列——上游 Megatron 的 4 条 dispatcher 路径都不做热路径均衡。
> **已有笔记：** [`papers/ultraep/`](../../papers/ultraep/README.md)（论文精读，当时未记录代码已开源）、[`papers/tempo.md`](../../papers/tempo.md)（同一问题的 serving 侧、代价模型视角）

---

## 结论

**先移植 UltraEP，把 MoonEP 当作下一代目标架构。**

理由一句话：**UltraEP 是叠加式的、有现成 Megatron 集成、而且论文显式点名 AMD Helios；MoonEP 的结构性收益更大，但它要求重排专家参数的物理布局，blast radius 穿透 distributed optimizer 与 grouped GEMM。**

**两者互斥，不是互补**——都在解同一个不均衡问题，一个靠构造消灭它，一个靠实时修复它。选了 MoonEP 就不需要 UltraEP，反之亦然。

⚠ **无论选哪个，roadmap 的 C0-4（多机数值正确性 CI，2 节点 bf16 + fp8 各一条 loss golden）必须先落地。** 两者**都会碰梯度路径**——UltraEP 要把 replica 的梯度归约回 main expert，MoonEP 要把冗余专家的梯度归约回 home rank。这正是我们在 DeepEP 上花 620 行调查才根治的那类静默错梯度的 bug 家族，而当前 CI 对它零覆盖。在 C0-4 之前动这两个中的任何一个都是拿正确性赌吞吐。

---

## 〇、补充：更广的厂商扫描改变了排序（2026-09-07 追加）

初版只比较了 MoonEP 与 UltraEP。国内厂商全量扫描回来后，多出两个候选，其中一个**可能应该排在两者之前**。

### Triton-distributed（ByteDance-Seed）— 新的第一顺位候选

[ByteDance-Seed/Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed)，MIT，2025-04 创建，**2026-08-12 仍在推**，★1535。基于 Triton 的**分布式编译器**，用 tile-centric primitive **生成** comm-compute overlap kernel（TileLink, MLSys'25），附一套 overlap kernel 库；2025-12 更新加了 EP function、low-latency mode、**Mega-EP**。

**它是这批候选里唯一有实打实 AMD 证据的。** README 原文 "Triton-distributed currently mainly targets Nvidia GPU and AMD GPU"，仓库里有 `python/triton_dist/test/amd/`，2026-09-07 逐个核实以下文件均返回 200：

```
test_ep_moe_fused.py      test_mori_shmem_api.py    test_rocshmem_api.py
test_ep_a2a_fused_kernel.py
```

**AMD 侧走的正是 ROCm/mori + rocSHMEM——和本文 §2 为 MoonEP 找的底座是同一个，也和 Primus ODC 已在用的是同一个。**

为什么可能该排第一：**它给的是 kernel 生成能力，不是又一个手写 kernel。** Primus 现在有 MegaMoE 这一个手写的 fused MoE；Triton-distributed 提供的是能产出这类 kernel 的编译器 + Mega-EP + megakernel 基础设施。而"编译器"这一格上游 Megatron 明确不会做，落在 roadmap 的独占列。移植成本在所有候选里最低（AMD 路径已存在），上限最高。

**训练侧证据（第二轮扫描补上，修正上一段的保留意见）**：初写时我说"训练侧端到端证据没有找到"，这不准确。仓库里有 [PR #175](https://github.com/ByteDance-Seed/Triton-distributed/pull/175) 的 AMD fused EP megakernel，对应论文 UniEP（"Unified Expert-Parallel MoE MegaKernel for **LLM Training**"），**训练导向**。2026-09-07 我把文件拉下来实读了：

`python/triton_dist/kernels/amd/ep_all2all_fused.py`，**1267 行，MIT**。它的 docstring 与实现要点：

- 两个 persistent mega kernel：`dispatch + grouped GEMM(1)` 与 `grouped GEMM(2) + combine`——**和 MegaMoE 的切分方式一致**
- 一个 expert 的 token 从所有 source rank 到齐即可开跑它的 GEMM tile，**tile 粒度 overlap**
- 跨 rank 数据移动与同步走 **mori_shmem device primitive**（`putmem_warp` / `signal_op` / `signal_wait_until` / `barrier_all`，经 `libshmem_device`）；对称 buffer 来自 `mori_shmem_create_tensor`，每 expert 一个 `uint64` signal 做 barrier
- **wavefront 宽度 64**，grouped-GEMM tile 是朴素指针算术 `tl.dot`
- split 交换与 GEMM tiling 元数据**默认在 device 上构建**（`kernel_build_fused_dispatch_metadata` / `kernel_build_gemm_tiling`），`use_device_metadata=False` 可退回 host torch 路径

⚠ **仍是 forward-only**。机制确认（**此处修正我先前的一次误判**）：kernel 文件 `ep_all2all_fused.py` 内确实没有 `NotImplementedError`，但 **autograd wrapper 不在这个文件里**——它是 `triton_dist/function/amd/ep_moe_fused.py` 里的 `TritonDistFusedEpMoeFunction`，其 `backward` 抛 `NotImplementedError`，测试第 300 行显式断言该行为：

```
# backward must raise NotImplementedError (check once)
assert raised, "TritonDistFusedEpMoeFunction.backward must raise NotImplementedError"
```

所以"backward 显式 `NotImplementedError`"的原始说法是对的；我先前只 grep 了 kernel 文件就否定它，范围搞错了。**用于训练必须自己补反向，这是第 2 顺位的主要工作量。**

它和 MoonEP / UltraEP **不在同一层**——后两者是可直接消费的 runtime，它是工具链 + 参考实现，所以严格说不是三选一。

### Triton-distributed EP 的性能证据（2026-09-07 补查）——AMD EP 侧同样是零

和 MORI 一样，这里也必须如实记：**AMD 上的 EP 性能，公开数据为零。**

| 证据 | 平台 | 场景 | 数字 |
|---|---|---|---|
| `docs/amd-perf.md` | **MI308X 单节点** | **AG-GEMM / GEMM-RS（稠密 TP）** | 仅两张图，**根本不含 EP** |
| README low-latency AllToAll | 32×H800 | **推理** | 137 us vs DeepEP 182 us（128 token/rank、topk=8、hidden 7168、fp8） |
| Seed-OSS-36B e2e | NVIDIA | 推理 | 1.33× |
| **UniEP 论文** [arXiv 2604.19241](https://arxiv.org/abs/2604.19241) | **NVIDIA Hopper** | **训练** | **1.03×–1.38× over SOTA** |

三点必须挑明：

1. **`docs/amd-perf.md` 全文只有 7 行、两张图，且都是 AllGather-GEMM 与 GEMM-ReduceScatter——这是稠密张量并行，不是 EP。** 硬件还是 **MI308X**（阉割版），不是 MI300X 更不是 MI355X。仓库里那套 AMD EP 测试能跑，但**没有任何公开的 AMD EP 性能结果**。
2. **那个 137 us vs 182 us 的 DeepEP 对比自带减分项**：README 原文括注 *"note: DeepEP doesn't use NVLink for inference"*——**基线是被限制住的**，不能当作 1.33× 的干净胜负。而且是推理。
3. **唯一的训练侧数字来自 UniEP 论文，1.03×–1.38×，全部在 NVIDIA Hopper 上测。** 注意**下沿是 1.03×，几乎等于没有**；1.38× 是上沿。这个区间在移到 MI355X + mori_shmem 之后会怎么变，无人知晓。

**不过 AMD EP 测试自带 benchmark harness**（`benchmark_latency_memory` / `print_benchmark_comparison`，报 forward latency + peak memory），所以**我们可以自己测**，不必等上游——这应与 §〇之二 那个 MORI-EP vs DeepEP 对比合并成同一个前置实验。

#### 自测尝试（2026-09-07 当日）：构建成功，但被 LLVM 版本耦合挡死

已在 `xiaoming-dev`（ROCm 7.15 / torch 2.12 / gfx950）把 Triton-distributed **从源码构建通过**，
`import triton`（其 fork 版本 **3.4.0**，amd + nvidia 双 backend）与 `import triton_dist` 均正常，
shmem 后端走 **mori_shmem**（因此 **rocshmem / pyrocshmem 整条链可以完全跳过**，这是个实打实的省事发现）。
mori_shmem 初始化、topology、8-rank bootstrap 全部通过，测试真的进到了 kernel 编译。

**但拿不到性能数字，卡在一个结构性的版本耦合上**：

```
ValueError: Failed to parse library at .../triton_dist/tools/compile/libmori_shmem_device.bc
  triton/backends/amd/compiler.py:400 in make_llir -> llvm.link_extern_libs()
```

| | LLVM |
|---|---|
| mori 的 device bitcode 由谁产生 | **AMD clang 23.0.0git**（ROCm llvm-project `caaa58b9`，即 ROCm 7.15 自带） |
| TD 的 Triton 链的是哪个 | 固定 pin 到上游 LLVM `8957e64a`（下载预编译 `llvm-8957e64a-ubuntu-x64`） |

**新版 LLVM 产生的 bitcode，旧版 LLVM 解析不了**——报错就是字面意思。这跟 mori 版本无关：
v1.1.0（本地编）与 1.2.2（官方 wheel）生成的 `.bc` 都由同一个 ROCm 7.15 clang 产出，两者都过不去。

**所以文档里"ROCm 7.1 + torch 2.7.1"不是保守写法，是硬约束**——它锁的其实是
"ROCm 的 clang 版本必须 ≤ Triton pin 的 LLVM"。要在本容器跑通只有两条路：
(a) 换成上游指定的 `rocm/pytorch:rocm7.1_ubuntu24.04_py3.12_pytorch_release_2.7.1` 镜像重建；
(b) 让 TD 的 Triton 改用 ROCm 7.15 的 LLVM 重编（它 pin 那个 hash 是有原因的，风险高）。

**另一个独立结论（对 §〇之二 的 MORI 排序有意义）**：本地源码编出来的 mori v1.1.0
**是坏的**——只调用一次 `shmem_get_unique_id()` 就 `free(): invalid size` 堆损坏，
8-rank 下 `shmem_init_attr` 直接 SIGSEGV；换成官方 `amd_mori` 1.2.2 wheel 后同一段代码完全干净、
初始化全过。**结论：MORI 一律用 PyPI wheel，不要自己源码编**。

构建过程踩到 6 个坑，按顺序记下（对任何人重做这件事都省时间）：

| # | 症状 | 原因 / 处置 |
|---|---|---|
| 1 | `git submodule update --init --recursive` 长时间不结束 | mori 的传递依赖里有 **SPDK / DPDK / xnvme**，与本需求无关。**只 init `3rdparty/triton` 和 `3rdparty/mori/3rdparty/spdlog`** 即可（mori 的 CMake 只 `add_subdirectory(3rdparty/spdlog)`） |
| 2 | mori `pip install` 报 `invalid pyproject.toml config: project.license` | TD 钉的 mori(v1.1.0~22) 用了新式 SPDX 字符串写法，本容器 setuptools 不收。改成 `license = {text = "MIT"}` |
| 3 | cmake `Could not find ROCM_SMI_LIB` | mori 硬编码 `HINTS /opt/rocm/lib`，而本容器 ROCm 来自 pip `_rocm_sdk_devel`。`ln -s .../_rocm_sdk_devel /opt/rocm` |
| 4 | cmake `hsakmt::hsakmt` 依赖 `numa::numa` 但 target 未定义 | 该 target 只由 ROCm 自己的 FindNUMA 建。装 `libnuma-dev` **不够**，须在 `src/application/CMakeLists.txt` 里 `find_package(hsakmt)` **之前**补 imported target（放顶层 CMakeLists 无效——那时还在 `project()` 之前，`find_library` 解析不到） |
| 5 | ninja `'/usr/lib64/libc.so' missing` | ROCm pip wheel 是在带 `/usr/lib64` 的发行版上打的。`ln -s /usr/lib/x86_64-linux-gnu/libc.so /usr/lib64/libc.so` |
| 6 | `ModuleNotFoundError: triton.backends.amd` | in-tree backend 靠 `add_links(external_only=False)` 落地，而 `develop`/`egg_info` 走的是 `external_only=True`，**不会链 amd/nvidia**。手工把 `3rdparty/triton/third_party/{amd,nvidia}/backend` 拷到 **`3rdparty/triton/python/triton/backends/`**（注意是 3rdparty 那棵树，不是 `Triton-distributed/python/`） |

另有一个**必须避开的陷阱**：`amd_mori`(PyPI wheel) 与 TD 自建的 `mori` **两个包提供同一个 `mori` 模块**，
上游 README 明确要求不可共存。若先装了 wheel 再让 TD 构建，`pip uninstall amd_mori` 会把两者重叠的文件
一起删掉，留下一个残缺的 `mori/`（只剩 `ops/`、`_jit_sources/`，没有 `__init__.py`）——必须
`rm -rf site-packages/mori` 后重装。**代价**：TD 钉的是 mori v1.1.0~22，装它就会顶掉跑出 §〇之二
那组数字的 1.2.2，两者不能同时在位。

还就地修了 mori v1.1.0 的一个 bug：`src/application/topology/gpu.cpp:80` 的
`ROCM_SMI_CHECK(rsmi_shut_down())` 会因 rsmi 已被别处关闭而返回 `RSMI_STATUS_INIT_ERROR` 并 abort
整个进程。**前面所有 rsmi 查询都成功、只有关闭这一步失败**，且实测 8-rank torchrun 下
`rsmi_init` 全部 rc=0（`probe_rsmi_under_torchrun.py`），确认是良性双重关闭。去掉该处返回值检查后
SIGABRT 消失、进展到 SIGSEGV。1.2.2 的 wheel 无此问题。

**UniEP 论文里真正对我们有价值的，可能不是那个加速比，而是它的问题陈述**（摘要原文）：现有 EP 优化 *"rely on ad-hoc, complex kernels that lack adaptability across diverse optimization configurations and **frequently neglect numerical stability**, failing to meet the strict precision requirements of large-scale training"*，因此 UniEP 引入 **deterministic token ordering，保证即使在激进 overlap 调度下也与串行执行数值一致**。⭐ **这和 §五 说的 C0-4 多机数值 golden CI 是同一件事的两面**——它把"overlap 不能破坏数值可复现性"直接做进了 kernel 设计。若要补反向，**这条确定性顺序约束必须一并继承，否则重蹈 DeepEP 静默错梯度的覆辙。**

### UCCL / UCCL-EP / UCCL-Zip — 命中"压缩集合通信"这一格

[uccl-project/uccl](https://github.com/uccl-project/uccl)，Apache-2.0，**2026-09-06 仍在推**，★1506。三件套：UCCL-collective（NCCL/**RCCL** drop-in 替换）、UCCL-P2P、UCCL-EP；UCCL-Zip 是无损压缩集合通信（arXiv 2604.17172）。

README 三处 AMD 证据：UCCL-EP "allows running DeepEP atop of heterogeneous hardware platforms, **including AMD and Nvidia GPUs**"；UCCL-P2P 的 benchmark 图是 **MI300X + Broadcom Thor-2**；以及 "UCCL has been adopted as part of the **AMD TheRock ecosystem**"。

它同时命中独占列的两格——**ROCm 可移植的 MoE 通信**与**压缩集合通信**（后者 roadmap 标为"无人认领"）。旁证：已有第三方 benchmark 把 UCCL-EP 与 DeepEP V1/V2、pplx-garden 并列对比，说明它被当作 DeepEP 的同级对手而非玩具。

注：不是中国厂商（UCCL 是学术/开源社区项目），是扫描时顺带捞到的。

### ⚠ DeepEP V2 换了通信后端——这会影响 §7 那个缺口的解法

DeepEP 已更名 "DeepEveryParallel" 并完成 V2 重写：**从 NVSHMEM backend 换成更轻量的 NCCL Gin backend**；高吞吐与低延迟 API 统一成单一 `ElasticBuffer`；支持到 EP2048；解析式 SM/QP 计算不再需要 autotune；V3 类训练 SM 占用 24 → 4–6；新增 **0-SM Engram (RDMA) / 0-SM PP (RDMA) / 0-SM CP (Copy Engine)** 实验原语。Requirements 写 SM90 + CUDA 12.3+ + **NCCL 2.30.4+**。

**含义**：我们现有的 DeepEP ROCm 集成（含那个 620 行调查根治的 intranode 静默错梯度）是针对 **V1 / NVSHMEM** 的。V2 之后 ROCm 的落地路径整体变了——需要 RCCL 侧的 Gin 等价物，而不是继续在 NVSHMEM 语义上做映射。另外 **0-SM PP / 0-SM CP 是全新能力**，不属于"MoE dispatch 逻辑"那个上游正在吸收的范畴。**这条应该在动 §7 那个 gfx950 缺口之前先查清楚。**

### 其余值得知道的

| 项 | 判断 |
|---|---|
| [`alibaba/atrex-kernel-agent`](https://github.com/alibaba/atrex-kernel-agent)（Apache-2.0，2026-09-06 活跃） | GPU kernel 优化的 agent 系统，profile 驱动 + correctness/performance 双闸门 + git 隔离 + 可恢复。README 明确支持 **NVIDIA / AMD / T-Head PPU**，campaign 类型含 **FlyDSL** 与 TileLang。**和我们 `iteration_rules.mdc` 几乎是同一套方法论的工程化实现**，与 `notes/MegaMoeFlydsl/` 直接对口。不是训练基础设施，但可能是对日常产出影响最直接的一项 |
| [`inclusionAI/linghe`](https://github.com/inclusionAI/linghe)（MIT，2026-04-28） | 蚂蚁的 MoE FP8 训练 kernel 库，对齐 **Megatron-LM blockwise quantization**，Triton 实现⇒大部分可直接跑 ROCm。范围与 Primus-Turbo 重叠，但几个 trick 值得单挑：**用 routing-map padding 替代 activation padding**、fused linear gate（backward 7.06×）、fp32 router GEMM。属于便宜的边角收益 |
| DeepSeek **DeepGEMM** | 2026-04-16 自己上线了 "fused MoE with overlapped communication (**Mega MoE**)"，与我们的 MegaMoE 正面撞车。SM90/SM100 CUDA-only，**当参考实现读，不要 port** |
| DeepSeek **TileKernels** | 创建后只推了一天即静默（2026-04-22 → 04-23），**DORMANT**。思路可借（TileLang 有 AMD 后端），代码不必跟 |
| **TEMPO** | `jeshxxx/TEMPO` **2026-09-07 核实为 404**，论文摘要声称有代码但仓库不存在，且方向是推理。已在 [`papers/tempo.md`](../../papers/tempo.md) 更正 |
| 明确劝退 | Alibaba Tessera / ChunkFlow 未开源；Ant GLake 已 ARCHIVED；Tencent Angel-PTM 从未开源；`bytedance/flux` 2026 年无活动（已被 Triton-distributed 取代，后者直接把它列为对标基线）；Pai-Megatron-Patch 2026 年零活动；`ByteDance-Seed/ByteScale` 404 |

### 排序更新

**Triton-distributed 与 MoonEP / UltraEP 不在同一层**，可以并行推进：前者是工具链（生成 overlap kernel），后两者是 runtime（解 EP 不均衡）。就 runtime 这一层，**下面 §一~§六 的结论不变：先 UltraEP，MoonEP 作为下一代架构。** 就工具链这一层，Triton-distributed 的 AMD EP / fused-MoE 测试套件已经在仓库里，是投入产出比最高的单项。

---

## 〇之二、国外厂商扫描（2026-09-07 追加）——其中一条直接松动了 MoonEP 的 gate

### ⭐ ROCm/mori 已注册为 torch SymmetricMemory 后端——§六 gate Q1 基本被回答了

下面 §六 我把"MoonEP 的 symmetric memory 契约能否干净映射到 MORI-SHMEM"列为 Phase-2 的第一个 gate 问题。**AMD 自己已经把这件事做掉了一半。** 2026-09-07 逐个拉取 PR 页面核对标题，逐字如下：

| PR | 标题（原文核实） |
|---|---|
| [#544](https://github.com/ROCm/mori/pull/544) | `allocator: register mori as a torch SymmetricMemory backend` — by carlushuang |
| [#594](https://github.com/ROCm/mori/pull/594) | `feat(cco): add Triton device API bindings` — by yangyuhuiling |
| [#606](https://github.com/ROCm/mori/pull/606) | `feat(EPv2): support wide EP (larger than 32) in intranode kernels` — by kawhil-amd |
| [#586](https://github.com/ROCm/mori/pull/586) | `perf(ep): optimize IntraNode dispatch kernel for MI350X` — by kudomcho |

含义有三层：

1. **#544 把 gate Q1 从"要不要自己造映射层"降级成"验证现成后端够不够用"。** MoonEP 依赖 symmetric memory；torch 的 `SymmetricMemory` 抽象现在在 ROCm 上有 first-party 实现。这不代表 MoonEP 能直接跑（它还要求专家权重按 `[E+B, H, H']` 连续布局，那才是真正的硬约束，见 §四），但**通信底座这一层不再是自研项**，MoonEP 的风险评估应当下调。
2. **#594 + 上面那个 `ep_all2all_fused.py` 合起来是完整链路**：mori 出 Triton device binding，字节的 AMD megakernel 正好就是用 `libshmem_device` 这套 primitive 写的。**在 MI355X 上用 Triton 写 MegaMoE 式 comm+GEMM 融合 kernel，两端都已就位。**
3. **#586 / #606 说明 mori EPv2 正在主动啃 MI350X 与 wide-EP(>32)**，这和 §七 记的"训练侧 EP 仍 gfx942-only"是同一个缺口的两端——**该缺口的上游侧正在被 AMD 自己填**，我们要做的是接，不是造。

✅ License 已核实（2026-09-07 直接拉 `LICENSE` 文件）：**MIT，Copyright © Advanced Micro Devices, Inc.**

### ⚠ 但 MORI 的性能证据是缺的——排第一的理由只能是"底座"，不能是"更快"

把 mori 排在第 1 位时我没有给任何性能数据。补查之后必须如实记下：**README 的 benchmark 表没有基线。**

对 `README.md` grep `deepep` / `baseline` / `compare` / `vs` / `nvshmem` / `rccl`，**零命中**。表格只有 MORI-EP 自己的绝对值，正文里的 "SOTA performance" 是没有对比数据支撑的散文声明。也就是说，**"换成 MORI-EP 会比现在的 DeepEP 路径快"这个命题，上游没有给出任何证据。**

绝对值本身（DeepSeek V3 配置，4096 token / 7168 hidden / top-8，FP8 dispatch + BF16 combine）：

| 硬件 | Kernel | Dispatch XGMI | Dispatch RDMA | Combine XGMI | Combine RDMA |
|---|---|---|---|---|---|
| MI300X + CX7 | EP8 | 307 GB/s | — | 330 GB/s | — |
| MI300X + CX7 | EP16-V1 | 171 | 52 | 219 | 67 |
| MI300X + CX7 | EP32-V1 | 103* | 57* | 91* | 50* |
| **MI355X + AINIC** | **EP8** | **345** | — | **420** | — |
| MI355X + AINIC | EP16-V1 | 179 | 54 | 234 | 71 |
| MI355X + AINIC | EP32-V1 | 85 | 46 | 110 | 61 |

`*` README 自注为"上一版 kernel 的过期数据，待重测"。

读法：EP8 是单节点全 XGMI（RDMA 列为空），MI355X 上 dispatch 345 / combine 420 GB/s，很强。**一旦跨节点，RDMA 就是瓶颈，稳定落在 46–71 GB/s**——EP16/EP32 的 XGMI 数字下滑有一部分是流量分流到 RDMA 所致，不能直接当作退化读，但**跨节点带宽这个绝对水位才是大规模训练真正要面对的数**。

**唯一的端到端训练数字**：MORI 的 SDMA AllGather 集成进 DeepSpeed ZeRO-3，*"up to 10% end-to-end training speedup"*，机理是把 AllGather 流量卸载到专用 SDMA copy engine。⚠ 这是 **ZeRO-3 的 AllGather，不是 MoE EP**，不能直接挪用来支持本文的排序。

**其余全部战绩都是推理**：SGLang / vLLM 的 MoE EP 与 PD 分离、KVCache 传输、Kimi agentic serving（AMD × Moonshot 合作）、SemiAnalysis InferenceX v2 的 WideEP。

**结论修正**：第 1 位的理由**只能是"底座 + 依赖前置 + first-party 覆盖 MI355X"**，不能宣称性能收益。真要论证性能，得我们自己在 MI355X 上做 MORI-EP vs 现有 DeepEP 路径的同工况对比——这应当作为动手前的第一个实验，而不是事后补。

**一条强佐证（与性能无关，但坐实了 1→2 的依赖）**：README 2026/03 条目写明 **MORI-SHMEM 已作为后端支撑字节 Triton-distributed 的 EP dispatch/combine kernel**（[PR #164](https://github.com/ByteDance-Seed/Triton-distributed/pull/164)、[PR #145](https://github.com/ByteDance-Seed/Triton-distributed/pull/145)）。这正是排序表第 2 行那个 `ep_all2all_fused.py` 的底座——**两步走的依赖关系由上游自己确认，不是我的推断**。

### ✅ 上面那个"应当作为动手前第一个实验"的对比——已经做了（2026-09-07 当日，MI355X ×8）

上一节写"得我们自己在 MI355X 上做同工况对比"。当天就把 MORI 装上实跑了，结论如下。

安装路径（比源码构建省事很多）：`pip install amd_mori` 直接拿到 **1.2.2 预编译 wheel**；
容器里另需 `apt install libprotobuf32t64 libgrpc++1.51t64`（wheel 不 bundle 这两个）。
用它自带的 `tests/python/ops/bench_dispatch_combine.py`，**仓库必须 checkout 到 `v1.2.2` 与 wheel 对齐**
——git HEAD 的 bench 引用了 wheel 里没有的 `_is_gfx125x`，且 HEAD 才有 `--kernel-type/--warmup/--iters/--routing`
这些 flag，v1.2.2 没有。

#### ⚠ 头号坑：调优配置不会自动生效，必须显式传 launch 参数（差 5–9%）

wheel 里**带了 gfx950 MI355X 的调优表**（`mori/ops/tuning_configs/gfx950_mi355x_IntraNode_ep8_{dispatch,combine}.json`），
但走 `bench_dispatch_combine.py --cmd bench` 时**并不会自动查表**——不传
`--dispatch-block-num / --dispatch-warp-per-block / --combine-block-num / --combine-warp-per-block`
就是默认值，实测因此低 5–9%。表里对我们这个 shape 的记录是
`block_num=256, warp_per_block=8`（dispatch）与 `block_num=128, warp_per_block=4`（combine，zero_copy）。

**这一条对后续任何 MORI 集成都成立：别用默认 launch 参数去评估 MORI。**

#### 复现 README 公布值（同口径：FP8 dispatch + BF16 combine、T=4096）

| | 默认参数 | **调优参数** | AMD 调优表记录 | README |
|---|---|---|---|---|
| dispatch | 308.3 | **332.9** | 334.77 | 345 |
| combine | 385.3 | **422.1** | 419.9 | 420 |

**结论：MORI 公布的量级基本属实。** combine 422.1 复现甚至略超 README 的 420；
dispatch 332.9 与 AMD 自己调优表记的 334.77 只差 0.6%，
但**README 写的 345 高于 AMD 自己调优库里的 334.77**——那 3% 是 README 侧的乐观取值，
以调优表为准更可信。

#### 我们真正关心的训练工况：DSv3、EP8、T=8192/rank、H=7168、E=256（32/rank）、topk=8、**bf16**

| | 延迟 | 带宽 |
|---|---|---|
| MORI-EP dispatch（默认参数） | 1.758 ms | 353.4 GB/s |
| **MORI-EP dispatch（调优参数）** | 1.676 ms | **370.8 GB/s** |
| MORI-EP combine（默认参数） | 1.595 ms | 389.6 GB/s |
| **MORI-EP combine（调优参数）** | 1.460 ms | **425.5 GB/s** |

10 轮 × 8 rank 抖动极小（调优后 dispatch 368.4–373.3，combine 421.5–430.5），可信度高。
注意 T=8192 超出了调优表覆盖范围（表只到 4096），这里是把 4096 的最优参数外推过来用，
仍然拿到 +4.9% / +9.2%，说明**还有进一步 tuning 空间**（`--cmd tuning` 可扫）。

#### 同层对手的正确对照：primus-turbo 树内 deep_ep

**这才是这一节要回答的问题**——MORI-EP 是 dispatcher，它的同层对手是 DeepEP，不是 MegaMoE。
`pip` 装的 `deep_ep` 0.1.0（UCCL kernels）跑不起来（见下），改用 **primus-turbo 0.4.1.dev33
树内的 `primus_turbo.pytorch.deep_ep`**，harness 是 `ep-bench/bench_turbo_deepep_dsv3_ep8.py`。

> ⚠⚠ **本节第一版结论是错的，已推翻重测。** 两个错误：
> (1) **跨节点比**——MORI 跑在 `n04-25`，turbo deep_ep 跑在 `n01-29`，两台机的 RCCL
> 参照差 19–23%，当时却写成"同日同机"；
> (2) **带宽口径不一致**——MORI 的 `disp_total_bytes = total_recv_num_token × hidden ×
> elem_size`（`bench_dispatch_combine.py:444`）**包含自己发给自己的 token**，而我的
> harness 排除了 self 列，两者差 ~15%。
> 下面是把 MORI 装到 `n01-29` 后同机重测、并统一到 MORI 口径的结果。

⚠ **顺带更正一条知识**：`knowledge/libraries/_patterns.md` 记的"primus-turbo 的 `deep_ep/`
目前 gfx942 only，gfx950 等待"**已经过时**——它在 MI355X（gfx950）上编译好且能跑通。

同样必须给它调优机会（`Buffer.num_sms` 默认只有 20，对 256 CU 的 MI355X 严重不足；
`buffer.py` 的 config_map 还带着 DeepSeek 的 H800 调优值和 `TODO: automatically tune`）：

| `num_sms` | dispatch 延迟 ms | combine 延迟 ms |
|---|---|---|
| 20（默认） | 3.575 | 3.820 |
| 32 | 2.606 | 2.826 |
| 64 | 2.182 | 2.449 |
| 128 | 2.067 | 2.313 |
| **256（最佳，需 `--nvl-gb 8`）** | **2.024** | 2.324 |

调优收益比 MORI 那 5–9% 大得多（dispatch 延迟 −43%），但**曲线在 128 SM 就基本饱和**，
128→256 只多 2%。`num_sms` 取 160/192/224 时 SIGABRT（128 与 256 均正常，
原因未查，疑似特定 channel 数下的死锁）；160 以上还需把 NVL buffer 从 2 GB 提到 8 GB，
否则撞 `deep_ep_hip.cpp:616` 的容量断言。

#### 同机（`n01-29`）、同一小时、统一口径的横向表

口径统一为 MORI 的定义：**本 rank 收到的全部 token（含 self）× hidden × elem_size ÷ 耗时**，
GB 按 `1000³`。两侧的 recv token 数实测一致（MORI 43 357 / turbo 43 332，差 0.06%），
所以**延迟列是完全免口径的直接对比**，带宽列只是它的换算。

| 实现（`n01-29`，MI355X EP8，DSv3 / bf16 / T=8192） | dispatch ms | dispatch GB/s | combine ms | combine GB/s |
|---|---|---|---|---|
| **MORI-EP 1.2.2（调优参数）** | **1.676** | **371.0** | **1.460** | **425.6** |
| primus-turbo deep_ep（`num_sms=256`） | 2.024 | 306.9 | 2.324 | 267.3 |
| primus-turbo deep_ep（`num_sms=128`） | 2.067 | 300.5 | 2.313 | 268.6 |
| primus-turbo deep_ep（`num_sms=20` 默认） | 3.575 | 173.8 | 3.820 | 162.6 |
| 朴素 RCCL all-to-all（同体积，参照地板） | 2.108 | 294.7 | — | — |
| DeepEP 0.1.0 / UCCL kernels（pip 包） | **init 期 SIGSEGV** | — | **kernel 挂死** | — |

MORI 在两台机上完全一致（`n04-25` 370.8/425.5 vs `n01-29` 371.0/425.6），
说明它的数不依赖具体节点。

#### ⚠ RCCL 参照线不稳，不要当硬地板用

同一脚本同一配置在 `n01-29` 连跑 7 次：264.3 / 267.7 / 270.9 / **195.6** / 268.4 /
266.2 / **176.8** GB/s（排除 self 口径）。中位数 ~267，但**约 1/4 的运行会掉 25–35%**。
而且 `n04-25` 上是 329.6，与 `n01-29` 中位数差 19–23%——**RCCL a2a 既跨节点不可比、
同节点也不稳**。原因未查（harness 迭代间无 barrier，rank 漂移会累积；也可能是 RCCL
channel 选择）。**结论：RCCL 只能用来判"有没有数量级问题"，不能用来判 10–20% 的胜负。**

#### 结论（已按同机同口径修正）

**MORI-EP 快于 primus-turbo 现有的 deep_ep 路径**：dispatch **1.21×**（1.676 vs 2.024 ms），
combine **1.58×**（1.460 vs 2.313 ms）。

第一版写的 1.39× / 1.82×，以及"turbo deep_ep 连朴素 RCCL 都跑不过"，**都是跨节点 +
口径不一致造成的假象**。同机同口径下 turbo deep_ep（306.9）实际**略高于**同体积
RCCL a2a（294.7），这才是正常的。

这仍然**推翻了本节上游那句"MORI-EP 的性能收益无证据、第 1 位只能靠底座撑"**——
只是收益幅度是 1.2–1.6×，不是 1.4–1.8×。

保留的限制：turbo deep_ep 侧除 `num_sms` 外没有做 per-shape 调优
（config_map 仍是 DeepSeek 的 H800 值），所以这是**当前可用状态**下的差距，
不等于两套 kernel 设计的理论上限差距。另外 turbo 的 combine 是 `pair − dispatch`
差分测的（combine 单独循环会让 rank 失步），MORI 是直接计时，差分法偏保守。

#### MegaMoE 那两个数为什么不放进上表

| MegaMoE | dispatch GB/s |
|---|---|
| 2026-07-16 基准 | 368–381 |
| 2026-09-07 当日（已回归） | 50.8 |

三条限制使它不能与上表并列：(1) 测量边界不同——MORI/DeepEP 的 dispatch 是独立通信
collective，MegaMoE 的是 FlyDSL 融合路径里的一段；(2) 368–381 是 7-16 旧 torch/ROCm
栈的数，不是同日；(3) 两个数都在 `n04-25`，不是上表的 `n01-29`。
**MegaMoE 的对照要等回归修好后在同一台机上重做。**

### RCCLX（在 torchcomms 树内）——压缩集合通信这一格的更强候选

[`meta-pytorch/torchcomms`](https://github.com/meta-pytorch/torchcomms)，**BSD-3-Clause**，2026-09-06 仍在推。RCCLX 是 Meta 增强版 RCCL，没有独立仓库，源码在 `comms/rcclx/develop/`。提供 Direct Data Access（GPU-resident collectives）、**low-precision collectives**、`AllToAllvDynamic`。

我拉了 README 核实 AMD 支持程度——**是一等公民，不是事后 hipify**：

```
TIP: Default builds both gfx942 and gfx950 and can take 1hr+. Narrow to your GPU:
  MI300X/MI325X (gfx942):   ./build_rcclx.sh --amdgpu_targets gfx942
  MI350X/MI355X (gfx950):   ./build_rcclx.sh --amdgpu_targets gfx950
```

**默认同时构建 gfx942 与 gfx950。** 近期 commit 标题里直接出现 wavefront 与 HIP graph（`Reduce the low-precision block absmax within a wavefront`、`Cover low precision under HIP graph capture`、`Carry the low-precision wire format through the pipelined A=4 all-to-all`）。

**与 §0 的 UCCL-Zip 相比**：两者都落在"压缩集合通信"这一格，但 RCCLX 的 AMD 证据更硬（gfx950 默认构建 + wave 级 LP 实现 + HIP graph 覆盖），且 LP all-to-all 直接对应 MoE 训练的 EP 通信。**这一格应以 RCCLX 为首选评估对象，UCCL-Zip 作为对照。** 扣分项：torchcomms 官方自称 experimental，CTran 特性未全部开源，公开数字偏推理。

### ROCm/iris — Triton 原生对称内存

[`ROCm/iris`](https://github.com/ROCm/iris)，MIT，AMD RAD 团队。在 Triton 里提供 SHMEM 式 symmetric heap + `iris.load/store/atomic`，纯 Python+Triton（无 bytecode 包装），明确列 MI300X / MI350X / MI355X。相对 rocSHMEM 的优势是**编译器能同时看见计算与通信**——正是写 super-kernel 需要的。扣分：以单节点为主，multi-node 仍在 roadmap；2026-07-21 之后约一个半月没动。

### 决策相关的"已死"清单

这几条会影响我们对生态的判断，值得单独记：

| 项目 | 状态 |
|---|---|
| `databricks/megablocks` | **实质 dormant**。2026 年唯一提交是 2026-03-25 的 `delete workflows`（拆 CI，弃养信号）；上一实质提交 2025-06-26。⚠ 搜索引擎摘要声称其"2026 仍活跃"，与 commit 历史矛盾 |
| `huggingface/nanotron` | **实质 dormant**，最后实质提交 2025-06-23 |
| `perplexityai/pplx-kernels` | **已废弃**，README 顶部挂 `!!!DEPRECATION!!!`；vLLM 已于 2026-02 移除该后端 |
| `Azure/msccl-tools` | **已归档** 2025-01-28，功能并入 `microsoft/mscclpp` |
| `mosaicml/llm-foundry` | 维护模式。MosaicML 血脉（composer / llm-foundry / megablocks）整体停止创新 |

其余否掉：`microsoft/tutel` 虽活跃且有真实 MI300 支持，但**已转型推理/chat agent**，MoE 训练价值消退；`ThunderKittens` 彻底 NVIDIA-only（tcgen05/TMA/wgmma）；`mirage` (MPK) megakernel 编译器但论文与实现都是**推理**；`Liger-Kernel` 2026 明显向 NVIDIA 倾斜且与 Primus-Turbo 重叠；`nvidia-resiliency-ext` 能力（hang 检测 / in-process restart / straggler / failure attribution）对我们可能是空白，但依赖 NVML+CUDA+NCCL，**只能重写不能移植**；`NVIDIA/TransformerEngine` 无需移植（[`ROCm/TransformerEngine`](https://github.com/ROCm/TransformerEngine) fork 已活跃）。**DeepSpeed 确实仍活跃**（v0.19.6，2026-08-27，约 20 天一版），但内容与我们已有的 offload/ZeRO 重叠——活跃不等于值得移植。

值得单独一提的想法来源：`allenai/OLMo-core`（2026 最活跃的开源训练框架之一）不是组件而是**想法来源**，可挖的具体点是 PP FLOPs 计算修正、ComposableScheduler/PowerLR、把 checkpoint 保存耗时做成 trainer metric、activation-checkpointing 的 determinism check。`microsoft/dion` 的卖点不是"又一个 Muon"，而是**先压缩再正交化**（只通信/正交化动量矩阵的 α 比例子矩阵 + error feedback），属系统侧而非算法侧——但加速 kernel 用 CuTeDSL，AMD 上要重写。

### 合并后的最终排序

| 顺位 | 目标 | 层次 | 理由 |
|---|---|---|---|
| 1 | **MORI-EP + torch SymmetricMemory 后端接进 Primus** | 通信底座 | AMD first-party（**MIT**）、覆盖 MI355X、每周在动、#544 已把 torch 抽象打通、且已是第 2 行那个 kernel 的实际后端。是下面一切的地基。⚠ **不是因为它更快**——上游 benchmark 无基线，MoE EP 侧无任何 vs DeepEP 的对比数据，见 §〇之二 |
| 2 | **照 `ep_all2all_fused.py` 补出 backward** | kernel | 1267 行 MIT 现成 forward，mori_shmem + wave64，切分方式与 MegaMoE 一致。1+2 合起来就是 MI355X 上的 MegaMoE，全程 ROCm-native。⚠ **AMD EP 性能公开数据为零**（`docs/amd-perf.md` 只有 MI308X 的稠密 AG-GEMM/GEMM-RS）；训练侧唯一数字是 UniEP 论文的 **1.03×–1.38×，且在 Hopper 上**。补反向时**必须继承其 deterministic token ordering** |
| 3 | **UltraEP** | runtime（EP 均衡） | §一~§六 结论不变，仍是 runtime 层第一顺位 |
| 4 | **RCCLX LP collectives 评估** | 通信优化 | 能否吃掉 MoE all-to-all 的带宽；gfx950 默认构建 |
| 5 | **MoonEP** | runtime（下一代） | gate Q1 已松动，但 `[E+B,H,H']` 布局约束仍在，见 §四 |

⚠ **C0-4 多机数值 CI 仍是 3/5 两项的硬前置**（见 §五），这一条不因排序变化而放松。

### 📊 自测补白（2026-09-07 当日在 MI355X 上实跑）

上文两处都写了"AMD EP 性能公开数据为零"，于是当天就在 `smci355-ccs-aus-n04-25`（8× MI355X，GPU 实测空闲）的
`xiaoming-dev` 容器里用自家 MegaMoE 立基线。**结果不能直接用于选型，但结论很重要**——详见
[`notes/MegaMoeFlydsl/2026-09-07_1034_...`](../MegaMoeFlydsl/2026-09-07_1034_mega_bf16_dispatch_regression_vs_0716_baseline_mi355x.md)。

同工况（DSv3 / EP8 / T=8192 / H=7168 / E=256 / topk=8 / bf16）：

| | 2026-07-16 自家基准 | 2026-09-07 实测 | MORI 公布（MI355X EP8） |
|---|---|---|---|
| XGMI dispatch | **368–381 GB/s** | 50.8 GB/s | 345 GB/s（FP8） |
| `grouped/dense` | 91.9–100% | 8.7% | — |
| overlap roofline | 87.6–92.1% | 57.1–74.3% | — |
| speedup vs 串行 | **1.46–1.61×** | 0.81–0.99× | — |

两条可用的结论：

1. **MegaMoE 在正确的栈上，XGMI dispatch 368–381 GB/s 是高于 MORI 公布的 345 GB/s 的**（后者还是 FP8，我们是 bf16）。
   也就是说"换 MORI-EP 会更快"这个假设，在自家基准面前更加没有依据——**排序表第 1 位的理由必须坚持是"底座"而非"性能"**。
2. 但**当前容器栈（torch 2.12+ROCm7.15 / turbo 0.4.1.dev33）上存在未定位的严重回归**：正确性全绿（`cos=1.00000`），
   dense GEMM 健康（1230 TFLOPS，与基准一致），却出现 grouped GEMM 塌到 8.7%、XGMI 塌到 50.8 GB/s。
   已排除 autotune 缓存（冷跑同样结果）与 token 规模区间（T 4096→8192 无改善）。
   **在回归定位之前，任何"MegaMoE vs MORI-EP vs Triton-distributed"的横向实验都做不了**——这是那个前置实验的新前置。

> 两轮调研合计覆盖国内厂商（DeepSeek / Moonshot / 字节 / 阿里 / 蚂蚁 / 华为 / 腾讯 / 快手 / 智谱 / MiniMax / StepFun / 百度 / 上海 AI Lab）与国外厂商（AMD / Meta / Microsoft / NVIDIA / AI2 / HuggingFace / Databricks / Perplexity / PyTorch 基金会）。上文凡标注核实的均为 2026-09-07 当日直接抓取；license 字段多处未能确认，动手前需逐个复核。

---

## 一、机制对比

| | **MoonEP** | **UltraEP** |
|---|---|---|
| 出处 | [MoonshotAI/MoonEP](https://github.com/MoonshotAI/MoonEP)，MIT，2026-07-24 随 Kimi K3 开源 | [Dots-Infra/UltraEP](https://github.com/Dots-Infra/UltraEP)，MIT，2026-07-11；论文 [2606.04101](https://arxiv.org/abs/2606.04101)（北大主导 + 小红书） |
| 均衡方式 | **构造性**：在线 GPU planner 规划少量冗余专家 + 预取权重，使每 rank 恰好收到 `S×K` 个 token，与倾斜程度无关 | **反应式**：按 post-gating 的 exact load，每 microbatch 每层复制热专家 + 重路由 token，把 rank 间不均衡从 1.30–4.01 压到 **1.01–1.04** |
| 与 dispatcher 的关系 | **它就是 dispatcher**——替换 DeepEP | **在 dispatcher 之上**——自己仍用 DeepEP（hybrid-ep 分支）做 dispatch/combine |
| 集成形态 | 替换式 | **叠加式**，官方称 <1K 行 |
| 规划器 | 在线 GPU kernel，作者称开销可忽略 | 单个 cooperative thread block / 单 SM，shared memory 暂存负载矩阵，warp 级归约做阈值二分；**全程 device 侧，热路径无 CPU 同步** |

**MoonEP 独有的额外收益**（这是它真正的价值，不在均衡本身）：

- **零拷贝**：token 直接写到远端 rank 上专家分组后的最终位置，permute/unpermute 融进通信，返回通信 buffer 的视图给计算，消掉 comm-buffer → user-buffer 那次主导 epilogue 的拷贝。
- **静态形状**：只需固定的 `S×K` buffer，**每层 MoE 的 host 同步被从构造上消除**。
- **不 OOM**：DeepEP 变动的激活形状会碎片化显存，官方 e2e 曲线显示高不均衡下直接 OOM；MoonEP 形状全静态，任何不均衡度下迭代时间平坦。

**UltraEP 独有的工程性质：**

- 与 token dispatcher 和 grouped GEMM **完全解耦**，独立 Python/CUDA runtime 约 9.6K 行。
- 支持 DP/PP/VPP、激活重算、低精度、**CUDA graph**。
- 跨层复用 replica 的 weight/grad buffer，**运行时零动态分配**。
- 暴露开销 <300 µs，大规模 EP 下占 step 的 1–2%。

---

## 二、硬件依赖与 AMD 落点

这是决定移植成本的部分。

### MoonEP

要 **symmetric memory**：每个专家投影一份连续的 `[E+B, H, H']` VMM range，**在每个 rank 上布局完全相同**；行 `[0,E)` 是所有 rank 的本地专家（每 rank `E/R` 行），物理上就是 home rank 的参数内存，通过 symmetric mapping 映射到各处；行 `[E, E+B)` 是本地预取槽。`reduce_grad` 让每个 rank 把 `R` 份 reduce buffer 映射成一个 `[R, B, H, H']` 视图，**跨 NVLink 远端读**自己专家的梯度槽再本地累加。

AMD 侧底座：**[ROCm/mori](https://github.com/ROCm/mori) 的 MORI-SHMEM 在 MI300X / MI325X / MI355X 上都是 ✅**（MI450X 🚧）。而且 **Primus 的 ODC 按需通信本来就建在 ROCm/MORI + rocSHMEM 上**，这个底座在代码库里已经是已知量。

### UltraEP

数据面用的是 **scale-up 域内的 peer load/store**，不是 RDMA 库：

- 权重分发与梯度归约共用一个执行单元——把专家的 weight/grad 切成固定大小 tile，把 placement plan 编译成 device 常驻的 transfer task，跑一个 **persistent kernel**，thread block 反复从全局 task 流里取下一个 tile。权重分发时 source tile 进一次 shared memory、再 store 到所有远端 replica；梯度归约时从远端 replica load 回来累加到本地。shared memory 双缓冲。
- 热专家 fan-out 超过阈值（4）时建两级 relay，relay 数取 √(|H(e)|−1)，按 chunk 而非整专家转发，chunk ready flag 触发第二级。
- **常驻 thread block 数是显式的资源旋钮**——前向关键路径上多给，反向 overlap 路径上按可配置的 SM 占用封顶，避免和并发的反向计算核抢 SM。

AMD 侧：XGMI 的 peer 映射本来就提供 load/store 语义，DeepEP 的 ROCm 训练路径 Primus 已经有。**这套原语（persistent kernel + tile 流 + shared memory 双缓冲 + ready flag）和 RocMoE 超级核在做的事是同一个家族**，`.cursor/skills/cco-pipeline-overlap/` 里已经沉淀了对应的模式。

> **注意 wave 宽度**：UltraEP 的 quota solver 重度使用 warp 级归约。AMD wave = 64 而 NVIDIA warp = 32，这部分要逐处核对，不是无脑 HIP-ify。

---

## 三、Helios 契合度

**UltraEP 的整个前提就是 rack-scale node（RSN）**——把 scale-up 域从单机 4/8 卡扩到整机架 64+ 卡，使「把整个 EP 组放进一个高带宽域、在热路径上做均衡」第一次物理可行。论文里 **显式点名 AMD Helios**（与 NVIDIA Rubin / UALink 并列）。

它自己写明的边界条件：

> Topology-aware replication. Confining expert replication to the scale-up fabric, UltraEP adapts to available NVLink domain size automatically. **When the NVLink domain is smaller than the EP group, the achievable imbalance floor sits slightly above the ideal 1.0.**

对我们的含义分两个时间点：

| 硬件 | scale-up 域 | UltraEP 状态 |
|---|---|---|
| 今天 8×MI355X 单节点 | 8 卡 XGMI | EP=8 时整组在域内 → **已经落在最佳情形**；EP=32/64 跨节点时降级 |
| Helios / MI455X | **72 卡** | EP ≤ 72 全部在域内 → **满血**，且这正是论文的设计目标 |

也就是说 **UltraEP 的价值会随我们从 MI355X 走到 Helios 而显著上升**，而且现在就能在 EP=8 上验证。这和 [`knowledge/systems/mi455x-parallelism-strategy-reference.md`](../../knowledge/systems/mi455x-parallelism-strategy-reference.md) 的 G2「把 EP/TP 通信关在 scale-up 域内」是同一条准则的两个层次：G2 说把通信关在域内，UltraEP 说**关在域内之后还能在热路径上做均衡**。

MoonEP 没有针对 RSN 设计，benchmark 只有 **H20 / EP=8 单节点**。它的 `B = E/R` 在大 `R` 下反而变小（`E=256, R=64 → B=4`），而且额外显存是"每个投影 `B` 份专家权重，全局而非每层"，所以规模上不构成障碍——但**它在大 EP、多机架上的行为没有公开证据**。

---

## 四、规模与证据强度

| | MoonEP | UltraEP |
|---|---|---|
| 通信 benchmark | H20，EP=8，扫 maxvio，对比 DeepEP v2 | — |
| 端到端训练 | 有曲线（扫不均衡度），未说明规模 | **最多 256 GPU（4 机架），GLM4.5-106B / Qwen3-235B / GLM4.7-358B / DeepSeek-V3-671B，bf16** |
| 生产验证 | Kimi K3（2.8T MoE）；官方称贡献于 2.5× scaling 效率提升，但未拆分 | 论文称真实生产 MoE 训练中稳定保持 >92% 理想吞吐并保收敛 |
| 对照基线 | DeepEP v2 | Megatron-LM、SGLang、EPLB、LPLB、**EPLB+（喂 exact load 的强化版）**、Ideal（强制均衡上界） |
| 参考集成 | vLLM PR [#52101](https://github.com/vllm-project/vllm/pull/52101) 进行中，**BF16 + eager only，无 CUDA graph**，且用 `torch._grouped_mm` 而非调优过的 grouped GEMM | **官方 Megatron-LM fork**（`Dots-Infra/Megatron-LM-UltraEP`），8×Hopper 可跑 demo + Qwen3-235B recipe；SGLang 集成 experimental |

UltraEP 的基线设置更硬——它把 **EPLB 喂上 exact load** 之后仍然赢，这排除了"收益只是来自信息更新鲜"这个解释，验证的是 quota-driven planning 本身。

---

## 五、决定性的工程论据

**MoonEP 要求专家参数的物理布局改成"每个投影一份连续的 `[E+B, H, H']`、每 rank 布局相同、通过 symmetric memory 互相映射"，并且连续性是硬要求（group GEMM 纯按行号寻址）。**

这条穿透三个地方：

1. **distributed optimizer / MFSDP 的分片方式**——参数不再是框架自己切的形状。
2. **grouped GEMM 的权重布局**——MegaMoE 与 Primus-Turbo 的 FP8/MXFP8 grouped GEMM 都有自己的排布假设。
3. **梯度 buffer**——训练要镜像一份 fp32 的 `[E+B, H, H']`，且行 `[E, E+B)` 必须由**独立的 reduce buffer** 支撑、对框架自己的 grad reduce 不可见。

对比之下，UltraEP 只需要"跨层复用的 replica weight/grad buffer + 运行时零动态分配"，且**明确宣称与 grouped GEMM 解耦**。

在 Primus 当前的状态下（C0 阶段，patch 层还在 fail-silent、Turbo 的 CI 与 benchmark 钉不同 commit、多机数值 CI 为零），选一个会重排参数物理布局的方案是错误的排序。

---

## 六、路线

**Phase 0（前置，不可跳）** — roadmap C0-4：2 节点 bf16 + fp8 loss golden，覆盖 EP 开/关。没有它，下面任何一步的"跑通了"都不可信。

**Phase 1 — 移植 UltraEP**
1. 先在 8×MI355X、EP=8 上跑通官方的 Megatron-LM 参考 fork（NVIDIA 侧），确立行为基线与验收判据（rank 间不均衡度、暴露开销、loss 曲线）。
2. HIP-ify 约 9.6K 行 runtime。重点：quota solver 的 warp 级归约（wave 64 vs 32）、tile 流 persistent kernel 的 peer store/load 换成 XGMI 路径、relay 的 ready flag 语义（**这里要过 `knowledge/kernels/memory-access-patterns.md` 的五问，尤其 Q5 push/pull 方向与 XGMI 的出站写 fire-and-forget / 入站读付 RTT 的不对称**）。
3. 接进 Primus 的 Megatron 后端。验收：不均衡度落到 1.0x 区间、暴露开销 <1–2% step、loss golden 不变。

**Phase 2 — 评估 MoonEP 作为下一代 dispatcher**
不急着写代码，先回答三个问题：
1. ~~MoonEP 对 symmetric memory 的依赖能否逐项映射到 MORI-SHMEM？~~ → **已大幅松动，见 §〇之二**：[ROCm/mori #544](https://github.com/ROCm/mori/pull/544) 已把 mori 注册成 **torch SymmetricMemory 后端**，通信底座不再是自研项。剩下的窄问题只有一个：`reduce_grad` 那个 `[R, B, H, H']` 跨 rank 远端读视图，能否用该后端表达。
2. `[E+B, H, H']` 连续布局与 MegaMoE / Primus-Turbo grouped GEMM 的权重排布能否调和，还是必须二选一？
3. 静态形状真的能换来整迭代 HIP graph capture 吗？——这才是 MoonEP 的最大奖励，也是 [`knowledge/systems/industry-training-optimization-2026.md`](../../knowledge/systems/industry-training-optimization-2026.md) §4 那条「sync-free 和整迭代 graph capture 应该合流，二者相乘才是大头」的兑现路径。

如果第 3 问是肯定的，MoonEP 的价值就不止是均衡，值得为它承受布局重排的代价；如果否，UltraEP 就是终局。

> 第一问被 mori #544 削掉大半之后，**MoonEP 的成败几乎完全压在第 2 问（权重布局）上**——它本来就是三问里唯一动 distributed optimizer 与 grouped GEMM 的那个。Phase 2 的调研顺序应据此调整：先查布局，别再花时间在通信底座上。

---

## 七、顺带发现的独立缺口

**Primus 的训练侧 EP 在 ROCm 上仍是 gfx942-only**（`knowledge/libraries/_patterns.md`：primus-turbo 的 `deep_ep/` 目前 gfx942 only，gfx950 等待），而 **MORI-EP 已经支持到 MI355X**（MI300X/MI325X/MI355X 全 ✅）。另注意 DeepSeek 官方 DeepEP 的 [`mori-ep` 分支](https://github.com/deepseek-ai/DeepEP/tree/mori-ep) 只覆盖 **low-latency 模式**，即推理解码路径，不是训练 dispatch。

这个缺口独立于上面任何一个移植决定，且**应该排在两者之前**——否则移植完了也只能在 gfx942 上跑。

---

## 八、待核实

- MoonEP 的代码规模与 kernel 数量（未拉源码统计）。
- MoonEP 端到端训练曲线的具体规模（卡数、模型）——README 只有图，无文字说明。
- Kimi K3 那个 "2.5× scaling 效率" 里 MoonEP 的贡献占比（官方未拆分）。
- UltraEP 与 MoonEP 是否有第三方独立复现（目前都只有作者自评）。
- AcclEP（阿里的 EP 通信库，MoonEP 在致谢里点名）是否开源——尚未找到仓库。

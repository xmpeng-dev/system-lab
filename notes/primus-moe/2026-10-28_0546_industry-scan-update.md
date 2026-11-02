# 业界进展扫描 2026-09-11 → 2026-10-28

> **When**: 2026-10-28 05:46 UTC+8
> **Where**: 桌面调研(GitHub / arXiv / 官方 release notes 当日抓取)+ 本地 Primus / Primus-Turbo 只读代码核对;无新上机实验
> **Context**: 续 [`2026-09-11_1541_industry_opt_port_to_rocm_primus.md`](./2026-09-11_1541_industry_opt_port_to_rocm_primus.md)。三路只读 subagent 并行扫描(仓库活性 / arXiv / vendor 与上游框架),关键结论由我手工复核

## TL;DR

1. **上一轮的 vendor gap 表有两条是读了已废弃仓库得出的。** `ROCm/rccl` 与 `ROCm/hipBLASLt` 在 GitHub 上现在都是壳,描述分别写着 `[DEPRECATED] Moved to ROCm/rocm-systems repo` 与 `[DEPRECATED] Moved to ROCm/rocm-libraries repo`。读活仓库后,四条 vendor gap 有**三条要改写** —— 最重要的是 **RCCL 2.30.7 已经有 GIN device API 并随 ROCm 10.0.0 发布**。
2. **[2026-09-03 roadmap](./2026-09-03_primus-roadmap-2026q4-2027h2.md) 的第三条支柱「护城河已被实证」塌了一半。** TorchTitan 的 ROCm 立场反转:`main` 的 README 有 `AMD GPU (ROCm)` CI 徽章、安装说明给 `rocm10.0` wheel、MI350X golden 在 10-20 还在刷新。**而且它已经把 Primus 缺的那套可信度设施做齐了。**
3. **本窗口唯一测在 MI300X 上的 MoE 通信论文,对我们的增量比初读时小。** Expert Coupling 的两个机制里,去重我们走 DeepEP 时本就有,token shuffling 的前提 `TP == EP` 在我们 293 份 recipe 里为真的数量是 0。
4. **格式缝那一格:别人把活干了。** Meta 的 LP-FA4 在生产训练里把 MXFP8 量化融进前一个 kernel 的 epilogue,并给出了 `[32,32]` 方块 scaling 让量化对转置不变这个技巧;dist_moe 的 `weight_block_size="32x32"` 是同一件事的第二个独立实现。
5. **低精度梯度通信窗口仍然是 0 篇,而且有反向佐证。** Ai2 的 Olmo-core 3 明说梯度仍在 FP32 bucket 里归约,MXFP8 只用于 forward 与 dgrad。

---

## 1. Vendor gap:三条要改写

### 1.1 方法论错误先说

9-11 那一轮把 RCCL 与 hipBLASLt 的状态记在了**不再接受开发的仓库**上:

| 仓库 | GitHub 描述 | 实际开发位置 |
|---|---|---|
| `ROCm/rccl` | `[DEPRECATED] Moved to ROCm/rocm-systems repo` | [`ROCm/rocm-systems`](https://github.com/ROCm/rocm-systems)(默认分支 `develop`) |
| `ROCm/hipBLASLt` | `[DEPRECATED] Moved to ROCm/rocm-libraries repo` | [`ROCm/rocm-libraries`](https://github.com/ROCm/rocm-libraries) |

「RCCL 停在 backport NCCL 2.28 算法、没有 Gin」这个结论是读死壳的产物。

### 1.2 改写后的表

| Gap | 9-11 记录 | 今天(2026-10-28) | 动了? |
|---|---|---|---|
| **Device-initiated EP / Gin** | 无 Gin,停在 NCCL 2.28 算法 backport | **RCCL 2.30.7 带 GIN device API**,两个 device-initiated 后端(Anvil SDMA、rocSHMEM GDA),对齐 NCCL GIN API v14;随 **ROCm 10.0.0(2026-08-26)** 发布 | **是,基本关闭** |
| **Copy-engine collectives(RCCL)** | 只有 mori 的 SDMA AllGather,无框架 API | **CE AllReduce 已有**(有维护 commit 为证);CE ReduceScatter 仍是开着的 PR([rocm-systems#12310](https://github.com/ROCm/rocm-systems/pull/12310),09-25 开) | **是,部分** |
| **Copy-engine collectives(mori)** | 无框架可消费的 API 面 | **`ccoSdma` 完整 put/get 已暴露给 FlyDSL 和 Triton**([#792](https://github.com/ROCm/mori/pull/792), 10-20);SDMA 默认编译([#710](https://github.com/ROCm/mori/pull/710));EP v2 AsyncLL dispatch/combine 跑在 cco 上并支持跨节点([#768](https://github.com/ROCm/mori/pull/768), 10-27) | **是** |
| **hipBLASLt grouped GEMM FP8** | "ROCm 7.1 穷举后只有 BF16" | **这个基线站不住**:[hipBLASLt#1316](https://github.com/ROCm/hipBLASLt/pull/1316)「FP8 TN Compute for Grouped GEMMs」**2024-11-07 就合了**;[rocm-libraries#13581](https://github.com/ROCm/rocm-libraries/pull/13581) 正在给 grouped GEMM C API 加 MX scaling | **基线有误** |
| **packed FP4/MXFP8 param all-gather** | FSDP2 fp8 只在 Flux 路径 | 未找到任何 ROCm 侧对等物 | 否 |

**Gin 剩下的障碍比想象中窄,而且是机械问题。** [`ROCm/rccl#2190`](https://github.com/ROCm/rccl/issues/2190)(09-09 开)报告 ROCm 10.0 确实装了 `nccl_device.h` 并导出了 `ncclDevCommCreate` / `ncclPutSignal` / `ncclWaitSignal` / `ncclCommWindowRegister` / `ncclMemAlloc`,但头文件在 host gcc 下编不过(`__builtin_nontemporal_store` 没加守卫、`min` 未限定),在 hipcc 下触发 `--extended-lambda` 的 static_assert。这正是 PyTorch 那个 [RCCL symmetric memory PR #192524](https://github.com/pytorch/pytorch/pull/192524) 从 08-07 开到现在还没合的原因。**能力在,卡的是头文件卫生。**

另外 `rocm-systems` 里合了 [#10891](https://github.com/ROCm/rocm-systems/pull/10891)「Enable AMD host VMM for DeepEP-style elastic windows」—— 是明确的 DeepEP 形状使能。

### 1.3 窗口内新开的 gap

NCCL [v2.32.3-1(2026-09-17)](https://github.com/NVIDIA/nccl/releases) 落地、ROCm 侧未找到对等物的:

| 新能力 | 是什么 |
|---|---|
| CFT counted-write and wait | Compute Fabric Transport 的 device 侧计数写 |
| socket-based GIN | GIN 走 TCP socket,无需 NIC 专用后端即可开发 device kernel |
| ring-based hierarchical CE AllGather | `NCCL_HIER_CE_COLL_AG_RAIL_RING_ENABLE`([#2299](https://github.com/NVIDIA/nccl/pull/2299)) |
| `ncclSetEncryption` | NCCL 自有 socket 流量的可选 TLS |
| `launchCompletionEvent` | 逐 collective 的 launch-completion event |
| CuTe DSL ReduceCopy device API | nccl4py v0.6.0(09-23);**RCCL 的绑定是 nccl4py v0.2.0 的 fork —— 这条 gap 在窗口内变宽了** |

方向性判断:NVIDIA 窗口内的工作(socket GIN、hierarchical CE AllGather、RMA/GIN 拆分)大体是在**把 device API 从专有传输层上抽离**,这让 RCCL 更容易跟而不是更难。RCCL 2.30.7 在 NCCL 2.30.7(2026-06-04)之后约四个月就适配了 GIN/RMA v14 拆分,**结构性滞后现在是一到两个 NCCL minor 版本,不是能力鸿沟。**

---

## 2. 上游框架:TorchTitan 反转,Megatron 的扩展点仍在变

### 2.1 TorchTitan 的 ROCm 立场反转(roadmap 支柱 ③ 的问题)

9-03 roadmap 的第三条核心判断是:上游 TorchTitan 停掉 ROCm CI、删掉 ROCm golden loss、把 ROCm 排除在发版验证之外、并把 AMD 用户指向独立 fork —— 据此判定护城河比预想的宽。我拉了 `main` 的 README 原文核对,这条**已经不成立**:

- Test status 表里有一行 **`AMD GPU (ROCm)`**,挂着 `integration_test_rocm.yaml` 在 `main` 的徽章
- 安装说明:"Replace `cu132` with another CUDA version or a ROCm build (e.g. **`rocm10.0`**)"
- AMD fork 现在只是 `**[2025/11]**` 的一条旧 news,不再是 ROCm 用户的推荐路径
- MI350X golden 在 10-20 还在刷新([#5557](https://github.com/pytorch/torchtitan/pull/5557) DeepSeek V3 CP MI350X golden refresh;[#4848](https://github.com/pytorch/torchtitan/pull/4848) 10-07 更新 MI350X loss baseline)

**更要紧的是它顺手把我们缺的可信度设施做齐了。** README 的 "Debugging and observability" 一节列着:`scripts/loss_compare.py` 跨 commit 比 loss、deterministic mode、**SDC replay**(`torchtitan/observability/SILENT_DATA_CORRUPTION.md`)、structured logging、fake-backend 单卡干跑大布局、Flight Recorder。对照 [10-12 delta](./2026-10-12_0101_primus-roadmap-update.md):Primus 的 C0-2(patch fail-loud)、C0-4(多机数值 CI)、C0-5(恢复被注释的测试)**在同期全部零位移**。

两条附带:TorchTitan 现在有真的 RL 栈(TitanRL,GRPO/DAPO + bitwise trainer/generator parity),这让 Primus `README.md:7` 那句 "reinforcement learning workflows" 的对照更难看;Features 里还有 **Dist-MoE**(见 §4.1)。

### 2.2 Megatron-LM

| 项 | 状态 |
|---|---|
| Megatron-Core 版本 | **core_v0.19.2(2026-09-18)**;NGC `26.09-lambda.rc1`(10-13)。v0.19.2 本身很薄:TE 2.17→2.18 + FP4 进 1F1B A2A overlap([#6135](https://github.com/NVIDIA/Megatron-LM/pull/6135)) |
| **`BackendSpecProvider` 变了** | [#6766](https://github.com/NVIDIA/Megatron-LM/pull/6766)(**09-10 合**,比 9-11 基线早一天)把所有 spec builder 改成问 `BackendSpecProvider`,PR 正文说移除了 60 处 `HAVE_TE`/`HAVE_APEX`/`HAVE_FUSED_LAYER_NORM`/`LNImpl` 与 7 处 TE 直接 import。**还有一次接口变更在路上**:[#4087](https://github.com/NVIDIA/Megatron-LM/pull/4087)「Refactor BackendSpecProvider to use Protocols」仍开着 |
| `GPTModel` → `HybridModel` | **推进了但没完成**。已合:GPT→Hybrid checkpoint 翻译([#5792](https://github.com/NVIDIA/Megatron-LM/pull/5792))、HybridModel 的 DSv4 层构造([#6405](https://github.com/NVIDIA/Megatron-LM/pull/6405))。仍开:Core 内的 HybridModel 一等配置列表([#7555](https://github.com/NVIDIA/Megatron-LM/pull/7555))及其训练接线([#7153](https://github.com/NVIDIA/Megatron-LM/pull/7153)) |
| `--fp4-param-gather` / GTP | 强化并扩展:现在**要求** dist optimizer([#8009](https://github.com/NVIDIA/Megatron-LM/pull/8009), 10-10);推理侧流式 checkpoint 反量化([#7405](https://github.com/NVIDIA/Megatron-LM/pull/7405), 10-16);CP 轴折进 GTP_remat 权重分片组([#6886](https://github.com/NVIDIA/Megatron-LM/pull/6886), 10-14);开着的 `fp4_dense_mxfp8`([#8581](https://github.com/NVIDIA/Megatron-LM/pull/8581)) |

对我们的直接后果:[10-12 delta](./2026-10-12_0101_primus-roadmap-update.md) 记的 C1-1 是「`PrimusTurboSpecProvider` 存在但仍靠 monkey-patch 安装,且带 `backend_versions=["<0.17"]`」。上游在 09-10 已经把这个接口改过一次、还有第二次在路上,而我们的 Megatron pin 冻在 **2026-03-06**。**C1-4 的窗口在以每次上游接口变更为单位地关闭。**

### 2.3 PyTorch

窗口内:v2.14.0(09-02)、v2.14.1(09-30);2.15 未发布。与训练系统相关的:

- **ROCm 上的 grouped GEMM**:"Support grouped and FP8-scaled grouped GEMM Triton lowering on compatible ROCm hardware"(#188600、#188742)—— 目前最接近可移植 FP8 grouped GEMM 的东西
- ROCm symmetric memory 维护:NCCL symm-mem 后端的 `barrier()`(#188051)、重复 `hipMemMap` 失败(#188673)、VA 复用时的陈旧 signal pad(#189088)。注意这是**与仍未合的 RCCL-backed 后端不同**的另一条路径
- 一条回归护栏:NCCL symm-mem pool 不再自动升级 `register_mem_pool(..., symm=True)` 之后分配的 segment —— 因为注册迟到的 segment 可能在持有 allocator 锁的情况下只在部分 rank 上发起 collective,导致不可恢复的挂死(#192112)

---

## 3. Expert Coupling:降级,但保留一半

完整笔记:[`../../papers/expert-coupling.md`](../../papers/expert-coupling.md)。

[arXiv 2610.09372](https://arxiv.org/abs/2610.09372)(Zyphra,10-07)是本窗口**唯一**把基线测在 **8×MI300X/节点**上的 MoE 通信论文,xGMI 全互联 + 每 GPU 一张 100 Gb/s RoCE。它量化了我们最想要的那个数:A2A 占 step 时间 EP8/top-2 **13.5%**、EP32/top-2 **44.8%**、EP8/top-6 **24.0%**、EP32/top-6 **60.4%**,并给出两个**不改路由决策、不改专家参数、不加副本**的机制。初读时我判它为"本窗口对我们最有价值的一篇"。**代码审计后这个判断要降级。**

### 3.1 去重 dispatcher:我们已经有了

论文自己就写明去重是 DeepEP 的已有技术,他们只是补进 Megatron-LM(Megatron 的 `alltoall` dispatcher 确实每个被选专家发一份拷贝)。我们的情况:

| dispatcher | 启用方式 | 每 token 发几行 | 去重? |
|---|---|---|---|
| Megatron `MoEAlltoAllTokenDispatcher`(**论文的基线**) | `moe_token_dispatcher_type: alltoall` | 每 **(token, 专家)** 一行 | **否** |
| **Primus-Turbo DeepEP**(`flex`) | `use_turbo_deepep: true` → patch 强制 `moe_enable_deepep=True` + `moe_token_dispatcher_type='flex'`,并把 `MoEFlexTokenDispatcher` 重绑为 `PrimusTurboDeepEPTokenDispatcher`(`primus/backends/megatron/patches/turbo/moe_dispatcher_patches.py:59-72`) | 每 **(token, 目的 rank)** 一行 | **是,早就是** |
| **MegaMoE 超级 kernel** | `use_turbo_mega_moe: true`(默认 false,门控 `1 < ep_size <= 8` + bf16) | 每 **(token, topk slot)** 一行 | **否** |

决定性证据(我手工核过 `third_party/Turbo/csrc/kernels/deep_ep/layout.cu`):

```
rank_idx = expert_idx / num_expert_per_rank - rank_begin_idx;
is_in_rank[rank_idx]++, is_in_rdma_rank[rank_idx / NUM_MAX_NVL_PEERS]++;
...
shifted_is_token_in_rank[j + rank_begin_idx] = (is_in_rank[j] > 0);
num_tokens_per_rank_per_thread[thread_id][j] += (is_in_rank[j] > 0);
```

先按 rank 累加,再写 `(is_in_rank[j] > 0)` —— **一个 token 在同一 rank 上选中 3 个专家只计 1 行。** 跨节点路径在 `internode.cu` 里进一步按 RDMA rank 折叠(`total_count += (is_token_in_rank_uint64 != 0)`)。

`use_turbo_deepep: true` 出现在 `qwen3_235B_A22B`、`qwen3_30B_A3B`、`deepseek_v3`、`kimi_k2`、`glm5`、`mixtral_8x7B`、`minimax_m3` 等全部旗舰 MI355X MoE recipe 里。**所以论文基线里"去重带来的那部分"(EP8 top-2 去 6%、top-6 去 25%)在我们这儿已经计入基线。**

MegaMoE 侧有一个 `dedup_src_row` 区域在 `symm_buffer.py:78,591,770` **被声明了但全树没有任何 kernel 引用** —— 预留未实现的基础设施。

### 3.2 Token shuffling:前提在我们这儿恒为假

它硬性要求 **`TP == EP`**。我把 `examples/**/configs/**` 下所有声明 `expert_model_parallel_size` 的 YAML 跑了一遍(解析 `${PRIMUS_TP:1}` 式默认值):

| `(TP, EP)` | 份数 |
|---|---|
| `(1, 1)` | 173 |
| `(${PRIMUS_TP:1}, ${PRIMUS_EP:8})` | 69 |
| `(1, 8)` | 36 |
| `(8, 1)` | 6 |
| `(${PRIMUS_TP:1}, 8)` | 4 |
| 其余(全部 `EP=1`) | 6 |

**`TP == EP > 1` 的份数是 0;`TP > 1` 与 `EP > 1` 同时成立的份数也是 0。** 带 `EP>1` 的 109 份全部是 `TP=1`;所有 `TP>1` 的都是 dense 模型配 `EP=1`。MegaMoE 路径上更是直接 `assert config.tensor_model_parallel_size == 1, "MegaMoE is EP-only (TP==1)"`(`primus/backends/megatron/core/extensions/mega_moe.py:310`)。

这不是"难"或"需要工作量",**是前提在 293 份 recipe 里全假**,而论文明说它不适用于 TP1。要让它适用得先把生产 MoE 搬到 `TP=EP=8` 布局 —— 论文自己的数据显示那本身就是吞吐倒退(TP 的集合通信抬高非 MoE 部分,top-2 下 A2A 只占 step 的 9–21%,step 收益封顶在 1.06×)。

### 3.3 剩下的那一半:相关性放置,而且落地极便宜

这是唯一真正的增量,好处是它对布局没要求(任何用 EP 的布局)、专家显存零增量(求解器每步都是 swap,每 rank 恒持 `E/EP` 个专家)、不改路由。

**落地形态是论文作者自己选的那个:per-layer 的全局专家 ID 置换,checkpoint 权重按全局 ID 索引。** 而我们的设备侧映射恰好全是硬编码 contiguous:

| 位置 | 公式 |
|---|---|
| `third_party/Megatron-LM/megatron/core/transformer/moe/moe_layer.py:124-132` | `local_expert_indices_offset = ep_rank * num_local_experts` |
| `third_party/Turbo/csrc/kernels/deep_ep/layout.cu:94` | `rank_idx = expert_idx / num_expert_per_rank - rank_begin_idx` |
| `third_party/Turbo/primus_turbo/flydsl/mega/fp8/dispatch_prologue.py:471` | `destination_rank = expert_id // experts_per_rank` |
| `primus/backends/megatron/core/extensions/mega_moe.py:167-172` | `first_expert = ep_rank * self.experts_per_rank` |

**所以在上游置换专家 ID,现有 contiguous kernel 就自然实现了相关性放置,零 HIP 改动。**

统计采集也只差一步:`primus/backends/megatron/core/transformer/moe/router.py:74-77` 已有一个逐 microbatch 的 hook,但累的是 `routing_map.sum(dim=0)` 这个**边际**计数,论文要的是成对共选矩阵 `routing_map.T @ routing_map`。`routing_map`(bool `[tokens, experts]`)就在那一行的作用域里。转储骨架可以克隆 `primus/backends/megatron/patches/kimi_k3_expert_load_probe_patches.py`(config 门控、逐 step 逐层写 JSONL,25 步一次全量数组)。论文的采集需求极轻:一个 rank 的一个 microbatch、几千 token、约每 1000 step 重拟合一次。

另外确认:Primus 全树 **没有 EPLB、没有冗余专家、没有放置计划生成器、没有负载驱动迁移**。`moe_router_force_load_balancing` 看起来像但不是 —— 它用 `arange(...) % num_moe_experts` 强制轮询,**改变路由决策**,代码里自己标为 "Benchmark aid"(`v4_moe.py:334`)。

### 3.4 为什么量级预期应该保守

论文自测的一条负面结果直接压在我们主力场景上:**单节点 top-6 时加 token shuffling 的 A2A 收益 1.46×,低于只做放置的 1.48×**,机理是"在 xGMI 上集合通信时间由最忙的那对 GPU 决定,而不是跟随聚合字节数"。同一现象在 EP8 top-2 表现为字节降 1.36× 但时间只降 1.16×。这是 Infinity Fabric 的性质而非 CDNA3 特有,**应预期 MI355X 同样成立**。

叠加三件事 —— 我们从已去重基线出发、我们默认 EP8 单节点(论文的 1.41× 是 top-6 EP16 两节点)、MI355X + FP8/MXFP8 进一步压缩 step 里的计算侧因而 A2A 占比更小 —— **诚实估计是个位数百分比低位的 step 收益**。这篇真正能付钱的配置是多节点 EP(EP16/EP32),不是我们今天 recipe 的样子。

把它和 §4.3 的"重叠不免费"放在一起看:这是**第四次**独立观察到"通信优化的收益不按字节线性兑现"(前三次是 Tessera 的 20 SM 稳定罚 10–20%、Motif-3 的"消除优于隐藏"、Instella-MoE 在 64K+CP 阶段关掉重叠)。

---

## 4. 仓库活性(9-11 → 10-28)

| 仓库 | 最后 push | 变化 | 判断是否改变 |
|---|---|---|---|
| [`ROCm/mori`](https://github.com/ROCm/mori) | 10-28 | v1.2.4(10-12);SDMA 默认编译、cco put/get 暴露给 Triton/FlyDSL、EP v2 AsyncLL 跨节点;新增 UMBP 子系统与多 NIC OFI/libfabric 后端 | 不变,**加强** |
| [`meta-pytorch/torchcomms`](https://github.com/meta-pytorch/torchcomms) | 10-27 | **整库 deprecated**(10-27,[#4532](https://github.com/meta-pytorch/torchcomms/pull/4532),功能上游进 c10d);NCCLX 在 OSS 构建中弃用([#4455](https://github.com/meta-pytorch/torchcomms/pull/4455));**CTran 被删除而非开源**([#4498](https://github.com/meta-pytorch/torchcomms/pull/4498)) | **推翻** |
| [`Dots-Infra/UltraEP`](https://github.com/Dots-Infra/UltraEP) | 10-15 | **解冻**。[#2](https://github.com/Dots-Infra/UltraEP/pull/2)(10-11)暴露求解器的放置图并接受外部放置;[#4](https://github.com/Dots-Infra/UltraEP/pull/4)(10-15)删掉 `nvshmem.cuh` / `ipc_manager.cu` / `runtime.cpp`,新增 `ultra_ep/symm_mem.py`,换成 **torch SymmetricMemory** | **推翻** |
| [`ByteDance-Seed/Triton-distributed`](https://github.com/ByteDance-Seed/Triton-distributed) | 09-18 | [#208](https://github.com/ByteDance-Seed/Triton-distributed/pull/208) 改为对 stock Triton 3.7 的 **out-of-tree plugin**,in-tree 构建文件删除;LLVM hash 从 `8957e64a` 移到 `1f126a6d`;新增 fused MXFP8 intra-node dispatch | **部分推翻**:构建模型变了;**EP 反向仍然没有**(`ep_all2all_fused.py` 现 1,298 行,无 `backward`、无 `autograd.Function`) |
| [`deepseek-ai/DeepEP`](https://github.com/deepseek-ai/DeepEP) | 10-21 | **V2.5**([#763](https://github.com/deepseek-ai/DeepEP/pull/763), 09-29)。`ElasticBuffer` 拆成 `EPBuffer`/`EngramBuffer`/`PPBuffer`/`BucketBuffer`;新增 `lb_prefetch_weights`/`lb_reduce_grads` 动态冗余专家(README 致谢 MoonEP 与 UltraEP);**V1 与 NVSHMEM 彻底删除**;门槛抬到 **CUDA 13.1+ / PyTorch 2.10+ / NCCL 2.32.3+ / Hopper+**。非 NV 支持走**另开的** [`DeepEP-Ascend`](https://github.com/deepseek-ai/DeepEP-Ascend) 仓库 | 确认:别追 |
| [`MoonshotAI/MoonEP`](https://github.com/MoonshotAI/MoonEP) | 10-12 | 09-20 "Public Release 26/09" 重写成完整训练库(`dispatch`/`combine`/`prefetch_weight`/`reduce_grad`,反向路径齐全) | 设备列表**仍然只有** NVIDIA + 真武 PPU |
| [`cursor/mixture-of-kittens`](https://github.com/cursor/mixture-of-kittens) | 10-01 | 只改 README。**Apache-2.0**,SM100/SM103,Python 424KB / CUDA 307KB;代码里 `hip`/`rocm` 命中 **0**,ROCm/AMD 相关 issue 与 PR 均为 **0** | 确认:读不搬 |
| [`inclusionAI/linghe`](https://github.com/inclusionAI/linghe) / [`infinigence/FlashOverlap`](https://github.com/infinigence/FlashOverlap) | 04-28 / 01-20 | 无 | 确认休眠 |

**UltraEP 是本窗口最有价值的仓库变化。** mori 从 08-26 起就是 torch SymmetricMemory 后端([mori#544](https://github.com/ROCm/mori/pull/544)),UltraEP 现在把 NVSHMEM 依赖换成了同一个抽象 —— "HIP-ify 9.6K CUDA" 的前提变了。但边界要说清:UltraEP README 仍写 SM90/SM100、`TORCH_CUDA_ARCH_LIST` 只给 `9.0`/`10.0`,仓库里 rocm/hip/amd 命中为 0,唯一 release tag 仍是 `v1.0.0`(07-17)。**依赖障碍没了,但 AMD 支持既没声称也没证明。**

`torchcomms` 被弃用要反着看:**RCCLX 本身没被弃,窗口内 AMD 侧还在推** —— gfx950 HIPIPC BF16 all-reduce([#4435](https://github.com/meta-pytorch/torchcomms/pull/4435), 10-15)、registered/captured HIPIPC workspace、`RCCL_EP_STREAM_DISPATCH`/`RCCL_EP_STREAM_COMBINE` 门控的 intranode EP 流式 dispatch/combine([#4453](https://github.com/meta-pytorch/torchcomms/pull/4453), 10-16)。但 9-11 排的"评估 RCCLX"这一项要重新定义:**别依赖 torchcomms 的 API 面**。

**AcclEP 仍未开源。** 唯一公开产物是 SIGCOMM'26 论文 [Balancing and Beyond: Communication-Centric Optimizations in Expert Parallelism](https://ennanzhai.github.io/pub/sigcomm26-acclep.pdf)(Jiamin Cao 等,阿里云,DOI [10.1145/3789240.3829201](https://doi.org/10.1145/3789240.3829201))。两个要留意的细节:**论文正文里系统叫 EPIC,"AcclEP" 只出现在托管文件名里**;§8 说他们的 CCL 是"built on DeepEP",不是从零造。与我们相关的一句:他们的推理集群"spans principal clusters across NVIDIA, **AMD**, and in-house GPU families",参考文献里引了 `ROCm/rccl`。

### 4.1 新仓库:`meta-pytorch/dist_moe`

窗口内唯一实质性的新仓库。[`meta-pytorch/dist_moe`](https://github.com/meta-pytorch/dist_moe),09-18 建仓,**BSD-3-Clause**,11 个 commit,无 release。

**先把它的身份说准,因为很容易读错。** 它**没有**进 `pytorch/pytorch`,**也不是** TorchTitan 的默认依赖。TorchTitan 的 `torchtitan/models/common/dist_moe/README.md` 原文:"Dist-MoE is **optional and released independently from TorchTitan**",要手动 `pip install "git+https://github.com/meta-pytorch/dist_moe.git@main"`;"Ordinary TorchTitan imports and recipes **do not import** Dist-MoE. A selected Dist-MoE recipe performs a **guarded import**…**TorchTitan's base requirements intentionally do not install this Blackwell-only backend**"。`requirements.txt` 里也没有。TorchTitan 全文称它为 **"the annex"**(附件),这个用词本身就是定位。

**它真正重要的地方不是"又一个 CUDA MoE 库进了官方栈",而是 TorchTitan 为此定义了一个厂商可插拔的 routed-expert 契约,dist_moe 是第一个插进去的东西。** 契约面有名字、有文档:

| 插入点 | 做什么 |
|---|---|
| `DistMoeTransform` | 直接替换 `RoutedExperts.Config`,但**保留 TorchTitan 的 `GroupedLinear` 参数布局**(`w13.weight [E,2,F,D]`、`w2.weight [E,D,F]`)→ 原文 "preserves parameter, optimizer, FSDP, and checkpoint ownership" |
| `TrainingEngine.Config.dist_moe_runtime` | 基类型 `Configurable.Config \| None`,不选就不构造 |
| 一个 "narrow forward-context registration capability" | 引擎把最终 model parts、topology、liveness schedule 交给 runtime,**不加 model kwargs、不要求自定义 pipeline-stage 子类** |
| GraphPP 的 slot resolver | tracing 时把 `activation_slot_id_1` 作为**显式 graph input**,一个 stage 一张图,无 per-microbatch 图变体 |
| `output_postprocess` | 仍是普通 TorchTitan 模块,参数/优化器/checkpoint/FSDP 所有权不变 |

**这正是 C1-1 / C1-2 的形状,也正是我们没做的那件事。** 对照:Primus 把 `PrimusTurboSpecProvider` 用 monkey-patch 覆写进 Megatron 的 4 个模块,每次上游改 `BackendSpecProvider`(§2.2:09-10 已改一次,第二次在路上)都要跟;TorchTitan 定义一个 transform + 一个可选 runtime 配置字段,外部包自己插。**所以这件事的正确读法不是「又多了一个要对标的 kernel」,而是「上游示范了一遍厂商 kernel 该怎么接,而我们还在打 patch」。**

它确实出现在 verified recipe 里(Features 列 "Dist-MoE: fused dispatch, expert compute, and combine on Blackwell";Supported models 表里 DeepSeek V3 那行是 `deepseek_v3_671b (+ Dist-MoE, NVFP4)`),但该打的折扣:dist_moe README 自己写 NVFP4 training = No,所以那行里的 NVFP4 多半是 TorchTitan 自己的 `torchtitan/quantization/nvfp4/`,别把两个读成一个。

**会不会以后并进 pytorch?** 有先例但大概率不适用。`meta-pytorch` 确是孵化器 —— `torchcomms` 就在 10-27 整库弃用、功能上游进 c10d(§4)。但 dist_moe 是 CUTLASS DSL + SM100-only 的 kernel 包,`pytorch/pytorch` 核心不会为单一架构 ship 一套 CuTe DSL kernel,**判断是长期保持独立可选包形态** —— 这是推断,无公开声明。

支持矩阵(README 原文):

| 精度 | Pipeline | 训练 | 推理 |
|---|---|---|---|
| BF16 | Standard | 是 | 是 |
| MXFP8 E4M3 | Staged 或 Mega | 是 | 是 |
| NVFP4 | Staged 或 Mega | **否** | 是 |

**卡在哪(对我们):**

1. **最硬的一条不是 CUDA,是 CUTLASS DSL。** kernel toolchain 写明 `Triton 3.8 or 3.9; NVIDIA CUTLASS DSL 4.6.1`。GitHub 把它报成 100% Python 很容易误读成"可移植性好",其实相反 —— 不可移植性从 `.cu` 搬进了一个 DSL 依赖,而 **DSL 是 hipify 不了的**,只能照语义在 FlyDSL 上重写。从移植角度这比一份 CUDA C++ 更难。
2. **通信基底**:context 拥有 "symmetric device memory used to exchange routed rows and gradients with expert-parallel peers"。映到 AMD 就是 mori / torch SymmetricMemory —— 而 RCCL 撑的那个后端(§1.2)还卡在头文件。
3. **VMM 几乎不卡**:它要的是"一段稳定虚拟地址 + pinned host 溢出",HIP 有 VMM API,且 `rocm-systems#10891` 已经合了 DeepEP 形状的 host VMM 使能。
4. **硬性门槛**:`Linux ARM64; NVIDIA Blackwell (SM100 class)`、`PyTorch 2.14.0+cu130`、`CUDA toolkit 13.0.3.0`、`NCCL 2.30.7`,且 "execution has no CPU fallback";连验证安装都要两张 Blackwell。**我们连行为基线都建不起来** —— 这点和 UltraEP 不同(后者至少能在 NVIDIA 参考 fork 上先跑基线再 HIP-ify)。

**它自己也还缺不少**,README 的 "Not supported in this release" 里有两条对我们有利:**MXFP4 不支持**(我们这条线领先:ALTO 配方 + Turbo 的 `ea35b348` fused MXFP4 wgrad 累加 + `a48595c8` de-osc),**communication quantization 不支持**(就是 Motif-3 那条"EP dispatch 之前做 MXFP8"的位置,官方栈现在是空的)。另外整仓没有一个吞吐数字。

**真正值得偷的是执行模型,不是 kernel:**

| 设计 | 原文要点 | 对应我们的格子 |
|---|---|---|
| **SyncFree 执行模型 + 设备侧 planner** | "the host launches one fixed sequence. Device predicates decide whether the recompute kernels perform work; the host does not branch on the planner result" —— 动态重算与 CUDA graph 兼容 | **C3-7「sync-free ∪ HIP graph」**。我们的障碍正是逐节点 Python dispatch 和那几处 `.item()`/`tolist()` D2H,它给的解法是"host 序列固定、分支下沉成 device predicate" |
| **三级容量预算,软硬限分明** | activation slot 是**软限**(超了退化成重算);total scratch 是**硬限**(超了由"第一个知道最终本地接收行数的 routing kernel"打印所需/可用行数后 trap);有纯函数 `plan_memory()` 与 `plan.explain()` | MegaMoE 的 symm buffer 没有 plan 阶段、没有 explain、溢出行为未定义 |
| **`weight_block_size="32x32"`** | "preserves training's quantized weight values while retaining only forward operands" —— 即方块 scaling 的转置不变,反向不二次量化。**而且量化发生在 FSDP 的 post-all-gather hook 里**:FSDP 以 BF16 all-gather 持久分片 → hook 准备 "grouped 32x32 MXFP8 qdata and FPROP/DGRAD scale layouts" → FSDP 释放临时 BF16 通信存储 | **格式缝的第三个独立实现**,且位置特殊 —— 不在 GEMM epilogue 也不在 norm epilogue,**在 FSDP unshard 边界上**。见 §5.3 |
| **归约顺序对形状敏感是被写进文档的** | "In batch-invariant mode calls are raised to more than 128 rows: **at or below 128 rows the annex's post-expert top-k reduction uses another, equally correct, accumulation order**, so a token's output would depend on how many tokens share its step" | 与 `HANDOFF_mxfp8_topk_grad.md` 同类(归约顺序对形状敏感),但他们把它当成**已知、写进文档、有明确规避手段**的性质 |
| **in-place WGRAD accumulation 被定义成显式契约** | "opt-in serialized-writer contract",附 `parameter.grad_dtype` × `Config.wgrad_dtype` 解析表,并警告 "Read WGRAD accumulation before integrating it with a distributed optimizer or pipeline schedule" | 与 `Primus/HANDOFF_mxfp8_topk_grad.md` 那个"topk 非 2 幂 + ≥2 µbatch ⇒ dW 爆"直接同构 —— 多次 backward 往同一块 buffer 累加。**他们把这当成需要显式契约的危险操作,我们这边是隐式的** |

**判定:偷思路,不搬代码,且暂时连基线都做不了。** 门槛不是工作量,是 CUTLASS DSL 无 ROCm 对等物 + 手上没有 SM100 机器。

其他新仓库:[`deepseek-ai/DeepEP-Ascend`](https://github.com/deepseek-ai/DeepEP-Ascend)(09-30,无 license 文件,Ascend 950DT / CANN 9.2.0,dispatch EP8 373–375 GB/s 降到 EP128 313–320 GB/s)—— 对 AMD 无关,但它是 DeepSeek 处理非 NVIDIA 硅的模板:**另开一个 fork 仓库,不是在主线加后端**。Prime Intellect 在 [10-23 的博客](https://www.primeintellect.ai/blog/prime-mega-moe)声称给 DeepGEMM 的 forward-only Mega MoE 加了重叠反向(MoE kernel 最高 3.11×、端到端对 Prime-RL 1.34×),**但对应的开源代码找不到**。

---

## 5. 论文与两个被跟踪的窗口

### 5.1 其余值得读的

| 论文 | 日期 | 机构 | 关键数字 | AMD? | 类目 |
|---|---|---|---|---|---|
| [ALTO / Training an MoE LLM in MXFP4](https://arxiv.org/abs/2610.32229) (2610.32229) | 10-23 (v2 10-26) | **AMD**(Barsoum, Khodamoradi, Dong Li) | GPT-OSS-20B MXFP4 val loss **3.3350** vs BF16 3.3283;到 MLPerf 目标 3.34 的步数开销 **+20.0% → +5.0%** | **原生(MI300X)** | 偷思路 |
| [MegaFlux](https://arxiv.org/abs/2610.00671) (2610.00671) | 09-30 | NVIDIA + Stanford/Princeton | 8×B200,147 配置:fwd **1.45×** / bwd **1.28×** 几何均值(相对同 megakernel 固定放置,峰值 2.14×/2.64×);backward 里藏掉 **91–100%** 的"副本权重传输 + 副本梯度归约"合计开销 | 无 | 偷思路 |
| [Low Precision Flash Attention 4](https://pytorch.org/blog/low-precision-flash-attention-4-end-to-end-block-scaled-attention-for-blackwell/) | 09-23 | Meta | B200 上 MXFP8 FA4:fwd **2.85 PF/s**、bwd **2 PF/s**;内部 shape 上 fwd 1.6× / bwd 1.52×。**在生产训练里** | 无(tcgen05/TMEM) | 偷思路 |
| [Broken Symmetry in BF16 Attention](https://arxiv.org/abs/2609.34272) (2609.34272) | 09-28 | Rutgers + CMU + NYU + MBZUAI + Oracle | 450M 模型 50B token:健康到 25B token,之后梯度范数涨 **1000×**、**零 NaN**、最终 loss 比 FP32 attention 高 **0.2 nats**。GProj 把 query/key 梯度误差中位数 219%/13% 压到 0.34%/0.37%,代价 +4.7% step 时间 | 无(但 bug 是格式级不是厂商级) | 偷思路 |
| [HyDra](https://arxiv.org/abs/2609.34318) (2609.34318) | 09-28 | SJTU + 腾讯 + 南大 | 生产 256K 上下文 >11K 卡 Megatron-Core DCP:**同 token 数下 per-rank microbatch 时间差最多 5×** → 46% 流水线气泡 + 13% DP 气泡;2,048 卡生产作业上气泡 36%→14%,吞吐 1.25× | 无 | 偷思路 |
| [Architecting for Hardware Diversity](https://arxiv.org/abs/2610.22230) (2610.22230, MICRO'26) | 10-15 | **Meta + AMD**(69 作者) | MI350X 相对直接移植的 NVIDIA 基线端到端训练吞吐最高 **1.5×**;利用 MI350X 相对 B200 的 1.6× HBM 容量 | **原生(MI350X)** | 偷思路 |
| [Olmo-core 3](https://huggingface.co/blog/allenai/olmocore3) | 10-01 | Ai2 + UW | 512×B300 上 1.200T 总参 / 58.36B 激活:**858 useful-model TFLOP/s/GPU**,38.1% MFU;MXFP8 相对 BF16 +21%(4 卡受控基准) | 无 | 偷思路 |
| [Echo](https://arxiv.org/abs/2610.14277) (2610.14277) | 10-09 | 作者含 Zhouchen Lin, Kim-Chuan Toh | A100 上 1GB 时比 BF16 NCCL all-reduce 快 **2.09×**;训练 step 1.09×(空闲 fabric)/ 1.38×(有背景流量);10B token 预训练 final loss 比 BF16 高 <0.9%。**不需要 FP8/FP4 硬件** | 无 | 偷思路 |
| [RailWave](https://arxiv.org/abs/2610.03415) (2610.03415) | 10-02 | 中大 / 东南 / Monash / NUS / 人大 | GLM-4.5-Air(106B)训练导出的通信负载:H800 上最高 **5.84×**、H20 上 **4.36×**(相对原生 DeepEP);[代码](https://github.com/CyberSecurityErial/RailWave-EP) | 无(建在 DeepEP 上) | 搬代码 |
| [BaGuaLu-2](https://arxiv.org/abs/2610.23064) (2610.23064) | 10-16 | 清华(翟季冬组) | 1040B 参数 / 12,288 GPU,**2.012 EFLOPS**,35% MFU;**2.9% 的迭代 OOM,重启 284 s = 正常迭代的 16×** | 摘要未点名厂商 | 偷思路 |
| [Flattening Every Memory Peak in Long-Context MoE](https://arxiv.org/abs/2609.14306) (2609.14306) | 09-13 | Salesforce AI Research | MoE 120B–667B @ 1M 上下文:上下文触达 8–32×、吞吐最高 10.4×(相对调优过的 FSDP2);dispatch 峰值 **−59.3%**、vocab projection 峰值 **−86.6%**;**loss/梯度精确不变** | 摘要未点名 GPU | 偷思路 |
| [mKernel](https://arxiv.org/abs/2609.13585) (2609.13585) | 09-11 | UC Berkeley / Sky | 两个 16 卡 H200 集群:GEMM+AllReduce **1.72×**、Ring Attention **1.88×**。**报告 IBGDA 相对 host-assisted GPU-initiated 通信收益甚微** | 无 | 偷思路 |
| [MAI-Base-2](https://microsoft.ai/news/frontier-pre-training-at-microsoft-ai/) | 10-15 | Microsoft AI | 2.8T 总参 / 80B 激活,50T token,16k GB200,38 天,21% 平均 MFU,**89% goodput**;MoE 层 MFU 从 24% 抬到 35% | 无 | 读 |

两篇特别点名:

- **ALTO 是 AMD 自己的 MXFP4 配方**,而且它报了**负面结果** —— differential gradient estimation、outlier clipping、macro-block scaling 三条在算子级 SNR 上都有增益但端到端没用,**替我们省掉三个死胡同**。重要 caveat:是 MI300X 上的 **simulated MXFP4**(量化后立刻反量化再走高精度 GEMM),数值验证了、wall-clock 明确没优化,**gfx950 原生 MXFP4 路径仍然没测过**。
- **Broken Symmetry 的失效签名和你手上那个 MegaMoE bug 同构**:训练前期健康、某个点之后梯度范数暴涨、**全程无 NaN**、loss 只是不降。它的两个独立成因都是格式级的(FlashAttention-3 forward softmax 里一个已知未修的 FMA;BF16 舍入 score 梯度后行和非零,把 key 均值漏进 query 梯度,且泄漏随后期 attention 变尖锐而增长)。诊断方法零成本:**把两层的 attention backward 用 FP32 重算一遍,看多出来的梯度是否消失。**

### 5.2 窗口 (a):低精度梯度通信 —— 仍然 0 篇

针对 `"gradient reduce-scatter"` / `"quantized all-reduce"` / `"FP8 gradient"` / `"low-precision gradient"` / `"compressed collective"` 的窗口内全文检索只回 **1 条**,还是 [TACO](https://arxiv.org/abs/2610.02199) 这篇优化器状态显存的论文,不是通信论文。

三个近似项都不占这一格:

| 论文 | 为什么不占 |
|---|---|
| [Echo](https://arxiv.org/abs/2610.14277) (2610.14277) | 压的是**张量并行的 activation all-reduce**,明确围绕"两个相依矩阵乘之间的 all-reduce",不是 DP 梯度路径 |
| [GeoMesh](https://arxiv.org/abs/2609.18388) (2609.18388) | 交换符号压缩的 pseudo-gradient(~32×),但只在**地理分布 WAN + 异构 GPU** 场景,基线是 65.8% 同步开销的体制 |
| [Ring-Based Coded Approximate All-Reduce](https://arxiv.org/abs/2609.23898) (2609.23898) | 编码理论构造 |

**反向佐证最有力:** Olmo-core 3 把自己的设计明说成相反方向 —— 参数梯度 "accumulate in dedicated FP32 bucket views, reduce in FP32, and feed FP32 main weights",MXFP8 只限于 forward 与 dgrad。**2026 年 10 月,一个出货的 MoE 栈的 state of the art 仍然是 FP32 梯度归约。** 9-03 立的 C3-5 窗口还开着,没被抢跑,也没有再拖的理由。

### 5.3 窗口 (b):quant 融进 GEMM epilogue —— 关了,而且是被生产关的

Meta 的 [Low Precision Flash Attention 4](https://pytorch.org/blog/low-precision-flash-attention-4-end-to-end-block-scaled-attention-for-blackwell/)(09-23)原文:"To eliminate the quantization overhead, we **fuse the MXFP8 quantization into the epilogue of the preceding kernel**",实现为 fused RMSNorm+Quantize 与 GEMM+Quantize,一趟出双 scale layout,**在 Meta 生产 GEM 训练里跑**,开源在 [`lp_fa4`](https://github.com/facebookresearch/ads_model_kernel_library/tree/main/lp_fa4)。

三条可迁移的点子,全是调度决策而非架构:

1. **把量化融进前一个 kernel 的 epilogue**
2. **用 `[32,32]` 方块 scaling 取代 `[32,1]`/`[1,32]` 双向量化** —— 量化对转置不变,反向只需量化一次。**他们明说先试了在 epilogue 里融双向量化,慢**
3. **"zero-gather jagged"**:FP8 数据留在未 pad 的位置,只 scatter 小得多的 scale factor

窗口内两个佐证:Olmo-core 3 把 MXFP8 量化融进 NVSHMEM PUT kernel,让 dispatch 直接携带 qdata+scales 供 expert grouped GEMM 消费;NVIDIA TE [v2.14](https://github.com/NVIDIA/TransformerEngine/releases/tag/v2.14) 加了 MXFP8 的 fused GEMM+SwiGLU grouped MLP(覆盖 MoE fwd/bwd)与 NVFP4 的 fused row-cast/RHT/transpose/column-cast。**三者都依赖 Blackwell/tcgen05,所以想法可迁、代码不可迁。**

对 [career-strategy 的 quant-GEMM seam](../career-strategy/2026-08-12_1505_the-quant-gemm-seam.md) 的影响,叠加 [10-12 delta §2](./2026-10-12_0101_primus-roadmap-update.md) 记的 `eb7c31da`(#1208,在 attention QKV 路径把 RMSNorm→MXFP4 量化做成 norm epilogue):**论证这个模式可行这件事已经不需要你来做了。** 现在缺的只有 MoE / FP8 wgrad 路径上那一段 colwise requant 的实现,以及 `experts.py:303,343` 那两处 `tolist()` D2H。**独占性在下降,立项阻力也在下降。**

### 5.4 窗口 (c):scale-up 域的 MoE 成本模型 —— 仍然只有 Mixture-of-Kittens

窗口内未找到任何在 NVL72/NVL144/Helios 级单跳域上建 MoE 成本模型的论文。最接近的三个都没过线:[ArchAgent](https://arxiv.org/abs/2610.23724)(2610.23724,Argonne,覆盖 UALink 与 Infinity Fabric,但是 1MW 预算下混合 AI+HPC 负载的 agentic 架构搜索,无 MoE 成本模型)、[2609.37828](https://arxiv.org/abs/2609.37828)(MoE **推理**的拓扑/并行/拥塞控制仿真)、[2610.20354](https://arxiv.org/abs/2610.20354)(GB 统一内存放置微基准,不涉 MoE)。

**[`../../papers/mixture-of-kittens.md`](../../papers/mixture-of-kittens.md) 仍然是唯一一篇 scale-up 域 MoE 成本分析,而 AMD Helios 侧的对等物还没有人写。** 结合 §6 的硬件进展,这是一个明确的空位。

---

## 6. 硬件:Helios 的部署进度

9-07 与 9-11 两轮都把 Helios 当成已知的未来(UltraEP 与 Mixture-of-Kittens 都点名它)。窗口内的增量是**部署,不是发布**:

- Helios 本身在 **2026-07-23 发布**(先于上一轮扫描窗口):72× MI455X 在**一个 scale-up 域**内经 UALoE fabric 互联,**260 TB/s** scale-up 带宽;配 6th Gen EPYC "Venice" 9006 与 Pensando 网络
- 窗口内新闻:AMD **2026-10-27** 的新闻稿说首批 Helios 机架**已在 Meta 安装并在测试中**;Meta 是 MI450 架构的早期定义客户,并将部署一款定制 Instinct GPU,合作延伸到未来的 MI500 系列

两条必须带上的限定:

1. **Helios 是参考设计不是在售产品**(AMD 自己的 FAQ 这么写),量产部署预期 2H 2026;Meta 的机架措辞是"being tested"
2. **所有对比性能数字(vs Vera Rubin NVL72)都是 AMD Performance Labs 在预生产硬件上的投影。** 引用 72 卡拓扑可以当事实,**吞吐声称按厂商建模对待**
3. UALoE 是 **UALink over Ethernet**,经四个 cartridge 与交换 tray 的 ASIC,所以"单跳"应读作"单一 scale-up fabric 域",**不等同 XGMI 的直连延迟**。未找到公开的 UALoE 逐跳延迟数据

对我们的含义:**"AMD 的 scale-up 域是 8 卡 XGMI"这个默认假设在规划 2027 时不能再用。** 一个 72 卡单跳域会定性地改变 EP 成本结构 —— 而 §3.4 那条"xGMI 上集合时间由最忙 GPU 对决定"的机理在 UALoE 上是否成立,是个开放问题,也正好是 §5.4 那个空位的内容。

---

## 7. 对计划的影响

### 7.1 要改的既有判断

| 文档 | 条目 | 改成 |
|---|---|---|
| [9-11 §3.4](./2026-09-11_1541_industry_opt_port_to_rocm_primus.md) vendor gap | "RCCL 无 Gin" | RCCL 2.30.7 有 GIN device API(ROCm 10.0.0);卡在 `nccl_device.h` 的编译卫生 |
| 同上 | "hipBLASLt 只有 BF16 grouped GEMM" | FP8 TN grouped GEMM 2024-11 已合;gfx950 的 dtype/layout 矩阵仍未知 |
| [9-11 §3.3](./2026-09-11_1541_industry_opt_port_to_rocm_primus.md) 搬代码顺位 #2 | "RCCLX 低精度 collective 评估" | torchcomms 整库已弃用;RCCLX 本身仍活,但**不要依赖 torchcomms API 面** |
| 同上 顺位 #3 | "UltraEP 仓库 2026-07-17 后再无 push,可 fork 但别等上游" | 已解冻,且 **10-15 换成 torch SymmetricMemory**;但仍无 AMD 声称或证明 |
| [9-11 §4.4](./2026-09-11_1541_industry_opt_port_to_rocm_primus.md) 仓库活性表 | Triton-distributed "LLVM 锁死我们的容器" | 构建模型已改为 stock Triton 3.7 的 out-of-tree plugin,LLVM hash 变了;**是否真能链通需要本地重建验证**;EP 反向仍然没有 |
| [9-03 roadmap](./2026-09-03_primus-roadmap-2026q4-2027h2.md) 支柱 ③ | "TorchTitan 停 ROCm CI、删 golden、指向 fork ⇒ 护城河比预想宽" | **反转**:ROCm CI 在 `main` 上绿、README 有 AMD 徽章、MI350X golden 10-20 还在刷新;且它已把我们缺的可信度设施做齐 |
| [10-12 delta](./2026-10-12_0101_primus-roadmap-update.md) C2-8 | "完成 —— Turbo 原生 `hipblaslt_grouped_gemm_fp8`" | **判错**。`HipblasltGroupedGemm::run()` 是把 `num_gemms` 个 group **逐个**调普通 `hipblasLtMatmul`、扇到 4 条 stream / 8 个 handle 上用 event 汇合(`csrc/kernels/grouped_gemm/hipblaslt_grouped_gemm.cu`),**没有调用任何 grouped-GEMM 入口**。roadmap 承诺的"单次 API 调用消掉全部 dispatch 开销"**没有拿到** |

### 7.2 排序增量(叠在 10-12 delta 的 R1–R5 之上)

**R1 不变,而且证据更强。** Broken Symmetry 给了"静默错梯度"的第二个独立案例,dist_moe 把多次 backward 累加 wgrad 定义成需要显式契约的危险操作 —— 两条都支持"把 MegaMoE mxfp8 bug 变成 C0-4 入场券"。额外可白嫖的诊断:**把两层 attention backward 用 FP32 重算,看多余梯度是否消失**。

**R3(patch fail-loud / `targets=`)的紧迫性上升了一档。** 上游 `BackendSpecProvider` 在 09-10 已变更一次、第二次(#4087)在路上,而我们的 Megatron pin 冻在 2026-03-06、194 个 patch 里只有 2 个声明版本门。**这把锁不只是越来越难开,锁住的门后面在持续改动。**

**R4(格式缝)的独占性在下降,应加速。** 见 §5.3。

**新增 R6:相关性专家放置,小、干净、零 HIP 改动。** 见 §3.3。做法:在 `router.py:76` 旁加一个 `routing_map.T @ routing_map` 的共选统计(克隆 kimi_k3 probe 的转储骨架)→ 离线跑 KL 风格局部搜索出 per-layer 专家 ID 置换 → 在加载时置换全局专家 ID。**验收标准要先定在通信量而不是 step 时间上**,因为 §3.4 的理由让 EP8 单节点的 step 收益预期只有个位数百分比低位。它与 EPLB 正交(§3.7 的 router 消融证明放置打的是通信量、不是倾斜),所以不与 UltraEP 冲突。

**新增 R7:把 TorchTitan 的可信度设施逐项对到 Primus 的 C0 空格。** `loss_compare.py`、deterministic mode、SDC replay、fake-backend 干跑 —— 这四项在上游是 BSD-3 的 PyTorch-native 代码,比我们从零立项便宜得多,而且"上游已经有了"本身就是最好的立项论据。

**不要做的新增项:** 搬 dist_moe(§4.1,CUTLASS DSL + 无机器);追 DeepEP V2.5(门槛已到 CUDA 13.1 / NCCL 2.32.3 / Hopper+,`mori-ep` 分支自 2026-02-05 起静止且其所 fork 的 V1 基座已被删除);把 MoonEP 排进本季度(设备列表仍无 AMD,尽管已重写成完整训练库)。

---

## 相关文件

- 上一轮业界扫描:[`2026-09-11_1541_industry_opt_port_to_rocm_primus.md`](./2026-09-11_1541_industry_opt_port_to_rocm_primus.md)
- EP 选型:[`2026-09-07_0900_moonep-vs-ultraep-port-selection.md`](./2026-09-07_0900_moonep-vs-ultraep-port-selection.md)
- Roadmap 与执行复盘:[`2026-09-03_primus-roadmap-2026q4-2027h2.md`](./2026-09-03_primus-roadmap-2026q4-2027h2.md) · [`2026-10-12_0101_primus-roadmap-update.md`](./2026-10-12_0101_primus-roadmap-update.md)
- 本轮新增 paper note:[`../../papers/expert-coupling.md`](../../papers/expert-coupling.md)
- 格式缝:[`../career-strategy/2026-08-12_1505_the-quant-gemm-seam.md`](../career-strategy/2026-08-12_1505_the-quant-gemm-seam.md)
- scale-up 域成本结构:[`../../papers/mixture-of-kittens.md`](../../papers/mixture-of-kittens.md)

## 核查溯源

三路只读 subagent,2026-10-28;其中 DeepEP 去重 kernel、Primus recipe 的 TP/EP 分布、TorchTitan README、Turbo 的 `hipblaslt_grouped_gemm`、两篇 AMD 相关论文的 ID 与数字由我手工复核。

- 仓库活性扫描 —— [`03fc34f0-5e84-4828-99d3-168563df7777`](03fc34f0-5e84-4828-99d3-168563df7777)
- arXiv 扫描 —— [`e3a152ac-f5b1-441e-945f-52e328febfcf`](e3a152ac-f5b1-441e-945f-52e328febfcf)
- vendor / 上游框架扫描 —— [`10048260-85c9-4d5a-889c-42eaaf3e3a8e`](10048260-85c9-4d5a-889c-42eaaf3e3a8e)
- Expert Coupling 深读 + Primus 接入点审计 —— [`0cd20b39-184d-41db-a711-495101224b55`](0cd20b39-184d-41db-a711-495101224b55)

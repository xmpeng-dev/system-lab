# papers — 论文阅读笔记

每篇论文一个文件(或子目录,当有复现脚本/补充材料时)。
slug 是 kebab-case;年份只在需要消歧义时加。

> 100+ 篇论文的全景分类索引在 [`../knowledge/moe/paper-landscape.md`](../knowledge/moe/paper-landscape.md);
> **arXiv 2025–2026 训练/推理优化速查**见 [`../knowledge/systems/training-optimization-landscape-2026.md`](../knowledge/systems/training-optimization-landscape-2026.md);
> **2026-08 增量扫描（27 篇待读 + 本轮精读的 5 篇）**见 [`../knowledge/systems/arxiv-digest-2026-08.md`](../knowledge/systems/arxiv-digest-2026-08.md);
> 这里只列**已有详细笔记**的论文。

## 论文清单

### 训练优化

| Paper | 发表 | Topic | 一句话结论 | File |
|---|---|---|---|---|
| Expert Coupling | arXiv'26 | MoE A2A 通信量削减 | Zyphra;**本窗口唯一把基线测在 8×MI300X/节点的 MoE 通信论文** —— A2A 占 step:EP8/top-2 13.5%、EP32/top-2 44.8%、EP8/top-6 24.0%、EP32/top-6 60.4%(EP32+top-6 下超过「专家计算+全部非 MoE 计算」之和)。预训练早期 router 已成形两类相关性:层内共选(0.8% 的专家对承载 42% token,独立路由只给 1.6%)与跨层可预测(128 个专家里 112 个把 ≥30% token 送往下层同一专家,中位 48%)。**相关性放置 + 去重 dispatcher** 去掉最多 58% 的 dispatch 行;**token shuffling** 把搬移融进 SP 的 reduce-scatter(不新增 collective),单节点本地率 12.5%→59%。A2A 时间 1.16–2.63×、step 最高 1.41×,**不改路由、不改专家参数、不加副本**。⚠ **对 Primus 增量只有一半**:去重我们走 DeepEP 时本就有(`layout.cu` 写 `(is_in_rank[j] > 0)`,论文自己承认去重是 DeepEP 技术);token shuffling 要求 `TP == EP`,我们 293 份 recipe 里为真的是 **0**。⚠ **关键负面结果:单节点 top-6 加洗牌的 A2A 收益 1.46× 低于只放置的 1.48×** —— "xGMI 上集合时间由最忙 GPU 对决定,不跟随聚合字节",这是「通信优化不免费」的第四次独立观察,且正压在我们 EP8 单节点主场景上。可用的那半落地极便宜:置换全局专家 ID 即可,零 HIP 改动 | [`expert-coupling.md`](./expert-coupling.md) |
| BASP | HPEC'26 | 序列并行通信(Clemson) | `N=K·B` 时把 N 卡切成 B 个不相交的 K 卡组、每组包一条序列,Ulysses 的全局 N 路 all-to-all 变成 B 个并行的 K 路 a2a;每卡 token 数 `BS/N` 与显存不变。**核心是让 K = 每节点卡数,把 a2a 完全关在 NVLink 内、不走 IB**。8 卡 A100(2×4)上 Llama/Qwen 端到端 **1.17–1.32×**,a2a 时间降 2.23–3.10×,loss 曲线重合。⚠ **不建议引用**:Table I 的 `SP=N/B` 基线疑似非等 batch 比较——等全局 batch 下 `DP=B × Ulysses-SP=K` 就已给出 BASP 那一行的全部指标(每卡 `BS/N`、a2a 跨 `N/B`),而论文称该配置"sequences are no longer split across GPUs"是明确错误的;若相对该组合真有差异,只可能是连续 rank 分配这个拓扑映射修复。⚠ 节点仅 4 卡而 `K≤4` 恒成立,**「组变小」与「不出节点」两个机制全程不可分离**;摘要里的 `85×` a2a 加速处 `K=1`,collective 已被消掉而非加速,该配置本身已退化为纯 DP。⚠ 无代码、无误差棒、无 MFU;Ring/Striped/USP/FlexSP 只在 related work 提及无一对比。**唯一可用的是序列扫描曲线**(1K/2K/4K 仅 5.7%/3.4%/1.8%,8K 跳到 13.1%,16K 18.2%,32K 25.9%),独立佐证我们 G5「序列 ≥8K 才开 CP」;Helios 72 卡单 scale-up 域下其 win region 基本消失 | [`basp.md`](./basp.md) |
| MoEBlaze | MLSys'26 | MoE 训练内存 | 元数据索引不物化 routed token + SwiGLU 融合 + SiLU backward 重算,单 H100 单层 vs Megablocks 内存↓4× / 加速≤6.2×;[html](./moeblaze.html) | [`moeblaze.md`](./moeblaze.md) |
| LAER-MoE / FSEP | ASPLOS'26 | MoE 并行 | FSEP 全分片专家并行 + 动态重排,1.69× 端到端 | [`laer-moe-fsep.md`](./laer-moe-fsep.md) |
| SwiftMoE | arXiv'25 | MoE 训练 | 参数-优化器解耦 + 动态 Expert 放置,+30.5% 收敛 | [`swiftmoe.md`](./swiftmoe.md) |
| MemFine | arXiv'25 | MoE 内存 | 细粒度 chunk 激活调度 + 选择性重计算,48% 内存↓ | [`memfine.md`](./memfine.md) |
| MoE Parallel Folding | arXiv'25 | MoE 并行 | 五维混合并行,Attn/MoE 解耦,49.3% MFU | [`moe-parallel-folding.md`](./moe-parallel-folding.md) |
| Comet | MLSys'25 | MoE 通信重叠 | shared-tensor 依赖分解 + thread-block 专用化(单融合 kernel),1.96× 单层 / 1.71× 端到端;[html](./comet.html) | [`comet.md`](./comet.md) |
| MegaScale-MoE | EuroSys'26 | MoE 大规模训练 | MoE 层锁节点内 + SP(attn)/EP(FFN) + inter/intra-op overlap + 通信压缩,1440×H800 训 352B 达 1.41M tok/s / 1.88× vs Megatron;[html](./megascale-moe.html) | [`megascale-moe.md`](./megascale-moe.md) |
| UniEP | arXiv'26 | MoE MegaKernel 训练 | Dispatch+GroupGEMM / GroupGEMM+Combine 融成单 kernel,SM 动态角色 + token scoreboard + 确定性映射保 bit-wise,vs COMET 1.03–1.38×;开源可移植 AMD;[html](./uniep/uniep.html) | [`uniep/`](./uniep/README.md) |
| Mixture-of-Kittens | arXiv'26 | MoE MegaKernel（NVL72） | Stanford+Cursor;精读版。**scale-up 上朴素 NCCL 在 21/48 格是最强基线**(集中在 EP16 小 token);三洞察=按算子 pull/push、可调 minibatch 粒度、ring buffer 去 CPU sync;层最高 **2.37×**(Table 1 最大 2.74×)、512×GB300 E2E **1.41×**;Apache 2.0。从 Figure 4 反推 push 纯搬运比 pull 快 5–9%,pull 只赢在信号,与 UniEP 不矛盾。对 MegaMoE:每次 forward 的 D2H 正是它要消的同步;CLC 解决的是"collective 有没有 SM 跑",不是我们 L1 的"能不能按时发出" | [`mixture-of-kittens.md`](./mixture-of-kittens.md) · [全文中译](./mixture-of-kittens-cn.md) |
| UltraEP | arXiv'26 | MoE 负载均衡(RSN) | Rack-Scale 节点上 exact-load 实时(每 microbatch/层)均衡,quota planner + RSN-native 通信,不均 1.3–4→1.0,训练 1.42×/serving 1.56×;点名 AMD Helios;[html](./ultraep/ultraep.html) | [`ultraep/`](./ultraep/README.md) |
| DisagMoE | arXiv'26 | MoE 训练 overlap | attention/FFN 解耦到不同 GPU 组 + AF-Pipe(all-to-all→M2N 一等流水阶段)+ roofline/MILP 分 GPU/NIC,up to 1.81× vs Megatron;[html](./disagmoe/disagmoe.html) | [`disagmoe/`](./disagmoe/README.md) |
| Piper | arXiv'26 | MoE 训练(AMD/HPC) | Frontier(MI250X+RCCL+Dragonfly)上资源建模 + PP×EP 局部化通信 + 拓扑感知 all-to-all + expert migration,2–3.5× MFU vs X-MoE;[html](./piper/piper.html) | [`piper/`](./piper/README.md) |
| AutoOverlap | arXiv'26 | comm-compute 编译器 | communication chunk 抽象 + 源到源 Triton 编译器自动 kernel 内细粒度 overlap(后端/chunk/tile 三维自适应),平均 1.3× 最高 4.7×;[html](./autooverlap/autooverlap.html) | [`autooverlap/`](./autooverlap/README.md) |
| FlowMoE | NeurIPS'25 | MoE 流水线 | 统一流水线调度 + chunk 优先级,-57% 训练时间 | [`flowmoe.md`](./flowmoe.md) |
| Megatron-Core MoE | -- | 工程参考 | NVIDIA 官方 MoE 实现细节(grouped GEMM / token dispatcher / load balance) | [`megatron-core-moe.md`](./megatron-core-moe.md) |
| veScale FSDP | -- | 分布式训练 | veScale 的 FSDP 设计与实现要点 | [`vescale-fsdp.md`](./vescale-fsdp.md) |
| DMuon | arXiv'26 | 分布式优化器 | Owner-centric Muon + Gram SYRK NS + MILP LB,FSDP2 drop-in 3 行,端到端 avg +2% vs AdamW / optim 6.85–163× vs Muon-AG | [`dmuon.md`](./dmuon.md) |
| MatrixFSDP | arXiv'26 | 分布式优化器 | 改 ZeRO-3 分片放置(每矩阵一个 owner 持整块)让 backward 归约天然落在 owner,optim step 零矩阵集合通信;64×A100 optim 4.2×→54.6× / E2E 1.37×→2.15×;不支持 TP、未谈 MoE | [`matrixfsdp.md`](./matrixfsdp.md) |
| AGoQ | arXiv'26 | 低精度梯度通信 | 激活近 4-bit + 8-bit 梯度,把 AllReduce 改成 A2A→本地 FP32 归约→AG 以避开"通信中做低精度加法";显存 −52%、1.34×(8B–32B LLaMA, ≤64 卡)。**但要打折看**:52% 里梯度量化只占 2.4 GB(5.2%),1.34× 几乎全来自激活省显存换掉重算(R=10→0)且该配置 DP=2、重构基本没起作用;3.4× 全来自位宽而非结构,从未做原生低精度 RS 对照组;Table 7/Table 4/Eq.21 三处内部矛盾。**最关键**:该结构上游已出货(`--gtp-remat-reduce-scatter-with-fp32-accumulation`,"same bytes on the wire"),ZeRO++ qgZ(2306.10209, 2023)更早发表且多 2-hop;**真缺口是格式(通信库无块缩放 dtype)不是结构**,真约束是**尾数宽度**(加 W 个数需 log₂W 位,W=72→6.2 位 vs FP8 E4M3 的 3 位) | [`agoq.md`](./agoq.md) |
| DynamiQ | SIGCOMM'26 | 低精度梯度通信 | 把单 chunk 的 RS 抽象成 in-arborescence,每跳误差正比子树大小(ring `O(εSM²n³)` / butterfly `O(εSM²n²)`);按 super-group 在**聚合后**梯度里的量级分 2/4/8/16 位 + 层次化 scale + 相关舍入 + decompress-accumulate-recompress 融合 kernel(瓶颈是 HBM 带宽非线速)。LLaMA-1B MMLU 达 BF16 的 99% 时比 MXFP8 快 34.5% / 比 BF16 快 40.8%;**消融里变量位宽分配单项 3.5–5.1×,超其余三项之和**。⚠ 只到 8 GPU / 1B 级微调,DP=8192 那组是纯模拟+合成数据;**testbed 的 HBM:线速 = 61:1,加速比不可外推**(盈亏门槛 9.5,MI455X 机架内只有 5.4–6.5→亏,跨机架 33–39→盈);开源 | [`dynamiq.md`](./dynamiq.md) |
| GIFT | arXiv'26 | 低精度梯度通信 | 用 K-FAC 输入侧因子把梯度白化到近各向同性坐标系再做 FP8 量化与归约(K-FAC 只当度量不当优化器),rank-32 低秩 + 只对 profile 出的 13 个脆弱层(**全是 fc2**)开、每 50 步刷新。FP8 单步往返 RelL2 −67.4%(对角近似几乎无效→有用几何是**跨维耦合**的)。⚠ **质量结论站不住**:同表里直接欧氏 FP8 端到端 −10.79% 而 GIFT 只 −7.6%(慢 3.19 pt);头对头 600M 上欧氏赢 8 项 GIFT 赢 6 项、14 任务均值欧氏 0.5186 > FP32 0.5060 > GIFT 0.5032;基线是 FP32 非 BF16,换 BF16 后直接 FP8 只 +3.9%、GIFT 只 +0.4%;**缺"欧氏+EF"这个关键消融**(GIFT 分支有 EF 和局部 scale,欧氏基线两样都没有);只到 300M/600M 且两点用了不同序列长度。**原创发现**:EF 缓冲累积在变换坐标里,刷新时旧基残差被新 `L_A^⊤` 映射回去→**基不匹配,注入 ∝(L_A^new−L_A^old)^⊤R 的伪更新**,论文一字未提 | [`gift.md`](./gift.md) |
| Tessera | OSDI'26 | 异构 MoE 流水并行(阿里) | 切分与 overlap 调度是循环依赖:先真机 profile 每个 overlap pair 的 post-overlap cost,再用 MILP 按实测边代价选切分,外加 DBO 用路由元数据预测气泡填 Wgrad;Qwen3/Qwen3-Next 生产 4,096–12,288 GPU **+20%~33%**,万亿 **39% MFU**;**§5 工程经验含金量最高**——生产里否掉了 Comet 式融合(后端耦合)、**EP 通信 kernel 占 ~20 SM 致 10–20% 减速**、代价模型尾部误差 15% 会翻转 MILP 排序 | [`tessera.md`](./tessera.md) |
| Instella-MoE | arXiv'26 | 16B MoE 模型报告(AMD 自训) | **AMD 在 MI300X/MI325X 上用 Primus(Megatron 后端)从零训的 16B-A2.8B 全开源 MoE**,7.1T token,六阶段 checkpoint + 配置 + 数据配比 + 代码全释出。系统侧只有一个贡献 **FarSkip-Collective**:改层间连接让下游消费**过时/不完整激活**,从而把 Dispatch/Combine 两个 A2A 与 attention + shared expert 重叠,预训练吞吐 **+12.7%**、SGLang EP 服务 TTFT **+39.2%**(8×MI325X);**刻意不写 custom kernel**,前向靠 `async_op=True`+`.wait()`,反向靠 race-safe 异步通信算子 + PyTorch Sequence Number API 重排。**最扎实的一条是 DPO 必须关负载均衡**:开着时 54% 的 token top-1 专家与 SFT 不同、top-6 Jaccard 0.42,关掉后是 5% / 0.94,dense 模型上不出现(因 bias 更新速率与 aux loss 按高 LR 标定,而 DPO LR 只有 8e-8)。⚠ **系统数字不可复算**:无 MFU / 无绝对吞吐 / 无预训练卡数 / 无 GPU-hours,12.7% 出自归一化柱状图且基线未定义;**EP 全程=8 从未跨节点**,而 FarSkip 明说依赖"通信与计算 kernel 共享 CU"——正是 Tessera 与 Motif 3 指出的病灶;**64K 长上下文阶段把重叠关掉了**;质量中性的证据是 200B token **单次运行**、平均分差仅 +0.05,7.1T 正式训练无对照组;Gated MLA 的 "+0.47 平均"实为 15 项里赢 7 输 8,全靠 MMLU/HE+/MBPP+ 三项撑起 | [`instella-moe.md`](./instella-moe.md) |
| Motif 3 | arXiv'26 | 314B MoE 模型报告(训练系统) | 韩国 Motif,314B-A13B / 384 专家 top-8 / B200 / TorchTitan。**mHC 的 post-mapping 必须从 `2σ(z)` 退火到 `1σ(z)`**,否则 `>1` 的映射逐层反复放大 sublayer 输出、累积激活离群值——**我们 V4 的 `hyper_connection.py:359` 正是硬编码 `2.0`**;**QK-Clip 在 GQA/MLA 下必须非对称劈分** `r=1/(1+√G)`,对称 `√γ` 会把共享 K 压到零;改 FA4 forward 直接吐 per-head logit max;**大规模下"消除"专家权重 All-Gather 优于"隐藏"**(通信 kernel 吃 SM,与 Tessera 的 20-SM 发现互证);长上下文**逐层选 CP**(full→Ulysses / SWA→window-aware Ring halo,解析 1024×)+ LPT 负载重排(rank 间 attention FLOPs 差 **3.54×**);MXFP8 在 EP dispatch **之前**量化(通信减半 + 量化成本降到 1/top-k);6 项专家健康度指标带阈值。**但 §3 十几项系统优化零吞吐/MFU/加速比数字**,只能当设计决策清单读 | [`motif-3.md`](./motif-3.md) |
| HyperParallel-MoE | arXiv'26 | MoE 训练(昇腾) | 算子串行→编译期静态调度的 tile 级 AIC/AIV 异构 taskflow,AIV 驱动单边通信 + SSC 离线编译,单次 kernel launch;Dispatch-to-Combine 1.49–1.58× / E2E 1.08–1.09×;**静态 vs 动态派发 0.1 vs 2.36 µs/任务**,ROCmoe P4 的量化论据 | [`hyperparallel-moe.md`](./hyperparallel-moe.md) |
| MXFP4 原生预训练 | arXiv'26 | FP4 训练数值(AMD) | Penn State + AMD;MI355X **原生** MXFP4 逐段打开,到 ppl 3.3 的 token 开销 Fprop 8–9% → +Dgrad 10–11% → **+Wgrad 跳 26–27%**;随机舍入与随机 Hadamard 在全流水**不收敛**,**确定性** Hadamard 打回 8–9%(H16/H32 同尺寸对照,已排除"旋转尺寸"这个变量);单步吞吐 +20% 但**端到端只剩 +9–10%**——FP4 收益被稳定性闸住;**dense-only,未碰 MoE/grouped GEMM** | [`mxfp4-pretraining.md`](./mxfp4-pretraining.md) · [全文中译](./mxfp4-pretraining-zh.md) |

### 推理系统

| Paper | 发表 | Topic | 一句话结论 | File |
|---|---|---|---|---|
| TEMPO | arXiv'26 | EP 负载均衡(快手可灵 KlingAI) | **把 grouped GEMM 的每专家耗时测清楚了**:不是 token 数的线性函数,而是双 regime——低于 **n\*≈156–168 token/专家**时 HBM 权重流式主导,代价挂在**激活副本数**上(拟合地板 1.7/3.0/14.8 µs 落在 roofline 预测的 **1.13–1.24×**,即峰值带宽 80–88%);高于则线性,但 grouped GEMM 把 token 向上取整到 **128 的 M-tile**,拆分专家会**取整多次、凭空制造填充**。`t=max(a+bG, c+βN)`,拟合误差 4–8%。枢纽事实:**92–100% 的真实 decode batch 同时含两种 regime**(B=512 时平坦段专家占激活 47%、线性段专家扛 91% token),所以没有 regime-free 的固定代理;NP 难恰好只在两 regime 交互时出现(纯极限下分别退化为 token LP 与 semi-matching)。⚠ **纯 serving/decode,不能搬到训练**(§5.14 明说训练侧均衡器靠梯度步间的权重迁移窗口,服务没有);tile 项到 16,384 token/卡失效。⚠ **头条数字在模型空间**,15.5% 是 EP32–64 外推;墙钟只有 +5.0%(GovReport)与 p99 TPOT −15.6%,饱和 decode 反而 −2~4%;**DSv3 完全无效**(每卡 32 专家已被统计平均抹平)。⚠ `+38–70% vs LPLB` 比的是工程架构,论文自己 like-for-like 移植后承认"most of the gap was architecture",真实优势只是计算瓶颈点的**稳定性**(token-LP 跨窗口摆动 20 pp vs TEMPO 4.4 pp)。**方法论诚实度极高**:撤回未复现结果、预声明窗口剔除规则、报告模型失效的负对照(Qwen3 形状拟合出负 b)、noop 对照分离数据通路与求解器 | [`tempo.md`](./tempo.md) |
| Perseus | arXiv'26 | 多节点 megakernel 通信 | proxy RDMA 上 `put-with-signal` 展开成 `Put→Fence→Signal`,每次 fence 排空 NIC 流水;96 并发/8 节点吞吐掉到 2%;decoupled signaling(每 PE 一个 fence)+ NIC 侧保序,Libfabric 10.3× / IBRC 2.47×,proxy 反超 IBGDA 1.2×;**节点内无此问题** | [`perseus.md`](./perseus.md) |
| X-Stage | arXiv'26 | 发射后流水阶段建模 | remote store 发出到远端可见之间存在软件可见的 X-Stage;三参数 Burst–Gap 模型 `(T_iss⁰, R=717 GB/s, Q=4.25 MiB)` 预测背压拐点;据此重排 DeepGEMM MegaMoE 的 L1/L2 交错,84 配置几何均值 1.18× / 最大 1.62× / 最差 0.94×(L2 局部性反噬) | [`x-stage.md`](./x-stage.md) |
| ExpertPlex | arXiv'26 | MoE 服务(自适应 APK) | 共享专家 + 分离 attention;Adaptive Persistent Kernel 在 tile 边界抢占(界 = 1 tile + 1 检查 epoch,与序列长无关),抢占决策沿 system→device→DSMEM→warp 传播;vs Green Context prefill 只慢 1.12×(对方 4.07×);goodput 2.01× vs PDD;**唯一质疑静态调度的一篇,但只反固定空间划分** | [`expertplex.md`](./expertplex.md) |
| Fleet | arXiv'26 | Chiplet megakernel | Chiplet-task + 协作 L2 tiling,MI350 decode 1.3–1.5× vs vLLM eager | [`fleet.md`](./fleet.md) |
| MoE Tile Signaling | ICPP'26 | MoE 推理 overlap | remote-owner layout + persistent producer/consumer + tile epilogue signal,combine A2A 与 GEMM overlap,4×A100 最高 2.64× E2E / 2.74× MoE layer vs FasterMoE | [`moe-tile-signaling.md`](./moe-tile-signaling.md) |
| MegaScale-Infer | SIGCOMM'25 | MoE 推理 | 分离式 EP,Prefill/Decode/Expert 解耦,3.2× 吞吐 / 55% 成本↓ | [`megascale-infer.md`](./megascale-infer.md) |
| SnapMLA | arXiv'26 | MLA FP8 解码 | RoPE-aware per-token 量化(RoPE 保 BF16)+ 预缩放域对齐 + S_V 折进 P 由 softmax 隐式反量化,8×Hopper 最高 1.91× 吞吐(主要来自 KV cache 减半后 batch 变大) | [`snapmla.md`](./snapmla.md) |
| KTransformers | SOSP'25 | 异构推理 | CPU+GPU 异构推理,$5K 跑 DeepSeek-V3 671B | [`ktransformers.md`](./ktransformers.md) |
| UBEP | SIGCOMM'26 | MoE EP 通信库(超节点) | 南大 + 华为;BSP 阶段+全局 barrier → 依赖驱动细粒度任务,**Data-as-Flag** 用 512B 原子写把 flag 融进 payload(TFF/DC/**SP 哨兵轮询**三变体),层次化 token 调度同时均衡负载与 hop 距离(Two-Hop 延迟 = 本地 **11.5×**);CM384/256 die 上 dispatch ↓34.7–52.4% / 带宽 +35.3–40.8% / P99 TPOT ↓11.1%;**MoE 通信占 per-token 延迟 ~50% 但只占硬件执行时间 ~20%**——"重排执行而非加带宽"的第三方证据;§6 点名批评 fused persistent kernel 的刚性资源划分与软件轮询,是 ROCmoe 路线要正面回答的一篇 | [`ubep.md`](./ubep.md) |

### 架构创新

| Paper | 发表 | Topic | 一句话结论 | File |
|---|---|---|---|---|
| OmniMoE | arXiv'26 | MoE 路由 | 原子专家 + 笛卡尔积路由 O(√N),10.9× 推理加速 | [`omnimoe.md`](./omnimoe.md) |
| LatentMoE | -- | MoE 架构 | Latent space MoE 设计要点 | [`latent-moe.md`](./latent-moe.md) |

### 编译器 / 内核生成

| Paper | 发表 | Topic | 一句话结论 | File |
|---|---|---|---|---|
| HipKittens | MLSys'26 | AMD kernel 原语(Stanford+AMD) | ⭐ tile 抽象可移植到 AMD,但**实例化算法必须重做**。**wave specialization 在 AMD 是负优化**(寄存器静态均分→producer 白占寄存器压低输出 tile:0-producer/256×256 得 1610 vs 4-producer/128×256 得 893 TFLOPS);替代方案 8-wave ping-pong(48 行热循环即追平裸汇编)/4-wave interleave;register pinning 绕过 HIPCC +20%;chiplet grid swizzle +19%;**附录逆出了 CDNA ISA 未文档化的 LDS phase/bank 表**;已产品化进 AITER | [`hipkittens.md`](./hipkittens.md) · [全文中译](./hipkittens-zh.md) |
| Swizzled Head-first Attention | arXiv'26 | AMD chiplet NUMA 调度(Duke+AMD) | FA2 的 workgroup 网格重排,把一个 head 的全部 Q-block 锁到单个 XCD 以复用私有 L2;提出 **ACC(共享同一份 K/V 的 WG 组)** 抽象;MI300X 上 MHA 最高 **+50%**,L2 命中 **90–96% vs block-first ≈1%**——**AITER 现用的 Swizzled Block-first 在多头+长序列下几乎全 miss**;但 GQA 上无优势(8 KV head 恰等于 8 XCD)、backward 仅 1.10×、**causal 下亏 7%**(FlyDSL 实测,论文未提);全文归一化无绝对 TFLOPS;**AMD 已实现进 FlyDSL** | [`swizzled-head-first-attention.md`](./swizzled-head-first-attention.md) |
| AutoMegaKernel | arXiv'26 | megakernel 合成 + 静态校验 | HF Llama → 单个常驻 cooperative kernel,零手写 CUDA;frozen schedule-IR 验证器用 9 条静态图检查证无死锁/无竞争,7160 个对抗调度**零假接受**;同源重定向 sm_80/90/120;**负面结果更有用**:瓶颈是每 tile 跨 SM 同步,且在带宽最高的训练级芯片上最严重 | [`automegakernel.md`](./automegakernel.md) |

### 其他

| Paper | 发表 | Topic | 一句话结论 | File |
|---|---|---|---|---|
| LEANN | -- | 向量索引 | Low-storage vector index for RAG / on-device | [`leann-low-storage-vector-index/`](./leann-low-storage-vector-index/README.md) |

## 编辑约定

- **slug**: kebab-case,通常去掉 "moe" 后缀(因为同类论文聚在一起)。
- **单文件 vs 子目录**: 默认单文件 `<slug>.md`;当有复现脚本、补充材料、原始 PDF 时升级为 `<slug>/` 目录。
- **新增论文**: 用 `.cursor/skills/read-paper/SKILL.md` 或 `paper-deep-analysis/SKILL.md`,写完后**回写本 README 的清单**(挑对应方向的表格加一行)。
- **必含字段**: Problem · Contribution · Method(含数据流) · Experiments · Limitations · Our take。
- 引用其他论文笔记用 `./<slug>.md` 相对路径。

## 历史索引

旧版 100+ 篇分类索引已搬到 [`../knowledge/moe/paper-landscape.md`](../knowledge/moe/paper-landscape.md)。

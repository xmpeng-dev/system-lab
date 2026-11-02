# Mixture-of-Kittens：面向 NVL72 的 MoE 训练 megakernel
# Mixture-of-Kittens: MoE Megakernel for NVL72s

> [arXiv 2609.36070](https://arxiv.org/abs/2609.36070)（v1，2026-09-28，cs.DC / cs.LG）· [HTML](https://arxiv.org/html/2609.36070v1) · [Cursor 博客](https://cursor.com/blog/mixture-of-kittens)（页面日期 2026-08-04）· [代码](https://github.com/cursor/mixture-of-kittens)（Apache 2.0，Copyright 2026 Cursor Research）
> Stuart H. Sul、Nash Brown（Stanford CS + Cursor Research）· Henry Wildermuth、William Lin、Federico Cassano（Cursor Research）· Christopher Ré（Stanford CS）
> 硬件：GB300 NVL72（72×B300，第五代 NVLink）；CUDA 13.0 / Python 3.13 / PyTorch 2.13。代码只支持 SM100 / SM103（README）。
> 发表：arXiv 预印本，页面上没有会议信息。
> **全文中译见 [`mixture-of-kittens-cn.md`](./mixture-of-kittens-cn.md)**（原文许可 CC BY 4.0）。
>
> 本篇按论文章节逐节精读，覆盖正文全部表格数字与附录 A。论文没写、但博客或 README 写了的内容标注「（博客）」「（README）」；两边对不上的地方集中在 §9。对 MegaMoE 论文与 Primus 的影响在 §10。
>
> **2026-11-02 重写。** 旧版有四处错误，已改：许可证是 Apache 2.0，不是 MIT；附录 A 里 Comet 的调度占比是 4.2–13.0%，不是 4.7–13%；Primus MegaMoE 的数字换成未发布 blog 里核对过的值；MoK 与 HipKittens 只有 Ré 一位共同作者，MoK 也没有引用 HipKittens。

## TL;DR

**论文问的是：为 scale-out 网络（IB / RoCE）调出来的 MoE 训练系统，搬进 NVL72 这种 72 卡单跳 scale-up 域，还对不对？** 答案是不对。最直接的证据在 Table 1：MXFP8 下 48 个格子里，朴素 NCCL + PyTorch 是最强基线的有 21 个；EP16、每卡 1024 或 2048 token 的 16 个格子全是它。

作者把 MoE 层的延迟拆成一个代价模型，然后用三个设计分别压其中三项：

\[
C = \max(T_{\text{comp}}, T_{\text{mem}}, T_{\text{comm}}) + T_{\text{non-overlap}} + T_{\text{sync}} + T_{\text{host}}
\]

1. **按算子选通信方向**，压 \(T_{\text{sync}}\)：forward 用 pull dispatch + push combine，backward 镜像。原则是让等信号的那块 GPU 自己发远端访存，这样完成信号不出卡。
2. **把 overlap 粒度做成一等可调参数**，压 \(T_{\text{non-overlap}}\)：最优 minibatch 在 512–32768 token 之间随负载漂移。
3. **设备侧 ring buffer**，压 \(T_{\text{host}}\)：消掉"按路由结果 D2H 再分配"的 CPU–GPU 同步。

三者装进一个确定性的持久 megakernel（forward、backward 各一个）。层吞吐相对最强公开基线最高 2.37×（MXFP8 forward，EP64、每卡 2048 token，Figure 7；Table 1 里实际最大值是 2.74×）。Cursor 生产栈 512×GB300、EP32 端到端 1.41×。

**对我们最直接的五条**（细节见 §10）：

1. **MegaMoE 恰好有 MoK 要消灭的那个同步。** `primus_turbo/pytorch/ops/moe/fused_mega_moe.py` 的 docstring 写明每次 forward 都要 D2H 读回本 rank 收到的行数，因此不能 CUDA graph 捕获。
2. **L1 不能写成"没人讨论过"，但也不是"同一个问题"。** 融合算子要和框架 collective 重叠，需要两个条件：框架得按时发出 collective（hook 粒度），硬件得留出 SM/CU 给它跑。MoK 用 CLC 解决后者，MegaMoE 的两阶段拆分解决前者。MegaMoE 在后者上是什么状况，目前没有核实。
3. **朴素 RCCL 基线必须补，而且在我们的 EP8 上它更可能是最强基线。** MoK 的数据里 EP 越小、每卡 token 越少，朴素 NCCL 越占优。
4. **pull 赢在信号，不赢在搬运。** 从 Figure 4 反推，去掉信号后 push 的纯搬运比 pull 快 5–9%。我们在 RocMoE M1 里测到 XGMI 上 pull 每 token 慢约 43%。MegaMoE 在 EP8 上用 push dispatch 有依据，但论文里要把这笔账算出来。
5. **粒度估算公式可以直接搬**（博客）：\(T \ge 2 \cdot C \cdot 128 \cdot 256 / \min(2I, H)\)。MegaMoE 的 dispatch 等待粒度目前是"一个 local expert 的全部 token"，由模型和路由决定，不能调。

---

## 1. 问题与代价模型（§1、§3 开头）

硬件正在从"8 卡 + 商品化 scale-out 网"转向单一 scale-up 域：论文列的例子是 GB200/GB300 NVL72、AMD MI455X 的 Helios、Google TPU Pod、AWS Trn2 UltraServer。NVIDIA 路线图往后还有 NVL144 / 576 / 1152。作者的说法是互连开始像"一个大内存系统"，而不像网络。

**AMD Helios 在全文只出现一次**，就在这一段的例子列表里。论文没有任何 AMD 实验，也没有讨论 AMD 硬件。旧版笔记说它"点名 Helios"，这个说法没错，但分量不能往上抬。

MoE 在大规模训练里占端到端一半以上时间，论文引的是 FSMoE（ASPLOS'25）和 Megatron-Core MoE（`2603.07685`）。博客的说法是"取决于负载与训练配置，可以超过一半"。

代价模型（Eq. 1，Figure 3）各项含义：

| 项 | 含义 | MoK 用哪个设计压它 |
|---|---|---|
| \(T_{\text{comp}}, T_{\text{mem}}, T_{\text{comm}}\) | 计算、片内访存、跨卡通信；完美重叠时取最大值 | 每项单独打满对应硬件单元：tensor core 持续喂数、所有链路都有足够在途请求且流量摊匀 |
| \(T_{\text{non-overlap}}\) | 没盖住的部分，例如第一次 dispatch 与最后一次 combine | 可调粒度（§4） |
| \(T_{\text{sync}}\) | 片内、跨卡的完成信号与 memory fence | 通信方向（§3） |
| \(T_{\text{host}}\) | CPU–GPU 同步、内存分配、kernel launch | ring buffer（§5）+ megakernel |

## 2. 背景里值得记的数字（§2）

GB300 的存储层级（论文脚注说明取 GB300 NVL72 作代表）：

| 层级 | 容量 | 带宽 |
|---|---|---|
| 寄存器 | 每 SM 65,536 个 32-bit | 每周期可访问 |
| Shared memory | 每 SM 228 KB | 约 40 TB/s |
| L2 | 129 MB | 约 22 TB/s |
| HBM | 279 GB | 8 TB/s |
| 远端 HBM（NVLink） | — | 900 GB/s 单向 |
| 远端 HBM（IB / RoCE） | — | 100 GB/s 单向 |

博客补充：第五代 NVLink 在 GB300 上标称 1.8 TB/s，这是双向合计；Blackwell 每卡 148 个 SM。

**相关工作的定位**有三点值得记：

- 作者自认**最接近的工作是 DeepGEMM 里的 MegaMoE**，并把它定性为"只做推理的 forward kernel，专为 DeepSeek-V4 serving 的 FP8×FP4 设计"。这个名字和 Primus-Turbo 的 MegaMoE 撞了（§10.1）。
- **没有引用** UniEP、Primus MegaMoE、HipKittens。引了 ParallelKittens（MLSys'26，Sul 是一作，MoK 的 inter-SM overlap 与 SM 划分来自它）、UEP（OSDI'26，作为 push 派的例子）、Triton-distributed（只在"编程原语"一类里出现）、FlashMoE、Comet。
- 对 megakernel 的立场：megakernel 只是把多个算子装进一个持久 kernel 的**机制**，不规定怎么融合、怎么重叠。论文说 MoK 的设计决策都在后者。

## 3. 通信方向（§3.1）

### 3.1 为什么方向决定信号量

细粒度重叠时，每个计算任务（比如一次 tensor core 矩阵乘）都要等到确认输入已到才能开始。

- **push dispatch**：每个 peer 每推一块 token 都要发一次完成信号。接收方要等所有 peer 的信号；博客的说法是在 NVL72 上最多等 71 个 peer，还要发一次全机架的 memory fence。而且信号在路上的时候，数据其实已经到了。
- **pull dispatch**：目的 GPU 自己发完所有 load，本地一个信号就够了。只有一个实体在发信号，开销不随 EP 度数增长。

Figure 4 的微基准（每卡 7,168 个宽 7,168 的 BF16 token，4 个 local expert，每 token 4 条均匀随机路由）：

| EP | pull 全延迟 (µs) | push 全延迟 (µs) | pull 信号开销 | push 信号开销 | push 信号占全延迟 | 去掉信号后 push / pull（推算） |
|---|---|---|---|---|---|---|
| 4 | 475 | 489 | 1 | 39 | 8.0% | 450 / 474 |
| 8 | 559 | 574 | 1 | 51 | 8.9% | 523 / 558 |
| 16 | 603 | 633 | 4 | 66 | 10.4% | 567 / 599 |
| 32 | 638 | 704 | −1 | 119 | 16.9% | 585 / 639 |
| 64 | 638 | 744 | −2 | 135 | 18.1% | 609 / 640 |

"信号开销"是同一个 kernel 加信号与不加信号的延迟差。**最后一列是我用"全延迟减信号开销"推出来的，论文没给。** 它说明两件事：

1. 在这个均匀路由的微基准里，**push 的纯数据搬运比 pull 快 5–9%**；pull 的全延迟优势完全来自信号。这和 UniEP 在 Hopper 上测到的"putmem 比 get 快"并不矛盾：两篇都同意 push 搬得更快，MoK 只是表明 EP 扇入一大，信号开销就盖过了这点差距。
2. pull 的全延迟从 EP32 起不再增长（638 → 638 µs），push 一直在涨。

### 3.2 方向选择规则

**让接收信号的 GPU 同时发起远端 load/store。** forward 每卡 pull 自己的输入、本地通知计算 worker、算完再 push 输出；backward 镜像为 pull reverse-combine + push reverse-dispatch。通信与计算之间的交接永远不离开本卡。

为什么以前的系统都是 push：在细粒度 RDMA 上 push 优于 pull（论文引 Wei et al. OSDI'18 与 Demystifying NVSHMEM `2606.05951`）。scale-up 上整个 EP 组都在域内，方向才可以按算子单独选。

### 3.3 调度表（Algorithm 1）

pull dispatch 每一步要决定三件事，各有一个要求：

| 决定 | 要求 | 为什么 |
|---|---|---|
| 从哪个 peer 拉 | peer 之间 round-robin | 所有链路的在途请求数大致相等 |
| 拉哪个 token | 按 local expert 非降序到达 | 每个 expert 的矩阵乘不会被打碎 |
| 存到本地哪里 | 连续填 | 喂 tensor core 前不需要再搬一次 |

MoK 用一个**单独的 kernel**（在 megakernel launch 之前）建一张两列表 `{src_rank, src_index}`，行号就是该 token 在目的 buffer 里的位置，dispatch 只是逐行走表。

Algorithm 1 用文字说：先数出每个（源 rank, local expert）的路由数 \(M_{r,\ell}\)，前缀和得到每个 expert 区域的起点 \(S_\ell\)；对每条路由，算出它在"同源、同 expert"里是第几个（序号 \(o\)），落点是 \(p = \sum_{r'} \min(M_{r',\ell}, o) + |\{r' < r : M_{r',\ell} > o\}|\)。等价于按键（\(\ell\), \(o\), \(r\)）排序：先按 expert 分区，区内按序号再按源 rank 交织，自然就是 round-robin。博客强调实现上**不需要排序**，两遍计数扫描即可。

开销：附录 A 在 EP4 上测得占 MoE 运行时 2.1–7.7%，相对 Comet 低 1.4–2.2×（§7.5）；这部分时间计入了所有基准。

**一张表复用四个通信算子**：push combine 把两列重新解释成 `{dst_rank, dst_index}`，推送仍然在 peer 间均匀分布。最坏情况只有几 MB，常驻没有压力。

博客补充了 push 侧为什么难：push dispatch 要一张三列表 `{src_index, dst_rank, dst_index}`，要做多次排序，而且每个 rank 的调度都得考虑所有其他 rank，因为两个源 rank 不能写同一个 `dst_index`。

### 3.4 博客给的第三条理由：带宽（论文没有）

博客用 NCU 测了一块 256×256 BF16 tile（131,072 B）跨 NVLink 的字节构成：

| | 总字节 | RX 方向 | TX 方向 | 协议 RX | 协议 TX | 载荷 RX | 载荷 TX |
|---|---|---|---|---|---|---|---|
| Push | 159.6 KB | 2.9 KB（1.84%） | 155.6 KB（99.16%） | 2.7 KB | 24.6 KB | 0 | 131.1 KB |
| Pull | 172.0 KB | 147.5 KB（85.71%） | 24.6 KB（14.29%） | 16.4 KB | 24.6 KB | 131.1 KB | 0 |

push 总字节少约 12.4 KB，但几乎全在一个方向上。NVLink 两个方向是分开的 lane，要跑到 1.8 TB/s 必须两个方向都满。MoE dispatch/combine 的单次传输只有几 KB、而且不均衡，链路上会有空泡。博客的结论是：**在 expert 不均衡时，pull 的 NVLink 带宽利用率最高比 push 高 29%**。这个 29% 没有给图、也没给实验设置。

## 4. Overlap 粒度（§3.2）

### 4.1 两头的惩罚

每一块"一起传、一起算"的 token 叫一个 **minibatch**，它的大小 \(b\) 就是重叠粒度。

- 最细：刚好够一条 tensor core MMA 指令（博客：Blackwell 上是 256），Comet 是这一头的代表。
- 最粗：几万 token 一次，DeepEP 的常见用法是这一头。

两头都不对。记路由后 token 数 \(N_{\text{routed}}\)、负载形状 \(w\)：

- 细粒度惩罚 \(P_{\text{fine}}(b,w) \approx \lceil N_{\text{routed}}/b \rceil \cdot C_{\text{ineff}}(b,w)\)：每个 minibatch 都要付一份 tensor core 吃不满的代价和同步开销。
- 粗粒度惩罚 \(P_{\text{coarse}}(b) \approx T_{\text{dispatch}}(b) + T_{\text{combine}}(b)\)：第一个 minibatch 的 dispatch 和最后一个的 combine 没被盖住。

\(b\) 增大，前者降、后者升，所以 \(b\) 应该按负载选。Figure 5（16 卡，MXFP8 forward）：吞吐先随 \(b\) 上升，然后持平或下降；**同一模型、同一 token 数下，最好的 \(b\) 是最差的 1.38–3.53×**；最优 \(b\) 在 512–32768 之间。

### 4.2 最大延迟同步

通信对所有 minibatch **广度优先**，连续 dispatch，不等计算；计算**深度优先**，一个 minibatch 跑完整个 FFN 再做下一个。两边只在两处异步会合：计算等某个 minibatch 到齐，combine 等它的输出算完。

博客补充调度细节：comm SM 做完所有 dispatch 任务才开始做 combine，所以等它做完 dispatch 时，计算侧的完成信号已经攒了很多；**shared expert 就塞在计算 SM 等第一个 dispatch 的空窗里**。

### 4.3 为什么必须是一个 megakernel

一部分 SM 只算 FFN，另一部分只做 dispatch/combine，两组之间用 GPU 本地 barrier 通信（inter-SM overlap，来自 ParallelKittens）。不做成 megakernel 的话：

- 每个 minibatch 要 launch 好几个 kernel、用多条 stream 重叠，每个 minibatch 要多付几十 µs 的 launch 与 stream 同步（\(T_{\text{host}}\)）；
- 交错执行交给硬件调度器，而它**不保证计算和通信真的并发**（论文引 CUDA Green Contexts 文档）。博客说得更直接：多 stream + green context 没法可靠地按预期切分 SM，软件分区才有精确保证。

博客：用 TMA 做 load/store 时，**不到三分之一的 SM 就能打满 NVLink**。README 推荐 comm SM 数在 4–52 之间。

### 4.4 粒度的估算公式（博客，论文没有）

经验规则：让 FFN 里每个 grouped GEMM 至少有**两个满 wave**。一个满 wave 已足以打满 tensor core；第二个 wave 的 MMA 能和第一个 wave 的 epilogue、下游算子（SwiGLU、下一个 GEMM）重叠，摊掉尾部效应。

Blackwell 上每个 SM 做 128×256 的输出 tile。记 minibatch 大小 \(T\)、hidden \(H\)、expert 中间维 \(I\)、SM 数 \(C\)：

- up 与 gate 两个投影可以并行：\((T/128)\cdot(2I/256) \ge 2C\)
- down 投影不能和别的并行：\((T/128)\cdot(H/256) \ge 2C\)
- 合起来：\(T \ge 2 \cdot C \cdot 128 \cdot 256 / \min(2I, H)\)

Kimi 2.5（博客说它是 Composer 2.5 的基座，\(H = 7168\)、\(I = 2048\)），\(C = 148\) 时得 \(T \ge 2368\)。实测 MoK forward：

| \(T\) | 512 | 1024 | 1536 | 2048 | **2560** | 3072 | 3584 | 4096 |
|---|---|---|---|---|---|---|---|---|
| ms | 5.981 | 4.669 | 3.981 | 3.666 | **3.425** | 3.447 | 3.524 | 3.473 |

最优在 2560，和公式的 2368 很接近；再往上基本持平。

## 5. 消掉 CPU–GPU 同步（§3.3）

### 5.1 动机：Grace 慢

Figure 6 左：常见 PyTorch 算子的主机侧调用时间，GB300 NVL72 比 DGX B300 **慢 1.46–2.97×**。在生产规模上，GPU 很容易被主机工作卡住；CPU–GPU 同步代价变得更高，因为它让 GPU stream 没法跑在 CPU 线程前面。博客给了现场描述：trace 里经常看到 GPU kernel 在等 CPU 做日志、往远端推训练指标，同一套栈在 Intel CPU 的 DGX 上没有这个问题。

EP 里最常见的同步来源是**内存分配**：每个 expert 收多少 token 要到运行时才知道，精确分配目的 buffer 就得先把计数拷回 CPU（论文引 MegaBlocks）。另一条路是超过固定容量就丢 token（GShard、Switch），但会伤训练质量。

### 5.2 Macrobatch ring buffer

MoK 在设备上管一个固定大小的 token 环，叫 **macrobatch**，大小可调。它在 dispatch→FFN、FFN→combine 之间形成生产者-消费者流水：dispatch 往第三格写，FFN 处理第二格，combine 读第一格；绕回时同样流水，环的边界几乎不增加开销。博客：环只有几百 MB；forward 里会把上一个 macrobatch 某个 minibatch 的 combine 和下一个 macrobatch 对应 minibatch 的 dispatch 交错，因为它们用同一块环区域，腾空就立刻重填。

Figure 6 右：**能装一半路由 token 的环**，对比**能装下全部路由 token 的过量分配**（生产上不现实，但代表上限），延迟增加：

| 形状 | 每卡 1,024 token | 2,048 | 4,096 |
|---|---|---|---|
| Kimi K2.7 Code | 0.3% | 1.2% | 0.7% |
| GLM-5.2 | 7.4% | 4.6% | 1.9% |
| Qwen3.5-397B-A17B | 0.7% | 1.3% | 2.5% |
| DeepSeek-V4-Pro | 0.1% | −0.5% | 1.0% |

几何平均 1.7%，最差 7.4%（GLM-5.2、1024 token）。

### 5.3 Ring-aware forward replay

生产训练里显存放不下 forward 的全部中间激活，backward 要靠 forward replay 重算一部分。问题在于，直接重跑 MoK forward 只能保住最后一个 macrobatch（前面的都被环覆盖了），而且会重算 backward 根本用不到的东西（比如 down 投影的输出）。

MoK 的做法：

- **把 replay 融进 backward megakernel**，只补缺的中间量，并和 backward 重叠。博客：replay 只需跑到 SwiGLU，因为 backward 只需要 down 投影的输入和输出梯度。
- **forward 按 macrobatch 逆序处理**。顺序处理的话只剩最后一个 macrobatch，而它通常是不满的；逆序处理后 forward 结束时环是满的（或者装下了全部 token），replay 最少。

README：默认 forward 返回的 context 保存**第一个** macrobatch 的中间量（和逆序一致，第一个是最后处理的）；也可以丢掉它，在 backward 前用 `functional.recompute_forward_context` 重建。README 还建议把 `macrobatch_size` 尽量调大到占满可用显存（例如 131072 意味着环只用一圈）。所以环的首要作用是**去掉同步**，不是省显存。

### 5.4 README 里的容量假设（论文没有）

`schedule_capacity_multiplier` 默认 0.5，含义是"路由到单个 rank 的 token 占全组的最坏比例"。设成 1 等于假设所有 token 都去一个 rank（因为 expert padding，实际最坏情况略高于 1.0），但会增加调度开销。README 建议训练初期路由不均衡时调高，之后降到 0.5 左右；调低它几乎不省显存，因为调度表最多几 MB。

论文说 MoK 既不同步、也不丢 token。这个参数说明**调度表这一层仍然有一个上界**。0.5 在 EP64 上意味着单个 rank 收下全组一半的 token，正常训练碰不到，但路由塌缩时可能碰到。**超出上界时的行为，论文和 README 都没写。**

## 6. 生产特性（§3.4）

| 特性 | 做法 | 来源 |
|---|---|---|
| 确定性 | 浮点运算顺序固定，同输入 bitwise 同输出，与硬件调度无关。主要的非确定性来源是原子操作：要么用软件调度让同址原子按固定顺序执行，要么写临时 buffer 再由另一个 worker 按固定顺序归约。用途是训练消融和 on-policy RL | 论文 |
| CLC 让出 SM | 任务通过 Blackwell 的 Cluster Launch Control 调度。CLC 是硬件原生的 work stealing，基于它的 kernel 不用跑完就能把 SM 让给更高优先级 stream 上的 kernel | 论文 |
| MXFP8 | routed expert 走 MXFP8；shared expert 保持 BF16（博客："我们发现它影响训练稳定性"）。权重量化用单独的优化 kernel 预先做（README：留在 kernel 外，方便 FSDP），激活量化融进 dispatch、grouped GEMM 和 gated activation。博客："我们用 MXFP8 训练，更快且没有数值问题" | 论文 / 博客 / README |
| Router 权重梯度 | SonicMoE 式：不物化 down 投影输出，用 gated activation 与 down 投影输入梯度的内积算，融进 gated activation 的 backward | 论文 |
| Shared expert | 负载在各 rank 均匀，且不依赖 dispatch/combine。forward 放进第一个 dispatch 空窗；backward 把整个 shared expert backward 和第一轮 reverse-combine 重叠 | 论文 / 博客 |
| wgrad 延后 | backward 每个 minibatch 只做 3 个 dgrad grouped GEMM + SwiGLU backward；3 个 wgrad 等所有 minibatch 的 dgrad 做完再算。理由一：reverse-dispatch 不依赖 wgrad，可以和最后几次 reverse-dispatch 重叠。理由二：wgrad 的归约维是 token 轴，一次在完整 token 轴上累加数值上最稳 | 博客 |
| SM 划分 | comm SM 数在 forward / backward 分开可调，推荐 4–52 | 论文 / README |
| Workspace | 用 PyTorch symmetric memory 分配跨卡对称 buffer。分配很贵，推荐每个模型一个 workspace、跨层复用（`get_workspace` 自动按 EP 组、设备、形状缓存） | README |

**CLC 这一条对我们最重要，要看清它说的是什么。** 原文：训练时经常要把机架内的通信与计算，和**跨机架通信**（比如 FSDP all-gather）重叠，而后者**需要自己的 SM**；没有 CLC，跨机架通信会排在 megakernel 后面串行执行。也就是说，CLC 解决的是**一个已经发到高优先级 stream 上的 collective 能不能拿到 SM**；什么时候发出这个 collective，由 FSDP 决定，论文没讨论。论文也**没有测 CLC 开关前后的差别**。

## 7. 实验（§4、附录 A）

### 7.1 设置

- 硬件：GB300 NVL72 机架，每架 72 张 B300，第五代 NVLink 全互连。作者说尽力调过所有基线，最终配置都在开源仓库里。
- 层基准测的是完整一层：全局 token 调度、dispatch、routed 与 shared expert FFN、combine、router 加权求和。router logits 和输入都从标准正态分布生成，各实现拿到 bitwise 相同的输入。forward / backward 分开测，BF16 与 MXFP8 都测；500 次 warmup 后计时 100 次，每次取**最慢 rank** 的延迟，再取 100 次的中位数。
- 吞吐口径：forward 有效 expert-FFN 吞吐 \(= 6N(k+1)DI/t\)，\(k+1\) 包含 \(k\) 个 routed expert 和 1 个 shared expert；backward 按两倍 FLOP 算。
- 基线的入选条件：同时支持 forward 与 backward、能在 NVL72 上跑。入选的是 NCCL + PyTorch、DeepEP + PyTorch、DeepEP + TransformerEngine、HybridEP + Megatron（Megatron 也支持 DeepEP，但 NVIDIA 在 NVL72 上推荐 HybridEP）。
- 形状（各带 1 个 shared expert）：

| 模型形状 | 路由 expert 数 \(E\) | \(D\) | \(I\) | top-\(k\) |
|---|---|---|---|---|
| Kimi K2.7 Code | 384 | 7168 | 2048 | 8 |
| GLM-5.2 | 256 | 6144 | 2048 | 8 |
| Qwen3.5-397B-A17B | 512 | 4096 | 1024 | 10 |
| DeepSeek-V4-Pro | 384 | 7168 | 3072 | 6 |

### 7.2 主结果（Figure 7，EP64，每卡 2048 token）

相对每个组合下最快的基线：MXFP8 forward 最高 **2.37×**，MXFP8 backward **1.78×**，BF16 forward **1.92×**，BF16 backward **1.58×**（前两个数和 Table 1 的 EP64 / Qwen3.5 / 2048 一致）。

作者的解释：GB300 上 BF16 tensor core 峰值是 MXFP8 的一半，backward 的 FLOP 大约是 forward 的两倍；这两个因素都让执行更偏计算瓶颈，通信占比变小，MoK 能盖住或消掉的部分也就变小。所以 MXFP8 forward 收益最大、BF16 backward 最小。外推：Vera Rubin 的 FP8 算力是 GB300 的 3.5×，scale-up 带宽只有 1.67×，会更偏通信瓶颈，MoK 类收益应该更大。

### 7.3 Table 1：EP 度数与每卡 token 数（MXFP8）

"Best" 是四个基线里最低的延迟，下标 N = NCCL + PyTorch、D = DeepEP + PyTorch、H = HybridEP + Megatron。

| EP | 形状 | Tokens | MoK fwd (ms) | Best fwd | MoK bwd (ms) | Best bwd | 加速 fwd | 加速 bwd |
|---|---|---|---|---|---|---|---|---|
| 16 | Kimi K2.7 Code | 1,024 | 0.901 | 2.334 N | 1.786 | 2.413 N | 2.59× | 1.35× |
| | | 2,048 | 1.332 | 2.581 N | 2.425 | 3.329 N | 1.94× | 1.37× |
| | | 4,096 | 2.555 | 3.308 H | 4.320 | 5.053 H | 1.29× | 1.17× |
| | GLM-5.2 | 1,024 | 0.849 | 2.321 N | 1.284 | 2.070 N | 2.73× | 1.61× |
| | | 2,048 | 1.180 | 2.367 N | 1.965 | 2.762 N | 2.01× | 1.41× |
| | | 4,096 | 2.051 | 2.913 H | 3.514 | 4.147 H | 1.42× | 1.18× |
| | Qwen3.5-397B-A17B | 1,024 | 0.857 | 2.316 N | 1.440 | 1.915 N | 2.70× | 1.33× |
| | | 2,048 | 0.901 | 2.307 N | 1.296 | 2.129 N | 2.56× | 1.64× |
| | | 4,096 | 1.705 | 2.777 N | 2.180 | 3.005 H | 1.63× | 1.38× |
| | DeepSeek-V4-Pro | 1,024 | 0.992 | 2.319 N | 2.321 | 2.571 N | 2.34× | 1.11× |
| | | 2,048 | 1.560 | 2.612 N | 3.139 | 3.343 N | 1.67× | 1.06× |
| | | 4,096 | 2.762 | 3.555 H | 5.157 | 5.462 D | 1.29× | 1.06× |
| 64 | Kimi K2.7 Code | 1,024 | 1.016 | 2.600 N | 1.555 | 2.295 H | 2.56× | 1.48× |
| | | 2,048 | 1.293 | 2.570 H | 2.045 | 2.636 H | 1.99× | 1.29× |
| | | 4,096 | 2.524 | 4.002 H | 3.871 | 4.709 H | 1.59× | 1.22× |
| | GLM-5.2 | 1,024 | 0.989 | 2.706 H | 1.612 | 2.287 H | 2.74× | 1.42× |
| | | 2,048 | 1.142 | 2.471 H | 1.737 | 2.414 H | 2.16× | 1.39× |
| | | 4,096 | 2.171 | 3.354 H | 3.294 | 4.147 H | 1.54× | 1.26× |
| | Qwen3.5-397B-A17B | 1,024 | 0.995 | 2.567 N | 1.442 | 2.348 H | 2.58× | 1.63× |
| | | 2,048 | 1.068 | 2.527 H | 1.273 | 2.267 H | 2.37× | 1.78× |
| | | 4,096 | 1.830 | 2.978 H | 2.137 | 2.959 H | 1.63× | 1.38× |
| | DeepSeek-V4-Pro | 1,024 | 0.981 | 2.670 N | 1.483 | 2.470 H | 2.72× | 1.67× |
| | | 2,048 | 1.392 | 2.564 H | 2.321 | 2.839 H | 1.84× | 1.22× |
| | | 4,096 | 2.594 | 3.796 N | 4.346 | 5.112 H | 1.46× | 1.18× |

以下四点是我从表里数出来的，论文没有这样总结：

1. **谁是最强基线。** 48 格里 N 占 21、H 占 26、D 占 1、DeepEP + TE 一次都没赢。按 EP 拆开差别很大：EP16 的 24 格里 N 占 17，其中每卡 1024 / 2048 token 的 16 格全是 N；EP64 的 24 格里 H 占 20。引言里"近一半配置朴素 NCCL 反超所有替代方案"成立，但**集中在 EP16 和小 token 数**。
2. **加速范围。** forward 1.29–2.74×，backward 1.06–1.78×。2.37× 是 Figure 7 那一列的最大值，不是全部实验的最大值；Table 1 里有三格超过 2.7×（EP64 GLM-5.2 1024 为 2.74×、EP16 GLM-5.2 1024 为 2.73×、EP64 DSv4-Pro 1024 为 2.72×）。
3. **小 token 时基线有一个地板。** 每卡 1024 token 时，EP16 最强基线的 forward 延迟四个形状分别是 2.334 / 2.321 / 2.316 / 2.319 ms，EP64 是 2.600 / 2.706 / 2.567 / 2.670 ms，**几乎和形状无关**。我的推断是这个区间的基线被固定的每层开销主导（主机侧调度、launch、同步），所以 MoK 在这里的 2.3–2.7× 有相当一部分来自消掉 \(T_{\text{host}}\) / \(T_{\text{sync}}\)，而不是重叠做得更好。主机比 Grace 快的平台上，这部分收益可能会缩水。
4. **MoK 对 EP 度数不敏感。** 同形状同 token 数下 EP16 与 EP64 的 MoK 延迟接近（例如 Kimi 2048：1.332 vs 1.293 ms），这正是 scale-up 单跳域的特征。

作者对"小 token 加速更大"的论证：global batch size 不能无限增大（引 McCandlish、Shallue、Kaplan），扩卡只能降低每卡 token 数，所以规模越大 MoK 越有效。

### 7.4 端到端（Table 2）

| | DeepEP-based | MoK |
|---|---|---|
| tokens / s / GPU（均值） | 761.0 | 1,070.2 |

512 张 GB300、跨多个 NVL72 机架；两边都是 EP32、routed expert MXFP8、shared expert BF16。DeepEP-based 是**他们自己生产栈的上一个版本**：DeepEP 通信和他们开源的 MXFP8 expert FFN kernel 重叠（论文引 Cursor 博客 "1.5x faster MoE training with custom MXFP8 kernels"）。只换 MoE 实现，提升 41%。作者说实际生产用的卡更多、每卡 token 更少，收益还会更大。

粗算一下这个数是否自洽（我的推算，忽略 CLC 与主机效应）：MoE 层加速 \(s\)、占基线步时比例 \(f\)，要 \(1/(1-f+f/s) = 1.41\)；取 \(s = 2.37\) 得 \(f \approx 0.50\)，取 \(s = 2\) 得 \(f \approx 0.58\)。也就是说，基线里 MoE 至少要占一半步时，这和"MoE 可超过一半"的说法一致。

### 7.5 附录 A：调度开销（Table 3）

MoK 的调度 = 本地 top-k 路由的 all-gather + Algorithm 1。Comet 的调度 = 同样的 all-gather + 每 expert 计数与 scatter 下标 + 按源 rank 排序 + **主机侧 GEMM tile 调度**。BF16 forward、EP4，计时口径同 §7.1。

| 形状 | Tokens | MoK 调度 (ms) | MoK 总计 | MoK 占比 | Comet 调度 (ms) | Comet 总计 | Comet 占比 |
|---|---|---|---|---|---|---|---|
| Kimi K2.7 Code | 1,024 | 0.068 | 2.250 | 3.04% | 0.331 | 4.892 | 6.76% |
| | 2,048 | 0.093 | 2.571 | 3.60% | 0.389 | 5.640 | 6.89% |
| | 4,096 | 0.133 | 4.507 | 2.96% | 0.515 | 10.537 | 4.89% |
| GLM-5.2 | 1,024 | 0.064 | 1.411 | 4.50% | 0.294 | 3.268 | 9.00% |
| | 2,048 | 0.077 | 2.050 | 3.74% | 0.349 | 4.774 | 7.30% |
| | 4,096 | 0.107 | 3.360 | 3.20% | 0.439 | 7.751 | 5.67% |
| Qwen3.5-397B-A17B | 1,024 | 0.076 | 1.228 | 6.20% | 0.321 | 2.630 | 12.22% |
| | 2,048 | 0.101 | 1.316 | 7.65% | 0.387 | 2.970 | 13.03% |
| | 4,096 | 0.156 | 2.342 | 6.66% | 0.473 | 5.165 | 9.16% |
| DeepSeek-V4-Pro | 1,024 | 0.068 | 3.195 | 2.14% | 0.325 | 6.878 | 4.72% |
| | 2,048 | 0.087 | 3.468 | 2.50% | 0.361 | 7.614 | 4.75% |
| | 4,096 | 0.115 | 5.201 | 2.22% | 0.504 | 11.927 | 4.23% |

"总计"= 调度 + forward 延迟。MoK 2.1–7.7%，Comet 4.2–13.0%；两者占比之比 1.38–2.22×。MoK 只能跑 Blackwell，Comet 只支持到 Hopper，所以**绝对时间不可比，只比占比**。作者指出 MoK 的 forward 更快，分母更小，这个比较对 MoK 不利。引言里的"预调度可占 MoE 运行时 13%"就是这张表 Comet 的最大值。

## 8. 证据强度：说了什么 vs 测了什么

| 主张 | 证据 | 缺口 |
|---|---|---|
| pull dispatch 省信号 | Figure 4 微基准 | 负载（每卡 7168 token、4 local expert、top-4、均匀路由）和 Figure 7 不同；没有"MoK 改成 push dispatch"的对照 |
| pull 带宽更好 | 只有博客的一句"最高 29%" | 没图、没设置；论文正文没有这条 |
| 粒度必须可调 | Figure 5 扫描（16 卡，MXFP8 forward） | 充分 |
| ring buffer 代价低 | Figure 6 右（半容量环） | 没区分 forward 与 backward replay 的代价；没报显存占用 |
| 三条洞察缺一不可 | 无 | **没有拆解 2.37× 各来自哪条**；每条只有孤立的微基准 |
| CLC 让 FSDP 重叠 | 文字描述 | 没测开关前后 |
| 确定性 | 文字描述 | 没测代价 |
| 可用于生产 | 一个 E2E 数字、NeMo AutoModel 集成、Composer 训练 | E2E 的模型、序列长度、其他并行维度、是否开 CLC 都没写；没有 loss 曲线；只报均值没报方差 |
| 不丢 token | 设计如此 | README 的 `schedule_capacity_multiplier` 是一个上界，超出时的行为没写（§5.4） |
| 对真实路由有效 | 层基准用标准正态 logits，路由接近均匀 | 没有偏斜路由的压力测试；唯一和不均衡有关的数字是博客的 29% |
| 基线公平 | 四个公开基线，配置开源 | 按入选条件，Comet 与 FlashMoE 跑不了 Blackwell，被排除；Megatron 侧 shared expert overlap 是否打开没写；Triton-distributed 没比 |

## 9. 论文与博客 / README 对不上的地方

| 项 | 论文 | 博客 / README |
|---|---|---|
| 信号开销 | Figure 4：push 39–135 µs、pull ≤4 µs（EP4–64，机架内） | 博客："多节点微基准"里 push 103 µs vs pull 18 µs，约 5.8× |
| 调度开销 | 附录 A：2.1–7.7%（EP4、BF16 forward） | 博客："不到 MoE 总运行时的 3%" |
| 最优 minibatch | 512–32768 | README 推荐 2048–16384 |
| pull 的理由 | 信号、调度表 | 博客多一条带宽利用率（最高 29%） |
| DeepEP-based E2E | 761.0 | 博客 760.9 |
| "单个 kernel" | megakernel 融 dispatch / FFN / combine；router 投影、top-k、all-gather、调度表都在 launch 之前（Figure 1 右） | README：functional 层负责协调多次 kernel launch；forward 与 backward 是两个 megakernel |
| 发布时间 | arXiv v1 2026-09-28 | 博客页面日期 2026-08-04，并称"同时发布论文" |

这些差异都不改变结论，但引用具体数字时要说明出处。论文正文的数字优先。

## 10. 对 MegaMoE 论文与 Primus 的影响

### 10.1 定位：与本文最接近的训练 megakernel

| 维度 | MoK | Primus-Turbo MegaMoE |
|---|---|---|
| 硬件与 EP | GB300 NVL72，EP16 / 32 / 64，交换式单跳 | MI355X 单节点 EP8，XGMI 点对点 full mesh |
| 融合边界 | forward、backward 各一个 megakernel（dispatch + shared/routed FFN + combine） | 两个算子 `dispatch_grouped_gemm` / `grouped_gemm_combine`；另有 `fused_mega_moe_stage1` / `stage2` 两个 autograd 边 |
| dispatch 方向 | pull | **push**：`primus_turbo/flydsl/mega/bf16/dispatch_grouped_gemm_bf16_kernel.py:67` 的 `_wait_for_expert` 写明"阻塞直到每个 sender 都推完这个 expert"，用 system scope 自旋等 flag |
| combine 方向 | push | push（`grouped_gemm_combine_bf16_kernel.py:217`、`ep_intranode.py:278`） |
| 重叠粒度 | `minibatch_size` 可调 | dispatch 等待粒度是**一个 local expert 的全部 token**（同上 `_wait_for_expert`），由模型与路由决定，不可调。这是我从代码读出来的，需要 Turbo 团队确认 |
| SM / CU 划分 | forward / backward 分开可调，4–52 | `num_dispatch_cu` 16/32/16、`num_combine_cu` 64/16 |
| 主机同步 | 无（ring + 设备侧调度表） | **每次 forward 一次 D2H**，见下 |
| shared expert | 融合 | 训练步实验里关闭 |
| wgrad | 在 megakernel 内、所有 dgrad 之后 | FC2 variable-K wgrad 在融合对之外 |
| 确定性 | bitwise，作为产品特性 | 未核实；本文 L3 用确定性哈希做测试手段 |
| 框架 | Cursor 内部栈（FSDP）、NeMo AutoModel | Primus 的 Megatron 后端；TorchTitan 后端零引用 |
| 实现 | CUDA，博客说没用 kernel 框架、靠 agent 从零写；依赖 TMA、CLC、PyTorch symmetric memory | FlyDSL |
| 数字 | 层最高 2.37×（2.74×）；E2E 1.41× | 算子 BF16 1.29–1.61×（15 个 stage）；训练步 BF16 1.162–1.286×、MXFP8 1.169–1.350×（来自未发布 blog） |

**名字撞车。** MoK 把 DeepGEMM 的 MegaMoE 称为"最接近的工作"。我们的论文如果也叫 MegaMoE，审稿人很可能混淆。正文第一次出现时要明确区分（例如统一写作 Primus-Turbo MegaMoE，并说明它和 DeepGEMM 的推理 kernel 无关）。

### 10.2 主机同步：MegaMoE 恰好有 MoK 要消灭的东西

`Turbo-base-main`（`ca1fdd67`）与 `Turbo-dev`（`826936e5`）里 `primus_turbo/pytorch/ops/moe/fused_mega_moe.py:26-27` 原文："Each forward reads the rows dispatched to this rank back once (a D2H) and sizes every per-layer tensor to them, so neither entry point is CUDA-graph capturable."

具体位置：

- BF16：`primus_turbo/flydsl/mega/bf16/dispatch_prologue_kernel.py:737`，`int(num_tokens_per_expert_prefix[-1].item())`
- MXFP8：`fp8/dispatch_grouped_gemm_mxfp8_kernel.py:783`（`host.copy_`）与 `:856`（`.tolist()`）、`fp8/symm_buffer.py:429`、`fp8/quant.py:398`（注释直接写着 "D2H (sizes output/grid)"）

MI355X 节点的主机是 EPYC，不是 Grace，MoK 的 1.46–2.97× 不能直接套用。但同步本身照样会让 GPU 没法跑在 CPU 前面，也挡住了 CUDA graph。论文里要么把它列为限制，要么照 MoK 的思路改成固定容量 ring + 设备侧调度表。

### 10.3 L1 要按两个条件来写

融合算子要和框架的 collective 重叠，需要同时满足两个条件：

| 条件 | 含义 | 谁解决了 |
|---|---|---|
| A. 发得出去 | 框架要在合适的时间发出 collective；这由 hook / bucket 挂在哪个模块边界决定 | **MegaMoE**：整层包成一个 autograd Function 后，Megatron 分布式优化器失去模块边界，参数 all-gather 与梯度 reduce-scatter 退化成在融合 kernel 之后串行；`stage1` / `stage2` 拆分把边界还回来（`fused_mega_moe.py` docstring："restores the DDP comm/compute overlap granularity"） |
| B. 跑得起来 | collective 发出去后，要有 SM / CU 给它跑；持久 megakernel 占满所有 SM 时它只能排队 | **MoK**：用 CLC 让 megakernel 把 SM 让给更高优先级 stream |

MoK 在 FSDP 下没有暴露条件 A：它替换的是整个 MoE 模块，FSDP 的 all-gather 预取挂在模块边界上，不受影响。MegaMoE 在 Megatron 下暴露的正是条件 A。所以 L1 的正确写法是："这个问题有两个必要条件；MoK 在 Blackwell 上处理了 B，我们在 Megatron 上碰到并处理了 A。"既不是首创"没人讨论过"，也不是"同一个问题的两种解法"。

**由此多出一个必须核实的 TODO：MegaMoE 在条件 B 上是什么状况？** blog 的 DDP overlap trace 里，RCCL kernel 到底是和 MegaMoE kernel 本身并发，还是只和非 MoE 的 kernel 并发？如果 MegaMoE 的 grid 占满了 256 个 CU，collective 只能在它的空隙里跑。ROCm / CDNA 上有没有与 CLC 等价的"持久 kernel 让出 CU"机制，目前没核实；可能的替代方案（例如带 CU mask 的 stream，或 grid 不占满 CU）都要先查文档、再实测，不能直接写进论文。

### 10.4 push vs pull：在论文里把账算出来

- MoK 的 Figure 4 表明：去掉信号后 push 搬得比 pull 快 5–9%；pull 的净优势来自信号，而 push 的信号开销随 EP 扇入增长（EP8 时 51 µs，占 8.9%）。
- 我们自己的数据：RocMoE M1 在 XGMI 上测到 pull 每 token 比 push 慢约 43%（`notes/rocmoe/2026-06-25_1025_ref_deepgemm_mega_moe_dispatch_fc1_overlap.md`），原因记录为 XGMI RTT。
- 如果 XGMI 上 pull 的搬运代价是 43%，而 EP8 下 push 省下的信号最多只有个位数百分比，push 就是对的。论文里应该把这两个数并排写出来，作为在 EP8 / XGMI 上选 push 的依据，并说明到了 Helios EP64 结论可能反过来（`knowledge/systems/mi455x-megamoe-ep64-architecture.md` 已把 pull dispatch 作为首轮验证项）。
- RocMoE 的笔记里还有一条：用多条在途 `global_load_lds` 藏 RTT 可能缩小 pull 的差距。在做出这个实验之前，"XGMI 上 pull 慢 43%"只适用于当时那个实现。

### 10.5 必须补的实验与写作项

| 优先级 | 项 | 依据 |
|---|---|---|
| P0 | 朴素 RCCL all-to-all + PyTorch + grouped GEMM 基线 | MoK Table 1 里 EP16 小 token 时 16/16 是朴素 NCCL 最强；我们的 EP8 更小 |
| P0 | Related Work 首位写 MoK，并按 §10.1 的表逐项对照 | 最接近的训练 megakernel，开源，有生产 E2E 数字 |
| P0 | L1 按 §10.3 的两条件重写 | 避免被 MoK 直接驳倒 |
| P1 | 开启 shared expert 的配对实验 | MoK 融了 shared expert，而且放在 dispatch 空窗里 |
| P1 | 把 D2H 写成限制，或者去掉它 | §10.2 |
| P1 | 核实条件 B（trace 里 RCCL 是否与 MegaMoE kernel 本身并发） | §10.3 |
| P1 | push 选择的定量论证 | §10.4 |
| P2 | 报告口径对齐：最慢 rank 中位数、有效 FLOP 计入 shared expert、调度时间计入 | MoK §4.1；对应本文 L2 |
| P2 | 用 §4.4 的公式估 MegaMoE 的粒度下界 | 例：DSv3 形状 \(H = 7168\)、\(I = 2048\)，若计算 CU 为 224、每 CU 输出 tile 为 256×256，则 \(T \ge 2 \cdot 224 \cdot 256 \cdot 256 / 4096 = 7168\)。tile 与 CU 数是假设值，要换成 MegaMoE 的实际值 |

### 10.6 不要做

- 本季度把 MoK 整库 HIP 化。它依赖 TMA、CLC、PyTorch symmetric memory，只支持 SM100 / SM103；博客也说没用 kernel 框架，所以 HipKittens 帮不上直接的忙。
- 把 2.37× 当 AMD 平台的承诺数字。
- 把 CLC + FSDP 重叠的故事原样搬到 ROCm。

但要注意：博客明说 MoK 设计成"借助 agent 可以很快移植到别的平台"，加上 Apache 2.0 许可，别人在我们投稿前把它移植到 AMD 的可能性不能排除。

### 10.7 时间线风险

MoK 的 arXiv v1 是 2026-09-28，MLSys 2027 截稿是 2026-10-30，ParallelKittens 是 MLSys'26。它投 MLSys 2027 的可能性不小。如果被接收（2027-02-28 出结果），它在我们投 MLSys 2028 时就是已发表的先行工作。这一点需要持续跟踪，目前是推测。

## 11. 开放问题与复现清单

**问作者 / 自己要查的问题：**

1. Table 1 里 2.3–2.7× 的大加速，有多少来自消掉固定的主机开销？在 x86 主机的 NVL72（如果有）或 DGX 上会剩多少？
2. `schedule_capacity_multiplier` 超出时会怎样：丢 token、报错，还是越界？
3. 偏斜路由下 ring buffer 的绕圈次数与 replay 代价是多少？
4. E2E 实验里 CLC 是否打开？关掉以后 1.41× 会剩多少？
5. pull 在不均衡路由下带宽高 29% 的实验设置是什么？

**复现清单：**

| 项 | 状态 |
|---|---|
| 代码 | 开源，Apache 2.0；层基准代码在仓库 `benchmarks/` |
| 硬件 | 需要 GB200 / GB300 NVL72（SM100 / SM103） |
| 依赖 | Python ≥ 3.12、PyTorch ≥ 2.10、CUDA ≥ 13.0（论文实验用 3.13 / 2.13 / 13.0） |
| 超参 | README 列出 5 个：`fwd_num_comm_sms`、`bwd_num_comm_sms`、`minibatch_size`、`macrobatch_size`、`schedule_capacity_multiplier`；论文说最终配置都在仓库里 |
| 输入 | 层基准用标准正态合成输入，可复现 |
| E2E | 内部生产栈，不可复现 |

## 附录：术语

| 术语 | 含义 |
|---|---|
| Scale-up 域 | 单跳、高带宽、任意到任意互连的一组 GPU，例如 NVL72、Helios |
| Push / Pull | 由数据的生产者发远端 store / 由消费者发远端 load |
| Minibatch（MoK 用法） | 通信与计算一起流水的 token 块，即重叠粒度；不是优化器意义上的 micro-batch |
| Macrobatch | 设备侧 token 环的总容量 |
| Schedule table | `{src_rank, src_index}` 两列表，行号即目的 buffer 位置；四个通信算子共用 |
| Inter-SM overlap | 一部分 SM 专做计算、一部分专做通信，两组用片内 barrier 交接 |
| Cluster Launch Control（CLC） | Blackwell 的硬件 work stealing，持久 kernel 可以把 SM 让给更高优先级 stream |
| HybridEP | NVIDIA 在 NVL72 上推荐的 Megatron EP 通信栈 |

# Mixture-of-Kittens：面向 NVL72 的 MoE Megakernel

> 全文中译。原文 [arXiv 2609.36070v1](https://arxiv.org/abs/2609.36070)（2026-09-28）· [HTML](https://arxiv.org/html/2609.36070v1) · [代码](https://github.com/cursor/mixture-of-kittens)
> 原文许可：[CC BY 4.0](http://creativecommons.org/licenses/by/4.0/)。本译文署名原作者 Stuart H. Sul、Nash Brown、Henry Wildermuth、William Lin、Federico Cassano、Christopher Ré。改动仅有三处：译成中文；把图内表格转成 Markdown 表格；参考文献作者较多的条目缩写为 "et al."。
> 精读、证据审视以及对 MegaMoE / Primus 的影响见 [`mixture-of-kittens.md`](./mixture-of-kittens.md)。本文只译论文本身，不含 Cursor 博客与 README 的内容。
>
> **翻译体例**：正文、表格、图注、脚注、算法逐段全译。dispatch、combine、megakernel、minibatch、macrobatch、tensor core、SM、push / pull、rank 等术语保留英文，首次出现处给出中文。图 1、2、3、5、7 和图 6 左是图片，只译图注，图见原文；图 4 右和图 6 右在原文是图内表格，这里转成表格。参考文献保留英文。

**作者**：Stuart H. Sul¹²、Nash Brown¹²、Henry Wildermuth²、William Lin²、Federico Cassano²、Christopher Ré¹

¹ 斯坦福大学计算机科学系　² Cursor Research

通信作者：`ssul@cs.stanford.edu`

---

## 摘要

AI 加速器系统正在迅速收敛为纵向扩展（scale-up）架构：数十到数千块 GPU 通过高带宽、单跳的互连 fabric 通信。我们发现，为传统横向扩展（scale-out）网络优化的现有混合专家（Mixture-of-Experts，MoE）训练系统迁移到这一设定时表现很差，常常比用 PyTorch 和 NCCL 搭建的朴素基线还慢。随着业界路线图指向更大的 scale-up 域，理解这一硬件形态下的性能权衡日益重要。我们提出 Mixture-of-Kittens（MoK），一个为 Nvidia NVL72 设计的 MoE 训练系统。MoK 建立在三条能在 scale-up 域上释放性能的洞察之上：（1）为每个算子选择基于 push（推）还是基于 pull（拉）的通信；（2）重构计算–通信重叠；（3）彻底消除 CPU–GPU 同步。MoK 把这些洞察凝结为一个确定性的训练 megakernel（巨型内核），融合了 token dispatch（分发）、共享专家与路由专家的 FFN，以及 token combine（汇合）。在四个广泛使用的开源权重模型的 MoE 层形状上，MoK 的吞吐最高可达最强公开基线的 2.37×。在一次跨越多个 GB300 NVL72 机架、使用 512 块 GPU 的生产训练中，MoK 把端到端训练吞吐提升了 1.41×。

---

## 1 引言

> **图 1**：MoK 概览。左上：通信 SM 从对端 GPU 把 token 拉取到一个固定大小的 macrobatch 环形缓冲区中，专家 FFN 逐个处理 minibatch，通信 SM 再把输出推回。左下：MoK 在计算与通信之间划分 SM，并以流水方式处理各个 minibatch（M0、M1、…），使专家计算与 dispatch、combine 重叠。右：在 megakernel 启动之前，MoK 在 GPU 端完成路由投影、top-\(k\) 选择与 all-gather，以及 dispatch 调度。

训练大语言模型（LLM）的硬件正在迅速转向 scale-up 架构。厂商不再只是把小组加速器接到商品化的 scale-out 网络上，而是把数十到数千个加速器放进同一个高带宽 scale-up 域，例如 Nvidia 的 GB200/GB300 NVL72 [30, 32]、AMD 基于 MI455X 的 Helios [4]、Google 的 TPU Pod [17] 和 AWS Trn2 UltraServer [3]。在这些平台上，互连不再表现得像网络，而开始表现得像一个巨大的存储系统。这为 AI 系统设计打开了新的机会，同时也让现有设计所依赖的假设失效。

我们针对这一机会展开研究，聚焦 Nvidia NVL72 系统 [32] 上的混合专家（MoE）[42] 训练。MoE 已成为前沿 AI 模型的核心组件，并被证明能显著提升模型能力 [26, 14, 37, 11]。然而，一旦扩展到多块 GPU，MoE 就同时变得通信密集和计算密集；据先前工作报告，它因此在大规模训练中占用超过一半的端到端执行时间 [35, 52]。与此同时，Nvidia 的路线图指向越来越大的 scale-up 域，包括 NVL144、576 和 1152 [5]。作为这一家族的第一代，NVL72 为研究那些在未来硬件上会愈发重要的系统权衡提供了一个恰当的平台。

先前工作通常通过让 token 通信与专家计算重叠来加速分布式 MoE [56, 54, 53]。然而我们发现，现有方法要么不支持 scale-up 平台，要么在这类平台上变得次优。在我们评测的近一半配置中，一个用 PyTorch 和 NCCL 搭建的朴素基线胜过了所有被评测的替代方案（第 4.1 节）。这些结果表明，当前的 MoE 系统设计无法干净地迁移到 scale-up 硬件的成本结构上。

为理解这一差距，我们分析了 NVL72 上 MoE 的性能特征，并把发现提炼为三条能在 scale-up 架构上释放性能的关键洞察：

1. **选择正确的通信方向（第 3.1 节）。** 与 scale-out 域不同，scale-up fabric 让基于 pull 的通信与基于 push 的通信同样高效。这使我们可以重新思考每个 MoE 算子的通信方向：push 还是 pull。我们观察到，先前设计的方向选择是次优的，在细粒度通信中引入了本可避免的开销：多达 18% 的通信延迟花在完成信号上，多达 13% 的 MoE 运行时间花在 dispatch 之前的调度上。选择正确的一组方向（例如 pull 式 dispatch 与 push 式 combine）后，我们能把信号开销控制在 1% 以内，以最多低 2.2× 的开销构建调度，并让所有算子共享同一张调度表。
2. **重构计算–通信重叠（第 3.2 节）。** scale-up 域使得在全部 72 块 GPU 上、以在 InfiniBand 或 RoCE 上不现实的粒度进行计算–通信重叠成为可能。然而，粒度过细时 tensor core 永远无法饱和，粒度过粗则让 dispatch 和 combine 的延迟暴露在外。我们发现，在我们的各个负载中，使吞吐最大化的粒度在 512 到 32,768 个 token 之间变化。通过把粒度做成一等的可调参数，并围绕它设计 GPU 端调度、重叠方案与 megakernel 融合，我们达到了固定在次优粒度上的设计的最多 3.53× 吞吐。
3. **彻底消除 CPU–GPU 同步（第 3.3 节）。** NVL72 上的集成 CPU 比基于 DGX 的机器上的 CPU 慢，常见 PyTorch 操作的主机侧执行最多慢 2.97×。因此，必须把内存缓冲区管理完全移到设备侧，以去除 CPU–GPU 同步。为此，我们让 token 在一个读写相互重叠的固定大小环形缓冲区中循环，并在反向传播中集成一个感知环形缓冲区的前向重放（ring-aware forward replay），以恢复被覆盖的 token。与过量分配相比，我们的环形缓冲在各负载上平均只增加 1.7% 的延迟。

基于这些洞察，我们提出 Mixture-of-Kittens（MoK），一个从第一性原理出发、为 NVL72 平台构建的 MoE 训练系统。MoK 把 MoE 的全部计算与通信（即 dispatch、专家 GEMM、门控激活和 combine）融合进单个持久化 megakernel [44, 7]。除了展示上述三条洞察之外，MoK 还是一个完全可用于生产的系统。它提供了真实 MoE 训练所必需的特性，包括确定性、面向 FSDP 的 RDMA 重叠、带融合量化的 MXFP8 支持、融合的路由权重梯度计算，以及可调的 SM 划分（第 3.4 节）。

我们在四个广泛使用的开源权重模型的形状上评测 MoK：Kimi K2.7、GLM 5.2、Qwen 3.5-397B-A17B 和 DeepSeek V4 Pro。与支持 NVL72 的公开实现相比，MoK 超过最强基线最多 2.37×（MXFP8 前向）、1.78×（MXFP8 反向）、1.92×（BF16 前向）和 1.58×（BF16 反向）。在 Cursor 的生产训练栈中，在跨越多个 GB300 NVL72 机架的 512 块 GPU 上，MoK 相对此前基于 DeepEP 的实现，把以每 GPU 每秒 token 数计的端到端训练吞吐提升了 1.41×。

MoK 已开源¹，为 Cursor 的智能体编程模型 Composer [8] 在数万块 GPU 上的训练提供支撑。MoK 已作为 MoE 执行后端集成进 Nvidia NeMo AutoModel [31]。

总结起来，我们的贡献如下：

- 我们提出 MoK，一个开源的 MoE 训练系统。它把 MoE 的全部计算与通信融合进单个确定性 megakernel，并具备生产特性，包括 MXFP8 支持、面向 FSDP 的 RDMA 重叠以及融合的路由权重梯度计算（第 3 节）。
- 我们找出了在 scale-up 平台上实现高效细粒度计算–通信重叠的三条洞察：逐算子的通信方向、支持任意粒度的重叠方案，以及用设备上的环形缓冲消除 CPU–GPU 同步（第 3.1–3.3 节）。
- 我们证明，在四个广泛使用的开源权重模型形状上，MoK 相对最强公开基线最多加速 2.37×（第 4.1 节），并在一个生产训练栈中带来 1.41× 的端到端吞吐提升（第 4.2 节）。

¹ <https://github.com/cursor/mixture-of-kittens>

---

## 2 背景

本节回顾 MoE 与专家并行、GPU 与 scale-up 架构，以及相关工作。

### 2.1 MoE 与专家并行

> **图 2**：EP 度数为 3、top-1 路由的专家并行示例。

对每个输入 token，MoE 算子用一次路由投影选出 top-\(k\) 个路由专家，并为每个选中的专家分配一个路由权重。随后，共享专家与每个被选中的路由专家各自执行一个前馈网络（FFN），由 up 投影与 gate 投影、一个门控激活和一个 down 投影组成。MoE 把这些输出按归一化后的路由权重加权求和²：

\[
\operatorname{MoE}(x,s)=f_{\mathrm{shared}}(x)+\sum_{i\in\mathcal{E}}\frac{s_{i}}{\sum_{j\in\mathcal{E}}s_{j}}\,f_{i}(x)
\]

其中 \(x\in\mathbb{R}^{D}\) 是隐藏维为 \(D\) 的输入 token，\(\mathcal{E}\) 是被选中的 \(k\) 个路由专家的集合，\(s\in\mathbb{R}^{|\mathcal{E}|}\) 是它们的路由权重；\(f_{i}(x)=W_{\mathrm{down}}^{(i)}\bigl(\phi(W_{\mathrm{gate}}^{(i)}x)\odot W_{\mathrm{up}}^{(i)}x\bigr)\) 是专家 \(i\) 的 FFN，\(\phi\) 为激活函数，\(I\) 为中间维，权重 \(W_{\mathrm{up}}^{(i)},W_{\mathrm{gate}}^{(i)}\in\mathbb{R}^{I\times D}\)，\(W_{\mathrm{down}}^{(i)}\in\mathbb{R}^{D\times I}\)。

专家并行（Expert Parallelism，EP）[20, 12] 把路由专家划分到多块 GPU（即 rank）上。我们把共同持有全部专家权重的 rank 数称为 EP 度数。图 2 给出一个例子：6 个路由专家、EP 度数为 3 时，每个 rank 持有 2 个路由专家。因此，token 必须在前馈计算之前和之后按照路由选择在 GPU 之间传输。我们把专家 FFN 之前的 all-to-all 传输称为 dispatch，之后的称为 combine。

许多系统通过让 dispatch 和 combine 与专家 FFN 重叠来优化 EP [56, 54, 1, 53]。这些方案以流水方式组织 token 流：在传输一块 token 的同时，对另一块计算 FFN。MoK 是这类方案的一个变体。

² 这里我们考虑 DeepSeek V3 式的 MoE 架构 [10, 9]，但这些原则可以推广到其他 MoE 架构。

### 2.2 GPU 与 scale-up 架构

从高层看，GPU 从本地或远端 HBM 加载数据、执行计算，再把结果写回本地或远端 HBM。为此，一个 GPU kernel 要在一百多个流式多处理器（Streaming Multiprocessor，SM）上执行数以万计的硬件线程，这些线程可以同时向不同的执行单元发射指令。由于这些单元在不同的速率下饱和，要达到峰值性能，就要把它们的使用重叠起来：隐藏非关键操作，同时让关键操作的吞吐最大化。

GPU 存储层级离 SM 越远，就越是以延迟换容量。每个 SM 有 65,536 个 32 位寄存器，每个时钟周期都可访问，还有 228 KB 的片上共享内存（约 40 TB/s）。所有 SM 共享 129 MB 的 L2 缓存（约 22 TB/s），其后是 279 GB 的 HBM（8 TB/s）。线程还可以通过 NVLink（单向 900 GB/s）或 InfiniBand/RoCE（单向 100 GB/s）访问远端 GPU 的 HBM³。

像 NVL72 这样的 scale-up 架构扩展了这一层级：它通过一个高带宽、无阻塞、具有均匀单跳任意到任意连通性的 fabric，连接多得多的 GPU。这改变了 EP 这类并行策略的权衡。过去必须穿过低带宽 scale-out 网络的并行维度，如今可以完全留在 scale-up 域内，从而抬高性能上限，并使更激进的重叠成为可能。

³ 这里我们以 Nvidia GB300 NVL72 [32] 的数字作为代表性示例。

### 2.3 相关工作

#### 计算–通信重叠

大量工作为各类 AI 负载重叠 GPU 计算与通信 [50, 6, 21, 56, 54, 1, 46, 58, 49, 43]。有些系统让主机触发的 GPU 间内存传输与设备 kernel 重叠 [50, 6, 56, 49]，另一些则提出设备上的调度器，发起细粒度的设备侧通信 [54, 1]。还有一些工作提供通用的抽象与编程原语，以支持灵活的重叠策略 [58, 46]。这些技术已被应用于单个算子，如 attention [21] 和 MoE [54]，也被应用于整个模型 [43, 44]。MoK 建立在这批工作之上，专门针对 scale-up 平台上的分布式 MoE 训练。

#### 高效 MoE kernel

MoK 受到三类优化 MoE 执行的系统启发。第一类通过块稀疏重构 [13]、融合的 token 置换 [48]、低精度的按专家分组 GEMM [55] 或 IO 高效的算法重设计 [15] 来加速专家 FFN。MoK 在本地计算中复用了其中一些技术，例如 SonicMoE 的路由梯度融合（第 3.4 节）。

第二类优化 EP 通信及其与计算的重叠，手段包括层次化 all-to-all [38]、all-to-all 与专家 FFN 的自适应流水 [16]、专为重叠设计的 dispatch 与 combine kernel [56]，或与专家 FFN 融合的细粒度设备侧通信 [54, 1]。与 MoK 最接近的是 MegaMoE，即 DeepGEMM 库中的融合 EP megakernel [55]。然而，MegaMoE 是一个只做推理的前向 kernel，专为 DeepSeek-V4 在 FP8×FP4 下的推理服务而特化；MoK 则支持在 scale-up 架构上以 BF16、MXFP8 等训练偏好的精度进行完整训练。

第三类是若干端到端框架 [43, 39, 29, 19, 57, 59]，它们把高效的逐算子 kernel 与优化技术集成进完整的训练或推理栈。MoK 与这些系统互补，可以作为它们的 MoE 后端，正如它已在 Nvidia NeMo AutoModel [31] 中所做的那样。

#### Megakernel

MoK 采用 megakernel 方法 [44, 45, 7]：由单个持久化 kernel 执行多个算子，以消除逐 kernel 的启动延迟、实现算子间重叠，并尽量减少拖尾任务。在 MoK 中，dispatch、专家 FFN 和 combine 被集成进单个 megakernel。不过，megakernel 只提供在算子边界上组合 kernel 的机制，并不规定算子融合或计算–通信重叠的方式，而 MoK 的主要设计决策恰恰在这里。

---

## 3 Mixture-of-Kittens

> **图 3**：代价模型各项在一次朴素的重叠 MoE 执行上的示意。未标注的灰框表示同步，箭头表示执行依赖。

MoK 的目标是最小化端到端 MoE 延迟 \(C\)。从高层看，它可以分解为：

\[
C=\max(T_{\text{comp}},\,T_{\text{mem}},\,T_{\text{comm}})+T_{\text{non-overlap}}+T_{\text{sync}}+T_{\text{host}}. \qquad (1)
\]

其中 \(T_{\text{comp}}\)、\(T_{\text{mem}}\)、\(T_{\text{comm}}\) 分别表示计算、GPU 内存储访问和 GPU 间通信的时间。如果这些部分完美重叠，它们的联合代价就是其中的最大值。\(T_{\text{non-overlap}}\) 计入这些操作中未被重叠的部分，例如最初的 dispatch 和最后的 combine。\(T_{\text{sync}}\) 涵盖花在完成信号与内存屏障（memory fence）上的 GPU 内和 GPU 间同步时间。\(T_{\text{host}}\) 刻画主机侧开销，如 CPU–GPU 同步、内存分配和 kernel 启动延迟。图 3 在一次简单的重叠 MoE 执行上示意了这些项。

\(T_{\text{comp}}\)、\(T_{\text{mem}}\) 和 \(T_{\text{comm}}\) 还必须各自通过把对应硬件单元推到峰值利用率来单独最小化。对 \(T_{\text{comp}}\) 而言，这意味着通过流水化输入操作数让 tensor core 持续有数可算。对 \(T_{\text{comm}}\) 而言，这意味着用足够多的在途传输让网络通道饱和，并让流量均匀分布。

本节余下部分介绍最小化 \(C\) 的三个设计选择。此外，第 3.4 节讨论进一步提升生产可用性与性能的其他特性与优化。

### 3.1 选择正确的通信方向

**图 4 右（原文为图内表格）：**

| EP 度数 | 完整延迟 pull（µs） | 完整延迟 push（µs） | 信号开销 pull（µs） | 信号开销 push（µs） |
|---|---|---|---|---|
| 4 | 475 | 489 | 1 | 39 |
| 8 | 559 | 574 | 1 | 51 |
| 16 | 603 | 633 | 4 | 66 |
| 32 | 638 | 704 | −1 | 119 |
| 64 | 638 | 744 | −2 | 135 |

> **图 4**：push 式与 pull 式 dispatch 下的信号。左：push 时，GPU 0 的计算 SM 要等待来自每个对端的完成信号；pull 时，它们只需要来自本卡通信 SM 的一个 GPU 本地信号。右：NVL72 上的 dispatch 延迟与信号开销（每块 GPU 7,168 个宽度为 7,168 的 BF16 token，4 个本地专家，每个 token 4 条均匀随机的路由）。信号开销是 kernel 内完成信号相对于不带信号的同一 kernel 所增加的延迟。

本节解释为什么通信方向对细粒度 MoE 重叠很重要、MoK 为每个算子选择了哪个方向、如何优化所选的方向，以及它还带来了哪些额外好处。

在细粒度的计算–通信重叠下，信号（\(T_{\text{sync}}\)）会成为瓶颈，因为每个任务（例如一次 tensor core 矩阵乘）都必须等到确认其输入数据已到达后才能开始。通信方向决定了需要多少信号。图 4（左）以 dispatch 为例说明这一点。在 push 式 dispatch 中，每个对端每推送一块 token 都必须发送一次完成信号；而在 pull 式 dispatch 中，一旦目的 GPU 发出了它的全部 load，一个本地信号就足够了。如图 4（右）所示，push 式 dispatch 在信号上花费 39–135 µs，即 dispatch 时间的 8–18%，而且这一代价随 EP 度数增长。相比之下，pull 式的信号开销保持在 4 µs 以内。细粒度 megakernel 在每次执行中都要多次承担这一开销，代价会不断累积。

先前的 EP MoE 系统主要使用 push 式传输 [56, 23, 53]，因为对细粒度 RDMA 传输而言 push 优于 pull [51, 22]。然而在 scale-up 平台上，整个 EP 组可以装进一个 scale-up 域，这让我们可以为每个 MoE 算子独立选择通信方向。

因此，MoK 为每个算子选择方向的原则是：接收信号的 GPU 同时也是发起远端 load 或 store 的那一方。在前向传播中，每块 GPU 拉取自己的输入（pull 式 dispatch），通知本地的计算 worker，并在它们发出完成信号后推送自己的输出（push 式 combine）。反向传播与之镜像：pull 式 reverse-combine 与 push 式 reverse-dispatch。这样一来，通信与计算之间的交接从不离开本卡，跨 GPU 的信号被彻底消除。

> **算法 1**　pull 式 dispatch 调度表的构建
>
> **输入：** \(R\) 个 rank，每个 rank 有 \(N\) 个 token 和 \(L\) 个专家；top-\(K\) 路由 \(A\in\{0,\dots,RL-1\}^{R\times NK}\)，其中 \(A_{r,j}\) 是源 rank \(r\) 上第 \(j\) 条路由所选的专家；当前 rank \(c\)；本地专家 \(\ell\) 即全局专家 \(cL+\ell\)。
>
> **输出：** 调度表 \(\Phi\)，其中 \(\Phi_{p}=(r,j)\) 表示本地缓冲区位置 \(p\) 接收来自源 rank \(r\) 的第 \(j\) 条路由。
>
> 1. \(M_{r,\ell}\leftarrow\bigl|\{\,j:A_{r,j}=cL+\ell\,\}\bigr|\quad\forall\,r,\ell\)　▷ 每个（源 rank，本地专家）的路由数
> 2. \(S_{\ell}\leftarrow\sum_{\ell^{\prime}<\ell}\sum_{r}M_{r,\ell^{\prime}}\quad\forall\,\ell\)　▷ 专家 \(\ell\) 缓冲区区域的起点
> 3. **for all** 本地专家 \(\ell\) 以及满足 \(A_{r,j}=cL+\ell\) 的路由 \((r,j)\) **do**
> 4. 　　\(o\leftarrow\bigl|\{\,j^{\prime}<j:A_{r,j^{\prime}}=A_{r,j}\,\}\bigr|\)
> 5. 　　\(p\leftarrow\sum_{r^{\prime}=0}^{R-1}\min(M_{r^{\prime},\ell},\,o)+\bigl|\{\,r^{\prime}<r:M_{r^{\prime},\ell}>o\,\}\bigr|\)
> 6. 　　\(\Phi_{S_{\ell}+p}\leftarrow(r,\ j)\)
> 7. **end for**

#### 调度

选定通信方向之后，还要决定 token 传输的顺序，使其既能让网络链路饱和（\(T_{\text{comm}}\)），又能让接收端的 tensor core 保持忙碌（\(T_{\text{comp}}\)）。

先考虑 pull 式 dispatch。在每一步，GPU 都要决定：（1）从哪个对端拉取；（2）从该对端拉取哪个 token；（3）把它存到本地内存的哪个位置。对于（1），应以 round-robin 方式选择对端，使所有网络通道承载大致相同数量的在途请求。对于（2），token 应按本地专家的非降序到达，使各专家的矩阵乘不被打碎。对于（3），token 应连续地填充本地缓冲区，这样在喂给 tensor core 之前不需要额外的本地数据搬运。

MoK 通过一个单独的 kernel 满足全部三个条件：该 kernel 构建一张调度表，再把它传给 megakernel（图 1 右）。表有两列 `{src_rank, src_index}`，行号就是该 token 在目的缓冲区中的位置。dispatch 只需逐行遍历这张表。算法 1 详述了表的构建方式。概念上，它按键（本地专家 \(\ell\)，每源 rank 序号 \(o\)，源 rank \(r\)）对路由排序，其中 \(o\) 统计来自同一源 rank、指向同一专家的更早路由的个数。先按 \(\ell\) 排序让每个专家得到一段连续区域，区域内再按 \((o,r)\) 排序，就以 round-robin 方式交织了各个源 rank。MoK 的 GPU 实现将其并行化，在我们的各个负载上以 MoE 运行时间的 2.1–7.7% 构建调度，开销比 Comet [54] 小 1.4–2.2×（附录 A）。我们的基准测试计入了这部分时间（第 4 节）。

最后，调度表只需构建一次，即可被全部四个通信算子复用。例如，push 式 combine 可以把 `{src_rank, src_index}` 重新解释为 `{dst_rank, dst_index}`，推送仍会均匀地分布在各个对端之间。这张表在最坏情况下也只占几 MB，因此在各算子之间保留它的内存代价可以忽略。

### 3.2 重构计算–通信重叠

本节描述计算–通信重叠的粒度，说明最优粒度取决于负载，并解释 MoK 如何高效地支持全部粒度范围。

计算与通信的重叠需要交换大量就绪信号，每传输并计算一块 token 就要一个。我们把这样一块 token 称为一个 minibatch，它的大小决定了重叠的粒度。一个极端是极细粒度：只够填满一条 tensor core 矩阵乘累加指令的 token 数，如 Comet [54]。另一个极端是粗粒度：数万个 token，这是 DeepEP [56] 的常见用法。

两个极端都不是最优的。记 \(N_{\mathrm{routed}}\) 为路由后的 token 数，\(b\) 为 minibatch 大小，\(w\) 为负载形状（包括隐藏维与中间维）。小的 \(b\) 永远无法让 tensor core 完全饱和，每个 minibatch 都要付出一份由计算利用不足和同步开销构成的代价 \(C_{\mathrm{ineff}}(b,w)\)，从而带来细粒度惩罚 \(P_{\mathrm{fine}}(b,w)\approx\lceil N_{\mathrm{routed}}/b\rceil\,C_{\mathrm{ineff}}(b,w)\)。大的 \(b\) 则让第一个 minibatch 的 dispatch 和最后一个 minibatch 的 combine 无法被重叠，带来粗粒度惩罚 \(P_{\mathrm{coarse}}(b)\approx T_{\mathrm{dispatch}}(b)+T_{\mathrm{combine}}(b)\)。增大 \(b\) 会降低前一项、抬高后一项，因此应按负载选择 \(b\)，使二者之和 \(P_{\mathrm{fine}}(b,w)+P_{\mathrm{coarse}}(b)\) 最小。

图 5（左）从实验上展示了这一权衡。在所评测的各负载上，吞吐起初随 minibatch 增大而上升，因为 tensor core 利用率提高、每个 minibatch 的开销被摊薄；之后则持平或下降，因为更粗的 minibatch 暴露了通信空泡与 tensor core 空泡。使吞吐最大化的 minibatch 大小随模型形状和 token 数在 512 到 32,768 个 token 之间变化。

因此，MoE 系统不应在设计时固定一种粒度，而应把粒度做成一等的可调参数，并在任何粒度下都保持低开销。MoK 通过两点做到这一点：（1）最大限度地推迟计算–通信同步；（2）把全部计算与通信融合进单个 megakernel。

> **图 5**：左：16 块 GPU 上 MoK MXFP8 前向吞吐随 minibatch 大小的变化。颜色区分模型形状，符号区分每块 GPU 的输入 token 数 \(N\)，放大的符号标出每条曲线上吞吐最大的 minibatch 大小。对固定的模型与 token 数，最佳 minibatch 大小的吞吐是最差的 1.38×–3.53×。右：四个 minibatch（M0–M3）下的最大延迟同步。通信（黄色）与计算（红色）以不同的顺序遍历工作，只在虚线箭头处会合。

#### 最大延迟同步

MoK 让计算与通信在数据依赖允许的范围内尽可能地向前推进。图 5（右）说明了这一点。通信对各 minibatch 广度优先地推进，连续 dispatch 全部 token，不等待计算完成；计算则深度优先地推进，让一个 minibatch 跑完整个专家 FFN 后再处理下一个。两侧只在虚线箭头处异步地相互通知：计算在那里等待某个 minibatch 到达，combine 在那里等待它的输出。

#### Kernel 融合

MoK 把全部计算与通信融合进单个 megakernel [44]，并通过 SM 间重叠（inter-SM overlapping）[46] 让它们并发执行。一部分 SM 专门分配给专家 FFN，另一部分专门负责 dispatch 和 combine。两组通过 GPU 本地的 barrier 相互通知。如果没有 megakernel，图 5（右）的调度就需要为每个 minibatch 启动好几个 kernel，并通过多条 stream 来重叠它们。这会让每个 minibatch 承担数十微秒的 kernel 启动与 stream 同步代价（\(T_{\text{host}}\)），还会把交错执行交给硬件调度器，而后者并不保证计算与通信的并发执行 [33]。

### 3.3 消除 CPU–GPU 同步

本节量化 NVL72 平台上 CPU–GPU 交互的代价，介绍 MoK 的 token 环形缓冲，并解释它如何在前向与反向传播中都保持高效。

相对传统的 DGX 系统，GB300 NVL72 系统上的主机侧执行很慢 [24]。在图 6（左）中，我们在两个平台上测量常见 PyTorch 操作的主机侧调用时间，发现 NVL72 的主机路径慢 1.46× 到 2.97×。我们观察到，在生产规模下，GPU 执行因此更可能被主机工作卡住，而 CPU–GPU 同步的代价也高得多，因为它阻止 GPU stream 跑在 CPU 线程前面。

EP 执行中这类同步的一个常见来源是内存分配。由于路由到每个专家的 token 数要到运行时才知道，分配精确大小的目的缓冲区就需要在分配之前把这些计数拷贝到 CPU [13]。另一种做法是丢弃超出固定缓冲区容量的 token [20, 12]，但这会降低训练质量 [13]。

MoK 通过完全在设备上管理一个 token 环形缓冲区，既避免了 CPU–GPU 同步，也避免了丢弃 token。我们把这个环形缓冲区称为 macrobatch，并把它的大小暴露为可调参数。环形缓冲区在 dispatch 与专家 FFN 之间、以及专家 FFN 与 combine 之间形成生产者–消费者流水线。每个生产者写入 macrobatch 的一个区域，同时它的消费者处理另一个区域。例如在图 1（左上）中，dispatch 把 token 写入第三个槽位，同时专家 FFN 处理第二个槽位，combine 从第一个槽位读取。环绕回起点时同样的流水继续进行，因此环的边界几乎不增加开销。过量分配一个能容纳全部路由 token 的缓冲区在生产中并不现实，但它代表了可达到的最佳性能；与之相比，环形缓冲在我们的各负载上按几何平均只增加 1.7% 的延迟（图 6 右）。

**图 6 右（原文为图内表格）：** 环形缓冲相对过量分配的延迟增加

| MoE 形状 | 每 rank 本地 token 1,024 | 2,048 | 4,096 |
|---|---|---|---|
| Kimi K2.7 Code | 0.3% | 1.2% | 0.7% |
| GLM-5.2 | 7.4% | 4.6% | 1.9% |
| Qwen3.5-397B-A17B | 0.7% | 1.3% | 2.5% |
| DeepSeek-V4-Pro | 0.1% | −0.5% | 1.0% |

> **图 6**：左：GB300 NVL72 上常见 PyTorch 操作的主机侧调用时间，以 DGX B300 为基准归一化。横轴为操作，纵轴为归一化的 CPU 时间。右：MoK 的 macrobatch 环形缓冲区的延迟开销。我们把一个能容纳一半路由 token 的环形缓冲区，与一个能容纳全部路由 token 的过量分配缓冲区相比较，报告延迟增加的百分比。

在生产训练中，设备内存放不下前向传播的全部中间激活，因此反向传播必须通过前向重放重算其中一部分。然而，朴素地重跑 MoK 前向传播只能保住最后一个 macrobatch 的激活，因为环形缓冲会覆盖之前的一切；而且前向 megakernel 还会重算反向传播根本用不到的激活（例如 down 投影的输出）。

因此，MoK 把一个感知环形缓冲区的前向重放直接融进反向 megakernel：它只恢复缺失的中间量，并与反向传播重叠。我们还通过让前向结束后留在环中的 token 尽可能多，来进一步减少重放。如果前向按升序处理各 macrobatch，就只有最后一个 macrobatch 能留下来，而除非全部 token 都装得进一个 macrobatch，它通常是不满的。MoK 的前向传播转而按 macrobatch 的逆序处理 token，于是结束时环是满的，反向传播需要重放的部分最少。

### 3.4 其他特性与优化

除了本节描述的三个设计选择之外，MoK 还实现了对生产 MoE 训练必不可少的其他特性与优化。

#### 确定性

为了支持训练消融与 on-policy 强化学习，MoK 是完全确定性的。浮点运算的顺序在各次执行之间是固定的，因此无论硬件如何调度，相同的输入都产生逐位相同的输出。GPU 上非确定性的主要来源是原子操作。MoK 要么通过软件调度，强制对同一地址的原子操作按固定顺序执行；要么把它们替换为写入临时缓冲区，再由另一个 worker 按确定的顺序归约。

#### 通过 Cluster Launch Control 实现 RDMA 重叠

MoK 通过 Cluster Launch Control（簇启动控制，CLC）[34] 调度其任务。CLC 是 Nvidia Blackwell 引入的一种硬件原生的工作窃取（work-stealing）机制。基于 CLC 的 kernel 可以自然地把 SM 让给更高优先级 stream 上的 kernel，而不必先完全执行完毕。这一点很重要，因为在训练中，我们经常需要让机架内的通信与计算和跨机架通信（例如 FSDP 的 all-gather）重叠，而后者需要自己的 SM。没有 CLC，这些跨机架通信就会排在 megakernel 后面串行执行，导致性能下降。

#### MXFP8 支持

为了更好地利用硬件，MoK 同时支持 BF16 与 MXFP8 [40] 执行。在 MXFP8 模式下，为了训练稳定性，共享专家 FFN 保持 BF16。MXFP8 带来的一项额外开销是张量在进入 tensor core 之前必须先量化。为了尽量减少这一开销，MoK（1）提供一个优化过的 MXFP8 量化 kernel，用于预先量化权重；（2）把激活量化融合进 dispatch、按专家分组的 GEMM 以及门控激活计算。

#### 路由权重梯度计算

MoK 的反向传播还采用 SonicMoE 式的计算 [15] 求出路由权重梯度。MoK 不物化 down 投影的输出，而是用门控激活与 down 投影输入梯度的内积来计算路由权重梯度。这一计算被融合进门控激活的反向，以避免额外的 HBM 流量。

#### 共享专家

许多开源权重的 MoE 架构包含一个共享专家，每个 token 不论路由如何都要经过它。共享专家的负载在各个 EP rank 上是均匀的，也不依赖 dispatch 或 combine。因此，MoK 把共享专家 FFN 放进第一个 dispatch 窗口，那时计算 SM 本来会闲置。

#### 计算–通信 SM 调优

计算 SM 与通信 SM 之间的最优划分取决于负载 [46]。本地 token 数、模型形状以及硬件的计算与网络带宽，都会改变 dispatch 和 combine 相对专家 FFN 任务所需的时间。即使在同一负载内，前向与反向的最优值也不同。因此，MoK 把通信 SM 的数量作为可调参数，前向与反向分别设置。

---

## 4 实验

第 3 节介绍了 MoK 及其三个关键设计选择：通信方向、重叠粒度和环形缓冲。为评估它们的效果，我们在两个层面上对 MoK 做基准测试：（1）独立的 MoE 前向与反向传播的有效计算吞吐（第 4.1 节）；（2）在一个使用 512 块 GPU 的生产训练栈上的端到端训练吞吐（第 4.2 节）。所有实验都在 GB300 NVL72 机架上运行，每个机架包含 72 块通过第五代 NVLink [32] 全互连的 B300 GPU，软件为 CUDA 13.0、Python 3.13 和 PyTorch 2.13。我们尽力调优了每个基线以获得最佳性能，所有最终配置都可以在我们的开源仓库中查看。

### 4.1 MoE 层基准

我们对单个 MoE 层的完整执行做基准测试，包括全局 token 调度、dispatch、路由专家与共享专家的 FFN、combine，以及按路由权重的加权求和。路由 logits 与模型输入从标准正态分布中生成，每个实现都收到它们逐位相同的副本。我们在 BF16 和 MXFP8 下分别测量前向与反向传播，在 500 次预热迭代之后对 100 次执行计时。

对每次执行，我们取最慢 rank 的延迟，记录 100 次执行的中位数，并换算为有效专家 FFN 吞吐。对于 \(N\) 个输入 token、隐藏维 \(D\)、专家中间维 \(I\)、top-\(k\) 路由和测得时间 \(t\)，前向吞吐为 \(6N(k+1)DI/t\)，其中 \(k+1\) 计入 \(k\) 个路由专家和一个共享专家。反向传播使用两倍于此的 FLOP 数。

> **图 7**：GB300 NVL72 上每块 GPU 的 MoE 层吞吐。横轴为 MoE 形状，纵轴为以 TFLOP/s 计的有效专家 FFN 吞吐。我们使用 EP 度数 64，路由前每块 GPU 2,048 个输入 token。越高越好。

我们与满足以下两个条件的公开基线比较：（1）同时支持 MoE 前向与反向传播；（2）支持在 NVL72 系统上运行：

- NCCL [27] + PyTorch [36]
- DeepEP [56] + PyTorch
- DeepEP + TransformerEngine [28]
- HybridEP [53]⁴ + Megatron [43]

⁴ Megatron 也支持 DeepEP，但 Nvidia 推荐在 NVL72 上使用 HybridEP 以获得最佳性能。

我们评测四个广泛使用的开源权重模型的 MoE 层形状。每个形状都包含一个共享专家、\(E\) 个路由专家、隐藏维 \(D\)、专家中间维 \(I\)，以及每个 token \(k\) 条路由：

- Kimi K2.7 Code [26]（\(E{=}384\)，\(D{=}7168\)，\(I{=}2048\)，top-\(k{=}8\)）
- GLM-5.2 [14]（\(E{=}256\)，\(D{=}6144\)，\(I{=}2048\)，top-\(k{=}8\)）
- Qwen3.5-397B-A17B [37]（\(E{=}512\)，\(D{=}4096\)，\(I{=}1024\)，top-\(k{=}10\)）
- DeepSeek-V4-Pro [11]（\(E{=}384\)，\(D{=}7168\)，\(I{=}3072\)，top-\(k{=}6\)）

图 7 给出了模型形状、传播方向（前向与反向）和精度（MXFP8 与 BF16）每种组合下的 MoE 层吞吐，使用 EP 度数 64、dispatch 前每块 GPU 2,048 个 token。相对每种组合下最快的基线，MoK 的加速比在 MXFP8 前向上最高达 2.37×，在 MXFP8 反向上为 1.78×，在 BF16 前向上为 1.92×，在 BF16 反向上为 1.58×。

我们观察到，MoK 的相对收益在 MXFP8 上大于 BF16，在前向上大于反向。这是因为 GB300 上 BF16 tensor core 的峰值吞吐只有 MXFP8 的一半 [32]，而反向传播执行的浮点运算大约是前向的两倍。这两个因素都让执行更偏计算瓶颈，花在 GPU 间通信上的时间比例更小，而这部分正是 MoK 能有效重叠或消除的。因此，最大的相对收益出现在 MXFP8 前向，最小的出现在 BF16 反向。我们预计这类收益在未来硬件上会增大；例如，与 GB300 相比，Nvidia Vera Rubin [2] 把 FP8 计算吞吐提高了 3.5×，而 scale-up 网络带宽只提高了 1.67×，执行会更偏通信瓶颈。

> **表 1**：MoK 对 EP 度数与本地 token 数的敏感性。"Tokens"为 dispatch 前每块 GPU 的本地 token 数。"Best"为该负载下四个基线中的最低延迟，括号内标明是哪个基线（N 为 NCCL+PyTorch，D 为 DeepEP+PyTorch，H 为 HybridEP+Megatron）。"加速比"为"Best"除以"MoK"。延迟越低、加速比越高越好。

| EP | MoE 形状 | Tokens | 前向 MoK（ms） | 前向 Best（ms） | 反向 MoK（ms） | 反向 Best（ms） | 加速比 前向 | 加速比 反向 |
|---|---|---|---|---|---|---|---|---|
| 16 | Kimi K2.7 Code | 1,024 | 0.901 | 2.334 (N) | 1.786 | 2.413 (N) | 2.59× | 1.35× |
| | | 2,048 | 1.332 | 2.581 (N) | 2.425 | 3.329 (N) | 1.94× | 1.37× |
| | | 4,096 | 2.555 | 3.308 (H) | 4.320 | 5.053 (H) | 1.29× | 1.17× |
| | GLM-5.2 | 1,024 | 0.849 | 2.321 (N) | 1.284 | 2.070 (N) | 2.73× | 1.61× |
| | | 2,048 | 1.180 | 2.367 (N) | 1.965 | 2.762 (N) | 2.01× | 1.41× |
| | | 4,096 | 2.051 | 2.913 (H) | 3.514 | 4.147 (H) | 1.42× | 1.18× |
| | Qwen3.5-397B-A17B | 1,024 | 0.857 | 2.316 (N) | 1.440 | 1.915 (N) | 2.70× | 1.33× |
| | | 2,048 | 0.901 | 2.307 (N) | 1.296 | 2.129 (N) | 2.56× | 1.64× |
| | | 4,096 | 1.705 | 2.777 (N) | 2.180 | 3.005 (H) | 1.63× | 1.38× |
| | DeepSeek-V4-Pro | 1,024 | 0.992 | 2.319 (N) | 2.321 | 2.571 (N) | 2.34× | 1.11× |
| | | 2,048 | 1.560 | 2.612 (N) | 3.139 | 3.343 (N) | 1.67× | 1.06× |
| | | 4,096 | 2.762 | 3.555 (H) | 5.157 | 5.462 (D) | 1.29× | 1.06× |
| 64 | Kimi K2.7 Code | 1,024 | 1.016 | 2.600 (N) | 1.555 | 2.295 (H) | 2.56× | 1.48× |
| | | 2,048 | 1.293 | 2.570 (H) | 2.045 | 2.636 (H) | 1.99× | 1.29× |
| | | 4,096 | 2.524 | 4.002 (H) | 3.871 | 4.709 (H) | 1.59× | 1.22× |
| | GLM-5.2 | 1,024 | 0.989 | 2.706 (H) | 1.612 | 2.287 (H) | 2.74× | 1.42× |
| | | 2,048 | 1.142 | 2.471 (H) | 1.737 | 2.414 (H) | 2.16× | 1.39× |
| | | 4,096 | 2.171 | 3.354 (H) | 3.294 | 4.147 (H) | 1.54× | 1.26× |
| | Qwen3.5-397B-A17B | 1,024 | 0.995 | 2.567 (N) | 1.442 | 2.348 (H) | 2.58× | 1.63× |
| | | 2,048 | 1.068 | 2.527 (H) | 1.273 | 2.267 (H) | 2.37× | 1.78× |
| | | 4,096 | 1.830 | 2.978 (H) | 2.137 | 2.959 (H) | 1.63× | 1.38× |
| | DeepSeek-V4-Pro | 1,024 | 0.981 | 2.670 (N) | 1.483 | 2.470 (H) | 2.72× | 1.67× |
| | | 2,048 | 1.392 | 2.564 (H) | 2.321 | 2.839 (H) | 1.84× | 1.22× |
| | | 4,096 | 2.594 | 3.796 (N) | 4.346 | 5.112 (H) | 1.46× | 1.18× |

#### 对 EP 度数与本地 token 数的敏感性

表 1 报告了 EP 度数为 16 与 64、每块 GPU 1,024、2,048 与 4,096 个 token 下的 MXFP8 延迟与加速比。本地 token 数较小时 MoK 的加速更大，因为此时通信与调度开销在执行时间中占比更大。这一区间在大规模训练中很重要，因为全局 batch size 不能无限增大而不出现优化收益递减。因此，扩展到更多 GPU 就必须减少每块 GPU 处理的 token 数 [25, 41, 18]，这使 MoK 随部署规模的扩大而愈发有效。

### 4.2 端到端基准

我们还在内部的生产训练栈中评测 MoK。对照组是该栈的上一个版本：它让基于 DeepEP 的 MoE 通信与我们开源的 MXFP8 专家 FFN kernel [47] 重叠。我们把这一版本称为 DeepEP-based。我们只改变 MoE 实现，在多个 GB300 NVL72 机架上测量以每 GPU 每秒 token 数计的端到端训练吞吐。

> **表 2**：512 块 GB300 GPU 上生产训练栈的端到端吞吐，比较 DeepEP-based 后端与 MoK。两次运行都使用 EP 度数 32、MXFP8 路由专家和 BF16 共享专家。我们报告每 GPU 吞吐的均值。越高越好。

| | DeepEP-based | MoK |
|---|---|---|
| 每 GPU 每秒 token 数 | 761.0 | 1,070.2 |

如表 2 所示，用 MoK 替换基于 DeepEP 的实现，使端到端训练吞吐提升了 41%。实际上，我们的生产训练运行使用的 GPU 远多于这次实验，每块 GPU 的 token 也更少；如第 4.1 节所述，这会进一步增大 MoK 的收益。

---

## 5 结论

随着 AI 加速器向大型 scale-up 域收敛，现有分布式 MoE 系统背后的设计选择日益次优。我们研究了 NVL72 上的 MoE 训练，找出了释放其性能的三条洞察：逐算子选择通信方向、围绕可调粒度重构计算–通信重叠，以及通过设备侧环形缓冲消除 CPU–GPU 同步。Mixture-of-Kittens（MoK）把这些洞察体现在单个确定性 megakernel 中，融合了 MoE 的计算与通信，并具备硬件加速的 MXFP8 执行、面向 FSDP 的 RDMA 重叠、融合的路由权重梯度计算等对生产至关重要的特性。在广泛使用的开源权重模型形状上，MoK 大幅超越最强的公开基线，并在生产规模的 LLM 训练中带来端到端吞吐的提升。

## 致谢

感谢 Cursor 使这项工作成为可能。感谢 Vivien Cheng、Dan Fu、Hermann Kumbong、Jerry Liu、Chen Lu、Sasha Rush、Nathan Wang 和 Less Wright 在本工作期间提供的有益反馈与讨论。

---

## 参考文献

保留英文；作者较多的条目缩写为 "et al."。

1. O. Aimuyo, B. Oh, and R. Singh. FlashMoE: Fast distributed MoE in a single kernel. NeurIPS 2025, vol. 38, pp. 100676–100699.
2. E. Alvarez, V. Mehta, and F. Ghodsian. Inside NVIDIA Rubin GPU architecture: Powering the era of agentic AI. NVIDIA Developer Blog, 2026. Accessed 2026-09-12.
3. Amazon Web Services. Amazon EC2 Trn2 instances and UltraServers. 2024. Accessed 2026-08-20.
4. AMD. AMD Helios rackscale solution. 2026. Accessed 2026-08-20.
5. R. Bhargava, T. Allison, and H. Petty. NVIDIA Vera Rubin POD: Seven chips, five rack-scale systems, one AI supercomputer. NVIDIA Developer Blog, 2026. Accessed 2026-09-03.
6. L. Chang et al. FLUX: Fast software-based communication overlap on GPUs through kernel fusion. arXiv:2406.06858, 2024.
7. X. Cheng et al. MPK: A compiler and runtime for mega-kernelizing tensor programs. OSDI 2026, pp. 1909–1926.
8. Cursor Research et al. Composer 2 technical report. arXiv:2603.24477, 2026.
9. D. Dai et al. DeepSeekMoE: Towards ultimate expert specialization in mixture-of-experts language models. ACL 2024, pp. 1280–1297.
10. DeepSeek-AI et al. DeepSeek-V3 technical report. arXiv:2412.19437, 2024.
11. DeepSeek-AI et al. DeepSeek-V4: Towards highly efficient million-token context intelligence. arXiv:2606.19348, 2026.
12. W. Fedus, B. Zoph, and N. Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. JMLR 23(120):1–39, 2022.
13. T. Gale, D. Narayanan, C. Young, and M. Zaharia. MegaBlocks: Efficient sparse training with mixture-of-experts. MLSys 2023, vol. 5, pp. 288–304.
14. GLM-5-Team et al. GLM-5: From vibe coding to agentic engineering. arXiv:2602.15763, 2026.
15. W. Guo, M. Mishra, X. Cheng, I. Stoica, and T. Dao. SonicMoE: Accelerating MoE with IO and tile-aware optimizations. ICLR 2026, pp. 67814–67839.
16. C. Hwang et al. Tutel: Adaptive mixture-of-experts at scale. MLSys 2023, vol. 5, pp. 269–287.
17. N. P. Jouppi et al. TPU v4: An optically reconfigurable supercomputer for machine learning with hardware support for embeddings. ISCA 2023.
18. J. Kaplan et al. Scaling laws for neural language models. arXiv:2001.08361, 2020.
19. W. Kwon et al. Efficient memory management for large language model serving with PagedAttention. SOSP 2023, pp. 611–626.
20. D. Lepikhin et al. GShard: Scaling giant models with conditional computation and automatic sharding. ICLR 2021.
21. H. Liu, M. Zaharia, and P. Abbeel. RingAttention with blockwise transformers for near-infinite context. ICLR 2024, pp. 3992–4008.
22. Y. Ma et al. Demystifying NVSHMEM: A system-level analysis on symmetric memory and device-initiated operations in GPU communication. arXiv:2606.05951, 2026.
23. Z. Mao et al. UEP: Portable expert-parallel communication. OSDI 2026, pp. 1107–1123.
24. Marks101. PyTorch: High CPU overhead on Grace systems. NVIDIA/TransformerEngine issue #2053, 2025. Accessed 2026-08-27.
25. S. McCandlish, J. Kaplan, D. Amodei, and OpenAI Dota Team. An empirical model of large-batch training. arXiv:1812.06162, 2018.
26. Moonshot AI. Kimi K2.7 Code. Hugging Face, 2026. Accessed 2026-08-20.
27. Nvidia. NCCL: Optimized primitives for collective multi-GPU communication. GitHub, 2015.
28. Nvidia. TransformerEngine: A library for accelerating Transformer models on Nvidia GPUs, including using 8-bit and 4-bit floating point (FP8 and FP4) precision on Hopper, Ada and Blackwell GPUs. GitHub, 2022.
29. Nvidia. TensorRT-LLM. GitHub, 2023.
30. Nvidia. Nvidia GB200 NVL72. 2024. Accessed 2026-08-20.
31. Nvidia. NeMo AutoModel: DTensor-native SPMD library for scalable and efficient training. GitHub, 2025.
32. Nvidia. Nvidia GB300 NVL72. 2025. Accessed 2026-08-20.
33. Nvidia. Green Contexts. CUDA Programming Guide, 2026. Accessed 2026-08-29.
34. Nvidia. Work stealing with Cluster Launch Control. CUDA Programming Guide, 2026. Accessed 2026-08-29.
35. X. Pan et al. FSMoE: A flexible and scalable training system for sparse mixture-of-experts models. ASPLOS 2025, pp. 524–539.
36. A. Paszke et al. PyTorch: An imperative style, high-performance deep learning library. NeurIPS 2019, pp. 8026–8037.
37. Qwen Team. Qwen3.5: Towards native multimodal agents. Blog, 2026. Accessed 2026-08-20.
38. S. Rajbhandari et al. DeepSpeed-MoE: Advancing mixture-of-experts inference and training to power next-generation AI scale. ICML 2022, pp. 18332–18346.
39. J. Rasley, S. Rajbhandari, O. Ruwase, and Y. He. DeepSpeed: System optimizations enable training deep learning models with over 100 billion parameters. KDD 2020, pp. 3505–3506.
40. B. D. Rouhani et al. Microscaling data formats for deep learning. arXiv:2310.10537, 2023.
41. C. J. Shallue et al. Measuring the effects of data parallelism on neural network training. JMLR 20(112):1–49, 2019.
42. N. Shazeer et al. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. ICLR 2017.
43. M. Shoeybi et al. Megatron-LM: Training multi-billion parameter language models using model parallelism. arXiv:1909.08053, 2019.
44. B. Spector, J. Juravsky, S. H. Sul, O. Dugan, D. Lim, D. Fu, S. Arora, and C. Ré. Look ma, no bubbles! Designing a low-latency megakernel for Llama-1B. Hazy Research Blog, 2025. Accessed 2026-08-20.
45. B. Spector, J. Juravsky, S. H. Sul, D. Lim, O. Dugan, S. Arora, and C. Ré. We bought the whole GPU, so we're damn well going to use the whole GPU. Hazy Research Blog, 2025. Accessed 2026-09-21.
46. S. H. Sul, S. Arora, B. F. Spector, and C. Ré. ParallelKittens: Systematic and practical simplification of multi-GPU AI kernels. MLSys 2026, vol. 8, pp. 1076–1089.
47. S. H. Sul. 1.5x faster MoE training with custom MXFP8 kernels. Cursor Blog, 2025. Accessed 2026-09-13.
48. S. Tan, Y. Shen, R. Panda, and A. Courville. Scattered mixture-of-experts implementation. COLM 2024.
49. V. Thakkar et al. CUTLASS: CUDA templates and Python DSLs for high-performance linear algebra. GitHub, 2017.
50. Y. Wang et al. Introducing async tensor parallelism in PyTorch. PyTorch Forums, 2024. Accessed 2026-08-20.
51. X. Wei, Z. Dong, R. Chen, and H. Chen. Deconstructing RDMA-enabled distributed transactions: Hybrid is better! OSDI 2018, pp. 233–251.
52. Z. Yan et al. Scalable training of mixture-of-experts models with Megatron Core. arXiv:2603.07685, 2026.
53. F. Yu, T. Liu, and K. Sun. Optimizing communication for mixture-of-experts training with hybrid expert parallel. NVIDIA Developer Blog, 2026. Accessed 2026-08-20.
54. S. Zhang et al. COMET: Fine-grained computation-communication overlapping for mixture-of-experts. MLSys 2025, vol. 7.
55. C. Zhao et al. DeepGEMM: Clean and efficient BLAS kernel library on GPU. GitHub, 2025.
56. C. Zhao et al. DeepEP: An efficient expert-parallel communication library. GitHub, 2025.
57. L. Zheng et al. SGLang: Efficient execution of structured language model programs. NeurIPS 2024, pp. 62557–62583.
58. S. Zheng et al. Triton-distributed: Programming overlapping kernels on distributed AI systems with the Triton compiler. arXiv:2504.19442, 2025.
59. K. Zhu et al. NanoFlow: Towards optimal large language model serving throughput. OSDI 2025, pp. 749–765.

---

## 附录 A　调度开销

本节比较 MoK 与 Comet [54] 的调度开销。在 MoK 中，调度包括本地 top-\(k\) 路由的 all-gather，以及随后按算法 1 构建 dispatch 调度表。在 Comet 中，调度包括本地 top-\(k\) 路由的 all-gather，随后是每专家的 token 计数与 scatter 下标计算、按源 rank 对路由排序，以及主机侧的 GEMM tile 调度。

我们在 EP 度数 4 下测量 BF16 前向传播，负载、输入和本地 token 数与第 4.1 节相同，在 500 次预热迭代后对 100 次迭代计时，报告最慢 rank 延迟的中位数。由于 MoK 需要 Blackwell GPU，而 Comet 只支持到 Hopper 为止的 GPU，两者的绝对时间不可比。因此我们比较两个系统各自的相对调度开销。由于 MoK 上的前向传播更快，这一比较对 MoK 不利、对 Comet 有利。结果见表 3。

> **表 3**：EP 度数 4 下 BF16 前向传播中 MoK 与 Comet 的调度开销。"Tokens"为 dispatch 前每块 GPU 的本地 token 数。"调度"为 dispatch 开始前构建调度所花的时间，"总计"为"调度"加上前向延迟，二者单位均为 ms；"占比"为"调度"除以"总计"。

| MoE 形状 | Tokens | MoK（GB300）调度 | MoK 总计 | MoK 占比 | Comet（H100）调度 | Comet 总计 | Comet 占比 |
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

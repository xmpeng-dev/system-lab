# MegaMoE 论文:路线 A(MLSys 工业轨 / 生产系统论文)写作计划

> **When**: 2026-10-28 21:15 UTC+8(21:30 按用户决定"肯定走路线 A"重写)
> **Decision**: 用户已定 —— **路线 A,生产系统论文,不声称机制新颖性**。目标下一班 MLSys(2028,预计 2027-10 月底)。不考虑 blog 发不发。
> **前置**: 会议普查 + 三个评审硬伤 + 对照实验方案见 [`2026-10-28_2050_megamoe-paper-plan.md`](./2026-10-28_2050_megamoe-paper-plan.md)
> **源材料**: `/perf_apps/xiaoming/rocm-blog/blogs/software-tools-optimization/primus-mega-moe/README.md`(561 行 + 5 图,未发布草稿)

## TL;DR

1. **工业轨的匿名规则对路线 A 是决定性利好,而且我上一轮不知道这条。** 官方原文:工业轨"authors are required to anonymize the author names, but they are allowed to include **company names, product names, URLs, and other identifying details** as appropriate to the work"。**AMD、MI355X、Primus、Primus-Turbo、MegaMoE 全部可以照写。**blog 的内容几乎可以逐字迁移,不用做那种把自己写成无名氏的扭曲。
2. **平台特异性从劣势翻成了资产。** 路线 B 里"这是 AMD"是减分项(把别人的想法移植一遍);路线 A 里**"这是 AMD"就是贡献本身** —— 至今没有人在这个深度上报告过 CDNA4 上的融合式 EP MoE 训练,而 gfx950 的逐指令 visibility scope、XCD 感知的 CU 静态划分这些恰恰是工业轨要的"真实平台经验"。
3. **资格条件有一条是新的硬门槛,必须正面处理:**"Submissions must describe a real system"(已部署系统的回顾性评估 / 产品或部署 roadmap 上的在建系统 / 带教训的已取消系统),且"first author and the majority of authors must be from industry"。后者满足(8 位全 AMD);前者要求论文里**如实刻画部署程度** —— 这是现有 blog 完全没写的一节。
4. **工业轨的评审标准是 impact、lessons、experiences,不是 novelty** ⇒ 论文的差异化不在数字,在**教训章节**。MegaMoE 手上有六条能写的教训,其中几条研究轨根本不会接受(比如对静默数值回归的处置是 vendoring),而工业轨明确欢迎。
5. **更正我上一轮的一个说法:artifact evaluation 大概率不影响录取决定**(自愿、录取之后进行)。我之前说"AE 徽章是把 borderline 推过线的现实手段",这是错的,P5 的优先级要相应下调。详见 §1 注。
6. **工作量约 6 个月,跑道 12 个月。** 多出来的 5 个月不要用来提前交,用来把论文从"能过"抬到"明确接收":gfx1250 第二平台(对冲硬件过时)+ liveness 压力实验。

---

## 1. 工业轨的硬性条件(2026-10-28 从 MLSys 2027 官方 CFP 核实)

| 项 | 值 | 对路线 A 的含义 |
|---|---|---|
| **匿名化** | 只隐作者名;**公司名、产品名、URL、其它识别信息均可保留** | **最大利好。**blog 内容可近乎逐字迁移 |
| **评审标准** | impact、lessons and experiences from building real-world large-scale ML systems、influence on the MLSys community。**明确不要求 novelty** | 差异化在教训,不在数字 |
| **作者资格** | 第一作者与**多数作者必须来自业界** | ✓ 8 位全 AMD |
| **系统资格** | **必须描述一个真实系统**:已部署系统的回顾性评估 / 产品或部署 roadmap 上的在建系统 / 带教训的已取消系统。"Speculative, small, or exploratory projects without a real system do not qualify" | **需新写一节刻画部署程度**(§4 A3) |
| **换轨** | **投稿截止后不可更换轨道**(两轨提交规则不同) | 轨道选择是不可逆决定,投前定死 |
| **篇幅** | 双栏 **10 页**(不含参考文献);附录页数不限但单独上传,**审稿人没有义务读** | 10 页放不下 blog 的 561 行 + 5 图,必须做取舍;细表进附录但不能指望被读 |
| **格式** | `mlsys2025style.zip`(2027 届沿用 2025 格式) | — |
| **AI 使用声明** | 2027 届起**要求在参考文献之后加一节** | 来源非官方页原文,**待核实** |
| **投稿流程** | 2027 届起走 OpenReview(此前为私有 HotCRP);所有作者需有完整 OpenReview profile,**提前两周建好** | 行政前置,别卡在这 |
| **作者篇数配额** | CFP 确有 "Authorship Quota" 一节,**但抓取时正文被截断,具体数字未知** | **必须核实**,它会约束同期投几篇 |
| **评审节奏** | 评审意见 → 作者回应**只有 3 天** → 约 6 周后通知 | 可预判的质疑必须写进正文,不能留到 rebuttal |
| **Artifact Evaluation** | **自愿、在录取之后进行、不影响录取决定** | 来源非官方页原文,**待核实**;若属实则 AE 只是加分徽章,不是过线手段 |

> **更正记录。** 上一版本我写"冲 AE 徽章是把 borderline 推过线的现实手段"。按本次抓到的描述,AE 在 MLSys 是录取后的自愿环节、与录取决定无关,那条建议不成立。该条来自第三方整理页而非 MLSys 官方页,所以我把它标为待核实而不是直接当定论 —— 但在核实之前,**不应把 AE 当作提升录取概率的手段来排期**。

---

## 2. 为什么 A 是对的选择(压缩自上一版的竞品分析)

MLSys 2026 的录取名单同时说明了两件事。

**一,这个品类在 MLSys 是主流。** AXLearn(Google,自述 production system)、veScale-FSDP(ByteDance)、GUARD(Meta,万卡 straggler 检测)、Sparing Strategies(Meta)、ML Fleet Efficiency(Google TPU 机群)、ExecuTorch、MLCommons Chakra、XProf —— "我们建了这个,这是详细基准"可以独立成篇。AMD 也不是生面孔:Wave(AMD)与 HipKittens(Stanford & AMD)都在 2026 届。

**二,研究轨那条路上,想 claim 的每一块已各有一篇。** ParallelKittens(Stanford,多 GPU kernel 通算重叠的通用原则)、FarSkip-Collective(NVIDIA,MoE A2A 重叠)、FP8-Flow-MoE(casting-free FP8 MoE 数据流、消除冗余 quant/dequant —— 直接占位格式缝卖点)、MoEBlaze(Meta,MoE 训练内存墙)、TokenWeave(MSR India,"最小集成代价"的通算重叠)。走研究轨会被读成这几篇之和再移植到 AMD。

**路线 A 正是绕开第二点、吃住第一点的那条路。** 代价是放弃"对 ParallelKittens 的可指名反例"这个高天花板位置(见 §6 对它的处置)。

---

## 3. 论文形状

**不是机制论文,是"一个真实系统 + 六条教训 + 一套可复现的测量口径"。**

Abstract 第一句的候选(数字全部来自 blog 实测):

> 我们报告 MegaMoE 的设计、实现与评估 —— 一个把专家并行通信融进 grouped GEMM 的融合式 MoE 训练算子,部署在 AMD Instinct MI355X 上的 Primus 训练栈中。算子级在 15 个 stage 上取得 1.29–1.61×(BF16),达到理想重叠上界的 82.7–94.2%;端到端训练步在 6 个模型上取得 1.16–1.29×(BF16)/ 1.17–1.35×(MXFP8)。

### 3.1 六条教训(这是工业轨真正买单的东西)

| # | 教训 | 为什么值钱 |
|---|---|---|
| **L1** | **框架设定融合上界。** 算子级的重叠收益不会自动变成训练步收益;把整层包成一个 autograd Function 后算子更快、训练步没更快,因为 Megatron 分布式优化器的 hook 失去了挂载点,DDP 通信退化成串行。处置是两阶段 autograd 拆分 | **最强的一条。**~~整个 megakernel 文献默认融得越多越好,没人讨论过这个天花板~~ **(2026-11-02 更正:这句不成立。)** Mixture-of-Kittens(2609.36070)是整层融合,生产端到端 1.41×,并且用 Blackwell CLC 让出 SM 给 FSDP 流,已经处理过框架通信共存的问题。但 MoK 解决的是"collective 发出后有没有 SM 跑"(CLC 让出 SM),我们碰到的是"collective 能不能按时发出"(Megatron hook 失去模块边界)。L1 按这两个必要条件来写:MoK 处理了后者,我们处理了前者。见骨架 §8.1 与 `papers/mixture-of-kittens.md` §10.3 |
| **L2** | **怎么让重叠声明可证伪。** 双基线(serial 隔离重叠本身 + DeepEP 生产路径作为真实替换对象)、η 对理想重叠 roofline 且 `T_gemm` 用独占全部 256 CU 的严格分母、用等效 FLOP 稠密 GEMM 反过来为分母辩护(grouped 达稠密的 91.5–99.7%)、speedup 与 η 必须合看因为二者极值方向相反 | 通算重叠这一类工作目前**没有标准化报告口径**。工业轨的评审标准里有一条就是 "influence on the MLSys community" —— 一套别人能抄的口径正好对应 |
| **L3** | **精度测试挡不住一致性 bug。** 确定性哈希测试(固定种子重复跑 + 顺序敏感哈希)抓到过一个 wait 常量放松一档的 bug,**当时精度测试全绿** | 具体、可复现、有血的教训。这类"我们的测试为什么不够"的章节是工业论文的高价值区 |
| **L4** | **融合不总是省显存。** 对称堆 8.87 GiB,每层省 0.766 GiB ⇒ 约第 12 层才回本;4 层 proxy 仍是净亏 | 诚实的会计。MoEBlaze 讲内存墙,我们讲内存代价,不正面撞 |
| **L5** | **静默数值回归的现行答案是 vendoring,不是测试。** `#486` 重建共享 FlyDSL BF16 grouped GEMM,静默把 MegaMoE 的 it1 grad norm 从 1.45 打到 1.88e6;处置是把 tile 钉到 `f6d5ab68` 快照私有化,代价是三套 mega 栈 / 三套 `symm_buffer` / 三套 `barrier` | **研究轨绝不会接受这种章节,工业轨明确欢迎**("challenge or reinforce existing solutions"、"canceled system with lessons learned")。这是真实工程现实,也是「前向对、无 NaN、dW 错」失效家族的第四例 |
| **L6** | **liveness 是设计假设不是证明。** gfx950 被观测到大致按 block index 升序填 CU,但这不是架构保证;不钉 occupancy、不缩 grid;MXFP8 combine 的在线搜索候选到 128 CU 时"the margin rests on fill order alone" | 把未证明的东西明说,比被审稿人挖出来强得多。**因为作者回应只有 3 天,这条必须在正文里就交代清楚** |

### 3.2 平台特异性怎么写

路线 A 里 gfx950 的细节是正资产,要展开而不是压缩:逐指令 visibility scope(cache modifier 18 = `sc1|nt` agent 域、19 = `sc0|sc1|nt` system 域)、按 block index 的角色特化、三张双 bank 的 ready-signal 表(`dispatch_flag` / `combine_flag` / `reduce_flag`)、XCD 感知的静态 CU 划分、MXFP8 的 scale 随 token 一起推送、`[32,32]` 方形块缩放使量化对转置不变。**这些就是"real-world platform experience"的具体内容。**

---

## 4. 必须补的证据(约 6 个月)

| # | 工作 | 性质 | 与 roadmap 的关系 |
|---|---|---|---|
| **A1** | **一条真实收敛曲线**,BF16 与 MXFP8 各一条,对 DeepEP 路径做 loss parity。blog 自陈 "These short runs measure throughput, not convergence" —— 一篇声称部署了训练系统的论文不能停在这里 | **强制**。训练系统论文的收敛证据没有替代品 | = **T1-4**(梯度范数回归门),做一次两处用 |
| **A2** | **配置补齐**:① 开 shared experts 的配对(现在关掉它等于替非融合调度卸掉一条本可摆在 dispatch 旁边的计算腿)② 不均衡路由下的**步级**(现在只有层级的 <1% 声明) | **强制**。工业轨的核心标准就是基准诚实度,这两条是 blog 自己标注的偏向项 | — |
| **A3** | **部署刻画**:多少模型、跑了多久、什么集群规模、是默认路径还是可选路径、谁在用。这是工业轨的资格条件,**blog 完全没有这一节** | **强制,且是新增内容**。写不出来就不符合 "real system" 资格 | 需要你提供事实,我核不到 |
| **A4** | **同硬件对照**:① Triton-distributed on MI355X —— 仅前向,其 `function/amd/ep_moe_fused.py:80-82` 是 `raise NotImplementedError`,且前向切分 docstring 与 MegaMoE 两算子切分一字不差 ② DeepEP 的**算子级**计时(blog 自陈缺,只有层级/步级) | **应做但非强制**。工业轨的两个现有基线(serial + DeepEP 生产路径)本来就是生产论文的正确基线;但"有没有和你硬件上的开源替代比过"在任何轨都是公平提问 | = **T3-3** |
| **A5** | **把六条教训写成正文** | **强制,且是全篇差异化所在** | — |

**次序建议:A3 最先**(它决定论文有没有资格,而且只需要整理事实不需要跑实验),**A1 与 A2 并行**(都要占 MI355X 节点),**A4 其次**,**A5 贯穿**。

---

## 5. 多出来的 5 个月拿来抬高度

| # | 工作 | 解决什么 |
|---|---|---|
| **A6** | **gfx1250 / MI455X 作为第二平台。**72 卡单 scale-up 域意味着 signal table 要多一级(blog Future Work 第一条) | **对冲硬件过时** —— 2028 年中开会时 MI355X 已是两代前;且业界扫描 §5.4 指出 AMD 侧至今没有 scale-up 域的 MoE 成本模型 |
| **A7** | **liveness 压力实验。**构造极端 occupancy / 多 kernel 竞争 / CU masking,找填充顺序会反转的条件;即使找不到反例,"我们试了这些都没反转"也远强于"我们没见过失败" | 把 L6 从免责声明升级成实证边界 |
| **A8** | 若 AE 确属录取后自愿环节(§1 待核实),则**降低优先级**,只在 A1–A7 都完成后做 | 避免把资源投到不影响录取的事情上 |

---

## 6. 路线 B 的那块材料怎么处置

放弃路线 B 不等于扔掉 L1。处置是:**在路线 A 里 L1 是一条教训(§3.1),同时把它留作下一篇的种子。**

如果执行 A1–A5 的过程中顺手做了"三档融合粒度的受控扫描"(不融合 / 两算子 / 整层单 Function,每档同时测算子级与步级),并且那条"算子级单调变好、步级掉头向下"的曲线稳定复现了,那么**第二篇研究轨论文的核心图就白捡了**。这个扫描的边际成本很低,基本是重跑已有配置。

**建议把它加进 A2 一起做,但不写进路线 A 的论文。** 理由:路线 A 的 10 页装不下,而且把它展开会让论文看起来像在声称机制新颖性 —— 那正是换轨后要避免的姿态。

---

## 7. 风险(相对路线 B 的变化)

| 风险 | 相对 B 的变化 | 对策 |
|---|---|---|
| **被抢发** | **大幅下降。**没人能抢发"我们自己生产系统的详细基准" | arXiv 早挂的紧迫性随之消失 |
| **硬件过时** | **上升。**生产系统论文的价值与系统的时效绑定;MI355X 的数字到 2028 年中已是两代前 | A6(gfx1250 第二平台)从"锦上添花"变成"必要" |
| **对外发表审批** | **上升。**一篇描述内部部署、点名 AMD/产品/集群的论文,审批重于机制论文;blog 自身的内部 review 还没走完(分支上最近两个 commit 仍是 "fix review findings"、"Apply review polish") | **这是最早需要解决的外部依赖,比任何实验都早。**我核不到,需你本人推动 |
| **"real system" 资格不过** | 新增风险 | A3 的部署刻画必须拿得出来;拿不出来要么改走研究轨(但那条路 §2 已论证走不通),要么等部署做实 |
| **录取率 26.8% + 无摘要缓冲 + 3 天 rebuttal** | 不变 | 可预判质疑全部写进正文(L6 尤其) |
| **10 页装不下** | 新增 | 细表进附录,但**审稿人没有义务读附录**,所以核心结论必须在正文自洽 |
| **AMD 内部选题重叠** | 不变。Wave 是 AMD 在 MLSys 2026 的 DSL 论文,本工作建在 FlyDSL 上 | "为什么是 FlyDSL 不是 Wave"是技术问题,正文要正面答;内部协调要你去问 |

---

## 核查溯源

MLSys 2027 官方 CFP(`mlsys.org/Conferences/2027/CallForResearchPapers`)原文核实的条目:工业轨的匿名规则(只隐作者名、可保留公司/产品/URL)、评审标准(impact / lessons / experiences,不要求 novelty)、作者资格(第一作者与多数作者须来自业界)、系统资格(必须是真实系统的三类之一,排除 speculative/small/exploratory)、**截止后不可换轨**、篇幅(双栏 10 页不含参考文献、附录单独上传且审稿人无义务阅读)、格式(`mlsys2025style`)、OpenReview 流程与 profile 提前两周、评审节奏(2027 届:投稿 10-19 开放 / 10-30 20:00 UTC 截止 / 评审 01-18 / 回应 01-21 / 通知 02-28)、双盲且允许 arXiv 预印。

**来自第三方整理页、非官方原文,待核实**:① AI 使用声明章节为 2027 届新增要求;② Artifact Evaluation 为自愿、录取后、不影响决定(**此条推翻了本文件上一版本的 AE 建议**);③ MLSys 2026 录取率 26.8%(135/504);④ §2 中各竞品论文的单位与一句话主张 —— 标题与单位可信,**但我没有逐篇读正文**,定稿前必须精读确认重叠程度。

**官方页抓取时被截断,必须补查**:CFP 的 "Authorship Quota" 一节(确认存在,具体篇数限制未知)。

**MLSys 2028 的实际 deadline 未知**(CFP 未发布)。2026 届 2025-10-30、2027 届 2026-10-30 连续两年同日,2025 届为 2024-10-06,故区间估计为 **2027 年 10 月初至月底**,CFP 预计 2027-09 前后发布,届时必须重核。

**超出我能核查的范围,需你本人确认**:A3 的部署事实;AMD 对外发表审批流程与内部选题协调。

`function/amd/ep_moe_fused.py:80-82` 的 `raise NotImplementedError`、其前向切分 docstring、以及 blog 的全部数字,由我本地只读核验,详见 [`2026-10-28_2050_megamoe-paper-plan.md`](./2026-10-28_2050_megamoe-paper-plan.md)。

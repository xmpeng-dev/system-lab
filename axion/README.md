# Axion Design Documents

> **Axion** is a communication-first sparse training runtime designed for large-scale MoE systems.  
> **核心理念:** Compile-time 感知 + 通信友好 Tensor 数据结构 + FSEP MoE 并行

---

## 文档索引

### 架构设计

| 文档 | 内容摘要 | 状态 |
|------|---------|------|
| [Axion_Architecture_Design.md](./Axion_Architecture_Design.md) | 系统整体架构、设计理念、为什么选择自定义 IR | ✅ 完成 |
| [ModelSpec_Design.md](./ModelSpec_Design.md) | Model Spec 三层设计：ModelShape + Parallelism + ExpertPlacement | ✅ 完成 |

### IR 设计系列

| 文档 | 内容摘要 | 状态 |
|------|---------|------|
| [ir/IR_Phase1_Type_System_Design.md](./ir/IR_Phase1_Type_System_Design.md) | 类型系统 + 指令集 + 5个典型场景 IR 手写验证 | ✅ 完成 |
| ir/IR_Phase2_Parser_Implementation.md | IR Parser 实现 + Pass 1/2（Type Inference + Comm Analysis） | 🔲 待写 |
| ir/IR_Phase3_FSEP_Overlap_Pass.md | Pass 3/4（FSEP Sharding Plan + Overlap Insertion） | 🔲 待写 |

### 运行时设计系列

| 文档 | 内容摘要 | 状态 |
|------|---------|------|
| runtime/CommTensor_Runtime.md | CommTensor 物理内存管理 + index map 机制 | 🔲 待写 |
| runtime/FSEP_Planner.md | Slow Planner + Fast Router 双层规划详细设计 | 🔲 待写 |
| runtime/CommFabric.md | NVLink / IB RDMA 统一抽象层 | 🔲 待写 |

---

## 核心设计决策记录

### ADR-001：为什么不用 FX Graph，而是自定义 IR

**决策:** 自定义 Axion IR，不基于 torch.compile / FX Graph

**理由:**
- FX Graph 的语义粒度 = 单个张量算子，通信是黑盒
- Axion 需要通信和计算的联合调度，类型系统必须原生支持 `CommTensor`、`ExpertShard`
- FX Graph 补丁方案在 Pass 分析时信息严重不足，是死路

**代价:** IR Parser 需要从零实现，冷启动成本高  
**缓解:** Bootstrap 阶段用 FX Graph 有损转换验证 IR 正确性，不作终态

---

### ADR-002：CommLayout 枚举而非为每种通信模式建独立类型

**决策:** `CommTensor<layout=BLOCKED_BY_DST>` 而非 `A2ADispatchTensor`

**理由:**
- 通信语义由 layout + comm_spec 共同描述，独立类型会爆炸式增长
- layout 枚举 5 种覆盖所有当前场景，扩展时添加枚举值即可
- 类型转换路径（layout 状态机）更清晰，Pass 分析更容易

---

### ADR-003：ExpertShard 是 IR 内置类型，不是 DenseTensor 的 metadata

**决策:** 专用 `ExpertShard` 类型，携带 `expert_id`、`migration_state` 等字段

**理由:**
- FSEP 的 Expert 迁移状态（STABLE / MIGRATING / SHADOW）必须在 IR 中可见
- 编译器 Pass 需要静态分析 Expert 分片分布，metadata dict 不可靠
- `Shard.Migrate` Op 的类型规则需要 `ExpertShard` 类型才能正确检查

---

## 理念来源

- 📄 **LAER-MoE** (FSEP) → arXiv:2602.11686 | [阅读笔记](../moe/LAER_MoE_FSEP_reading_notes.md)
- 📄 **veScale-FSDP** → arXiv:2602.22437 | [阅读笔记](../moe/veScale_FSDP_reading_notes.md)

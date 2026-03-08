# ROCflow: A Communication-First MoE Training Framework for AMD GPUs

> **Version:** 0.1 (Design Draft)  
> **Status:** Pre-release design phase  
> **Target Hardware:** AMD MI300X / MI325X and future Instinct series  
> **Backed by:** ASPLOS'26 · MLSys'25 · NeurIPS'25 · EuroSys'26 · arXiv'26

---

## Why ROCflow?

### The Problem No One Has Fully Solved

MoE (Mixture-of-Experts) models — DeepSeek-V3, Mixtral, Grok-2 and their successors — have become
the dominant architecture for frontier LLMs. Yet the training frameworks used to train them (Megatron-LM,
torchtitan, DeepSpeed) were designed around **Dense models first**, and MoE support was bolted on later.

The consequence is architectural: these frameworks treat MoE as a compute problem with some communication
overhead. ROCflow starts from the opposite premise.

> **MoE is fundamentally a communication problem. Compute is the easy part.**

A single MoE Transformer block generates **9 distinct communication operations** per forward+backward pass
(2× All-to-All dispatch/gather, TP All-Reduce, CP ring-attention, 2× A2A backward, 2× gradient
All-Reduce, FSDP shard sync) — 3–4× the communication density of a Dense block of equivalent size. At
10,000-GPU scale, All-to-All latency alone can consume **50–60% of total step time**.

No existing open-source framework has a unified, communication-aware scheduler that governs all 9 of
these operations simultaneously. ROCflow does.

### The AMD Gap

AMD's MI300X GPU offers capabilities that are uniquely suited to solving the MoE communication problem:

| AMD Hardware Feature | Capability | Relevance to MoE |
|---------------------|------------|-----------------|
| XGMI (Infinity Fabric) | 896 GB/s intra-node | Intra-node Expert A2A at near-memory bandwidth |
| HBM3e | 5.3 TB/s memory bandwidth | ExpertSlotTensor zero-copy dispatch |
| 192 GB unified HBM pool | Largest per-node memory | FSEP shard hosting without OOM |
| Independent XGMI + RDMA paths | Dual comm channels | True concurrent intra+inter-node communication |

No existing framework exposes these capabilities in a way that benefits MoE training. NCCL/RCCL treats
all communication uniformly. Megatron has no topology-aware routing. ROCflow is built around AMD's
hardware topology from day one.

### The Research Foundation

ROCflow is not a research project; it is an engineering synthesis. Every core feature is backed by
peer-reviewed work from top-tier venues:

| Feature | Source | Venue |
|---------|--------|-------|
| FSEP dynamic expert sharding | LAER-MoE | ASPLOS '26 |
| Tile-level GEMM-RDMA overlap | Comet | MLSys '25 |
| Cross-block DAG scheduling | FlowMoE | NeurIPS '25 |
| Attention/MoE parallel decoupling | MoE Parallel Folding | NVIDIA/arXiv '25 |
| Smart activation checkpointing | MoEBlaze + MemFine | arXiv '26/'25 |
| Topology-aware A2A | MegaScale-MoE | EuroSys '26 |
| Comm-aware token routing | ROCflow original | — |
| Comm-native tensor layout | ROCflow original | — |

---

## Core Design Principles

### Principle 1: Communication as a First-Class Citizen

Every design decision in ROCflow — tensor layout, parallelism configuration, kernel design, memory
management — is evaluated first through the lens of **communication cost and overlap potential**.

Compute efficiency matters. Communication efficiency matters more for MoE.

### Principle 2: Three-Layer Overlap Coverage

Existing frameworks address communication overlap at one or two levels. ROCflow covers all three:

```
Layer 3 │ Cross-Block Pipeline    │ FlowMoE-style DAG scheduler
        │                         │ Block i comm overlaps Block i+1 compute
────────┼─────────────────────────┼──────────────────────────────────────────
Layer 2 │ Intra-Block Decoupling  │ Attn and MoE use separate comm paths
        │                         │ XGMI (intra-node) ≠ RDMA (inter-node)
────────┼─────────────────────────┼──────────────────────────────────────────
Layer 1 │ Kernel-level Tile       │ HIP Wavefront Specialization
        │ GEMM-RDMA Pipeline      │ Compute wavefronts + comm wavefronts
        │                         │ run concurrently within one kernel
```

No existing open-source framework delivers all three layers together.

### Principle 3: Comm-Native Tensor Layout

The standard `[Batch, Sequence, Hidden]` tensor layout is optimized for computation, not communication.
Before an All-to-All dispatch, existing frameworks must first gather scattered tokens into a contiguous
buffer — a full memory copy that happens **twice per MoE layer per forward pass** (and again in backward).

ROCflow introduces **ExpertSlotTensor**: a tensor layout organized by communication destination from
the moment of allocation. All-to-All dispatch becomes a direct contiguous send with zero intermediate
copies.

### Principle 4: Routing Awareness of Communication Cost

The Gate network in MoE models selects experts purely based on token-expert affinity scores. It has
no knowledge of whether the selected expert lives on the same GPU, the same node, or a remote node
three network hops away.

ROCflow introduces **Comm-Aware Routing**: a soft topology-cost penalty term in the routing score that
gradually steers token routing toward lower-cost experts, reducing cross-node All-to-All volume without
compromising model quality.

### Principle 5: Unified Communication DAG

All 9 communication operations in a MoE block — A2A dispatch, A2A gather, TP All-Reduce, gradient
sync, FSDP shard operations — are modeled as nodes in a single directed acyclic graph (DAG). A
centralized scheduler assigns each operation to the optimal hardware path (XGMI or RDMA), resolves
dependencies, and maximizes concurrent utilization of all available bandwidth.

---

## System Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║                        ROCflow Architecture                          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ┌──────────────────────────────────────────────────────────────┐   ║
║  │                     User API Layer                            │   ║
║  │                                                               │   ║
║  │  rocflow.init(...)          rocflow.MoELayer(...)             │   ║
║  │  rocflow.train(...)         rocflow.ParallelConfig(...)       │   ║
║  │  rocflow.profile(...)       @rocflow.compile                  │   ║
║  └──────────────────────────────────────────────────────────────┘   ║
║                              │                                       ║
║  ┌───────────────────────────▼──────────────────────────────────┐   ║
║  │                  Parallel Strategy Engine                     │   ║
║  │                                                               │   ║
║  │  • FSEP (Fully Sharded Expert Parallel)                       │   ║
║  │    - Expert parameters sharded across all EP GPUs             │   ║
║  │    - Load-Adaptive Planner: periodic expert re-layout         │   ║
║  │    - No token drop; capacity managed through re-distribution  │   ║
║  │                                                               │   ║
║  │  • Attention / MoE Parallel Decoupling                        │   ║
║  │    - Attn layer: TP=N, EP=1 (tensor parallel, small params)   │   ║
║  │    - MoE layer:  TP=1, EP=M (expert parallel, large N_expert) │   ║
║  │    - Different GPU groups activated per layer type             │   ║
║  │                                                               │   ║
║  │  • 5D Parallel Support: TP × EP × DP × PP × CP               │   ║
║  └───────────────────────────┬──────────────────────────────────┘   ║
║                              │                                       ║
║  ┌───────────────────────────▼──────────────────────────────────┐   ║
║  │              Unified Communication DAG Scheduler              │   ║
║  │                                                               │   ║
║  │  ┌─────────────────────────────────────────────────────────┐ │   ║
║  │  │  DAG Node Types:                                         │ │   ║
║  │  │   [A2A_D] [A2A_G] [TP_AR] [DP_AR] [CP_Ring] [FSDP_AG]  │ │   ║
║  │  │   [Expert_GEMM] [Attn_GEMM] [Gate] [LayerNorm]          │ │   ║
║  │  └─────────────────────────────────────────────────────────┘ │   ║
║  │                                                               │   ║
║  │  • Critical-path-first priority scheduling                    │   ║
║  │  • Hardware path assignment: XGMI vs RDMA per operation       │   ║
║  │  • Cross-block overlap: Layer i comm || Layer i+1 compute     │   ║
║  │  • Gradient AllReduce overlapped with backward compute        │   ║
║  │  • FSDP AllGather pipelined with MoE A2A                      │   ║
║  └───────────────────────────┬──────────────────────────────────┘   ║
║                              │                                       ║
║  ┌───────────────────────────▼──────────────────────────────────┐   ║
║  │                  Comm-Aware MoE Layer                         │   ║
║  │                                                               │   ║
║  │  ┌──────────────────────┐   ┌──────────────────────────────┐ │   ║
║  │  │  Comm-Aware Router   │   │   ExpertSlotTensor Layout    │ │   ║
║  │  │                      │   │                              │ │   ║
║  │  │  score(t,e) =        │   │  [GPU0_slot][GPU1_slot]...   │ │   ║
║  │  │   affinity(t,e)      │   │  Organized by comm dest,     │ │   ║
║  │  │   - λ·comm_cost(t,e) │   │  not by sequence position.   │ │   ║
║  │  │                      │   │  Zero-copy A2A dispatch.     │ │   ║
║  │  │  comm_cost levels:   │   │                              │ │   ║
║  │  │   local GPU    = 0   │   │  Forward/backward buffers    │ │   ║
║  │  │   XGMI node   = α    │   │  share same memory region.   │ │   ║
║  │  │   RDMA remote = β    │   │                              │ │   ║
║  │  └──────────────────────┘   └──────────────────────────────┘ │   ║
║  └───────────────────────────┬──────────────────────────────────┘   ║
║                              │                                       ║
║  ┌───────────────────────────▼──────────────────────────────────┐   ║
║  │              HIP Kernel Layer (AMD Native)                    │   ║
║  │                                                               │   ║
║  │  ┌─────────────────────────────────────────────────────────┐ │   ║
║  │  │  rocflow_expert_kernel (HIP)                             │ │   ║
║  │  │                                                           │ │   ║
║  │  │  Wavefront Group 0,1,2 → Expert GEMM (matrix compute)    │ │   ║
║  │  │  Wavefront Group 3     → XGMI/RDMA tile send             │ │   ║
║  │  │                                                           │ │   ║
║  │  │  Tile N GEMM completes → written to LDS (64KB/CU)        │ │   ║
║  │  │  Comm wavefront reads LDS → DMA to peer GPU              │ │   ║
║  │  │  Tile N+1 GEMM begins  ← concurrently                    │ │   ║
║  │  │                                                           │ │   ║
║  │  │  Target: 85%+ GEMM-RDMA overlap on MI300X                │ │   ║
║  │  └─────────────────────────────────────────────────────────┘ │   ║
║  │                                                               │   ║
║  │  • Gate+Dispatch+GEMM+Gather fused kernel (MoE-compile pass) │   ║
║  │  • Smart Activation Checkpointing (per-chunk, MoE-aware)     │   ║
║  │  • Expert Group GEMM via hipBLASLt                            │   ║
║  └───────────────────────────┬──────────────────────────────────┘   ║
║                              │                                       ║
║  ┌───────────────────────────▼──────────────────────────────────┐   ║
║  │                AMD Hardware Abstraction                       │   ║
║  │                                                               │   ║
║  │  XGMI Fabric    896 GB/s  ←→  Intra-node Expert A2A          │   ║
║  │  RDMA/RoCE      400 Gbps  ←→  Inter-node A2A + grad sync     │   ║
║  │  HBM3e          5.3 TB/s  ←→  ExpertSlotTensor R/W           │   ║
║  │  RCCL                     ←→  Fallback collective ops         │   ║
║  └──────────────────────────────────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## Key Features vs. Competing Frameworks

| Feature | Megatron-LM | torchtitan | DeepSpeed | **ROCflow** |
|---------|:-----------:|:----------:|:---------:|:-----------:|
| MoE Expert Parallel | ✅ Basic EP | ✅ Basic EP | ✅ Basic EP | ✅ **FSEP + Dynamic Re-layout** |
| Overlap granularity | Layer (PP) | Micro-batch | Layer | **3-layer: Block/Layer/Tile** |
| Tile-level kernel overlap | ❌ | ❌ | ❌ | ✅ **HIP Wavefront Spec.** |
| Comm-native tensor layout | ❌ | ❌ | ❌ | ✅ **ExpertSlotTensor** |
| Attn/MoE parallel decoupling | ❌ | ❌ | ❌ | ✅ |
| Topology-aware A2A | ❌ | ❌ | ❌ | ✅ **XGMI-first routing** |
| Comm-aware token routing | ❌ | ❌ | ❌ | ✅ **Cost-penalized Gate** |
| Unified comm DAG scheduler | ❌ | ❌ | ❌ | ✅ |
| FSDP + MoE A2A co-scheduling | ❌ | ⚠️ Partial | ❌ | ✅ **Pipelined** |
| AMD ROCm native kernels | ❌ | ❌ | ❌ | ✅ **First-class** |
| torch.compile MoE path | ❌ Unstable | ⚠️ Partial | ❌ | ✅ **MoE-compile pass** |
| Token drop free | ❌ | ❌ | ❌ | ✅ **Via FSEP re-layout** |

---

## User API

### 1. Initialization

```python
import rocflow

rocflow.init(
    backend="rccl",           # Communication backend: "rccl" | "nccl" (for NVIDIA compat)
    topology="auto",          # Hardware topology detection: "auto" | "xgmi" | "rdma_only"
    log_level="info",         # Logging verbosity
)
```

### 2. Parallel Configuration

```python
from rocflow import ParallelConfig

# ROCflow allows different parallel strategies for Attention and MoE layers
config = ParallelConfig(
    # Attention layers: TP-heavy (small params, compute-intensive)
    attn_tp=4,
    attn_dp=2,
    attn_cp=1,               # Context parallel for long sequences

    # MoE layers: EP-heavy (many experts, communication-intensive)
    moe_ep=8,                # Expert parallel degree
    moe_tp=1,                # Expert-internal tensor parallel (for very large experts)
    moe_dp=2,

    # Pipeline parallel (applies to both)
    pp=2,

    # FSEP: enable fully sharded expert parallel
    fsep=True,
    fsep_rebalance_interval=50,   # Re-layout experts every N steps

    # Comm-aware routing
    comm_aware_routing=True,
    routing_comm_lambda=0.1,      # Communication cost penalty weight
                                  # 0.0 = standard routing, increase for comm savings
)
```

### 3. MoE Layer (Drop-in Replacement)

```python
from rocflow.nn import MoELayer

# Use as a drop-in replacement for any standard MoE FFN layer
moe = MoELayer(
    hidden_size=4096,
    ffn_hidden_size=14336,
    num_experts=256,
    num_experts_per_token=8,     # Top-K routing

    # ROCflow-specific options
    tensor_layout="comm_native",  # Use ExpertSlotTensor layout (default: "comm_native")
    overlap_mode="tile",          # Overlap granularity: "tile" | "chunk" | "layer"
    smart_ac=True,                # Smart activation checkpointing (MoE-aware)
    ac_chunk_size=64,             # Tokens per activation checkpoint chunk

    # Comm-aware routing (overrides ParallelConfig if set)
    comm_aware_routing=None,      # None = inherit from ParallelConfig
)
```

### 4. Model Definition

```python
import torch
import torch.nn as nn
from rocflow.nn import MoELayer
from rocflow.nn import wrap_attention  # Wraps standard attention with ROCflow parallel logic

class MyMoETransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn    = wrap_attention(MyAttention(config))  # TP/CP aware
        self.norm1   = nn.RMSNorm(config.hidden_size)
        self.norm2   = nn.RMSNorm(config.hidden_size)
        self.moe_ffn = MoELayer(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.ffn_hidden_size,
            num_experts=config.num_experts,
            num_experts_per_token=config.top_k,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.moe_ffn(self.norm2(x))
        return x
```

### 5. Training Loop

```python
import rocflow
from rocflow import ParallelConfig, Trainer

parallel_config = ParallelConfig(
    attn_tp=4, attn_dp=2,
    moe_ep=8,  moe_dp=2,
    pp=2,
    fsep=True,
    comm_aware_routing=True,
)

trainer = rocflow.Trainer(
    model=model,
    optimizer=optimizer,
    parallel_config=parallel_config,

    # Scheduler and memory
    grad_clip=1.0,
    precision="bf16",
    compile=True,             # Enable rocflow.compile (MoE-aware torch.compile pass)

    # Overlap configuration
    overlap_attn_moe=True,    # Enable Attn/MoE comm path decoupling
    overlap_grad_ar=True,     # Overlap gradient AllReduce with backward

    # Checkpointing
    checkpoint_dir="./checkpoints",
    checkpoint_interval=1000,
    async_checkpoint=True,    # Non-blocking checkpoint save
)

for batch in dataloader:
    loss = trainer.step(batch)
    if trainer.is_rank_zero():
        print(f"step={trainer.step_count}, loss={loss:.4f}, "
              f"mfu={trainer.mfu:.1%}, "
              f"comm_overlap={trainer.comm_overlap_rate:.1%}")
```

### 6. Profiling and Diagnostics

```python
# Analyze communication bottlenecks before full training run
from rocflow.profiler import CommProfiler

profiler = CommProfiler(model, parallel_config)
report = profiler.analyze(sample_batch)

print(report.summary())
# Output:
# ┌─────────────────────────────────────────────────────────────┐
# │ ROCflow Comm Profile Report                                  │
# ├──────────────────────┬─────────────┬────────────────────────┤
# │ Operation            │ Time (ms)   │ Overlap Opportunity    │
# ├──────────────────────┼─────────────┼────────────────────────┤
# │ A2A Dispatch (fwd)   │ 8.3ms       │ ✅ Tile-overlap w/ GEMM│
# │ A2A Gather (fwd)     │ 7.9ms       │ ✅ Tile-overlap w/ GEMM│
# │ TP All-Reduce        │ 2.1ms       │ ✅ XGMI path, parallel │
# │ Grad All-Reduce (dp) │ 12.4ms      │ ✅ Overlap w/ backward │
# │ FSDP All-Gather      │ 5.6ms       │ ✅ Pipeline w/ A2A     │
# ├──────────────────────┼─────────────┼────────────────────────┤
# │ Total exposed comm   │ 3.2ms       │ 91.3% overlap rate     │
# └──────────────────────┴─────────────┴────────────────────────┘
#
# Expert load distribution: max/avg ratio = 1.08 (healthy, FSEP active)
# Routing cross-node traffic: 23% of tokens (comm_lambda=0.1)
# Recommended: increase comm_lambda to 0.15 to reduce cross-node traffic

# Detailed per-layer breakdown
print(report.per_layer_breakdown())
print(report.expert_load_heatmap())
print(report.topology_traffic_map())
```

### 7. torch.compile Integration

```python
# ROCflow provides a MoE-aware compile pass that handles dynamic routing
from rocflow import rocflow_compile

# Option A: Compile the full model (recommended for production)
model = rocflow_compile(
    model,
    mode="max-autotune",      # Use hipBLASLt autotuning for Expert GEMM
    dynamic_routing=True,     # Enable symbolic shape tracing for variable token counts
    fuse_gate_dispatch=True,  # Fuse Gate + Dispatch into single HIP kernel
)

# Option B: Compile only the MoE layers (for debugging non-MoE parts)
from rocflow.nn import compile_moe_layers
model = compile_moe_layers(model, mode="reduce-overhead")
```

### 8. Fault Tolerance (Production Use)

```python
# Enable online fault tolerance for large-scale runs
trainer = rocflow.Trainer(
    model=model,
    optimizer=optimizer,
    parallel_config=parallel_config,

    # Fault tolerance
    fault_tolerance=True,
    ft_detection_interval=60,     # Check GPU health every 60 seconds
    ft_strategy="reroute",        # "reroute" | "checkpoint_restart"
    ft_spare_gpus=2,              # Keep N spare GPUs for hot-swap
)
```

---

## Performance Targets (MI300X, 8×MI300X node)

| Model | Config | MFU Target | vs Megatron baseline |
|-------|--------|-----------|---------------------|
| Mixtral 8×7B | 64 GPUs, EP=8 | **52%+** | ~+30% |
| DeepSeek-V3 scale (671B MoE) | 1024 GPUs, EP=64 | **45%+** | ~+25% |
| Custom 100B MoE | 256 GPUs, EP=32 | **48%+** | ~+28% |

Key expected improvements over Megatron-LM on equivalent AMD hardware:
- **Comm-compute overlap rate**: 85–92% (vs ~20% in Megatron)
- **Expert load imbalance ratio**: < 1.1× (vs 2–3× without FSEP)
- **Cross-node A2A volume**: −30–40% reduction via Comm-Aware Routing
- **Peak activation memory**: −40–50% via Smart AC + ExpertSlotTensor

---

## Roadmap

### Phase 1 — Core Framework (Q2 2026)
- [ ] ExpertSlotTensor layout and zero-copy A2A dispatch
- [ ] Unified Communication DAG scheduler
- [ ] FSEP implementation with Load-Adaptive Planner
- [ ] Attention/MoE parallel decoupling
- [ ] Basic HIP kernel for Expert GEMM (hipBLASLt Group GEMM)
- [ ] CommProfiler tool

### Phase 2 — Overlap Maximization (Q3 2026)
- [ ] HIP Wavefront Specialization kernel (tile-level GEMM-RDMA overlap)
- [ ] Cross-block pipeline overlap (FlowMoE-style DAG)
- [ ] XGMI-first topology-aware routing
- [ ] Gradient AllReduce + backward overlap
- [ ] FSDP2 + MoE A2A co-scheduling

### Phase 3 — AMD Optimization + compile (Q4 2026)
- [ ] MoE-compile pass (Gate+Dispatch+GEMM+Gather fusion)
- [ ] Comm-Aware Routing with adaptive λ tuning
- [ ] Smart Activation Checkpointing (MoE-aware chunk policy)
- [ ] MI325X / next-gen Instinct support
- [ ] Full fault tolerance for 10K+ GPU runs

### Phase 4 — Ecosystem (Q1 2027)
- [ ] DeepSeek-V3 / Mixtral model configs included
- [ ] Public benchmark suite vs Megatron-LM
- [ ] Integration with AMD ROCm profiling tools (Omniperf / rocProf)
- [ ] Community plugin API for custom Expert implementations

---

## Design Discussion Notes

> *This section tracks open design questions under active discussion.*

**Q1: Comm-Aware Routing λ scheduling**  
Should λ be a static hyperparameter, or auto-tuned online based on observed cross-node traffic ratio?
The LAER-MoE Load Planner provides a precedent for online adaptation. Candidate: start λ=0 for first
warmup steps, ramp to target λ over a schedule tied to routing entropy stabilization.

**Q2: ExpertSlotTensor + torch.compile compatibility**  
Variable-length slots (due to dynamic routing) break static shape assumptions in torch.compile.
Candidate solution: pad slots to `capacity = max_tokens_per_expert` during compile, actual dispatch
uses a mask. This recovers static shapes at the cost of ~5% padding waste.

**Q3: FSDP2 + FSEP + A2A deadlock avoidance**  
When FSDP2 AllGather, FSEP expert re-layout, and MoE A2A are all active simultaneously, the unified
DAG must enforce ordering constraints to prevent RCCL deadlocks. This is the highest-priority
correctness challenge in Phase 2.

**Q4: Backward A2A scheduling freedom**  
Forward A2A order is fixed by layer dependencies. Backward A2A (being the reverse) has more scheduling
flexibility — the grad of layer N's A2A-Gather can potentially be overlapped with layer N-1's Expert
backward GEMM. This has not been systematically exploited in any existing framework.

---

## Related Work

| Paper | Venue | What ROCflow Builds On |
|-------|-------|------------------------|
| LAER-MoE | ASPLOS '26 | FSEP architecture, Load-Adaptive Planner |
| Comet | MLSys '25 | Tile-level GEMM-comm overlap, Warp Specialization |
| FlowMoE | NeurIPS '25 | Unified DAG scheduler, cross-block overlap |
| MoE Parallel Folding | arXiv '25 | 5D parallelism, Attn/MoE decoupling |
| MegaScale-MoE | EuroSys '26 | Topology-aware A2A, production fault tolerance |
| MoEBlaze | arXiv '26 | Token dispatch data structure, Smart AC |
| MemFine | arXiv '25 | Chunk-level activation scheduling |
| SwiftMoE | arXiv '25 | Optimizer-parameter decoupling |

---

## Contributing

ROCflow is designed as a modular framework. Each subsystem (DAG Scheduler, FSEP Planner, HIP Kernels,
Comm-Aware Router) has a clean interface and can be contributed to independently.

See `CONTRIBUTING.md` for development setup on AMD hardware and the testing framework.

---

*ROCflow — Because MoE training deserves a framework that treats communication as the first problem, not the last.*

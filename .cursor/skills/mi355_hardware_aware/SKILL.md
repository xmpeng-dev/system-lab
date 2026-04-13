---
name: mi355-hardware-aware
description: >-
  Hardware-aware programming guide for AMD Instinct MI355X (gfx950) GPUs.
  Covers CDNA4 architecture specs, memory hierarchy, MFMA instructions,
  wavefront scheduling, and performance tuning. Use when optimizing kernels
  for MI355X, understanding AMD GPU microarchitecture, or debugging performance issues.
---

# MI355X Hardware-Aware Programming

## When to Apply

Use this skill when:
- Writing performance-critical kernels for MI355X
- Debugging GPU performance issues (low occupancy, memory bottlenecks)
- Understanding why certain optimizations work or don't work
- Porting kernels from other architectures (NVIDIA, MI300X)

## MI355X Architecture Overview

### Chip Specifications

| Spec | Value | Notes |
|------|-------|-------|
| Architecture | CDNA4 (gfx950) | AMD compute-focused architecture |
| Compute Units (CUs) | 256 | Each CU is an independent processor |
| SIMD Units per CU | 4 | Each SIMD has 16 ALUs |
| Wavefront Size | 64 threads | Fixed, unlike NVIDIA's configurable warps |
| Max Wavefronts/CU | 32 | Theoretical max with minimal resources |
| Max Workgroups/CU | 32 | Limited by resources |
| Clock Speed | ~2.1 GHz (boost) | Varies with power/thermal |
| HBM Capacity | 288 GB (HBM3e) | 8 stacks |
| HBM Bandwidth | 8 TB/s | Peak theoretical |
| TDP | 750W | Maximum power |

### Memory Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│                    Global Memory (HBM3e)                │
│                    288 GB @ 8 TB/s                      │
└─────────────────────────────────────────────────────────┘
                            ↓ (~400 cycles latency)
┌─────────────────────────────────────────────────────────┐
│                    L2 Cache (Shared)                    │
│                    256 MB @ ~12 TB/s                    │
└─────────────────────────────────────────────────────────┘
                            ↓ (~100 cycles latency)
┌─────────────────────────────────────────────────────────┐
│                 L1 Cache (Per CU)                       │
│                 32 KB @ ~20 TB/s                        │
└─────────────────────────────────────────────────────────┘
                            ↓ (~20 cycles latency)
┌─────────────────────────────────────────────────────────┐
│              LDS - Local Data Share (Per WG)            │
│              Up to 160 KB @ ~25 TB/s                    │
└─────────────────────────────────────────────────────────┘
                            ↓ (~10-20 cycles latency)
┌─────────────────────────────────────────────────────────┐
│              Vector Registers (Per Thread)              │
│              256 VGPRs × 4 bytes = 1 KB/thread          │
└─────────────────────────────────────────────────────────┘
```

### Per-CU Resources

| Resource | Amount | Impact on Occupancy |
|----------|--------|---------------------|
| LDS | 160 KB | WG size limited by LDS usage |
| VGPRs | 512 per SIMD (2048 total) | High VGPR = low wavefronts |
| SGPRs | 800 per CU | Scalar registers, less impactful |
| Wavefront slots | 32 | Hard limit |
| Workgroup slots | 32 | Hard limit |

## Compute Unit Architecture

### CU Block Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                      Compute Unit (CU)                       │
├──────────────────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │  SIMD 0  │ │  SIMD 1  │ │  SIMD 2  │ │  SIMD 3  │        │
│  │ 16 ALUs  │ │ 16 ALUs  │ │ 16 ALUs  │ │ 16 ALUs  │        │
│  │ 512 VGPR │ │ 512 VGPR │ │ 512 VGPR │ │ 512 VGPR │        │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘        │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Matrix FMA Unit (MFMA)                    │ │
│  │         Shared across all 4 SIMDs                      │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─────────────────┐  ┌─────────────────────────────────┐   │
│  │   Scheduler     │  │    LDS (160 KB)                 │   │
│  │   (per SIMD)    │  │    32 banks × 4 bytes           │   │
│  └─────────────────┘  └─────────────────────────────────┘   │
│                                                              │
│  ┌─────────────────┐  ┌─────────────────────────────────┐   │
│  │  Texture/L1 $   │  │    Scalar Unit + SGPRs          │   │
│  │    (32 KB)      │  │                                 │   │
│  └─────────────────┘  └─────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

### Wavefront Execution Model

A wavefront (64 threads) executes in **lockstep**:
- All threads execute the same instruction
- Divergent branches cause serialization
- Each SIMD executes 16 threads per cycle → 4 cycles for full wavefront

```
Cycle 0: Threads  0-15 execute
Cycle 1: Threads 16-31 execute
Cycle 2: Threads 32-47 execute
Cycle 3: Threads 48-63 execute
```

## MFMA (Matrix Fused Multiply-Add)

### Available Instructions (gfx950)

| Instruction | Shape (M×N×K) | Input | Output | Cycles |
|-------------|---------------|-------|--------|--------|
| `v_mfma_f32_16x16x16_f16` | 16×16×16 | f16 | f32 | 32 |
| `v_mfma_f32_16x16x16_bf16` | 16×16×16 | bf16 | f32 | 32 |
| `v_mfma_f32_32x32x8_f16` | 32×32×8 | f16 | f32 | 64 |
| `v_mfma_f32_32x32x8_bf16` | 32×32×8 | bf16 | f32 | 64 |
| `v_mfma_f16_16x16x16_f16` | 16×16×16 | f16 | f16 | 32 |
| `v_mfma_f16_32x32x8_f16` | 32×32×8 | f16 | f16 | 64 |
| `v_mfma_i32_16x16x32_i8` | 16×16×32 | i8 | i32 | 32 |
| `v_mfma_i32_32x32x16_i8` | 32×32×16 | i8 | i32 | 64 |

### MFMA Thread-Data Mapping

For `v_mfma_f32_16x16x16_f16`:
- All 64 threads in wavefront participate
- Each thread contributes 4 elements to A and B
- Each thread holds 4 accumulator elements
- Result: 16×16 output tile

```
Thread Layout in 16×16 Output:
┌────┬────┬────┬────┐
│T0-3│T4-7│... │T60 │  ← Row 0-3
├────┼────┼────┼────┤
│T0-3│T4-7│... │T60 │  ← Row 4-7
├────┼────┼────┼────┤
│... │... │... │... │
├────┼────┼────┼────┤
│T0-3│T4-7│... │T60 │  ← Row 12-15
└────┴────┴────┴────┘
```

### MFMA Throughput Calculation

```python
# Theoretical MFMA TFLOPS for MI355X
mfma_per_cu = 1  # One MFMA unit per CU
cus = 256
clock_ghz = 2.1
ops_per_mfma = 2 * 16 * 16 * 16  # For 16x16x16: 2*M*N*K FLOPs
cycles_per_mfma = 32

tflops = (cus * mfma_per_cu * ops_per_mfma * clock_ghz) / cycles_per_mfma / 1000
# ≈ 1350 TFLOPS (bf16/fp16)
```

## LDS (Local Data Share)

### LDS Banking

LDS has **32 banks** with 4-byte granularity:
- Address `A` maps to bank `(A / 4) % 32`
- Bank conflict: Multiple threads access same bank with different addresses
- Broadcast: Multiple threads access exact same address (no conflict)

```
Bank Assignment:
Address:  0   4   8   12  ...  124  128  132  ...
Bank:     0   1   2   3   ...  31   0    1    ...
```

### Avoiding Bank Conflicts

```python
# Bad: All threads hit bank 0
lds[tid * 32]  # Addresses: 0, 32, 64, ... → all bank 0!

# Good: Sequential access
lds[tid]  # Addresses: 0, 1, 2, ... → banks 0, 1, 2, ...

# Good: Stride of 1 element (4 bytes)
lds[tid * 1]  # No conflicts

# Swizzling for matrix tiles
def swizzled_lds_offset(row, col, stride):
    base = row * stride + col
    # XOR-based swizzle to spread across banks
    return base ^ ((row % 8) * 4)
```

### LDS Allocation and Occupancy

| LDS per WG | Max WGs/CU | Notes |
|------------|------------|-------|
| ≤40 KB | 4 | Good occupancy |
| ≤53 KB | 3 | Acceptable |
| ≤80 KB | 2 | Minimum for hiding latency |
| ≤160 KB | 1 | Low occupancy, avoid if possible |

## Register Pressure

### VGPR Allocation

Each thread has up to 256 VGPRs (32-bit each). Occupancy depends on VGPR usage:

| VGPRs/Thread | Max Waves/SIMD | Occupancy |
|--------------|----------------|-----------|
| ≤64 | 8 | 100% |
| ≤80 | 6 | 75% |
| ≤96 | 5 | 62.5% |
| ≤128 | 4 | 50% |
| ≤168 | 3 | 37.5% |
| ≤256 | 2 | 25% |

### Register Spilling

When VGPRs exceed allocation, compiler spills to:
1. **Scratch memory** (slow, global memory)
2. **LDS** (if available)

**Signs of spilling:**
- `scratch_` instructions in ISA
- Unexpectedly high memory traffic
- Performance cliff at certain tile sizes

## Memory Access Patterns

### Global Memory Coalescing

MI355X coalesces accesses within a wavefront into cache lines (64 bytes):

```python
# Good: Coalesced (64 threads access 64 consecutive elements)
data[tid]  # Single 256-byte transaction

# Bad: Strided (64 threads access with stride 64)
data[tid * 64]  # 64 separate transactions!

# Acceptable: Partial coalescing
data[tid * 2]  # 2 transactions
```

### Vectorized Loads

Use vector loads for better bandwidth:

```cpp
// Scalar load: 4 bytes
float a = ptr[idx];

// Vector load: 16 bytes (4x better)
float4 a = *reinterpret_cast<float4*>(&ptr[idx]);

// HIP intrinsic: 32 bytes
float8 a = __builtin_amdgcn_global_load_dwordx8(ptr);
```

### Async Copy (gfx950)

```cpp
// Async global → LDS copy
__builtin_amdgcn_global_load_lds(
    global_ptr,    // Source in global memory
    lds_ptr,       // Destination in LDS
    size,          // Bytes to copy
    offset,        // LDS offset
    aux            // Auxiliary data
);

// Wait for completion
__builtin_amdgcn_s_waitcnt_lgkmcnt(0);
```

## Synchronization Primitives

### Workgroup Barriers

```cpp
// Full workgroup barrier (LDS + memory)
__syncthreads();

// Equivalent in HIP
__builtin_amdgcn_s_barrier();

// Triton
tl.debug_barrier()
```

### Memory Fences

```cpp
// Workgroup scope fence
__threadfence_block();

// Device scope fence
__threadfence();

// System scope fence (for multi-GPU)
__threadfence_system();
```

### Wavefront Operations

```cpp
// Wavefront-level reduce (no barrier needed)
int sum = __reduce_add_sync(0xFFFFFFFFFFFFFFFF, value);

// Shuffle within wavefront
int other = __shfl_xor_sync(0xFFFFFFFFFFFFFFFF, value, 1);

// Ballot (which threads have condition true)
uint64_t mask = __ballot_sync(0xFFFFFFFFFFFFFFFF, condition);
```

## Performance Tuning Checklist

### 1. Occupancy Analysis

```bash
# Check with rocprof
rocprof --stats ./your_kernel

# Or use hipcc with stats
hipcc -Rpass-analysis=kernel-resource-usage your_kernel.cpp
```

**Target**: 2+ wavefronts per SIMD to hide latency

### 2. Memory Bandwidth Check

```python
# Calculate achieved bandwidth
bytes_moved = input_bytes + output_bytes
achieved_bw = bytes_moved / kernel_time

# MI355X peak: 8 TB/s
efficiency = achieved_bw / 8e12 * 100  # percent
# Target: >60% for memory-bound kernels
```

### 3. Compute Utilization

```python
# Calculate achieved TFLOPS
flops = 2 * M * N * K  # For GEMM
achieved_tflops = flops / kernel_time / 1e12

# MI355X peak: ~1350 TFLOPS (bf16)
efficiency = achieved_tflops / 1350 * 100  # percent
# Target: >70% for compute-bound kernels
```

### 4. Common Bottlenecks

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Low occupancy | High VGPR/LDS | Reduce tile size |
| Low bandwidth | Uncoalesced access | Check access patterns |
| Low compute | Insufficient ILP | Increase tile size, unroll loops |
| Kernel launch overhead | Small problem | Use persistent kernels |
| Bank conflicts | Bad LDS layout | Add swizzling |

## XGMI Interconnect (Multi-GPU)

### Topology

MI355X supports XGMI for GPU-to-GPU communication:
- 8 GPUs fully connected (all-to-all)
- ~900 GB/s per link
- Lower latency than PCIe

### Programming Considerations

```python
# Check GPU topology
import torch
for i in range(torch.cuda.device_count()):
    for j in range(torch.cuda.device_count()):
        if i != j:
            can_access = torch.cuda.can_device_access_peer(i, j)
            print(f"GPU {i} → GPU {j}: {'XGMI' if can_access else 'PCIe'}")
```

### IPC (Inter-Process Communication)

```cpp
// Share memory across GPUs via IPC
hipIpcMemHandle_t handle;
hipIpcGetMemHandle(&handle, device_ptr);
// Send handle to other process/GPU
hipIpcOpenMemHandle(&remote_ptr, handle, hipIpcMemLazyEnablePeerAccess);
```

## Profiling Tools

### rocprof

```bash
# Basic profiling
rocprof --stats ./your_app

# Detailed metrics
rocprof -i metrics.txt ./your_app

# Trace for timeline
rocprof --hip-trace --hsa-trace ./your_app
```

### Omniperf

```bash
# Comprehensive analysis
omniperf profile -n my_profile -- ./your_app
omniperf analyze -p my_profile/
```

### Key Metrics to Watch

| Metric | Good Value | Meaning |
|--------|------------|---------|
| `VALUUtilization` | >80% | Vector ALU busy |
| `MFMAUtilization` | >70% | Matrix unit busy |
| `LDSBankConflict` | <5% | LDS access efficiency |
| `L2CacheHit` | >90% | Data reuse |
| `MemUnitBusy` | <80% | Memory not bottleneck |

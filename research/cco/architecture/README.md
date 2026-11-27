# Architecture

Research hypothesis: reusable transport, synchronization and dependency protocols can coexist with operator-specific compute scheduling.

Layers: memory and topology; transport (compute-unit or copy engine where supported); tile readiness and ordering; persistent or staged scheduling; collective GEMM, MoE and attention; framework integration.

Compare independent-input, reduction-input and output-distribution dependencies. Preserve numerical semantics and forward progress. Choose fusion boundaries by communication volume, readiness granularity and compute efficiency, not fusion count.

This is a proposed architecture, not a claim of existing shared runtime implementation.

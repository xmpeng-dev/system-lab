# Repository maintenance guide

## Canonical ownership

- AGENTS.md and .cursor/rules: shared agent instructions.
- .cursor/skills: executable procedures, not experiment findings.
- knowledge: durable technical concepts with provenance.
- papers: external literature reviews; distinguish source claims from interpretation.
- notes: project evidence, decisions, dated progress and project status.
- benchmarks: reusable executable harnesses; keep experiment-specific scripts with evidence.
- weekly: historical reporting, linking to evidence instead of duplicating it.
- artifacts: derived visualizations; retain source references.
- summary: legacy archive; do not add new projects here.

## Research lifecycle

Question -> experiment plan -> reproducible evidence -> interpretation -> reusable knowledge -> publication-ready output.

Record experiment ID, code revision, hardware/software environment, workload, baseline, measurement protocol, raw evidence reference, result, limitations and verification status. Preserve negative results. Distinguish measured, reported, hypothesis and superseded conclusions.

## Maintenance

Weekly: update project indexes and progress links.
Monthly: review stale conclusions and consolidate duplicate indexes.
Quarterly: review active projects, archival candidates and skill usefulness.
Keep one canonical source for each conclusion. Generated reports should link to it.

## Publication boundary

This repository is public. New internal research, raw logs, unpublished manuscripts, private infrastructure details and employer-specific plans must stay in an authorized private environment. Review existing history for disclosure risks separately; moving or deleting a file does not remove its Git history.

## Migration policy

Phase 1: improve navigation without moving files.
Phase 2: audit references and external symlinks before moving any legacy directory; migrate one logical group per commit.
Phase 3: add automated link and metadata checks after conventions stabilize.
Preserve existing project paths and bootstrap behavior until compatibility has been tested. Do not rewrite history or delete raw evidence as part of routine cleanup.

---
name: team-weekly-report
description: >-
  Turn raw, per-person team updates into the team weekly report in the
  fixed house format (English, milestone section first, then one section
  per workstream titled "<Topic> - <Last>, <First>", number-dense past-tense
  bullets), save it under weekly/<year>/ and index it in weekly/README.md.
  Use when the user pastes team weekly content, says "写团队周报",
  "整理 team 周报", "按周报格式整理", "team weekly report", or sends
  teammates' updates to be formatted. Do NOT use for Xiaoming's personal
  cross-project weekly (existing weekly/<year>/*_weekly_report_* files) or
  for single-project progress notes (use progress-note).
---

# Team Weekly Report

> Xiaoming Peng (`Peng, Xiaoming`) collects the team's weekly updates and
> needs them reformatted into one paste-ready report (Teams / Outlook /
> Confluence). This skill fixes the format so every week looks the same.
> The personal cross-project weekly in `weekly/` is a different document;
> do not mix the two.

## When to apply

- The user pastes raw updates from one or more teammates and asks to
  organize them into the team weekly.
- The user asks to revise / extend a team weekly already in `weekly/<year>/`.

## When NOT to apply

- Xiaoming's personal weekly (`weekly/<year>/YYYY-MM-DD_weekly_report_*.md`
  without `team_`) — follow the existing files in that folder instead.
- Logging one experiment — use `.cursor/skills/progress-note/`.

## Workflow

### 1. Fix the week

Run `date "+%Y-%m-%d %H:%M %A"`. The report covers Monday–Sunday of the
current week unless the user names a range. File name uses the Friday
(report day) date:

```
weekly/<year>/YYYY-MM-DD_team_weekly_report_<MMDD>_<MMDD>_EN.md
```

### 2. Parse the raw input

For every item extract: owner, workstream, action, metric, before, after,
unit, measurement mode (boost / sustained / steady state, shapes), baseline
it is compared against, PR / commit references.

Use only numbers present in the input. Never compute a new headline number
the input does not state, except the obvious `(+x%)` or `x.xx×` derived
from a given before/after pair — and say so when you do.

### 3. Ask before guessing

Collect gaps into one short list and ask once (AskQuestion or plain
question): missing owner, missing unit, missing baseline, ambiguous
before/after direction, unclear which release a change landed in.
If the user says "just write it", leave a visible `[TBD: ...]` marker.

### 4. Organize sections

Order:

1. **Milestone section** (only if there is a release / merge / major
   deliverable this week): `<Milestone> (led by <Last>, <First>)`.
   This is the **overall platform progress overview** (e.g. MI455X
   Primus/Turbo): release facts, a one-sentence feature roll-up, the
   per-model P0 lines, and what is next for the platform. Kernel /
   workstream details (tuning tables, key changes, design notes) go in
   the owner's own section below, not as nested bullets here.
   - First bullet: what was released, where (link
     [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo) /
     [Primus](https://github.com/AMD-AGI/Primus) when mentioned), when it
     merged, and a one-sentence roll-up of the headline features with
     their key numbers in parentheses.
   - Second bullet: a group label such as `P0 model optimization`, with
     nested per-model lines:
     `<Model>: <before> → <after> <unit>, <before> → <after> <unit> (+x%) - <Last>, <First>`.
   - Keep each per-model line to headline numbers only (throughput,
     ms/iter, peak memory, with a short parenthetical at most) plus the
     owner. PRs, model-specific kernel numbers, and ongoing / next items
     go in that owner's per-person section below. A model with no
     detail beyond its P0 line (e.g. an ongoing item) adds nothing there.
2. **Per-person sections**: exactly one section per person, title
   `<Topic> - <Last>, <First>`. When a person has several topics, list
   them in the title (`Kimi K3.1 Pro / DeepEP / MoonEP - Yang, Bo`) and
   give each topic one bullet that starts with `<Topic>:`, with `Next:`
   inline at the end of that bullet. A single-topic person keeps plain
   bullets (no prefix). Put higher-impact / release-feeding people first;
   keep `Peng, Xiaoming`'s own section last unless the user says
   otherwise.

A topic jointly owned by several people gets one section titled
`<Topic> - <Last>, <First>; <Last>, <First>`.
Numbers repeated in the milestone roll-up and a workstream section must
match exactly.

### 5. Write the bullets

| Rule | Example |
|---|---|
| Past tense, verb first | Tuned / Optimized / Sent / Finished / Released / Added / Fixed |
| Improvement phrasing | `rose from 2032.5 to 2734.4 TF/s (+34.5%)` |
| Compact before/after | `763 → 719 ms/iter` (use `→`, not `->`) |
| Speedup | `1.30×` (use `×`, not `x`) |
| Percent | signed, in parentheses: `(+6.1%)`, `(−28% expert layer time)` |
| Ranges | `1–6% faster`, `1.48–2.01×` (en dash) |
| Shapes | `MiniMax-M3 (M=1024, N=6144, K=6144):` as bullet lead-in |
| Measurement mode | `boost (20 ms)`, `sustained (100 ms)`, `true steady state (2.5 s)` |
| Baseline named | `at 1.15× / 1.18× of AITER ASM`, `over serial RCCL + hipBLASLt` |
| References | `(PR #625)` |
| Names | always `Last, First` |
| Platform | name the GPU when results are hardware-specific: `On MI455X, ...` |

One bullet = one result. A short intro bullet (`Tuned grouped GEMM for
the MiniMax-M3 and Kimi K3.1 shapes.`) is fine when followed by per-shape
bullets. Add one clause of "why" only when the input gives it (e.g. why a
guard was made opt-in). No emojis, no bold inside bullets, no tables.

### 6. Save and index

File layout:

```markdown
# Team Weekly Report (MM/DD - MM/DD)

> **When**: YYYY-MM-DD HH:MM UTC+8
> **Where**: compiled by Peng, Xiaoming from team updates
> **Context**: team weekly; paste-ready body below the rule

---

<report body exactly as it will be pasted>
```

Section titles in the body are plain text lines (not `#` headings, not
bold), followed by a blank line and `-` bullets; nested bullets indent by
two spaces. Then add a row to the "团队周报清单" table in
`weekly/README.md` (period, headline, file link).

### 7. Reply

Show the paste-ready body in a fenced `markdown` block in chat, then list
any `[TBD]` items and derived numbers.

## Canonical example

Transcribed from the format reference the user provided (2026-11):

```markdown
MI455x Primus/Turbo v5 released (led by Zhang, Lihuan)

- Released mi455x_v5 in [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo) and [Primus](https://github.com/AMD-AGI/Primus): v5 merged into main on 11-18. It adds the FlyDSL MLA optimization for DeepSeek-V3 attention (forward 2,501 TF/s, 1.30×; backward 2,313 TF/s, 1.88×; 1.15× / 1.18× vs AITER ASM), grouped GEMM variable-K wgrad accumulating directly into fp32 main_grad (−28% expert layer time), Muse Spark grouped GEMM tuning (+12% geomean), and FlyDSL DeepEP with metadata carried in the rows (dispatch 1–11% faster).
- P0 model optimization
  - DeepSeek-V3: 763 → 719 ms/iter, 85,899 → 91,130 tok/s/GPU (+6.1%) - Zhang, Lihuan
  - MiniMax-M3.1-Flash: 168.6 → 123.8 ms/iter (1.36×, including no recompute and Turbo RMSNorm) - Huang, Wei
  - Kimi K3.1 Pro: 21,466 → 25,047 tok/s/GPU (+16.7%) - Yang, Bo

Grouped GEMM - Hu, Botao

- Tuned grouped GEMM for the MiniMax-M3 and Kimi K3.1 shapes.
- MiniMax-M3 (M=1024, N=6144, K=6144): boost (20 ms) rose from 2032.5 to 2734.4 TF/s (+34.5%), sustained (100 ms) from 1997.5 to 2517.5 TF/s (+26%), and true steady state (2.5 s) from 1741.3 to 2184.9 TF/s (+25.4%).
- Kimi K3.1 (M=293, N=6144, K=3584): boost rose from 1401 to 1905.2 TF/s (+35.9%), sustained from 1385.8 to 1843.3 TF/s (+33%), and true steady state from 1288.7 to 1628.9 TF/s (+26.3%).
- Kimi K3.1 (M=585, N=6144, K=3584): boost rose from 1830.9 to 2213.7 TF/s (+20.9%), sustained from 1829.2 to 2132.2 TF/s (+16.5%), and true steady state from 1723.6 to 2003.1 TF/s (+16.2%).

Attention - Zhang, Lihuan

- Optimized DeepSeek-V3 attention on FlyDSL. Forward reached 2501 TF/s (1.30×) and backward 2313 TF/s (1.88×), at 1.15× / 1.18× of AITER ASM and 89% / 83% of the MAIDAS B0 projection.

DeepEP - Yang, Bo

- Sent the metadata with each row in DeepEP dispatch (PR #625), making dispatch 1–6% faster and the worst rank 4–12% faster. Made the TDM guards opt-in, since NUMA balancing already removes the queue evictions they protect against. On MI455X, dispatch is 7–14% faster and combine is 3–6% faster.

MegaUlysses - Peng, Xiaoming

- Finished end-to-end validation of MegaUlysses on both the GQA and MLA paths, fusing the all-to-alls with the QKV, o_proj, and MLA latent up-projection GEMMs (GemmA2A / A2AGemm, AgGemm / GemmRS, including backward wgrad). The fused kernels are 1.48–2.01× over serial RCCL + hipBLASLt. Against unfused Ulysses with the same attention kernel, training is 1.7–8.6% faster for GQA and 4.0–9.9% faster for MLA.
```

## Anti-patterns

- Inventing or "rounding up" numbers, owners, PR numbers, or dates.
- Writing a narrative paragraph per person instead of one result per bullet.
- `First Last` name order, or dropping the owner from a section title.
- Mixing in personal-weekly material (ROI tables, negative-evidence logs)
  — that belongs in Xiaoming's personal weekly or `notes/`.
- Markdown headings / bold section titles in the paste-ready body.

## Related

- Personal weekly template: existing files in `weekly/2026/`
- Style rule: `.cursor/rules/40-notes-style.mdc`
- Per-experiment logs that may feed Xiaoming's own section: `notes/<project>/`

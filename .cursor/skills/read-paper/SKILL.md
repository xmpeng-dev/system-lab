---
name: read-paper
description: >-
  Reads research papers (PDF, arXiv, HTML, pasted text) and produces structured
  notes—metadata, problem, contributions, method, experiments, limitations, and
  follow-ups. Use when the user asks to read, summarize, digest, or take notes
  on a paper, preprint, article, or thesis chapter.
---

# Read Paper

## When to apply

Use this skill whenever the user wants help **understanding**, **summarizing**, or **note-taking** for a research paper or long technical article—not casual blog skims unless they ask for the same structure.

## Inputs

1. **Source**: path to PDF/markdown in the workspace, URL (arXiv, OpenReview, publisher), or pasted abstract + key sections if full text is unavailable.
2. **Goal** (infer if omitted): survey vs deep dive vs implementation vs related-work mining.

If only abstract is available, state that explicitly and avoid inventing full-method details.

## Workflow

1. **Locate text**: open the file or fetch accessible content; quote sparingly; prefer paraphrase with section/page pointers when possible.
2. **Skim structure**: title, abstract, introduction, method, experiments, conclusion; note figures/tables that carry the main claim.
3. **Extract claims**: separate *what the paper asserts* from *what is empirically shown*.
4. **Synthesize**: fill the output template below; flag missing or weak evidence.
5. **Optional tail**: if the user wants action next, add 3–7 concrete follow-ups (e.g., reproduce one experiment, read cited baseline paper X).

## Required output (use this template)

Produce **one** markdown document (or message) with the following sections, in order. Use `N/A` only when the paper truly omits that content.

```markdown
# [Paper title]

## Bibliographic metadata
- **Authors**:
- **Venue / year** (or arXiv id + version):
- **Identifiers**: DOI / arXiv / URL:
- **Code / data** (if any):

## One-paragraph thesis
[3–6 sentences: what problem, what approach at a high level, what evidence, why it matters]

## Problem & motivation
- **Task / setting**:
- **Gap vs prior work**:
- **Assumptions / scope**:

## Main contributions
- [Contribution 1 — crisp, falsifiable where possible]
- [Contribution 2]
- (add bullets until the paper’s claimed contributions are covered)

## Method
- **Core idea** (mechanism / algorithm / architecture):
- **Key definitions / notation** (only what is needed to understand results):
- **Training / inference procedure** (if applicable):
- **Complexity / compute** (if stated; else "not quantified"):

## Experiments & evidence
- **Datasets / benchmarks**:
- **Baselines / comparisons**:
- **Primary metrics**:
- **Main quantitative results** (table-level summary in prose or compact bullets):
- **Ablations / analyses** (what component matters and why):

## Limitations & risks
- **Stated limitations** (from paper):
- **Unstated / methodological risks** (fair critique, tied to evidence gaps):

## Positioning vs related work
- **Closest prior lines** and how this work differs:
- **Suggested reading next** (2–5 papers/citations worth opening):

## Glossary (optional, short)
- **Term** → one-line definition (only non-obvious domain terms used above)

## Open questions & reproducibility checklist
- **Questions** for the authors or for your own reading group:
- **Repro checklist**: data available? code? hyperparameters? random seeds? hardware?
```

## Style rules

- Prefer **precise, neutral** language; avoid hype unless analyzing author framing.
- Distinguish **claim** vs **evidence** (e.g., “Authors report … on …” vs “This implies …”).
- For equations: restate **in words** what they optimize or bound unless the user asks for full derivations.
- If the paper is **wrong or unclear**, say so with **specific** references to sections/figures, not generic doubt.

## Progressive disclosure

- For very long papers, keep **Method** and **Experiments** to the **minimum** needed to understand the headline results; offer to expand a section if the user asks.

---
name: canvas-to-html
description: >-
  Convert a Cursor `.canvas.tsx` file into one self-contained `.html` page that
  anyone can open in a plain browser without Cursor (no SDK, no fetch, no
  external assets). Use when the user wants to share, export, send, or
  download a canvas for non-Cursor viewers (colleagues, managers, external
  reviewers), email a canvas as an attachment, archive a canvas, or asks how
  to view a `cursor/canvas` artifact outside of Cursor.
---

# Canvas → standalone HTML

Cursor canvases use the `cursor/canvas` SDK that **only renders inside Cursor**.
This skill rewrites a canvas as a single, self-contained HTML file: inline
CSS, inline SVG, no JS framework, no network. The output opens in any
modern browser, prints to PDF cleanly, and works in both light and dark mode
via `prefers-color-scheme`.

## Workflow

Copy this checklist and tick as you go:

```
- [ ] 1. Read the source canvas top to bottom.
- [ ] 2. Read template.html and mapping.md from this skill.
- [ ] 3. Decide the output path (default: same directory as the source data).
- [ ] 4. Generate one .html file by filling the template.
- [ ] 5. Self-check (see § Self-check) before handing back to the user.
- [ ] 6. Tell the user the absolute path and that it can be sent as-is.
```

### 1. Read the source canvas

Locate the file (canvases live in `~/.cursor/projects/<workspace>/canvases/`,
but the source could be anywhere). Read it once end-to-end and note:

- Every component imported from `cursor/canvas`.
- Every data constant (arrays, objects, RUN config, etc.).
- Every inline `<svg>` block — these will be copied almost verbatim.
- Every call to `useHostTheme()` — these need to be swapped for CSS vars.

### 2. Read this skill's two helpers

- `template.html` — the HTML scaffold with all CSS classes, theme tokens, and
  `prefers-color-scheme` light/dark support. Start every conversion from this
  exact file; do not invent new class names.
- `mapping.md` — the `cursor/canvas` → HTML reference. Look up each component
  and copy the suggested HTML pattern.

### 3. Choose the output path

Default rule: write the HTML next to the **data** the canvas analyses, not
inside `canvases/` (the canvases dir is reserved for `.canvas.tsx` files Cursor
renders). Examples:

- Canvas about `b200/full_breakdown.py` data → write `b200/<name>.html`.
- Canvas about `<repo>/results/run42.json` → write `results/<name>.html`.
- If unclear, ask the user. Do **not** silently overwrite an existing `.html`.

Keep the kebab-case base name from the canvas (`foo.canvas.tsx` →
`foo.html`).

### 4. Generate the HTML

Open `template.html`, copy it into the new file, then:

1. Replace `{{TITLE}}` with the canvas's `<H1>` text.
2. Replace the placeholder `<main>` body with the converted content.
3. For each `cursor/canvas` element in the source, emit the HTML pattern from
   `mapping.md` § "Component reference".
4. For each inline `<svg>` block in the source, paste it verbatim and replace
   every `useHostTheme()` token access with the matching CSS variable
   (see `mapping.md` § "Theme tokens").
5. For every chart (`BarChart`, `LineChart`, `PieChart`), render it as inline
   SVG using the recipes in `mapping.md` § "Charts". Embed the data as
   literal numbers in the SVG path/rect attributes — do **not** generate it
   in JS at runtime.

### 5. Self-check

Before handing back to the user, verify ALL of:

- [ ] The file is **one** `.html`. No sibling `.css`, `.js`, or image files.
- [ ] No `<script src=...>`, no `<link rel="stylesheet" href=...>`, no
      `fetch()`. The file works fully offline.
- [ ] Opens in Chrome and Firefox without console errors. (Run
      `xdg-open <path>` or report the path so the user can click.)
- [ ] The numbers, labels, ordering, and colors match the source canvas.
- [ ] Both light and dark mode render readably (use the OS theme switch or
      Chrome DevTools' "Emulate CSS prefers-color-scheme").
- [ ] No slop: no `linear-gradient`, no `radial-gradient`, no
      `background-clip: text`, no emojis used as icons or status, no
      `box-shadow`, no rainbow coloring on every element.

### 6. Hand back

Reply with:

- The absolute path to the new HTML file.
- A one-line note: "Open it in any browser, send via email/chat, or print
  to PDF (Ctrl+P → Save as PDF)."

## Component & token reference

The full `cursor/canvas` → HTML mapping lives in `mapping.md`. The most
common cases:

| `cursor/canvas` | HTML scaffold |
|---|---|
| `<Stack gap={N}>` | `<div class="stack" style="gap:Npx">` |
| `<Row gap={N}>` | `<div class="row" style="gap:Npx">` |
| `<Grid columns={N} gap={M}>` | `<div class="grid grid-N" style="gap:Mpx">` |
| `<H1>` / `<H2>` / `<H3>` | `<h1>` / `<h2>` / `<h3>` |
| `<Text tone="secondary">` | `<p class="text text-secondary">` |
| `<Card>` + `<CardHeader trailing={X}>` + `<CardBody>` | `<div class="card">` + `<div class="card-header"><span>title</span><span>X</span></div>` + `<div class="card-body">` |
| `<Stat value="X" label="Y" tone="success">` | `<div class="stat stat-success"><div class="stat-value">X</div><div class="stat-label">Y</div></div>` |
| `<Pill tone="info">X</Pill>` | `<span class="pill pill-info">X</span>` |
| `<Callout tone="warning" title="T">…</Callout>` | `<div class="callout callout-warning"><div class="callout-title">T</div><div>…</div></div>` |
| `<Divider />` | `<hr class="divider">` |
| `<Table headers=[…] rows=[[…]] rowTone=[…]>` | `<table class="table">…</table>` with `<tr class="row-warning">` |
| `<BarChart …>` | inline `<svg>` — see `mapping.md` § Charts |
| `useHostTheme().text.secondary` | `var(--text-secondary)` |

## Anti-patterns

- **Do not** use a JS framework or CDN script — output must be 100% offline.
- **Do not** wrap the page in a Card. The canvas convention applies: most
  sections are open `<H2>` + content; cards are reserved for named entities.
- **Do not** add emojis or gradients to "make it pretty". The canvas
  aesthetic is flat and minimal — preserve it.
- **Do not** skip the `prefers-color-scheme` dark variables — managers
  reading on a dark-mode laptop will get a white flashbang otherwise.
- **Do not** generate data in `<script>` at runtime. Hardcode it into the
  HTML/SVG so the page works with JS disabled and prints cleanly to PDF.

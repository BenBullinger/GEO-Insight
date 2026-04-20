# Geo-Insight — Presentation

Local reveal.js deck for the Datathon 2026 final presentation. Clean sans-serif, deep-red single accent, muted palette.

## Requirements

Only Python 3 (already on macOS and Linux). No Node, no npm, no build step.

reveal.js itself is vendored into `vendor/reveal.js/` — cloning the repo and serving the folder is enough to present.

## Run locally

```bash
cd presentation
python3 -m http.server 8000
```

Then open <http://localhost:8000> in any browser.

## Keyboard

- `Space` / `→` — next slide
- `←` — previous slide
- `S` — speaker notes view
- `F` — fullscreen
- `O` — slide overview
- `?` — help overlay with all shortcuts

## Export to PDF

Append `?print-pdf` to the URL and use the browser's Print-to-PDF:

1. Open <http://localhost:8000/?print-pdf>
2. `Cmd/Ctrl + P`
3. Destination: **Save as PDF**
4. Layout: **Landscape**
5. Margins: **None**
6. Options: **Background graphics: on**

Chrome works best for this. The output matches what's on screen slide-for-slide.

## Files

```
presentation/
├── index.html             Main deck (slide content lives here)
├── css/theme-un.css       Custom theme
├── vendor/reveal.js/      Vendored reveal.js 6.0.1 (do not edit)
└── README.md
```

## Editing slides

All slides are `<section>` elements inside `index.html`. The theme provides some helpers:

- `<section class="title-slide">` — title slide layout
- `<section class="section-divider">` — blue divider between parts
- `<div class="cols">` / `<div class="cols-3">` — two/three-column layout
- `<div class="callout">` / `<div class="callout-warn">` — emphasis boxes
- `<div class="stat"><span class="value">42</span><span class="label">…</span></div>` — big-number stat
- `<p class="sources">…</p>` — footnote-style citation line
- `<div class="footer"><span>left</span><span>right</span></div>` — bottom-of-slide caption

Math renders via KaTeX:

- Inline: `\(G(u) = \sum_i w_i\,U_i\)`  
- Display: `\[ \bar{G}_k(u) = \tfrac{1}{k}\!\sum_t G_t(u) \]`


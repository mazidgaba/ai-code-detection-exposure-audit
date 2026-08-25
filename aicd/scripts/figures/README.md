# Figure sources

Two of the paper's figures are line art rather than plots, so they are drawn
directly rather than generated from a result file:

- `fig_concept.svg` — the distinction between benchmark-relative and
  detector-relative withholding
- `fig_methodology.svg` — the four stages of the design

They are rendered to PDF for the manuscript with a headless browser, which is
what produces selectable text at the sizes the two-column layout needs:

```bash
chrome --headless --disable-gpu --no-pdf-header-footer \
       --print-to-pdf=fig_concept.pdf fig_concept.html
```

where the HTML wraps the SVG with `@page{size:330px 338px;margin:0}` so the page
is exactly the artwork. Any SVG renderer will do; the browser is used because it
is the one that was to hand.

The remaining figures are plots and come from `make_kaggle_figures.py`, which
reads the stored results.

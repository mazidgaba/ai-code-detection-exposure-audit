# Verification

Two scripts that check the manuscript against the results in this repository.
Neither is needed to run the analyses; they are here because the paper says
its numeric claims are checked mechanically, and this is that check.

`review.py` parses the manuscript, finds every table, and compares each cell
against the report file that produced it. It writes
`research_state/numbers_ledger.csv`, which lists every figure the paper quotes
next to its source and whether the two agree. It also guards several claims
directly rather than only their digits: it fails if the compound twin effect
stops being many standard deviations from zero, if the in-distribution null
drifts, if an AICD Bench withheld language stops being present in the training
split, or if design variation overtakes seed variation.

`validate.py` checks the LaTeX itself: unbalanced environments and braces,
dangling references, uncited or undefined bibliography entries, missing figure
files, and rows whose cell count disagrees with the column spec.

Both need `main.tex`, which is not distributed here because the paper is
unpublished. Run without it they say so and exit cleanly.

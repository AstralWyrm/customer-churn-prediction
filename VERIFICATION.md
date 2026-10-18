# Verification record

Verified on 2026-10-18 with Python 3.12.14.

- `python -m unittest discover -s tests -v`: 7 tests passed.
- Every notebook code cell was executed successfully; all outputs are saved.
- All Python sources parse successfully.
- Core CLI entry points run with `python -S`, without site packages.
- All sections 2.1 through 2.4.4 executed, including both stopping-rule controls.
- Numerical thresholds agree with a brute-force reference on 80 small datasets.
- The requested four root thresholds reproduce 10.5 / 71.41 / 40.25 / 1.5.
- All-feature random forest predictions match the same-seed bagging model.
- Recorded metrics include 17 model settings, each evaluated on three splits.
- Fit timings are one run per setting and depend on the execution environment.

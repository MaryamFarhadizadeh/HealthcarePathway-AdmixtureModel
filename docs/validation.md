# Verification of the reproduction workflow

## Repeated and fresh-environment runs

The three scenarios (400 patients, three 80/20 splits, EM and SLSQP) were run
three times on 10 September 2026:

1. in the project environment;
2. again in the same environment;
3. in a fresh Python environment installed from `requirements-reproduction.txt`.

The fresh environment used Python 3.12.5 and installed all 18 pinned runtime
packages; `pip check` reported no broken requirements. All runs used the same
macOS arm64 machine and Graphviz 12.1.0, so cross-platform reproducibility has
not been tested.

All 115 CSV output files were byte-identical across the three runs. PDF files
and logs were not compared byte by byte, because they contain timestamps and
run-specific paths. Every recorded output hash was verified against its file.

## Additional checks

- Nine train-test splits: 320 training / 80 test patients, no patient in both.
- Eighteen admixture weight files: 80 test patients each, exactly three
  components, nonnegative weights summing to one.
- Twenty-seven transition matrices: nonnegative entries; each row sums to one,
  or to zero for states without observed outgoing transitions.
- The model input contains no latent backbone or theta columns.

Automated tests (exact recovery under component permutation, rejection of
unequal component counts, missing or duplicate patients, and protection of
existing output directories):

```bash
.venv-reproduction/bin/python -B -m unittest discover -s tests -v
```

## Results

EM means over the three splits, rounded to three decimals, as reported in
Supplementary Table S2. Each run writes this comparison to
`table_s2_comparison.csv`.

| Scenario | MAE | Correlation | ARI | Silhouette |
|---|---:|---:|---:|---:|
| Low | 0.234 | 0.834 | 0.940 | 0.923 |
| Moderate | 0.257 | 0.785 | 0.774 | 0.897 |
| High | 0.254 | 0.784 | 0.774 | 0.876 |

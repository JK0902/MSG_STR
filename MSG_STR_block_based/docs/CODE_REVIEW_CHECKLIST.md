# Independent code-review checklist

Reviewer name: ____________________  
Review date: ____________________  
Commit reviewed: ____________________

## Reproducibility

- [ ] Fresh environment installs successfully from `pyproject.toml`.
- [ ] All automated tests pass.
- [ ] Python sources pass compilation.
- [ ] Random seeds and analysis thresholds match the manuscript.
- [ ] The manuscript-to-code crosswalk is accurate.

## Data integrity and leakage

- [ ] Required columns and identifier uniqueness are checked.
- [ ] Model fitting and probability evaluation use the intended data partitions.
- [ ] Threshold selection is performed in the intended derivation block and then locked for validation.
- [ ] No outcome or post-index information is included in predictors.
- [ ] Missing-value handling matches the manuscript.

## Statistical calculations

- [ ] Confusion-matrix orientation is correct.
- [ ] Sensitivity and specificity reproduce independently calculated values.
- [ ] Prevalence-adjusted PPV and NPV reproduce Bayes-formula calculations.
- [ ] Alert burden equals `(TP + FP) / N`.
- [ ] GNN/no-graph ranking comparisons reproduce from exported symptom-level inputs.

## Privacy and security

- [ ] No patient identifiers, raw message text, credentials, or private paths are committed.
- [ ] Message-text processing remains disabled by default.
- [ ] Only aggregate outputs are written by the public workflow.

## Reviewer comments

______________________________________________________________________________

______________________________________________________________________________

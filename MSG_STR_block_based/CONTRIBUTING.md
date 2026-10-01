# Contributing and review

Changes should be made through a branch and pull request whenever possible.

Before requesting review:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src scripts
```

Pull requests should describe:

1. the manuscript method affected;
2. the rationale for the change;
3. any change to inputs, outputs, thresholds, or random seeds;
4. tests added or updated; and
5. whether numerical study results changed.

At least one independent analyst or coauthor should review changes affecting cohort construction, labels, model validation, threshold selection, or reported results.

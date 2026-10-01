# Patient-Centered, Graph-Augmented AI-Enabled Passive Surveillance for Early Stroke Risk Detection

This repository contains the modular, public-facing analysis code for the study **Patient-Centered, Graph-Augmented Artificial Intelligence-Enabled Passive Surveillance for Early Stroke Risk Detection in High-Risk Individuals**.

The refactor replaces monolithic notebook cells with small manuscript-aligned notebooks and tested Python modules. No patient-level data or portal-message text are distributed.

## Study overview

The project evaluates whether patient-reported language from secure portal messages can be transformed into interpretable signals for early stroke-risk detection among individuals with diabetes. The framework combines an iterative LLM-guided symptom taxonomy, structured symptom annotation, Elastic Net/LASSO, graph-derived symptom importance, and a conservative hybrid screening simulation.

The study objectives were to:

1. construct a hierarchical, patient-centered symptom representation;
2. identify early stroke-associated symptom patterns;
3. translate model outputs into a clinically interpretable screening signal; and
4. evaluate the feasibility of pre-event risk detection using passively collected messages.

The reported taxonomy included 35 main topics, 109 subtopics, and 495 granular concepts. The screening simulation evaluated 3–90-day windows and prioritized specificity above 0.90 and low alert burden.

## Analysis blocks

| Manuscript component | Stepwise notebook | Reusable implementation |
|---|---|---|
| Iterative LLM-guided taxonomy | `notebooks/01_MSG_Stroke_Supple_Modular.ipynb` | `taxonomy.py` |
| BERTopic semantic validation | Notebook 01, Blocks 3–4 | `bertopic_validation.py` |
| LLM symptom annotation | Notebook 01, Blocks 5–6 | `llm_annotation.py` |
| Hybrid screening simulation | `notebooks/02_Hybrid_Screening_Modular.ipynb` | `screening.py`, `metrics.py` |
| Separate GNN graph ablation | `notebooks/03_GNN_Ablation_Modular.ipynb` | `gnn_ablation.py`, `gnn_ablation_report.py` |

The GNN ablation is intentionally separate from the primary workflow. It measures changes in symptom prioritization after graph-derived terms are introduced; it is not interpreted as evidence of improved patient-level discrimination.

## Repository structure

```text
MSG_STR_block_based/
├── config/                 # Public-safe configuration template
├── data/                   # Input contracts; no study data
├── docs/                   # Manuscript crosswalk and review checklist
├── notebooks/              # Small, numbered execution blocks
├── scripts/                # Command-line entry points
├── src/msg_str/            # Tested reusable analysis functions
├── tests/                  # Unit and synthetic integration tests
├── .github/workflows/      # Automated checks on every push/PR
├── pyproject.toml
└── requirements.txt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Optional dependencies:

```bash
# BERTopic and sentence embeddings
python -m pip install -e ".[topic]"

# Google GenAI message annotation
python -m pip install -e ".[llm]"
```

## Privacy guard

Message-text processing is disabled by default:

```python
from msg_str.config import PrivacyConfig

privacy = PrivacyConfig(
    safe_mode=True,
    allow_text_input=False,
)
```

`allow_text_input=True` should be used only inside an approved secure environment. The public code logs aggregate counts only and does not export raw message text.

## Block 1: taxonomy and BERTopic validation

```bash
PYTHONPATH=src python scripts/run_topic_validation.py \
  --messages-csv data/deidentified_messages.csv \
  --taxonomy-csv data/publishable_seed_taxonomy.csv \
  --out-dir results/topic_validation \
  --allow-text-input
```

This command should be run only in the approved environment containing de-identified message text.

## Block 2: hybrid screening simulation

```bash
PYTHONPATH=src python scripts/run_screening.py \
  --messages-csv data/screening_messages.csv \
  --rubric-csv data/symptom_risk_rubric.csv \
  --out-dir results/screening \
  --windows 3,7,14,30,60,90 \
  --prevalence 0.10 \
  --minimum-specificity 0.90
```

The refactored screening implementation:

- uses a single parameterized function for all window lengths;
- removes duplicated 3-day rule functions;
- obtains probabilities from the fitted logistic estimator instead of undefined global coefficients;
- evaluates probabilities on held-out rows;
- calculates sensitivity, specificity, precision, NPV, prevalence-adjusted PPV/NPV, F1, accuracy, and alert burden; and
- applies deterministic operating-point selection under the specificity constraint.

## Block 3: separate GNN graph-component ablation

Full ablation:

```bash
PYTHONPATH=src python scripts/run_gnn_ablation.py \
  --input-csv data/symptom_importance_with_elasticnet.csv \
  --out-dir results/gnn_ablation \
  --percentile-threshold 80 \
  --top-k-values 10,20 \
  --graph-weight 0.5
```

Only the two manuscript-reported results:

```bash
PYTHONPATH=src python scripts/report_gnn_ablation.py \
  --input-csv data/symptom_importance_with_elasticnet.csv \
  --out-dir results/gnn_ablation_reported \
  --percentile-threshold 80 \
  --top-k-values 10,20 \
  --graph-weight 0.5
```

The focused script reports:

1. Spearman correlation, mean absolute rank change, and top-10/top-20 overlap between graph-augmented and no-graph event-association rankings.
2. Symptom topics selected by the graph-augmented candidate rule but not by the no-graph rule.

## Automated verification

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src scripts
```

The tests cover privacy guards, taxonomy seed construction, LLM-output parsing, screening feature aggregation, held-out logistic scoring, screening metrics, hybrid threshold scans, GNN composite formulas, rank statistics, and graph-only candidate-set differences.

GitHub Actions runs the same checks on each push and pull request.

## Reproducibility and code review

- All public functions have explicit inputs and outputs.
- Random-state parameters are fixed and configurable.
- Input schemas are validated before analysis.
- Generated results are excluded from source control.
- The manuscript-to-code mapping is documented in `docs/MANUSCRIPT_CODE_CROSSWALK.md`.
- An independent reviewer can use `docs/CODE_REVIEW_CHECKLIST.md` to document review.

Automated tests reduce implementation risk but do not replace independent scientific code review. A coauthor or independent analyst should review the repository before the final journal submission.

## Intended use

This repository is provided for research transparency, reproducibility, and academic use. It is not a deployed clinical decision-support system and is not intended for direct clinical use without prospective validation, institutional approval, and regulatory review.

## Data and ethics

All original analyses were conducted in a secure, HIPAA-compliant environment using de-identified data under institutional review. This repository does not include raw messages, identifiable EHR data, model credentials, or patient-level outputs.

## Citation

The article citation will be added after publication.

## Contact

For questions about methods or collaboration inquiries, please contact the corresponding author listed in the manuscript.

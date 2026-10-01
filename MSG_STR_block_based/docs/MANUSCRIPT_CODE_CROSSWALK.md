# Manuscript-to-code crosswalk

| Manuscript methods subsection | Notebook block | Source module | Principal output |
|---|---|---|---|
| Iterative LLM-guided symptom taxonomy | Notebook 01, Block 2 | `msg_str.taxonomy` | Seed-topic lists and documented prompt template |
| BERTopic semantic validation | Notebook 01, Blocks 3–4 | `msg_str.bertopic_validation` | Aggregate topic information and ontology alignment scores |
| LLM symptom annotation | Notebook 01, Blocks 5–6 | `msg_str.llm_annotation` | Validated single- or multi-label response parsers |
| Screening feature construction | Notebook 02, Blocks 2–4 | `msg_str.screening.compute_screening_features` | Person-level risk-category counts by screening window |
| Logistic screening component | Notebook 02, Block 5 | `msg_str.screening.fit_logistic_screening` | Held-out probabilities, coefficients, AUROC, and AUPRC |
| Symptom-count rule | Notebook 02, Block 6 | `msg_str.screening.grid_search_symptom_rule` | Candidate threshold combinations and metrics |
| Hybrid screening system | Notebook 02, Blocks 7–8 | `msg_str.screening.scan_hybrid_thresholds` | Sensitivity, specificity, adjusted PPV/NPV, and alert burden |
| Graph-component ablation | Notebook 03, Blocks 2–5 | `msg_str.gnn_ablation*` | Ranking agreement, rank changes, top-k overlap, and graph-only symptoms |

## Separation of concerns

The taxonomy, screening, and graph-ablation workflows are independent modules. This prevents notebook state from silently carrying variables between manuscript analyses and makes each method independently testable.

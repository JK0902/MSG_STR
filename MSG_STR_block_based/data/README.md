# Data inputs

No patient-level data or message text are distributed in this repository.

The public code expects de-identified inputs prepared inside an approved secure environment. Do not commit message text, patient identifiers, dates, model credentials, or cloud-storage credentials.

The screening script expects:

- a message-level table containing `user_id`, `classifications`, `time_difference_days`, and `stroke_event`; and
- a symptom rubric containing `symptom_id`, `symptom_risk_category`, and `stroke_risk_score`.

The GNN ablation expects the aggregate one-row-per-symptom table described in `docs/GNN_ABLATION_DATA_DICTIONARY.md`.

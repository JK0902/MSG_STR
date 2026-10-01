# Input data dictionary

The ablation takes a **symptom-level** CSV with one row per unique symptom topic.

| Column | Required | Definition |
|---|---|---|
| `symptom_id` | Yes | Unique symptom-topic identifier. |
| `symptom_name` | No | Human-readable topic name. Defaults to `symptom_id`. |
| `coverage` | Yes | Number of unique messages linked to the symptom. |
| `coverage_recency` | Yes | Recency-weighted sum of linked messages. |
| `learned_link_prob` | Yes | Mean message-to-symptom edge weight used by the primary pipeline. |
| `pagerank` | Yes | Symptom-node PageRank from the weighted bidirectional message–symptom graph. |
| `en_perm_importance` | Yes | Decrease in AUROC after permuting the symptom feature in the Elastic Net model. |
| `gnn_event_delta` | Yes | Increase in binary cross-entropy event loss after removing all edges incident to that symptom. Positive values indicate that removing the symptom worsened event classification. |
| `elasticnet_coef` | Yes | Signed Elastic Net coefficient; positive coefficients are required for high-risk candidate designation. |
| `event_assoc_score` | No | Submitted composite event-association score, used only for a reproduction check. |
| `importance_score` | No | Submitted composite structural-importance score, used only for a reproduction check. |

The analysis does not require or read patient identifiers, message text, timestamps, or individual diagnoses.

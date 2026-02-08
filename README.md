# Patient-Centered, Graph-Augmented AI-Enabled Passive Surveillance for Early Stroke Risk Detection

## Overview

This repository accompanies the study:

**Patient-Centered, Graph-Augmented Artificial Intelligence-Enabled Passive Surveillance for Early Stroke Risk Detection in High-Risk Individuals**

The project demonstrates how **patient-reported language from secure portal messages**, analyzed using **large language models (LLMs)** and **graph-based machine learning**, can be transformed into interpretable signals for **early stroke risk detection** among individuals with diabetes.

Rather than relying on structured EHR data or acute presentations, this work focuses on **passively collected, real-world symptom descriptions written in patients’ own words**, enabling **proactive risk recognition prior to stroke events**.

---

## Objectives

The primary objectives of this project are to:

### 1. Develop a patient-centered symptom representation  
Construct a hierarchical, interpretable symptom taxonomy grounded in how patients naturally describe their concerns in secure messages.

### 2. Identify early stroke-associated symptom patterns  
Use a dual machine-learning framework (Elastic Net / LASSO and heterogeneous Graph Neural Networks) to detect symptom patterns that are statistically and temporally associated with future stroke.

### 3. Translate model outputs into a clinically usable screening signal  
Design and evaluate a high-specificity, low-burden hybrid screening system suitable for EHR-based passive surveillance.

### 4. Demonstrate feasibility of proactive, pre-event stroke risk detection  
Show that patient-reported language alone can support early risk detection with meaningful lead time for clinical evaluation.

---

## Methods Summary

### Data Source and Cohort

- Retrospective cohort from a large academic health system (2014–2024)
- Adults with diabetes, with and without subsequent ischemic stroke
- De-identified secure patient portal messages (HIPAA Safe Harbor)
- Case–control matching by age, sex, race, ethnicity, and marital status

---

### Iterative LLM-Guided Symptom Taxonomy

- Built a **3-level hierarchy (MAIN → SUB1 → SUB2)** using an iterative LLM-guided process  
- Existing topics preserved whenever possible; new topics added only when clinically novel  
- Independent investigator review with high inter-rater reliability (Gwet’s AC1)  
- External validation using **BERTopic** to assess semantic overlap  
- Messages converted into structured symptom annotations using LLMs  

---

### Graph-Augmented Dual Machine Learning Pipeline

- **Elastic Net / LASSO logistic regression** for stable feature selection  
- **Heterogeneous Graph Neural Network (GraphSAGE)** modeling:
  - Patients, symptoms, comorbidities, and temporal message structure  
  - Optional semantic similarity edges via sentence embeddings  

**Symptom importance estimation:**
- GNN event-loss attribution  
- Elastic Net permutation importance  
- Combined into a unified **event association score**

---

### Hybrid Stroke Risk Screening Simulation

- Simulated EHR-based screening across **3–90 day windows**
- Combined:
  - Interpretable symptom-count rules  
  - Logistic regression risk scores  
- Screening thresholds optimized to **prioritize specificity (>0.9)** and minimize false alerts  
- Performance evaluated via **temporal validation blocks**

---

## Key Findings

### Rich symptom diversity
- **35** main topics  
- **109** subtopics  
- **495** granular symptom concepts identified from patient language  

### Clinically meaningful early signals
High-risk symptoms clustered into domains including:
- Vascular monitoring and cardiology concerns  
- Frailty and functional decline  
- Early neurologic symptoms (e.g., dizziness, gait issues)  
- Acute inflammatory or biologic triggers  

### High-precision screening performance
- **Specificity:** 1.00  
- **Prevalence-adjusted PPV:** 1.00  
- **Sensitivity:** up to 0.72 (highest in 90-day window)  
- **Alert burden:** 16–35%, depending on window  

### Actionable lead time
Longer screening windows (60–90 days) provided improved sensitivity while maintaining conservative alerting behavior.

---

## Repository Scope

This public repository provides:

- Documentation of the methodological framework  
- Example code structures and pseudocode for:
  - LLM-guided taxonomy building  
  - Graph-augmented machine learning pipelines  
  - Hybrid screening logic  
- Configuration templates for privacy-preserving analysis  

### This repository does **NOT** include:
- Raw patient messages  
- Identifiable EHR data  
- Deployed clinical decision support tools  
- Real-time surveillance systems  

---

## Intended Use

This work is intended for:

- Research transparency and reproducibility  
- Methodological reference for clinical NLP and graph ML  
- Academic and educational use  

**This repository is not intended for direct clinical deployment** without prospective validation, institutional approval, and regulatory review.

---

## Ethics and Privacy

- All analyses were conducted in a HIPAA-compliant environment  
- Only de-identified data were used  
- IRB approval obtained  
- No patient-level text is shared in this repository  

---

## Citation

If you use this work, please cite the associated manuscript (citation to be added upon publication).

---

## Contact

For questions about methods or collaboration inquiries, please contact the corresponding author listed in the manuscript.

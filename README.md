# EEG Foundation Challenge 2025  
**From Cross-Task to Cross-Subject EEG Decoding**

This biosignal challenge, accepted into the [NeurIPS 2025 Competition Track](https://neurips.cc/), aims to advance EEG decoding by tackling two core tasks:

1. **Cross-Task Transfer Learning** – develop models that transfer knowledge from passive EEG tasks to active tasks.  
2. **Subject-Invariant Representation** – build robust models that generalize across subjects to predict clinical factors.

See the [challenge paper on arXiv](https://doi.org/10.48550/arXiv.2506.19141) for full details.

---

## Competition Tasks

### Challenge 1: Cross-Task Transfer Learning  
Participants will predict behavioral metrics—**response time** (regression) and **success rate** (classification)—for the active **Contrast Change Detection (CCD)** task using EEG data from the passive **Surround Suppression (SuS)** paradigm.  
Utilizing unsupervised or self-supervised pretraining followed by fine-tuning is encouraged to generalize across both subjects and paradigms.

### Challenge 2: Psychopathology Factor Prediction (Subject-Invariant Representation)  
Participants must predict four continuous psychopathology scores—**p-factor**, **internalizing**, **externalizing**, and **attention**—based on EEG data collected from various paradigms.  
Pretraining on large-scale EEG data to learn generalized neural representations is recommended.

---

## Dataset Overview

The competition leverages the **HBN-EEG dataset**, which includes:

- **Participants**: Over 3,000 subjects  
- **EEG System**: High-density 128-channel recordings  
- **Tasks**:  
  - **Passive**: Resting State (RS), Surround Suppression (SuS), Movie Watching (MW)  
  - **Active**: Contrast Change Detection (CCD), Sequence Learning (SL), Symbol Search (SyS)  
- **Metadata**: Demographics (age, sex, handedness) and four psychopathology dimensions derived from CBCL.  
- **Format**: BIDS (Brain Imaging Data Structure)

---

## Starter Kit & Baselines

A starter kit will soon be released and includes code written using the **Braindecode** and **EEGDash** libraries. It provides utilities for data loading, model training, and evaluation.  
Contestants are free to build on top of these or use their own implementations.

---

## Timeline

| Phase                    | Date            |
|--------------------------|-----------------|
| Warm-up Phase Begins     | 15 August 2025  |
| Final Phase Begins       | 15 September 2025 |
| Competition Ends         | 31 October 2025 |
| Reports & Analysis Paper | 30 November 2025 |
| NeurIPS Workshop         | 6–7 December 2025 |

- **Warm-up Phase**: Open code submissions to a validation set; unlimited submissions allowed.  
- **Final Phase**: Submission of models evaluated on unreleased test data; submission limits in place. A 2-page method description is required.

---

## Awards & Recognition

Top-tier prize packages include:
-  **Spotlight Talk** at the NeurIPS Workshop *Foundation Models for the Brain and Body*  
- **$2,500 cash prize** for each of the top three teams (sponsored by Meta)  
- **Travel and registration** support for the main authors of the top teams

---

## Motivation

EEG signal decoding is challenged by signal noise, inter-subject variability, non-stationary recording conditions, and heterogeneity across tasks. This challenge aims to:

- Provide a large, diverse dataset for benchmarking  
- Promote transferable and generalized EEG decoding methods  
- Establish cross-task and cross-subject evaluation standards  
- Foster interdisciplinary collaboration between machine learning and neuroscience

---

## Organizing Team & Affiliations

The challenge is coordinated by:

- **Bruno Aristimunha** (INRIA / Université Paris-Saclay / Braindecode lead)  
- **Dung Truong** (UCSD / EEGDash developer)  
- **Pierre Guetschel** (Donders Institute)  
- **Seyed Yahya Shirazi** (UCSD / HBN-EEG curation lead)  
- **Arnaud Delorme** (ICM / EEGLAB)

Strategic oversight and domain expertise from leaders at DeepMind, Child Mind Institute, SDSC, McGill University, and others.

---

## Contact & Community

Questions? Join the discussion or contact the organizers:

- **Email / Google Group**: `neurips2025-eeg-competition@googlegroups.com`  
- **Website**: [eeg2025.github.io](https://eeg2025.github.io)

---

## How to Use This Repo

Clone the repo and watch for the starter kit release. Once published, follow the structure and add:

- **scripts/**: training and evaluation pipelines  
- **models/**: baseline and custom model definitions  
- **datasets/**: BIDS-based data handling  
- **README.md**: this document  
- **CONTRIBUTING.md**, **LICENSE** as needed

---


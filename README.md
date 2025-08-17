# EEG Foundation Challenge 2025 – Challenge 1  
**Cross-Task Transfer Learning for EEG Decoding**  

[![NeurIPS 2025](https://img.shields.io/badge/Conference-NeurIPS%202025-blue)](https://neurips.cc/)  
[![Codabench](https://img.shields.io/badge/Register-Codabench-orange)](https://www.codabench.org/)  
[![Dataset](https://img.shields.io/badge/Dataset-HBN--EEG-green)](https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/)  

---

## 📖 Overview  

This repository contains my work for **Challenge 1** of the [EEG Foundation Challenge 2025](https://eeg2025.github.io/):  

> **Goal:** Predict *behavioral outcomes* (response time and success rate) in the **Contrast Change Detection (CCD)** active task using EEG signals recorded during the **Surround Suppression (SuS)** passive task.  

This problem highlights the importance of **cross-task transfer learning**, where models must generalize between different paradigms.  

---

## 🎯 Task Description  

- **Inputs:**  
  - EEG data from the **Surround Suppression (SuS)** task.  
- **Outputs:**  
  - **Response Time (RT):** continuous regression target.  
  - **Success Rate:** binary classification target (success/failure).  

Participants are encouraged to apply **unsupervised or self-supervised pretraining** on passive EEG before fine-tuning for CCD outcomes.  

---

## 🧠 Dataset  

- **Dataset:** Healthy Brain Network (HBN) EEG  
- **Size:** >3,000 participants  
- **EEG System:** 128-channel high-density recordings  
- **Paradigms Used in Challenge 1:**  
  - **Passive:** Surround Suppression (SuS)  
  - **Active (Target):** Contrast Change Detection (CCD)  

- **Format:** BIDS (Brain Imaging Data Structure)  
- **Metadata:** demographics (age, sex, handedness)  

> ⚠️ Access details and preprocessing pipelines are included in the official starter kit.  

---

## ⚙️ Setup  

### Installation  

```bash
git clone https://github.com/<your-username>/eeg-challenge1.git
cd eeg-challenge1
conda create -n eeg2025 python=3.10 -y
conda activate eeg2025
pip install -r requirements.txt
```

### Dependencies  

- [PyTorch](https://pytorch.org/) ≥ 2.0  
- [MNE-Python](https://mne.tools/) ≥ 1.5  
- [Braindecode](https://braindecode.org/)  
- [EEGDash](https://github.com/eeglabdevelopers/EEGdash)  
- NumPy / SciPy / scikit-learn  

---

## 🚀 Usage  

### 1. Data Preparation  
```bash
python scripts/prepare_data.py --bids_root /path/to/HBN_BIDS
```

### 2. Train Model  
Example: EEGNet baseline with transfer learning  

```bash
python scripts/train.py   --task cross_task_transfer   --input SuS   --target CCD   --targets rt,success   --model eegnet   --epochs 50   --out_dir runs/eegnet_ctt
```

### 3. Evaluate Model  
```bash
python scripts/eval.py --run_dir runs/eegnet_ctt
```

### 4. Package for Submission  
```bash
python scripts/package_submission.py   --run_dir runs/eegnet_ctt   --out submissions/eegnet_ctt.zip
```

---

## 📊 Evaluation Metrics  

- **Regression (RT):** Mean Absolute Error (MAE)  
- **Classification (Success):** ROC-AUC (primary), Accuracy/F1 (secondary)  

Leaderboard scoring is handled automatically via **Codabench**.  

---

## 📅 Timeline  

| Phase                  | Date |
|------------------------|------|
| Warm-up Phase Begins   | 15 Aug 2025 |
| Final Phase Begins     | 15 Sep 2025 |
| Competition Ends       | 31 Oct 2025 |
| Reports Due            | 30 Nov 2025 |
| Workshop @ NeurIPS     | 6–7 Dec 2025 |  

---

## 📚 References  

- [EEG Foundation Challenge 2025 Website](https://eeg2025.github.io/)  
- [HBN-EEG Dataset](https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/)  
- Aristimunha et al. (2025). *EEG Foundation Challenge 2025: From Cross-Task to Cross-Subject EEG Decoding.* [arXiv:2506.19141](https://doi.org/10.48550/arXiv.2506.19141)  

---

## ✍️ Author  

This repository is maintained by **Erin Moore** and collaborator **Michael Sousa** as part of participation in the **EEG Foundation Challenge 2025 (Challenge 1)**.  

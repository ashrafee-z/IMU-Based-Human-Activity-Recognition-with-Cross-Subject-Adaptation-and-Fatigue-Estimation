# Extending IMU-Based Human Activity Recognition with Cross-Subject Adaptation, Early Activity Recognition & Fatigue Estimation

This repository contains the implementation and analysis for an extended Human Activity Recognition (HAR) framework using IMU data. The project explores **cross-subject domain adaptation**, **early activity recognition**, **fatigue estimation**, **sensor-failure robustness**, and **explainability** using traditional ML and deep learning models.

> **Author:** Zarif Ashrafee
> **Dataset:** PAMAP2 (IMU + Heart Rate)

---

## Overview

Most IMU-based HAR systems struggle with real-world deployment issues such as:

* Inter-subject variability
* Real-time latency
* Sensor dropouts
* Limited interpretability
* Absence of physiological context (fatigue, exertion)

This project evaluates these challenges and proposes practical extensions to improve HAR reliability and generalization.

---

## Key Contributions

### 1. **Cross-Subject Domain Adaptation**

Implemented and compared:

* **CORAL** (Correlation Alignment)
* **DANN** (Domain-Adversarial Neural Network)

Both methods outperformed the baseline Logistic Regression classifier, with **DANN reaching ~89.6% accuracy**.

---

### 2. **Early Activity Recognition (EAR)**

Predicted user activity from **partial windows** (100–512 samples).
Although classical models yield ~52–56% accuracy, results suggest deeper sequential models (LSTM/Transformer) are needed for reliable forecasting.

---

### 3. **Fatigue / Exertion Estimation**

Heart rate values were normalized and converted into:

* Low
* Moderate
* High

A baseline classifier reached **~76% accuracy**, showing viability of proxy exertion estimation without explicit survey labels.

---

### 4. **Robustness Under Sensor Failure**

Simulated:

* Full sensor-group dropout (Hand / Chest / Ankle)
* Probabilistic dropout augmentation

The ankle sensor had the strongest impact, but augmentation restored accuracy—**even exceeding baseline in some cases (~91.65%)**.

---

### 5. **Explainable AI (XAI)**

Used **LIME** and **SHAP** to identify top contributing features:

* Hand-IMU x-axis accelerometer statistics
* PSD-filtered chest magnetometer signals
* Ankle orientation frequencies

This supports interpretability for healthcare and safety-critical use cases.

---

## Project Structure

```
├── notebook.ipynb         # Full implementation (preprocessing, training, evaluation)
├── Optional (Folder)      # From the PAMAP2 Dataset
├── Protocol (Folder)      # From the PAMAP2 Dataset
```

---

## Methodology Summary

### **Dataset**

* PAMAP2 (2.8M rows, 55 features)
* IMU sensors: Hand, Chest, Ankle
* Heart rate sampled at 9 Hz

### **Pre-processing Pipeline**

1. Data cleaning (imputation, removal of transient activities)
2. Sliding windows (512 samples, 1s stride)
3. PSD-based noise filtering
4. Feature extraction & scaling
5. Subject-based LOSO validation

### **Models**

* Baseline: Logistic Regression, Random Forest
* CORAL (domain alignment)
* DANN (deep domain adaptation)
* Fatigue estimation classifier
* EAR with prefix-based windowing

---

## Main Results

| Task                             | Best Accuracy |
| -------------------------------- | ------------- |
| Baseline (RF)                    | **94.62%**    |
| CORAL                            | **86.27%**    |
| DANN                             | **89.61%**    |
| Fatigue Estimation               | **76.62%**    |
| Early Activity Recognition       | **~55%**      |
| Sensor Dropout (Augmented Ankle) | **91.65%**    |

---

## Hardware Used

* **RTX 4060 GPU**
* **Ryzen 7 7840HS**
* **16 GB LPDDR5X RAM**

---

## Future Work

* Collect dataset with true exertion labels
* Explore multimodal fusion (EEG, video, audio)
* Smartphone-only IMU HAR deployment
* Deep domain adaptation for improved generalization
* Sequence-to-sequence activity forecasting
* Privacy-preserving HAR (federated learning, differential privacy)

---

# Two-Stream SepConvLSTM for Real-Time Violence Detection (RWF-2000)

This project implements an efficient deep learning solution for classifying video clips as **"Fight"** or **"NonFight"** using a **Two-Stream MobileNetV2 + SepConvLSTM Network**. The model is optimized for **real-time CCTV surveillance**.

**Reference Repository:** [TwoStreamSepConvLSTM_ViolenceDetection](https://github.com/zahid58/TwoStreamSepConvLSTM_ViolenceDetection)

---

## 1. Core Model Architecture

The model uses the **Two-Stream principle** to capture both spatial and temporal information:

| Stream | Purpose | Architecture |
|--------|---------|--------------|
| **Spatial Stream ("What")** | Recognizes static scene content (people, objects) | MobileNetV2 on individual frames |
| **Temporal Stream ("How")** | Detects motion between successive frames | MobileNetV2 on motion difference frames |
| **Sequential Modeling** | Learn long-term dependencies | SepConvLSTM layers |
| **Fusion** | Combine both streams | Concatenation Fusion (`fusionType C`) |
| **Sequence Length** | Number of frames per clip | 8 frames (`T = 8`) |

---

## 2. Environment & Setup

- **Python Virtual Environment** with GPU support.  
- **GPU:** NVIDIA RTX 3050 Ti (CUDA + cuDNN).  
- **Dependencies:** Install via `requirements.txt`:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

---

## 3. Training & Performance

**RWF-2000 Dataset Training Summary (≈57 epochs):**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Peak Training Accuracy | 78% | High on training data |
| Peak Validation Accuracy | 70% | Moderate generalization |
| Gap (Overfitting) | 8% | Needs hyperparameter tuning |

**Next Steps:** Improve generalization by adjusting **dropout, L2 regularization**, and **data augmentation**.

---

## 4. File Structure & Paths

| Component | Local Path | Purpose |
|-----------|------------|---------|
| Preprocessed Data | `D:\rwf2000\processed\val` | Input to `--dataPath` |
| Model Weights | `D:\fela_results` | Input to `--weightsPath` |
| Best Checkpoint | `D:\fela_results\rwf2000_best_val_acc_Model` | Saved weights prefix |

> ⚠️ Ensure dataset and weights paths match the arguments used in scripts.

---

## 5. Evaluation Fix (Shape Mismatch)

**Issue:** `ValueError` during weight loading due to **sequence length mismatch**:

- SepConvLSTM expects **8 frames**  
- Original DataGenerator sampled **32 frames**

**Fix:**

1. Set `--modelVidLen 8` when building the model.  
2. Ensure DataGenerator samples **8 frames per clip**.  
3. Load weights using:

\`\`\`python
model.load_weights().expect_partial()
\`\`\`

**Evaluation Command:**

\`\`\`bash
python evaluate.py \
  --dataset rwf2000 \
  --batchSize 4 \
  --mode both \
  --lstmType sepconv \
  --fusionType C \
  --dataPath D:/rwf2000/processed \
  --weightsPath D:/fela_results \
  --vidLen 32 \
  --modelVidLen 8
\`\`\`

---

## 6. Next Steps

- Run evaluation to obtain **final test accuracy**.  
- Tune hyperparameters to reduce **8% overfitting gap**.  
- Explore strategies like **increased dropout, L2 regularization, and data augmentation**.

---

## 7. References

- [RWF-2000 Dataset](https://github.com/andrewssobral/real-world-fighting-dataset)  

---

**Author:** Your Name  
**Date:** November 2025

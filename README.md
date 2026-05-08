# Morphologically Grounded ECG Arrhythmia Classification

**CS437/CS5317/EE414/EE513 — Deep Learning, Spring 2026**  
Hassan Iftikhar (27100204) and Saim Bilal (27100208)

---

## Overview

Standard ECG deep learning models achieve high benchmark accuracy by exploiting global amplitude statistics and spectral shortcuts rather than the localised P-QRS-T morphology a cardiologist actually inspects. This project addresses that gap through three progressive stages on PTB-XL, restricted to Lead II and five diagnostic superclasses: NORM, MI, STTC, CD, and HYP.

The central question throughout is not just whether the model predicts correctly, but whether it attends to the right parts of the signal — the same morphological landmarks a clinician would use to justify the same diagnosis.

---

## Repository Structure

```
ecg-morphological-grounding-dl-project/
│
├── notebooks/
│   ├── baseline_final.ipynb
│   ├── first_improvement.ipynb
│   └── final_second_improvement.ipynb
│
├── research-paper/
│   ├── 26_27100204_27100208_Report.pdf
│   ├── main.tex
│   ├── main.bib
│   ├── gradcam_A.png
│   ├── gradcam_E.png
│   ├── icml2021.sty
│   ├── icml2021.bst
│   ├── icml_numpapers.eps
│   ├── fancyhdr.sty
│   ├── algorithm.sty
│   └── algorithmic.sty
│
├── soa_survey/
│   └── 26_27100204_27100208_SoA_Survey.pdf
│
├── README.md
└── .gitignore
```

---

## Project Stages

### Task 3 — Baseline

**Notebook:** `notebooks/baseline_final.ipynb`

A 1D-ResNet trained on three leads (I, II, V1) with Butterworth bandpass filtering at 0.5 to 40 Hz and per-lead z-score normalisation. Explainability is measured through a Morphological Alignment Score (MAS) computed by comparing Grad-CAM heatmaps against binary P/QRS/T masks derived from NeuroKit2 Pan-Tompkins delineation on Lead II.

**Architecture:** 4-stage 1D-ResNet (stem to 32 to 64 to 128 to 256 channels) with global average pooling and a sigmoid MLP head for multi-label prediction.

**Training:** AdamW (lr=1e-3, wd=1e-4), ReduceLROnPlateau (patience=5), BCEWithLogitsLoss with per-class positive weights, 70 epochs. Best checkpoint selected by validation Macro-AUC.

---

### Task 4 — Preprocessing Ablation

**Notebook:** `notebooks/first_improvement_final.ipynb`

Five preprocessing conditions trained on the same frozen architecture. LBWA (Lead-wise Beat-Warp Augmentation) was designed as a beat-preserving augmentation combining the STAR principle with piecewise time-warping theory from Iwana and Uchida (PLOS ONE, 2021). Despite the principled design it consistently hurt both classification performance and MAS compared to no augmentation, and was dropped.

| Condition | Preprocessing | Macro-AUC | Macro-F1 | Mean MAS |
|-----------|---------------|-----------|----------|----------|
| A | Butterworth, no aug | 0.8975 | 0.6952 | 0.478 |
| B | Butterworth + LBWA | 0.8946 | 0.6722 | 0.456 |
| C | SWT, no aug | 0.8984 | 0.6840 | 0.620 |
| D | SWT + LBWA | 0.8931 | 0.6582 | 0.535 |
| **E** | **Raw + Z-score, no aug** | **0.8987** | **0.6949** | **0.711** |

**Key finding:** Raw ECG with z-score normalisation alone outperformed every filtered and augmented variant. Both Butterworth and SWT retain only around 62% of P-wave amplitude on average, which degrades morphological learning for P-wave-dependent classes.

---

### Task 5 — Physiological Alignment Supervision

**Notebook:** `notebooks/final_run_2.ipynb`

Keeps the raw + z-score pipeline and adds class-conditioned physiological alignment supervision to the training objective. Class-specific CAMs (computed from classifier weights multiplied by pre-GAP features) are regularised toward Gaussian soft priors encoding the diagnostically relevant wave region per arrhythmia class.

**Class-wave prior mapping:**

| Class | Relevant wave regions |
|-------|-----------------------|
| NORM | P + QRS + ST + T (uniform) |
| MI | ST + T + QRS edge |
| STTC | ST + T |
| CD | QRS only |
| HYP | QRS + ST |

**Four-term loss:**

```
L = L_cls
  + lambda_a * conf * L_align    (asymmetric KL toward physiological prior, alpha=0.7)
  + lambda_o * conf * L_out      (penalise CAM mass outside prior region)
  + lambda_e *        L_entropy  (sharpen temporal localisation)
```

**Ablation results relative to baseline:**

| Condition | Delta AUC | Delta F1 | Delta MAS | Delta CD MAS |
|-----------|-----------|----------|-----------|--------------|
| Align | +0.0004 | +0.0039 | +0.0026 | +0.0096 |
| Align+Out | +0.0003 | +0.0037 | +0.0026 | +0.0094 |
| Full | +0.0008 | +0.0026 | +0.0026 | +0.0097 |

**Key finding:** 58% relative improvement in class-specific MAS with minimal classification cost. The Conduction Disturbance class sees a roughly tenfold MAS increase because QRS widening maps to the narrowest and most spatially precise physiological prior (sigma = 40 ms).

---

## Dataset

PTB-XL is publicly available and is not included in this repository.

- **PhysioNet:** https://physionet.org/content/ptb-xl/1.0.1/
- **Kaggle:** https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset

After downloading, update `DATA_PATH` at the top of each notebook to point to your local copy. All notebooks were originally run on Kaggle with a P100 GPU. The Kaggle dataset path used in the notebooks is:

```
/kaggle/input/datasets/khyeh0719/ptb-xl-dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/
```

---

## Dependencies

```bash
pip install wfdb neurokit2 torch torchvision scikit-learn \
            scipy matplotlib numpy pandas PyWavelets captum
```

---

## Reproducing Results

Run notebooks in this order:

1. `baseline_final.ipynb`
2. `first_improvement.ipynb`
3. `final_second_improvement.ipynb`

Each notebook is self-contained with markdown cells explaining every section. The inter-patient split uses PTB-XL strat_fold: train on folds 1 to 8, validation on fold 9, test on fold 10. Patient-ID overlap between train and test is verified to be zero before every training run. Checkpoints are saved to `CHECKPOINT_DIR` defined at the top of each notebook.

---

## Metrics

**Macro-AUC / Macro-F1:** Primary classification metrics. Raw accuracy is not reported because NORM comprises 44.5% of the dataset and would inflate accuracy trivially.

**MAS (Morphological Alignment Score):** Measures whether the model's saliency map activates within clinically correct ECG regions. Two variants are used across tasks and are not directly comparable to each other. See Section 3.2 of the report for full definitions and the reason for the definitional shift between Task 4 and Task 5.

---

## Report

Full paper: `research-paper/26_27100204_27100208_Report.pdf`

To recompile from source:

```bash
cd research-paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

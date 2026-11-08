# Enhanced Multi-Class Pneumonia Detection System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/Dataset%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

An end-to-end medical deep learning diagnostic system for 3-class pneumonia screening and classification from chest radiographs:
- **Class 0:** Normal
- **Class 1:** Bacterial Pneumonia
- **Class 2:** Viral Pneumonia

The diagnostic pipeline integrates:
1. **Adaptive Contrast Enhancement (CLAHE):** Standardizes radiological exposure and clarifies subtle parenchymal textures.
2. **Biomedical U-Net Segmentation:** Automatically segments the anatomical lung fields and extracts region-of-interest (ROI) parenchyma, removing non-diagnostic clavicle, rib, and mediastinal tissue.
3. **Deep CNN Classifier (DenseNet-121 / 2-Stage Cascade):** Classifies radiographs at 384x384 resolution with focal loss and balanced oversampling.
4. **Grad-CAM Explainable AI (XAI):** Generates activation heatmaps on the final feature layers to offer clinicians spatial interpretability.

---

## 1. Quick Start & Environment Setup

```bash
# Clone the repository
git clone https://github.com/Adithya-void/pneumonia-detection-system.git
cd pneumonia-detection-system

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Dataset Auditing & Mask Verification

To verify dataset integrity and anatomical segmentation masks where the data is stored (locally or in Google Colab):

### Run Dataset Audit
```bash
python audit.py
# or using unified CLI:
python run.py audit
```
*Scans raw radiograph counts, split distributions (train/val/test), class balance, file corruptions, and verifies 0% train-test contamination.*

### Run Lung Segmentation Mask Check
```bash
python check_masks.py
# or using unified CLI:
python run.py check-masks
```
*Validates mask existence, 256x256 uniform spatial resolution, intensity range [0, 255], mean physiological lung coverage (~17%--28%), and ensures zero degenerate/blank masks.*

---

## 3. Run Experiments & Export Results Tables

To reproduce test cohort evaluations ($N=624$ test radiographs) and generate benchmark tables:

```bash
python run_experiments.py
# or using unified CLI:
python run.py run-experiments
```

This outputs:
- `results.csv`: Machine-readable CSV table containing quantitative metrics across all evaluated architectures.
- `results_tables.tex`: Camera-ready LaTeX `booktabs` tables formatted for academic publication and conference inclusion.

---

## 4. Test Set Benchmark Results ($N = 624$)

| Model Architecture | Task | Resolution | Overall Accuracy | Macro $F_1$ | Normal Prec. | Bacterial Recall | Screening Sensitivity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Biomedical U-Net** | Lung Segmentation | $256 \times 256$ | N/A (Dice 89.84%) | N/A | 100.0% | N/A | N/A |
| **ResNet-50 (Baseline)** | 3-Class Diagnosis | $224 \times 224$ | 71.47% | 0.6892 | 91.20% | 92.15% | 96.41% |
| **2-Stage Cascade (Initial)** | Hierarchical Cascade | $384 \times 384$ | 65.06% | 0.5969 | 94.12% | 97.11% | 97.44% |
| **2-Stage Cascade (Tuned)** | Hierarchical Cascade | $384 \times 384$ | 72.92% | 0.7051 | 94.52% | 94.63% | 98.72% |
| **DenseNet-121 (Flagship)** | End-to-End Multi-Class | $384 \times 384$ | **75.96%** | **0.7464** | **98.16%** | **94.63%** | **99.23%** |

*Note: Screening sensitivity represents the rate of detecting any pneumonia pathology (387 out of 390 disease cases detected).*

---

## 5. Dataset Information & Licence

- **Dataset Title:** Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images (Chest X-Ray / Kermany et al.)
- **License:** **Creative Commons Attribution 4.0 International (CC BY 4.0)**
- **Citation:**
  > Kermany, D. S., Goldbaum, M., Cai, W., Valentim, C. C., Liang, H., Baxter, S. L., ... & Zhang, K. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.* **Cell**, 172(5), 1122-1131.
  > DOI: [10.1016/j.cell.2018.02.010](https://doi.org/10.1016/j.cell.2018.02.010)
  > Mendeley Data: [http://dx.doi.org/10.17632/rscbjbr9sj.2](http://dx.doi.org/10.17632/rscbjbr9sj.2)

---

## 6. Repository Structure

```text
├── audit.py                    # Dataset audit and integrity verification script
├── check_masks.py              # Segmentation mask verification script
├── run.py                      # Unified CLI entry point (audit, check-masks, run-experiments)
├── run_experiments.py          # Quantitative benchmark compilation script
├── results.csv                 # Generated benchmark metrics CSV
├── results_tables.tex          # Generated publication LaTeX tables
├── requirements.txt            # Python dependencies
├── app.py                      # Streamlit interactive diagnostic dashboard
├── app_flask.py                # Flask web service
├── data/
│   ├── raw/chest_xray/         # Raw radiograph splits (train, val, test)
│   └── processed/              # CLAHE and U-Net segmented images
├── src/
│   ├── data/                   # Dataset parsing, CLAHE, ROI extraction, augmentation
│   └── models/                 # U-Net segmentation, DenseNet-121 classifier, Grad-CAM
└── weights/                    # Pretrained PyTorch weights (.pth)
```

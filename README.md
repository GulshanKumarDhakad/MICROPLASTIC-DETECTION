# Microplastic Detection from Spectral Peaks (Random Forest)

Classify a sample as a common microplastic polymer, or as a natural material, using only the **positions of the strongest peaks (in cm⁻¹)** from its FTIR/Raman spectrum.

**Target classes:** `PE`, `PP`, `PS`, `PET`, `PVC`, `Nylon`, `no_microplastic`

> **Project status:** work in progress. The notebook currently covers setup, dataset generation, and exploratory analysis (Parts 1–3 of the plan below). Feature engineering, model training, and evaluation are planned but **not implemented yet**, so there are no model results to report.

---

## Repository contents

| File | Description |
|---|---|
| `microplastic_detection.ipynb` | Main notebook: problem definition, simulated dataset, EDA |
| `microplastics_simulated.csv` | Generated when the notebook runs (2,240 rows) |

## Getting started

**Requirements:** Python 3.12 (tested), `numpy`, `pandas`, `matplotlib`, `jupyter`

```bash
pip install numpy pandas matplotlib jupyter
jupyter notebook microplastic_detection.ipynb
```

Run all cells top to bottom. A fixed seed (`SEED = 42`) makes the dataset reproducible.

## Notebook plan

| # | Section | Status |
|---|---|---|
| 1 | Setup & problem definition | Done |
| 2 | Build a realistic dataset (noise, shifts, missing peaks, contaminants, batches) | Done |
| 3 | Exploratory data analysis | Done |
| 4 | Feature engineering (binned peak-presence vectors) | Planned |
| 5 | Models + evaluation (baseline → Random Forest → tuning), batch-aware CV | Planned |
| 6 | Explainability (feature importance) | Planned |
| 7 | Save model + predict on new samples | Planned |
| 8 | Limitations / next steps | Planned |

## Dataset

The data is **simulated** from approximate textbook reference peaks (see `REF_PEAKS` and `NATURAL` in the notebook). The generator is deliberately noisy, because fixed peaks with tiny noise would give a meaningless ~100% accuracy.

Each sample gets:

- **Peak shift noise:** Gaussian jitter (SD = 3 cm⁻¹) to mimic instrument precision
- **Batch calibration offset:** a per-batch shift to mimic different days or instruments
- **Missing peaks:** each reference peak is dropped with probability 0.15
- **Contaminant peaks:** 0–3 random extra peaks between 500 and 3500 cm⁻¹ (dyes, additives, residue)
- **Top-10 truncation:** at most 10 peaks kept, sorted by position, with `NaN` for empty slots
- **Natural look-alikes:** `no_microplastic` samples are drawn from cellulose, calcite, silica, or protein

**Schema:**

| Column | Description |
|---|---|
| `batch` | Batch ID (0–39), simulates instrument/day |
| `Label` | One of the 7 classes |
| `Peak_1` … `Peak_10` | Peak positions in cm⁻¹, ascending; `NaN` = no peak |

**Size:** 40 batches × 7 classes × 8 samples per class per batch = **2,240 samples** (320 per class, perfectly balanced).

### Observations from EDA

- Classes overlap heavily (e.g. PE and PP both have peaks near 2915 cm⁻¹; cellulose and PS both near 1030 cm⁻¹), which is what makes this a real classification problem.
- `no_microplastic` samples have the fewest detected peaks (mean ≈ 5.4), while PS has the most (mean ≈ 8.9).
- `batch` is kept in the data on purpose, so that evaluation can split **by batch** and mimic unseen instruments or days instead of just unseen rows from known batches.

## Using your own data

If you have real spectra, replace the dataset section with a loader that produces the same schema (`batch`, `Label`, `Peak_1..Peak_10`). The rest of the notebook is meant to work unchanged.

## Caveats

- Reference peak positions are **approximate**. Verify them against a spectral library (e.g. SLoPP or other open libraries) or your own lab data before drawing conclusions.
- Results on simulated data say little about real-world performance. Real spectra have intensity information, baseline effects, weathering, and mixtures that this simulation does not capture.
- Peak positions only are used here; peak intensities and widths are ignored.

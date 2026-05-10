# NovaAI — Unified Domestic Audio Classification

A multi-domain audio classification project that builds a unified dataset from five public sources and benchmarks two deep learning models: **EfficientNet-B0** (spectrogram-based CNN) and **Audio Spectrogram Transformer (AST)**. The system classifies 40 sound classes spanning domestic environments, industrial machine anomaly detection, and outdoor audio.

> **Course:** CECS 551 — Term Project Phase 4  
> **Environment:** Kaggle / Google Colab · NVIDIA Tesla T4 GPU · Python 3.12

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Dataset Access](#dataset-access)
3. [Installation & Dependencies](#installation--dependencies)
4. [Reproducing the Results](#reproducing-the-results)
5. [Sound Classes](#sound-classes-40-total)
6. [Pipeline Details](#pipeline-details)
7. [Models & Results](#models--results)
8. [Expected Outputs](#expected-outputs)
9. [License](#license)

---

## Project Structure

```
NovaAI/
├── NovaAI.ipynb                  # Step 1 — Dataset pipeline (build unified dataset)
├── preliminaryeda_novaai.ipynb   # Step 2 — Exploratory data analysis
├── efficientnetb0-tvte.ipynb     # Step 3A — EfficientNet-B0 training & evaluation
├── ast_audio.ipynb               # Step 3B — AST fine-tuning & evaluation
└── README.md
```

---

## Dataset Access

The unified dataset is built from five publicly available sources. Download each and place it under a `raw_data/` directory as shown below.

| Dataset | Domain | Download Link | Local Folder Name |
|---|---|---|---|
| DESED | Domestic (SED) | https://desed-task.github.io | `raw_data/DESED` |
| WildDESED | Outdoor (SED) | https://github.com/DCASE-Community/WildDESED | `raw_data/WildDESED` |
| SINS | Activity detection | https://zenodo.org/record/1247102 | `raw_data/SINS` |
| MIMII | Industrial anomaly | https://zenodo.org/record/3384388 | `raw_data/MIMII` |
| DataSEC / DataSED | Domestic/outdoor | https://zenodo.org/record/4568821 | `raw_data/DataSEC_DataSED` |

> **Note:** All datasets are free to download. DESED requires agreeing to a research license. MIMII and DataSEC are released under Creative Commons (CC BY 4.0 / CC0). Total uncompressed size is approximately **20–40 GB** depending on subset selection.

### Pre-built Dataset (Kaggle)

If you are running on Kaggle, the unified dataset is also available as a Kaggle dataset:

```
asaavitupsounder/unified-domestic-audio-dataset
```

Add it as a data source in your Kaggle notebook to skip the pipeline step entirely.

---

## Installation & Dependencies

### Option A — Kaggle / Colab (recommended)

All notebooks install their own dependencies via `pip` at the top of each cell. No additional setup is needed — just open the notebook and run all cells.

### Option B — Local environment

```bash
pip install torch torchaudio torchvision
pip install transformers>=5.5.0
pip install librosa soundfile
pip install numpy pandas scikit-learn
pip install tqdm matplotlib seaborn
```

**Python version:** 3.10 or 3.12  
**GPU:** A CUDA-capable GPU is strongly recommended. The models were trained on an NVIDIA Tesla T4 (~2.3 hours total for EfficientNet-B0 at 15 epochs).

---

## Reproducing the Results

Run the notebooks **in order**:

### Step 1 — Build the unified dataset

Open and run all cells in `NovaAI.ipynb`.

This will:
- Ingest raw audio from the five source directories under `raw_data/`
- Standardize all clips to 16 kHz mono WAV, 10-second segments
- Deduplicate (exact SHA-256 hash + near-duplicate cosine similarity)
- Map all labels to the 40-class ontology
- Produce a stratified 70/15/15 train/val/test split

**Output:** `unified_dataset/` directory containing:
```
unified_dataset/
├── audio/
│   ├── train/   # ~30,338 clips
│   ├── val/     # ~6,488 clips
│   └── test/    # ~6,418 clips
└── metadata/
    ├── train.csv
    ├── val.csv
    ├── test.csv
    ├── master.csv
    ├── class_map.json
    └── stats.json
```

> **Shortcut:** If using the Kaggle pre-built dataset, skip this step entirely.

---

### Step 2 — Exploratory Data Analysis (optional)

Open and run all cells in `preliminaryeda_novaai.ipynb`.

Produces class distribution plots, split balance charts, and audio statistics. No outputs are required for training — this step is for inspection only.

---

### Step 3A — Train EfficientNet-B0

Open and run all cells in `efficientnetb0-tvte.ipynb`.

- Converts audio clips to 128-band log-mel spectrograms on the fly
- Fine-tunes EfficientNet-B0 for 40-class classification
- Saves the best checkpoint to `efficientnet_b0_best.pt`
- Runs evaluation on the test set and prints a full classification report

**Expected training time:** ~34 minutes on a T4 GPU (15 epochs).

**Expected output:**
```
Best val acc: 98.23%
Test accuracy (top-1): 98.11%
```

---

### Step 3B — Fine-tune AST

Open and run all cells in `ast_audio.ipynb`.

- Loads `MIT/ast-finetuned-audioset-10-10-0.4593` from Hugging Face
- Fine-tunes the transformer head on the 40-class dataset
- Saves the final model to `ast_finetuned_40class.pth`
- Prints per-class accuracy including anomaly vs. normal breakdown

**Expected output (anomaly classes):**
```
machine_fan_anomalous:        0.9487
machine_pump_anomalous:       0.8462
machine_slide_rail_anomalous: 1.0000
machine_valve_anomalous:      1.0000
```

---

## Sound Classes (40 total)

Classes are organized into three domains:

**Domestic** — alarm_bell_ringing, blender, cat, dog, dishes, electric_shaver_toothbrush, frying, running_water, speech, vacuum_cleaner, music_indoor, activity_cooking, activity_watching_tv, activity_eating, activity_absence, activity_other, ...

**Anomaly (Industrial Machines)** — machine_fan_normal, machine_fan_anomalous, machine_pump_normal, machine_pump_anomalous, machine_slide_rail_normal, machine_slide_rail_anomalous, machine_valve_normal, machine_valve_anomalous

**Outdoor** — birds, bells, cicadas_crickets, jet_aircraft, lawn_mower_brush_cutter, siren_alarm_outdoor, thunder_fireworks_gunshot, wind_turbine, voices_outdoor, crows_seagulls_magpies, glass_break_outdoor, horn, train, ...

Full class list and label mappings are defined in `configs/ontology.py` (written by `NovaAI.ipynb`).

---

## Pipeline Details

`NovaAI.ipynb` builds the dataset through five stages:

1. **Standardization** — resample to 16 kHz, trim silence (`top_db=30`), peak-normalize, pad short clips or segment long clips with a 5-second hop
2. **Quality filtering** — reject clips shorter than 0.5 s or with >90% silence ratio
3. **Deduplication** — exact dedup via SHA-256 waveform hash, then near-dedup using 128-band log-mel embeddings at cosine similarity ≥ 0.95
4. **Label mapping** — ontology in `configs/ontology.py` maps all heterogeneous source labels to 40 canonical unified classes
5. **Stratified splitting** — 70/15/15 train/val/test split, stratified jointly by `unified_label` and `source_dataset` (seed = 42)

**Mel spectrogram settings (for model training):**

| Parameter | Value |
|---|---|
| n_mels | 128 |
| n_fft | 1024 |
| hop_length | 512 |
| f_min | 20 Hz |
| f_max | 8,000 Hz |

---

## Models & Results

### EfficientNet-B0

| Metric | Value |
|---|---|
| Best Validation Accuracy | **98.23%** |
| Test Accuracy (top-1) | **98.11%** |
| Training Time | ~34 min (15 epochs, T4) |

Selected per-class test results:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| machine_slide_rail_anomalous | 1.000 | 1.000 | 1.000 |
| machine_valve_anomalous | 1.000 | 1.000 | 1.000 |
| machine_fan_anomalous | 1.000 | 0.983 | 0.991 |
| machine_pump_anomalous | 0.882 | 0.923 | 0.902 |
| activity_watching_tv | 1.000 | 1.000 | 1.000 |
| activity_absence | 0.983 | 0.984 | 0.983 |

### Audio Spectrogram Transformer (AST)

Fine-tuned from `MIT/ast-finetuned-audioset-10-10-0.4593`.

| Class | Test Accuracy |
|---|---|
| machine_slide_rail_anomalous | 1.0000 |
| machine_valve_anomalous | 1.0000 |
| machine_slide_rail_normal | 1.0000 |
| machine_valve_normal | 1.0000 |
| machine_fan_normal | 0.9935 |
| machine_fan_anomalous | 0.9487 |
| machine_pump_anomalous | 0.8462 |

---

## Expected Outputs

After running all notebooks, the following files should exist:

| File | Produced by |
|---|---|
| `unified_dataset/` | `NovaAI.ipynb` |
| `efficientnet_b0_best.pt` | `efficientnetb0-tvte.ipynb` |
| `ast_finetuned_40class.pth` | `ast_audio.ipynb` |

---

## License

Source datasets carry their own licenses (CC BY 4.0, CC BY 3.0, CC0 where applicable). See each dataset's Zenodo page for details. Code in this repository is released for academic use.

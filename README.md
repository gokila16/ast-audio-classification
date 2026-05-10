# NovaAI — Unified Domestic Audio Classification

A multi-domain audio classification project that builds a unified dataset from five public sources and benchmarks two deep learning models: **EfficientNet-B0** (spectrogram-based CNN) and **Audio Spectrogram Transformer (AST)**. The system classifies 40 sound classes spanning domestic environments, industrial machine anomaly detection, and outdoor audio.

---

## Project Structure

| Notebook | Description |
|---|---|
| `NovaAI.ipynb` | Dataset pipeline — ingestion, standardization, deduplication, and stratified splitting |
| `preliminaryeda_novaai.ipynb` | Exploratory data analysis — class distributions, split balance, and audio statistics |
| `efficientnetb0-tvte.ipynb` | EfficientNet-B0 fine-tuning on mel spectrograms |
| `ast_audio.ipynb` | Audio Spectrogram Transformer (AST) fine-tuning |

---

## Dataset

The unified dataset aggregates audio from five public sources:

- **DESED** — Domestic Environment Sound Event Detection
- **WildDESED** — Outdoor extension of DESED
- **SINS** — Sensor-Independent Noise Suppression (activity detection)
- **MIMII** — Malfunctioning Industrial Machine Investigation & Inspection
- **DataSEC/DataSED** — Supplementary domestic/outdoor sounds

### Audio Settings

| Parameter | Value |
|---|---|
| Sample Rate | 16,000 Hz |
| Clip Duration | 10 seconds |
| Hop Duration | 5 seconds (for long clips) |
| Channels | Mono |
| Format | WAV / PCM-16 |

### Split Sizes

| Split | Size |
|---|---|
| Train | ~30,338 clips |
| Validation | ~6,488 clips |
| Test | ~6,418 clips |

---

## Sound Classes (40 total)

Classes are organized into three domains:

**Domestic** — alarm_bell_ringing, blender, cat, dog, dishes, electric_shaver_toothbrush, frying, running_water, speech, vacuum_cleaner, music_indoor, and activity labels (cooking, watching_tv, eating, absence, other, ...)

**Anomaly (Industrial Machines)** — machine_fan_normal/anomalous, machine_pump_normal/anomalous, machine_slide_rail_normal/anomalous, machine_valve_normal/anomalous

**Outdoor** — birds, bells, cicadas_crickets, jet_aircraft, lawn_mower_brush_cutter, siren_alarm_outdoor, thunder_fireworks_gunshot, wind_turbine, voices_outdoor, ...

---

## Pipeline (`NovaAI.ipynb`)

The data pipeline handles the full path from raw source files to a clean, split-ready dataset:

1. **Standardization** — resample to 16 kHz, trim silence, normalize amplitude, pad or segment to 10-second clips
2. **Quality filtering** — remove clips shorter than 0.5s or with >90% silence
3. **Deduplication** — exact dedup via SHA-256 waveform hashes, then near-dedup using mel-spectrogram embeddings (cosine similarity threshold: 0.95)
4. **Label mapping** — a unified ontology maps heterogeneous source labels to 40 canonical classes
5. **Stratified splitting** — 70/15/15 train/val/test split, stratified by label and source dataset

---

## Models & Results

### EfficientNet-B0 (mel spectrogram CNN)

Trained on 128-band log-mel spectrograms. Best validation accuracy reached after 15 epochs.

| Metric | Value |
|---|---|
| Best Val Accuracy | **98.23%** |
| Test Accuracy | **98.11%** |

Selected per-class highlights (test set):

| Class | F1 |
|---|---|
| machine_slide_rail_anomalous | 1.000 |
| machine_valve_anomalous | 1.000 |
| activity_watching_tv | 1.000 |
| machine_fan_anomalous | 0.991 |
| machine_pump_anomalous | 0.902 |

### Audio Spectrogram Transformer (AST)

Fine-tuned from `MIT/ast-finetuned-audioset-10-10-0.4593` on the 40-class dataset.

Selected per-class highlights (test set):

| Class | Accuracy |
|---|---|
| machine_slide_rail_anomalous | 1.0000 |
| machine_valve_anomalous | 1.0000 |
| machine_fan_anomalous | 0.9487 |
| machine_pump_anomalous | 0.8462 |

---

## Requirements

```
torch
torchaudio
transformers>=5.5.0
librosa
soundfile
numpy
pandas
scikit-learn
tqdm
```

All notebooks were run on **Kaggle / Google Colab** with a **NVIDIA Tesla T4 GPU**.

---

## Getting Started

1. Download the source datasets (DESED, WildDESED, SINS, MIMII, DataSEC) and place them under `raw_data/` following the directory names in `settings.py`.
2. Run `NovaAI.ipynb` to build the unified dataset.
3. (Optional) Run `preliminaryeda_novaai.ipynb` to inspect class distributions.
4. Run `efficientnetb0-tvte.ipynb` or `ast_audio.ipynb` to train and evaluate a model.

---

## License

Source datasets carry their own licenses (CC BY 4.0, CC BY 3.0, CC0 where applicable). See individual dataset documentation for details.

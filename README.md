# ECGBERT Reproduction Project

This project implements ECGBERT, a BERT-based model for ECG signal analysis and classification. The project includes data preprocessing, pretraining, and fine-tuning pipelines for ECG signal processing.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Key Components](#key-components)
- [MIT-BIH Database Reader](#mit-bih-database-reader)
- [Installation](#installation)
- [Usage](#usage)
- [Data Format](#data-format)
- [File Descriptions](#file-descriptions)

## 🎯 Overview

ECGBERT is a transformer-based model that learns representations from ECG signals through self-supervised pretraining. The project includes:

- **Data Preprocessing**: Noise removal, baseline wander correction, and signal cleaning
- **Pretraining**: Self-supervised learning on ECG signals using masked language modeling
- **Fine-tuning**: Task-specific fine-tuning for ECG classification tasks
- **Utilities**: Tools for reading and visualizing MIT-BIH database files

## 📁 Project Structure

```
ECGBERT-reproduce-project-/
├── mitdb/                          # MIT-BIH Arrhythmia Database files
│   ├── 100.dat, 100.hea, 100.atr   # ECG record files
│   └── ...                         # Additional records
├── pre_train/                      # Pretraining pipeline
│   ├── ECGBERT_pretrain_main.py    # Main pretraining script
│   ├── ECGBERT_pretrain_engine.py  # Training engine
│   ├── ECGgetdata.py               # Data loading utilities
│   ├── ECGpreprocessing.py         # Signal preprocessing
│   ├── ECGsigpreprocessing.py      # Signal-level preprocessing
│   ├── ECGsegmentation.py          # ECG wave segmentation
│   ├── ECGClustering.py            # Waveform clustering
│   ├── ECGSentenceGenerator.py     # Sentence generation for BERT
│   ├── ECGDataset.py               # Dataset classes
│   ├── pre_train.yaml              # Configuration file
│   └── utils/
│       └── misc.py                 # Utility functions
├── fine_tune/                      # Fine-tuning pipeline
│   ├── Fine_tune_main.py           # Main fine-tuning script
│   ├── Fine_tune_engine.py         # Fine-tuning engine
│   ├── ECG_Preprocessing.py        # Preprocessing for fine-tuning
│   ├── ECG_Segmentation.py         # Segmentation
│   ├── ECG_Beat_Sentence.py        # Beat-level sentence generation
│   └── models.py                   # Model definitions
├── preprocessed/                   # Preprocessed data storage
│   ├── ecg_raw_data.hdf5          # Raw ECG data
│   ├── 0.2_ECGBERT_dataset.hdf5    # Processed dataset
│   └── clustering_models/         # Clustering models
├── read_mitdb_files.py            # MIT-BIH file reader utility
├── custom_vocab.txt               # Custom vocabulary
└── sf1.0_bs32_lr0.0005_ep500_ecgbert_model.pth  # Pretrained model
```

## 🔧 Key Components

### 1. Preprocessing Pipeline (`pre_train/ECGpreprocessing.py`)

The preprocessing pipeline cleans and filters ECG signals:

- **Missing Value Handler**: Interpolates missing/NaN values using linear interpolation
- **Bandstop Filter**: Removes 50-60 Hz power line interference using Butterworth filter
- **Baseline Wander Removal**: Removes low-frequency drift using wavelet decomposition (Daubechies db4)
- **Signal Processor**: Orchestrates the entire preprocessing pipeline

**Processing Flow:**
```
Raw ECG Signal → Missing Value Handling → Bandstop Filtering → Baseline Removal → Clean Signal
```

### 2. Pretraining Pipeline

The pretraining pipeline includes:

- **Data Loading**: Reads ECG data from HDF5 files
- **Segmentation**: Identifies P, QRS, and T waves in ECG signals
- **Clustering**: Groups similar waveforms using K-means clustering
- **Sentence Generation**: Creates tokenized sentences for BERT training
- **Masked Language Modeling**: Self-supervised pretraining on ECG signals

### 3. Fine-tuning Pipeline

Task-specific fine-tuning for downstream ECG classification tasks, including:
- **Heartbeat Classification**: Binary (Normal/Abnormal) or multi-class beat classification
- **AFIB Detection**: Atrial fibrillation detection
- Custom downstream tasks can be added by implementing the preprocessing pipeline

See `fine_tune/FINETUNING_GUIDE.md` for detailed instructions.

## 📊 MIT-BIH Database Reader

The `read_mitdb_files.py` script provides a utility to read and display MIT-BIH Arrhythmia Database files in a human-readable format.

### Features

- **Header Information**: Displays record metadata, signal details, and patient information
- **Signal Data**: Shows sample ECG values and statistics
- **Annotations**: Displays beat annotations with types and timestamps

### Usage

```bash
python read_mitdb_files.py
```

The script reads record `100` from the `mitdb/` directory by default. To read a different record, modify the `record_name` variable in the `main()` function.

### Output

The script displays:
- **Header Information**: Record name, sampling frequency, signal channels, patient info
- **Signal Samples**: First N samples from each channel with values in mV
- **Signal Statistics**: Min, max, mean, and standard deviation for each channel
- **Annotations**: Beat type distribution and detailed annotation list with timestamps

### Example Output

```
================================================================================
HEADER INFORMATION for record: 100
================================================================================

Record Name: 100
Number of Signals: 2
Sampling Frequency: 360 Hz
Number of Samples: 650000
Duration: 1805.56 seconds (30.09 minutes)

Signal 1: MLII
  Format: 212
  Gain: 200.0
  Units: mV
  ...

Annotation Symbols Distribution:
  'N': 2239 occurrences - Normal beat
  'A':   33 occurrences - Atrial premature beat
  'V':    1 occurrences - Premature ventricular contraction
  '+':    1 occurrences - Rhythm change
```

## 💻 Installation

### Dependencies

The project requires the following Python packages:

```bash
pip install torch torchvision
pip install numpy scipy
pip install h5py
pip install pywavelets
pip install wfdb
pip install pyyaml
```

### Required Libraries

- **PyTorch**: Deep learning framework
- **NumPy/SciPy**: Numerical computing and signal processing
- **h5py**: HDF5 file handling
- **PyWavelets**: Wavelet transforms for baseline removal
- **wfdb**: MIT-BIH database file reading
- **PyYAML**: Configuration file parsing

## 🚀 Usage

### 1. Reading MIT-BIH Files

```bash
python read_mitdb_files.py
```

### 2. Pretraining

```bash
cd pre_train
python ECGBERT_pretrain_main.py --config pre_train.yaml
```

### 3. Fine-tuning

```bash
cd fine_tune
python Fine_tune_main.py [arguments]
```

## 📝 Data Format

### Input Format

- **MIT-BIH Database**: Standard `.dat`, `.hea`, and `.atr` files
- **HDF5 Files**: Processed ECG data stored in HDF5 format with groups for each record

### HDF5 Structure

```
ecg_data.hdf5
├── record_100/
│   ├── signal: (channels, samples) array
│   ├── fs: sampling frequency (attribute)
│   ├── seq_len: sequence length (attribute)
│   └── Source: source file name (attribute)
└── record_101/
    └── ...
```

## 📄 File Descriptions

### Pretraining Files

- **`ECGBERT_pretrain_main.py`**: Main entry point for pretraining
- **`ECGBERT_pretrain_engine.py`**: Training loop and optimization
- **`ECGgetdata.py`**: Loads ECG data from files into HDF5 format
- **`ECGpreprocessing.py`**: Main preprocessing pipeline (noise removal, baseline correction)
- **`ECGsigpreprocessing.py`**: Signal-level preprocessing utilities
- **`ECGsegmentation.py`**: Identifies P, QRS, T waves in ECG signals
- **`ECGClustering.py`**: Clusters similar ECG waveforms
- **`ECGSentenceGenerator.py`**: Generates tokenized sentences for BERT
- **`ECGDataset.py`**: PyTorch dataset classes for ECG data

### Fine-tuning Files

- **`Fine_tune_main.py`**: Main entry point for fine-tuning
- **`Fine_tune_engine.py`**: Fine-tuning training loop
- **`ECG_Preprocessing.py`**: Preprocessing for fine-tuning tasks
- **`ECG_Segmentation.py`**: Beat-level segmentation
- **`ECG_Beat_Sentence.py`**: Beat-level sentence generation
- **`models.py`**: Model architecture definitions

### Utility Files

- **`read_mitdb_files.py`**: MIT-BIH database file reader and visualizer
- **`custom_vocab.txt`**: Custom vocabulary for tokenization

## 🔍 Beat Type Annotations

The MIT-BIH database uses the following beat type annotations:

- **N**: Normal beat
- **L**: Left bundle branch block beat
- **R**: Right bundle branch block beat
- **A**: Atrial premature beat
- **a**: Aberrated atrial premature beat
- **J**: Nodal (junctional) premature beat
- **S**: Supraventricular premature beat
- **V**: Premature ventricular contraction
- **E**: Ventricular escape beat
- **F**: Fusion of ventricular and normal beat
- **Q**: Unclassifiable beat
- **/**: Paced beat
- **f**: Fusion of paced and normal beat
- **+**: Rhythm change
- **~**: Signal quality change

## 🚀 Running on Kaggle

This project is optimized for running on Kaggle with GPU support. See `KAGGLE_SETUP.md` for detailed instructions.

**Quick Start on Kaggle:**
1. Upload your data (MIT-BIH database, pretrained model, clustering models) as Kaggle datasets
2. Create a new GPU-enabled notebook
3. Clone this repository or upload as dataset
4. Install dependencies: `!pip install -q wfdb neurokit2 pywavelets`
5. Run: `python fine_tune/Fine_tune_heartbeat_main_kaggle.py`

See `KAGGLE_NOTEBOOK_TEMPLATE.ipynb` for a ready-to-use notebook template.

## 📚 References

- MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/1.0.0/
- WFDB Python Package: https://github.com/MIT-LCP/wfdb-python

## 📝 Notes

- The preprocessing pipeline is designed to handle various ECG signal qualities and artifacts
- All preprocessing steps include error handling to prevent data corruption
- The project uses HDF5 format for efficient storage and loading of large ECG datasets
- Configuration files (YAML) allow easy parameter tuning without code changes

## 🤝 Contributing

This is a reproduction project for research purposes. For issues or improvements, please refer to the original ECGBERT paper and implementation.

---

**Last Updated**: 2024


# Fine-tuning Guide: Heartbeat Classification

This guide explains how to fine-tune the pretrained ECGBERT model for heartbeat classification using the MIT-BIH Arrhythmia Database.

## Overview

The fine-tuning process consists of 4 main steps:
1. **Preprocessing**: Extract ECG signals and beat-level labels from MIT-BIH annotations
2. **Segmentation**: Identify P, QRS, and T waves in ECG signals
3. **Sentence Generation**: Create tokenized sentences using clustering models from pretraining
4. **Fine-tuning**: Train the model for heartbeat classification

## Prerequisites

1. **Pretrained Model**: You should have completed pretraining and have:
   - `sf1.0_bs32_lr0.0005_ep500_ecgbert_model.pth` (embedding model)
   - Clustering models in `preprocessed/clustering_models/0.1/`

2. **MIT-BIH Database**: ECG records in `mitdb/` directory with `.dat`, `.hea`, and `.atr` files

3. **Dependencies**: All required packages installed (wfdb, torch, scipy, sklearn, etc.)

## Step-by-Step Instructions

### Option 1: Run Complete Pipeline (Recommended)

Simply run the main script which executes all steps:

```bash
cd fine_tune
python Fine_tune_heartbeat_main.py
```

This will:
- Process all MIT-BIH records
- Extract beat-level labels
- Segment waveforms
- Generate sentences
- Fine-tune the model

### Option 2: Run Steps Individually

If you want to run steps separately or debug:

```python
from ECG_Heartbeat_Classification import ECG_Heartbeat_Preprocessing
from ECG_Segmentation import ECG_Segmentation
from ECG_Beat_Sentence import ECG_Beat_Sentence
from Fine_tune_engine import Fine_tune_engine

# Step 1: Preprocessing
dataset_paths = [['Heartbeat_Classification', 'mitdb', '.dat']]
ECG_Heartbeat_Preprocessing(dataset_paths, 'fine_tune_output', binary_classification=True)

# Step 2: Segmentation
ECG_Segmentation(['Heartbeat_Classification'], 'fine_tune_output')

# Step 3: Sentence Generation
cluster_dir = 'preprocessed/clustering_models/0.1'
ECG_Beat_Sentence(['Heartbeat_Classification'], 'fine_tune_output', cluster_dir)

# Step 4: Fine-tuning
pre_train_model_dir = '.'  # Directory containing pretrained model
Fine_tune_engine(['Heartbeat_Classification'], pre_train_model_dir, 'fine_tune_output')
```

## Configuration

### Classification Type

The script supports two classification modes based on AAMI standard:

1. **Binary Classification**:
   - Category N (Normal): N, L, R, e, j, +, ~ → Label 0
   - All other categories (S, V, F, Q) → Label 1

2. **5-Class AAMI Classification** (default):
   - **Category N (Normal)**: N, L, R, e, j, +, ~ → Label 0
   - **Category S (Supraventricular)**: A, a, J, S → Label 1
   - **Category V (Ventricular)**: V, E → Label 2
   - **Category F (Fusion)**: F → Label 3
   - **Category Q (Unclassifiable/Paced)**: /, f, Q → Label 4

The default is 5-class AAMI standard classification. To use binary, set `binary_classification=True` in the preprocessing call.

### Model Configuration

Edit `Fine_tune_engine.py` to adjust:
- `batch_size`: Batch size for training (default: 1)
- `lr`: Learning rate (default: 0.001)
- `epochs`: Number of training epochs (default: 13)
- `extra_layers`: Classification head architecture

## Data Structure

After preprocessing, the following structure will be created:

```
fine_tune_output/
└── Heartbeat_Classification/
    ├── ECG_Preprocessing/
    │   ├── Heartbeat_Classification_train_processed_signals.pkl
    │   ├── Heartbeat_Classification_val_processed_signals.pkl
    │   ├── Heartbeat_Classification_train_labels.pkl
    │   └── Heartbeat_Classification_val_labels.pkl
    ├── ECG_Segmentation/
    │   ├── Heartbeat_Classification_train_*_segments.pkl
    │   └── Heartbeat_Classification_val_*_segments.pkl
    ├── ECG_Sentence/
    │   ├── train/
    │   │   └── sentence_*.pkl
    │   └── val/
    │       └── sentence_*.pkl
    └── results/
        └── fine_tune_model_*.pth
```

## Beat Type Mapping

The preprocessing maps MIT-BIH annotation symbols to labels:

| Symbol | Description | Binary Label | 5-Class AAMI Label |
|--------|------------|--------------|-------------------|
| N | Normal beat | 0 | 0 (Category N) |
| L | Left bundle branch block | 0 | 0 (Category N) |
| R | Right bundle branch block | 0 | 0 (Category N) |
| e | Atrial escape | 0 | 0 (Category N) |
| j | Nodal escape | 0 | 0 (Category N) |
| + | Rhythm change | 0 | 0 (Category N) |
| ~ | Signal quality change | 0 | 0 (Category N) |
| A | Atrial premature beat | 1 | 1 (Category S) |
| a | Aberrated atrial premature | 1 | 1 (Category S) |
| J | Nodal premature beat | 1 | 1 (Category S) |
| S | Supraventricular premature | 1 | 1 (Category S) |
| V | Premature ventricular contraction | 1 | 2 (Category V) |
| E | Ventricular escape beat | 1 | 2 (Category V) |
| F | Fusion of ventricular and normal | 1 | 3 (Category F) |
| / | Paced beat | 1 | 4 (Category Q) |
| f | Fusion of paced and normal | 1 | 4 (Category Q) |
| Q | Unclassifiable beat | 1 | 4 (Category Q) |

## Troubleshooting

### Issue: Pretrained model not found

**Solution**: Ensure the pretrained model file exists. The script looks for:
- `sf1.0_bs32_lr0.0005_ep500_ecgbert_model.pth` in the project root

If your model has a different name or location, update `pre_train_model_dir` in the script.

### Issue: Clustering models not found

**Solution**: Ensure clustering models exist in:
- `preprocessed/clustering_models/0.1/P_cluster.pkl`
- `preprocessed/clustering_models/0.1/QRS_cluster.pkl`
- `preprocessed/clustering_models/0.1/T_cluster.pkl`
- `preprocessed/clustering_models/0.1/BG_cluster.pkl`

These should have been created during pretraining.

### Issue: Out of memory errors

**Solution**: 
- Reduce `batch_size` in `Fine_tune_engine.py`
- Process fewer records at a time
- Use a GPU with more memory

### Issue: No annotations found

**Solution**: Ensure MIT-BIH `.atr` files are present alongside `.dat` files. The script requires annotation files to extract beat labels.

## Model Evaluation

After fine-tuning, the model will be evaluated on the validation set. Metrics include:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **Specificity**: True negatives / (True negatives + False positives)

Results are logged during validation and saved with the model.

## Next Steps

After successful fine-tuning:
1. Evaluate the model on test data
2. Adjust hyperparameters if needed
3. Use the fine-tuned model for inference on new ECG data

## Notes

- The preprocessing uses patient-level splitting to avoid data leakage
- Default train/validation split is 80/20
- All signals are preprocessed (noise removal, baseline correction) before segmentation
- The model uses the same vocabulary and clustering from pretraining


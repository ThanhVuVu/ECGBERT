# ECGBERT Fine-Tuning: Complete Step-by-Step Guide

## Overview
This guide explains the complete fine-tuning pipeline for ECGBERT on heartbeat classification tasks. All critical parameters and numbers are documented.

---

## **STEP 1: Preprocessing ECG Signals**

### Function: `ECG_Heartbeat_Preprocessing()`
**File**: `fine_tune/ECG_Heartbeat_Classification.py`

### Purpose
- Loads raw ECG signals from MIT-BIH database
- Applies signal preprocessing (filtering, baseline removal)
- Extracts beat-level labels
- Splits data into train/validation sets

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `test_size` | **0.2** | Validation split ratio (20% validation, 80% train) |
| `binary_classification` | `False` | Classification type: `True` = Binary (Normal/Abnormal), `False` = 5-class AAMI |
| `file_extension` | `'.dat'` | MIT-BIH file format |

### Classification Labels

**Binary Classification** (`binary_classification=True`):
- `0` = Normal
- `1` = Abnormal

**5-Class AAMI** (`binary_classification=False`):
- `0` = N (Normal)
- `1` = S (Supraventricular)
- `2` = V (Ventricular)
- `3` = F (Fusion)
- `4` = Q (Unclassifiable/Paced)

### Output Structure
```
{output_dir}/{task_name}/ECG_Preprocessing/
├── {task_name}_train_processed_signals.pkl
├── {task_name}_val_processed_signals.pkl
├── {task_name}_train_labels.pkl
└── {task_name}_val_labels.pkl
```

### Important Notes
- Each signal is preprocessed with bandstop filtering (50-60 Hz) and baseline wander removal
- Labels are extracted at beat-level (same length as signal)
- Train/val split is done at patient level to avoid data leakage

---

## **STEP 2: ECG Waveform Segmentation**

### Function: `ECG_Segmentation()`
**File**: `fine_tune/ECG_Segmentation.py`

### Purpose
- Detects R-peaks in ECG signals
- Segments waveforms into P, QRS, T, and background (BG) waves
- Saves segment boundaries for each lead

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `fs` | **360 Hz** | Sampling frequency (MIT-BIH standard) |
| `max_level` | **4** | Wavelet decomposition level |
| `wavelet` | `'db4'` | Wavelet type (Daubechies 4) |

### Segmentation Timing (relative to R-peak)

| Wave Type | Start | End | Duration |
|-----------|-------|-----|----------|
| **QRS** | R-peak - 0.05s | R-peak + 0.05s | 0.10s (36 samples @ 360Hz) |
| **P** | QRS_start - 0.1s | QRS_start | 0.10s (36 samples) |
| **T** | QRS_end | QRS_end + 0.15s | 0.15s (54 samples) |
| **BG** | Remaining intervals | - | Background segments |

### Output Structure
```
{output_dir}/{task_name}/ECG_Segmentation/
├── {prefix}_{idx}_segments.pkl
├── {prefix}_{idx}_label.pkl
├── p/{prefix}_{idx}_p_segments.pkl
├── qrs/{prefix}_{idx}_qrs_segments.pkl
├── t/{prefix}_{idx}_t_segments.pkl
└── bg/{prefix}_{idx}_bg_segments.pkl
```

### Important Notes
- Uses NeuroKit2 for R-peak detection (`hamilton2002` method)
- Each ECG lead is processed separately
- Segments are clipped to signal boundaries

---

## **STEP 3: Sentence Generation (Tokenization)**

### Function: `ECG_Beat_Sentence()`
**File**: `fine_tune/ECG_Beat_Sentence.py`

### Purpose
- Assigns vocabulary tokens to each wave segment using clustering models
- Creates tokenized sentences for BERT input
- Generates sentence-signal-label triplets

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `wave_types` | `['p', 'qrs', 't', 'bg']` | Wave types to tokenize |
| `vocab_size` | **75** | Total vocabulary size (0-70 wave tokens + special tokens) |
| `cls_token` | **71** | [CLS] token ID |
| `sep_token` | **72** | [SEP] token ID |
| `mask_token` | **73** | [MASK] token ID (for pretraining) |

### Vocabulary Structure
- **Tokens 0-70**: Wave cluster assignments (P/QRS/T/BG clusters)
- **Token 71**: [CLS] - Classification token
- **Token 72**: [SEP] - Separator token
- **Token 73**: [MASK] - Masking token (pretraining only)
- **Token 74**: Padding token

### Clustering Models Required
```
{clustering_dir}/
├── P_cluster.pkl      # P-wave clusters
├── QRS_cluster.pkl    # QRS-wave clusters
├── T_cluster.pkl      # T-wave clusters
└── BG_cluster.pkl     # Background clusters
```

### Sentence Format
```
Sentence: [71] + [wave_tokens] + [72]
Signal:   Preprocessed ECG signal (same length as sentence)
Label:    Beat-level labels (same length as sentence)
```

### Output Structure
```
{output_dir}/{task_name}/ECG_Sentence/
├── train/
│   ├── sentence_0.pkl
│   ├── sentence_1.pkl
│   └── ...
└── val/
    ├── sentence_0.pkl
    ├── sentence_1.pkl
    └── ...
```

### Data Format in `.pkl` files
Each `sentence_{idx}.pkl` contains:
```python
[
    sentence,      # List of token IDs: [71, token1, token2, ..., 72]
    signal,        # NumPy array: ECG signal values
    labels         # NumPy array: Label values (same length)
]
```

### Important Notes
- Uses parallel processing (`Parallel(n_jobs=-1)`) for speed
- Each lead generates a separate sentence
- Sentence length = number of wave segments + 2 (CLS + SEP)

---

## **STEP 4: Fine-Tuning**

### Function: `Fine_tune_engine()`
**File**: `fine_tune/Fine_tune_engine.py`

### Purpose
- Loads pretrained ECGBERT models
- Fine-tunes on downstream task (heartbeat classification)
- Evaluates model performance

---

### **4.1 Model Architecture Parameters**

#### Embedding Model (`ECGEmbeddingModel`)
| Parameter | Value | Description |
|-----------|-------|-------------|
| `vocab_size` | **75** | Vocabulary size (must match pretraining) |
| `embedding_dim` | **32** | Embedding dimension (must match pretraining) |
| `max_seq_len` | **3600** | Maximum sequence length for positional encoding |
| | | (360 Hz × 10 seconds = 3600 samples) |

#### BERT Model (`ECGBERTModel`)
| Parameter | Value | Description |
|-----------|-------|-------------|
| `embedding_dim` | **32** | Must match embedding model |
| `num_layers` | **4** | Number of transformer encoder layers |
| `num_heads` | **8** | Number of attention heads |
| `dim_feedforward` | **64** | Feedforward network dimension |
| `vocab_size` | **75** | Output vocabulary size |

#### UNet CNN Embedding
| Parameter | Value | Description |
|-----------|-------|-------------|
| `in_channels` | **1** | Input signal channels |
| `embed_dim` | **32** | Embedding dimension |
| Kernel sizes | `7, 5, 4, 4, 3` | Convolution kernel sizes |
| Strides | `2, 2, 2, 2, 1` | Downsampling/upsampling strides |

---

### **4.2 Training Hyperparameters**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | **1** | Batch size (small due to long sequences) |
| `learning_rate` | **0.001** | Adam optimizer learning rate |
| `epochs` | **13** | Number of training epochs |
| `num_classes` | **5** | Output classes (1 for binary, 5 for AAMI) |
| `small_batch_seq_len` | **3600** | Sequence chunk size (matches positional encoding) |

### **4.3 Training Configuration**

```python
experiments = [{
    "batch_size": 1,
    "lr": 0.001,
    "epochs": 13,
    "num_classes": 5,  # 5-class AAMI: N, S, V, F, Q
    "extra_layers": [
        {'name': 'fc1', 'module': nn.Linear(32, 75)},   # embed_size → vocab_size
        {'name': 'fc2', 'module': nn.Linear(75, 5)}     # vocab_size → num_classes
    ]
}]
```

### **4.4 Loss Functions**

- **Binary Classification** (`num_classes=1`):
  - Loss: `BCEWithLogitsLoss()`
  - Output: Sigmoid activation

- **Multi-Class** (`num_classes=5`):
  - Loss: `CrossEntropyLoss()`
  - Output: Logits (no softmax - handled by loss)

---

### **4.5 Training Process**

#### Data Loading
1. **Load batch data** (`get_batch_data()`):
   - Loads pickle files: `sentence_{idx}.pkl`
   - Pads sequences to max length in batch
   - Converts to tensors on GPU

2. **Sequence Chunking**:
   - Sequences longer than 3600 are split into chunks
   - Each chunk processed separately
   - **CRITICAL**: `small_batch_seq_len = 3600` must match `max_seq_len` in positional encoding

#### Forward Pass
```
Input: tokens [batch, seq_len], signals [batch, seq_len]
  ↓
ECGEmbeddingModel:
  - Token embedding (vocab_size=75, embed_dim=32)
  - Positional encoding (max_len=3600)
  - UNet CNN embedding (processes signal)
  - Combined: token_emb + pos_emb + wave_emb
  ↓
Output: embeddings [batch, seq_len, 32]
  ↓
ECGBERTModel:
  - 4 transformer encoder layers
  - FC layer (32 → 75)
  ↓
Output: logits [batch, seq_len, 75]
  ↓
Extra Layers:
  - FC1: 32 → 75
  - FC2: 75 → 5
  ↓
Final: logits [batch, seq_len, 5]
```

#### Backward Pass
- **Gradient accumulation**: Each chunk gets separate backward pass
- **Optimizer step**: After each chunk (can be optimized with gradient accumulation)
- **Model saving**: After every batch (can be optimized to save less frequently)

---

### **4.6 Pretrained Model Loading**

The code tries to load pretrained models in this order:

1. **Separate files** (preferred):
   - `emb_model_1_results.pth`
   - `bert_model_1_results.pth`

2. **Single file**:
   - `sf1.0_bs32_lr0.0005_ep500_ecgbert_model.pth`
   - Automatically splits weights between embedding and BERT models

3. **Any .pth file**:
   - Falls back to any `.pth` file in directory

### **4.7 Model State**

- **Embedding Model**: `eval()` mode (frozen, not trained)
- **BERT Model**: `train()` mode (fine-tuned)
- **Extra Layers**: `train()` mode (trained from scratch)

---

### **4.8 Output Files**

```
{output_dir}/{task_name}/results/
├── pre_batch_fine_tune_model.pth      # Saved after each batch
├── pre_batch_emb_model.pth             # Saved after each batch
├── fine_tune_model_{epoch}_results.pth # Final epoch model
├── emb_model_{epoch}_results.pth      # Final epoch embedding
└── pre_batch_trotal_loss.pkl          # Loss history
```

---

## **Complete Pipeline Execution**

### Main Entry Point
**File**: `fine_tune/Fine_tune_heartbeat_main_kaggle.py`

```python
run_finetuning(
    mitdb_path='/path/to/mitdb',
    pretrained_model_path='/path/to/pretrained/models',
    clustering_dir='/path/to/clustering/models/0.1',
    output_dir='/path/to/output',
    binary_classification=False,  # False = 5-class, True = binary
    task_name='Heartbeat_Classification'
)
```

### Execution Order
1. **Step 1**: `ECG_Heartbeat_Preprocessing()` - Preprocess signals
2. **Step 2**: `ECG_Segmentation()` - Segment waveforms
3. **Step 3**: `ECG_Beat_Sentence()` - Generate sentences
4. **Step 4**: `Fine_tune_engine()` - Fine-tune model

---

## **Critical Parameters Summary**

### Must Match Between Steps
| Parameter | Value | Used In |
|-----------|-------|---------|
| `vocab_size` | **75** | Pretraining, Sentence Generation, Fine-tuning |
| `embedding_dim` | **32** | Pretraining, Fine-tuning |
| `max_seq_len` | **3600** | Positional encoding, Sequence chunking |
| `fs` | **360 Hz** | Segmentation, Signal processing |

### Training Hyperparameters
| Parameter | Value | Impact |
|-----------|-------|--------|
| `batch_size` | **1** | Memory usage, training speed |
| `learning_rate` | **0.001** | Convergence speed, stability |
| `epochs` | **13** | Training duration, overfitting risk |
| `small_batch_seq_len` | **3600** | Memory usage, must match positional encoding |

### Model Architecture
| Component | Parameters | Total |
|-----------|------------|-------|
| Embedding | vocab=75, embed=32 | ~2.4K params |
| Positional | max_len=3600, embed=32 | ~115K params (buffer) |
| UNet CNN | embed=32 | ~50K params |
| BERT | 4 layers, 8 heads, embed=32 | ~100K params |
| Extra FC | 32→75→5 | ~400 params |
| **Total** | | **~150K trainable params** |

---

## **Performance Considerations**

### Current Bottlenecks
1. **Model loading/saving every batch** - Major slowdown
2. **Multiple optimizer steps per batch** - Inefficient
3. **Sequential data loading** - No parallelization
4. **Sequence chunking** - Multiple forward/backward passes

### Optimization Opportunities
- Remove model reloading between batches
- Use gradient accumulation instead of multiple optimizer steps
- Implement DataLoader with multiple workers
- Save models less frequently (e.g., every N batches or epoch end)
- Consider mixed precision training (FP16)

---

## **Important Notes**

1. **Sequence Length**: Maximum sequence length is **3600 tokens** (10 seconds @ 360 Hz). Longer sequences are chunked.

2. **Memory**: With `batch_size=1` and `seq_len=3600`, expect ~2-4 GB GPU memory usage.

3. **Training Time**: With current implementation, expect **several hours** per epoch due to:
   - Model loading/saving overhead
   - Sequence chunking
   - Inefficient data loading

4. **Model Compatibility**: Pretrained model must have:
   - Same `vocab_size` (75)
   - Same `embedding_dim` (32)
   - Compatible architecture

5. **Data Format**: All intermediate data is stored as pickle files (`.pkl`).

---

## **Troubleshooting**

### Common Issues
1. **Out of Memory**: Reduce `small_batch_seq_len` or `batch_size`
2. **Model Loading Error**: Check pretrained model paths and file names
3. **Shape Mismatch**: Verify `vocab_size`, `embedding_dim` match pretraining
4. **Slow Training**: Remove model reloading, optimize data loading

---

## **Quick Reference: All Critical Numbers**

```
Vocabulary: 75 tokens (0-70 waves + 71 CLS + 72 SEP + 73 MASK + 74 PAD)
Embedding: 32 dimensions
Sequence: 3600 max length (10 seconds @ 360 Hz)
Sampling: 360 Hz
Batch: 1
Learning Rate: 0.001
Epochs: 13
Classes: 5 (N, S, V, F, Q) or 1 (Binary)
Transformer: 4 layers, 8 heads, 64 feedforward
Train/Val Split: 80/20
```

---

**End of Guide**



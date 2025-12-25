# Quick Start: Heartbeat Classification Fine-tuning

This is a quick guide to get you started with fine-tuning ECGBERT for heartbeat classification.

## Prerequisites Checklist

- [ ] Pretrained model exists: `sf1.0_bs32_lr0.0005_ep500_ecgbert_model.pth`
- [ ] Clustering models exist in: `preprocessed/clustering_models/0.1/`
- [ ] MIT-BIH database files in: `mitdb/` directory
- [ ] All dependencies installed (wfdb, torch, scipy, sklearn, etc.)

## Quick Start (3 Steps)

### Step 1: Prepare Pretrained Models (if needed)

If your pretrained model needs to be converted to the format expected by fine-tuning:

```bash
cd fine_tune
python load_pretrained_model.py
```

This will create separate embedding and BERT model files.

### Step 2: Run Fine-tuning Pipeline

```bash
cd fine_tune
python Fine_tune_heartbeat_main.py
```

This will automatically:
1. Preprocess MIT-BIH data and extract beat labels
2. Segment ECG waveforms
3. Generate tokenized sentences
4. Fine-tune the model

### Step 3: Check Results

After completion, check:
- Model checkpoints: `fine_tune_output/Heartbeat_Classification/results/`
- Training logs in console output
- Validation metrics (accuracy, precision, recall, specificity)

## Configuration Options

### Binary vs 5-Class AAMI Classification

Edit `Fine_tune_heartbeat_main.py`:

```python
# For binary classification (Normal/Abnormal)
binary_classification = True

# For 5-class AAMI standard (N/S/V/F/Q) - DEFAULT
binary_classification = False
```

The 5-class AAMI standard includes:
- **Category N (Normal)**: Normal beats, bundle branch blocks, escapes
- **Category S (Supraventricular)**: Atrial and nodal premature beats
- **Category V (Ventricular)**: Ventricular premature contractions and escapes
- **Category F (Fusion)**: Fusion beats
- **Category Q (Unclassifiable/Paced)**: Paced beats and unclassifiable beats

### Adjust Training Parameters

Edit `fine_tune/Fine_tune_engine.py`:

```python
experiments = [
    {
        "batch_size": 1,      # Increase if you have more GPU memory
        "lr": 0.001,          # Learning rate
        "epochs": 13,         # Number of training epochs
        ...
    }
]
```

## Expected Output Structure

```
fine_tune_output/
└── Heartbeat_Classification/
    ├── ECG_Preprocessing/        # Preprocessed signals and labels
    ├── ECG_Segmentation/          # Waveform segments
    ├── ECG_Sentence/              # Tokenized sentences
    │   ├── train/
    │   └── val/
    └── results/                   # Fine-tuned models
        ├── fine_tune_model_*.pth
        └── emb_model_*.pth
```

## Troubleshooting

### "Model file not found"
- Check that `sf1.0_bs32_lr0.0005_ep500_ecgbert_model.pth` exists in project root
- Or update `pre_train_model_dir` in the script

### "Clustering models not found"
- Ensure you completed pretraining which creates clustering models
- Check `preprocessed/clustering_models/0.1/` exists

### "No annotations found"
- Ensure `.atr` files are present alongside `.dat` files in `mitdb/`
- MIT-BIH database should have all three file types: `.dat`, `.hea`, `.atr`

### Out of Memory
- Reduce `batch_size` in `Fine_tune_engine.py`
- Process fewer records at a time

## Next Steps

1. **Evaluate**: Use the fine-tuned model on test data
2. **Inference**: Apply to new ECG signals
3. **Tune**: Adjust hyperparameters based on validation results

For more details, see `FINETUNING_GUIDE.md`.


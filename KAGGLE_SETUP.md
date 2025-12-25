# Kaggle Setup Guide for ECGBERT Fine-tuning

This guide will help you set up and run the ECGBERT fine-tuning pipeline on Kaggle with GPU support.

## Prerequisites

1. **Kaggle Account**: Sign up at [kaggle.com](https://www.kaggle.com)
2. **Kaggle API**: Install Kaggle API locally (for uploading data)
3. **Git**: To clone the repository

## Step 1: Prepare Your Data

### Option A: Using Kaggle Datasets (Recommended)

1. **Upload MIT-BIH Database**:
   - Go to [Kaggle Datasets](https://www.kaggle.com/datasets)
   - Create a new dataset
   - Upload your `mitdb/` folder with all `.dat`, `.hea`, and `.atr` files
   - Make the dataset public or private (your choice)

2. **Upload Pretrained Model and Clustering Models**:
   - Create another dataset or add to the same dataset
   - Include:
     - `sf1.0_bs32_lr0.0005_ep500_ecgbert_model.pth` (pretrained model)
     - `preprocessed/clustering_models/0.1/` folder (clustering models)

### Option B: Upload via Kaggle Notebook

You can also upload files directly in a Kaggle notebook using the "Add Data" button.

## Step 2: Create Kaggle Notebook

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click "New Notebook"
3. Select **GPU** as the accelerator (P100 or T4 recommended)
4. Enable internet access if needed for additional packages

## Step 3: Clone Repository

In your Kaggle notebook, run:

```python
# Clone the repository
!git clone https://github.com/your-username/ECGBERT-reproduce-project-.git

# Navigate to the project directory
import os
os.chdir('ECGBERT-reproduce-project-')
```

Or if you've uploaded the code as a dataset:

```python
# Add the dataset containing your code
# Then copy files to working directory
import shutil
shutil.copytree('/kaggle/input/your-dataset-name', '/kaggle/working/ECGBERT-reproduce-project-')
```

## Step 4: Install Dependencies

```python
# Install required packages
!pip install -q wfdb neurokit2 pywavelets

# Verify installation
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

## Step 5: Set Up Data Paths

### If using Kaggle Datasets:

```python
import os

# Set up paths (adjust dataset names to match your uploads)
MITDB_PATH = '/kaggle/input/mit-bih-arrhythmia-database/mitdb'  # Adjust to your dataset path
PRETRAINED_MODEL_PATH = '/kaggle/input/ecgbert-pretrained-models'
CLUSTERING_MODELS_PATH = '/kaggle/input/ecgbert-clustering-models/preprocessed/clustering_models/0.1'

# Create working directories
os.makedirs('/kaggle/working/fine_tune_output', exist_ok=True)
```

### If files are in working directory:

```python
import os

# Assuming you've copied everything to /kaggle/working/
MITDB_PATH = '/kaggle/working/ECGBERT-reproduce-project-/mitdb'
PRETRAINED_MODEL_PATH = '/kaggle/working/ECGBERT-reproduce-project-'
CLUSTERING_MODELS_PATH = '/kaggle/working/ECGBERT-reproduce-project-/preprocessed/clustering_models/0.1'
```

## Step 6: Run Fine-tuning

Create a new cell and run:

```python
import sys
sys.path.append('/kaggle/working/ECGBERT-reproduce-project-')

from fine_tune.Fine_tune_heartbeat_main import *
import os

# Update paths for Kaggle
os.chdir('/kaggle/working/ECGBERT-reproduce-project-')

# Modify the main script paths if needed, or create a Kaggle-specific runner
```

Or use the Kaggle-specific runner script (see below).

## Step 7: Save Results

After training completes:

```python
# Save model outputs
import shutil

# Copy results to output directory
output_dir = '/kaggle/working/fine_tune_output/Heartbeat_Classification/results'
if os.path.exists(output_dir):
    # Results will be automatically saved in Kaggle's output
    print("Results saved to:", output_dir)
```

## Kaggle-Specific Considerations

### 1. Session Time Limits
- Free tier: 9 hours GPU time per week
- Pro tier: 30 hours GPU time per week
- Plan your training accordingly

### 2. Disk Space
- Kaggle provides ~20GB of disk space
- Clean up intermediate files if needed:
  ```python
  # Remove large intermediate files
  !rm -rf /kaggle/working/fine_tune_output/Heartbeat_Classification/ECG_Preprocessing/*.pkl
  ```

### 3. Memory Management
- Kaggle GPU notebooks have limited RAM
- If you encounter OOM errors:
  - Reduce batch size in `Fine_tune_engine.py`
  - Process fewer records at a time
  - Use gradient checkpointing

### 4. Output Persistence
- Files in `/kaggle/working/` are saved when you save the notebook
- Files in `/kaggle/input/` are read-only
- Save important models to `/kaggle/working/` or download them

## Quick Start Template

Here's a complete template for a Kaggle notebook:

```python
# Cell 1: Setup
import os
import sys

# Clone or set up repository
!git clone https://github.com/your-username/ECGBERT-reproduce-project-.git
os.chdir('ECGBERT-reproduce-project-')

# Install dependencies
!pip install -q wfdb neurokit2 pywavelets

# Verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Cell 2: Configure paths
MITDB_PATH = '/kaggle/input/mit-bih-database/mitdb'  # Adjust to your path
PRETRAINED_MODEL = '/kaggle/input/pretrained-model/sf1.0_bs32_lr0.0005_ep500_ecgbert_model.pth'
CLUSTERING_DIR = '/kaggle/input/clustering-models/preprocessed/clustering_models/0.1'

# Cell 3: Run fine-tuning
sys.path.append('/kaggle/working/ECGBERT-reproduce-project-')
from fine_tune.Fine_tune_heartbeat_main_kaggle import run_finetuning

run_finetuning(
    mitdb_path=MITDB_PATH,
    pretrained_model_path=PRETRAINED_MODEL,
    clustering_dir=CLUSTERING_DIR,
    output_dir='/kaggle/working/fine_tune_output'
)
```

## Troubleshooting

### Issue: "Module not found"
**Solution**: Ensure you've added the project to Python path:
```python
sys.path.append('/kaggle/working/ECGBERT-reproduce-project-')
```

### Issue: "CUDA out of memory"
**Solution**: 
- Reduce batch size
- Use smaller sequence lengths
- Process data in smaller chunks

### Issue: "File not found"
**Solution**: 
- Check dataset paths match your uploads
- Use absolute paths starting with `/kaggle/input/` or `/kaggle/working/`
- Verify files exist: `!ls /kaggle/input/your-dataset/`

### Issue: "Dataset not accessible"
**Solution**:
- Ensure dataset is public, or
- Add dataset to your notebook using "Add Data" button

## Tips for Kaggle

1. **Use GPU efficiently**: Enable GPU accelerator in notebook settings
2. **Save checkpoints**: Save model checkpoints regularly
3. **Monitor progress**: Use tqdm progress bars (already included)
4. **Download results**: Download trained models before session ends
5. **Version control**: Commit your notebook with results

## Next Steps

After successful fine-tuning:
1. Download the trained model from Kaggle
2. Evaluate on test data
3. Use for inference on new ECG signals

For more details, see `fine_tune/FINETUNING_GUIDE.md`.


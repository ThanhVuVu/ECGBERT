"""
Kaggle-specific fine-tuning script for Heartbeat Classification
Optimized for Kaggle GPU environment with configurable paths
"""

from ECG_Heartbeat_Classification import ECG_Heartbeat_Preprocessing
from ECG_Segmentation import ECG_Segmentation
from ECG_Beat_Sentence import ECG_Beat_Sentence
from Fine_tune_engine import Fine_tune_engine

import os
import sys
import logging
import torch

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

def run_finetuning(
    mitdb_path='/kaggle/input/mit-bih-database/mitdb',
    pretrained_model_path='/kaggle/input/pretrained-model',
    clustering_dir='/kaggle/input/clustering-models/preprocessed/clustering_models/0.1',
    output_dir='/kaggle/working/fine_tune_output',
    binary_classification=False,
    task_name='Heartbeat_Classification'
):
    """
    Run fine-tuning pipeline on Kaggle.
    
    Args:
        mitdb_path: Path to MIT-BIH database directory
        pretrained_model_path: Path to directory containing pretrained model
        clustering_dir: Path to clustering models directory
        output_dir: Output directory for fine-tuning results
        binary_classification: True for binary, False for 5-class AAMI
        task_name: Name of the downstream task
    """
    
    # Verify GPU availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available! Training will be slow on CPU.")
    else:
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Verify paths exist
    if not os.path.exists(mitdb_path):
        raise FileNotFoundError(f"MIT-BIH database not found at {mitdb_path}")
    
    if not os.path.exists(clustering_dir):
        raise FileNotFoundError(f"Clustering models not found at {clustering_dir}")
    
    # Check for pretrained model
    pretrained_model_file = os.path.join(pretrained_model_path, 'sf1.0_bs32_lr0.0005_ep500_ecgbert_model.pth')
    if not os.path.exists(pretrained_model_file):
        logger.warning(f"Pretrained model not found at {pretrained_model_file}")
        logger.warning("You may need to extract BERT and embedding models separately")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("ECGBERT Fine-tuning for Heartbeat Classification (Kaggle)")
    logger.info("=" * 80)
    logger.info(f"MIT-BIH directory: {mitdb_path}")
    logger.info(f"Pretrained models: {pretrained_model_path}")
    logger.info(f"Clustering models: {clustering_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Classification type: {'Binary (Normal/Abnormal)' if binary_classification else '5-class AAMI (N/S/V/F/Q)'}")
    logger.info("=" * 80)
    
    # Dataset paths: [task_name, dataset_path, file_extension]
    dataset_paths = [[task_name, mitdb_path, '.dat']]
    downstream_tasks = [item[0] for item in dataset_paths]
    
    try:
        # ========== Step 1: Preprocessing ==========
        logger.info("\n[Step 1/4] Preprocessing ECG signals and extracting beat labels...")
        ECG_Heartbeat_Preprocessing(dataset_paths, output_dir, binary_classification=binary_classification)
        
        # ========== Step 2: Segmentation ==========
        logger.info("\n[Step 2/4] Segmenting ECG waveforms (P, QRS, T waves)...")
        ECG_Segmentation(downstream_tasks, output_dir)
        
        # ========== Step 3: Sentence Generation ==========
        logger.info("\n[Step 3/4] Generating tokenized sentences for BERT...")
        ECG_Beat_Sentence(downstream_tasks, output_dir, clustering_dir)
        
        # ========== Step 4: Fine-tuning ==========
        logger.info("\n[Step 4/4] Fine-tuning ECGBERT model...")
        Fine_tune_engine(downstream_tasks, pretrained_model_path, output_dir)
        
        logger.info("\n" + "=" * 80)
        logger.info("Fine-tuning complete!")
        logger.info(f"Results saved to: {output_dir}/{task_name}/results/")
        logger.info("=" * 80)
        
        # List output files
        results_dir = os.path.join(output_dir, task_name, 'results')
        if os.path.exists(results_dir):
            logger.info("\nOutput files:")
            for file in os.listdir(results_dir):
                if file.endswith('.pth'):
                    file_path = os.path.join(results_dir, file)
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    logger.info(f"  - {file} ({size_mb:.2f} MB)")
        
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    # Default Kaggle paths - modify as needed
    run_finetuning(
        mitdb_path='/kaggle/input/mit-bih-database/mitdb',
        pretrained_model_path='/kaggle/input/pretrained-model',
        clustering_dir='/kaggle/input/clustering-models/preprocessed/clustering_models/0.1',
        output_dir='/kaggle/working/fine_tune_output',
        binary_classification=False
    )


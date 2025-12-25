"""
Fine-tuning script for Heartbeat Classification using pretrained ECGBERT model
"""

from ECG_Heartbeat_Classification import ECG_Heartbeat_Preprocessing
from ECG_Segmentation import ECG_Segmentation
from ECG_Beat_Sentence import ECG_Beat_Sentence
from Fine_tune_engine import Fine_tune_engine

import os
import logging

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    
    # ========== Configuration ==========
    # Base directory for fine-tuning outputs
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fine_tune_output')
    
    # MIT-BIH database directory
    mitdb_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mitdb')
    
    # Clustering models directory (from pretraining)
    cluster_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        'preprocessed', 
        'clustering_models', 
        '0.1'
    )
    
    # Pretrained model directory
    pre_train_model_dir = os.path.dirname(os.path.dirname(__file__))
    
    # Task configuration
    task_name = 'Heartbeat_Classification'
    
    # Classification type: 
    # True for binary (Normal/Abnormal), 
    # False for 5-class AAMI (N=0, S=1, V=2, F=3, Q=4)
    binary_classification = False  # Default to 5-class AAMI standard
    
    # Dataset paths: [task_name, dataset_path, file_extension]
    dataset_paths = [[task_name, mitdb_dir, '.dat']]
    
    downstream_tasks = [item[0] for item in dataset_paths]
    
    logger.info("=" * 80)
    logger.info("ECGBERT Fine-tuning for Heartbeat Classification")
    logger.info("=" * 80)
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"MIT-BIH directory: {mitdb_dir}")
    logger.info(f"Clustering models: {cluster_dir}")
    logger.info(f"Pretrained models: {pre_train_model_dir}")
    logger.info(f"Classification type: {'Binary (Normal/Abnormal)' if binary_classification else '5-class AAMI (N/S/V/F/Q)'}")
    logger.info("=" * 80)
    
    # ========== Step 1: Preprocessing ==========
    logger.info("\n[Step 1/4] Preprocessing ECG signals and extracting beat labels...")
    ECG_Heartbeat_Preprocessing(dataset_paths, base_dir, binary_classification=binary_classification)
    
    # ========== Step 2: Segmentation ==========
    logger.info("\n[Step 2/4] Segmenting ECG waveforms (P, QRS, T waves)...")
    ECG_Segmentation(downstream_tasks, base_dir)
    
    # ========== Step 3: Sentence Generation ==========
    logger.info("\n[Step 3/4] Generating tokenized sentences for BERT...")
    ECG_Beat_Sentence(downstream_tasks, base_dir, cluster_dir)
    
    # ========== Step 4: Fine-tuning ==========
    logger.info("\n[Step 4/4] Fine-tuning ECGBERT model...")
    
    # Check if pretrained models exist
    emb_model_path = os.path.join(pre_train_model_dir, 'sf1.0_bs32_lr0.0005_ep500_ecgbert_model.pth')
    
    if not os.path.exists(emb_model_path):
        logger.warning(f"Pretrained model not found at {emb_model_path}")
        logger.warning("Please ensure the pretrained model file exists.")
        logger.warning("The model file should be: sf1.0_bs32_lr0.0005_ep500_ecgbert_model.pth")
        logger.warning("This appears to be the embedding model. You may need to extract BERT model separately.")
    
    # Note: The Fine_tune_engine expects specific model file names
    # You may need to adjust the model loading in Fine_tune_engine.py
    # to match your pretrained model file structure
    
    Fine_tune_engine(downstream_tasks, pre_train_model_dir, base_dir)
    
    logger.info("\n" + "=" * 80)
    logger.info("Fine-tuning complete!")
    logger.info("=" * 80)


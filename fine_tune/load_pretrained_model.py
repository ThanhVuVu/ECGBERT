"""
Helper script to load and convert pretrained ECGBERT model for fine-tuning
"""

import torch
import os
import logging
from fine_tune.models import ECGEmbeddingModel, ECGBERTModel

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

def load_pretrained_models(model_path, vocab_size=75, embed_size=32, device='cuda'):
    """
    Load pretrained ECGBERT models from a single checkpoint file.
    
    Args:
        model_path: Path to the pretrained model file
        vocab_size: Vocabulary size (default: 75)
        embed_size: Embedding dimension (default: 32)
        device: Device to load models on
    
    Returns:
        emb_model: ECGEmbeddingModel with loaded weights
        bert_model: ECGBERTModel with loaded weights (if available)
    """
    logger.info(f"Loading pretrained model from {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the state dict
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize models
    emb_model = ECGEmbeddingModel(vocab_size, embed_size).to(device)
    bert_model = ECGBERTModel(embed_size).to(device)
    
    # Check if checkpoint is a state dict or contains multiple models
    if isinstance(checkpoint, dict):
        # Try to load embedding model
        emb_keys = [k for k in checkpoint.keys() if 'token_embedding' in k or 'cnn_embedding' in k or 'positional_embedding' in k]
        
        if emb_keys:
            # Filter state dict for embedding model
            emb_state_dict = {k: v for k, v in checkpoint.items() 
                            if 'token_embedding' in k or 'cnn_embedding' in k or 'positional_embedding' in k}
            
            # Try to load (may need key mapping)
            try:
                emb_model.load_state_dict(emb_state_dict, strict=False)
                logger.info("Loaded embedding model weights")
            except Exception as e:
                logger.warning(f"Could not load embedding model directly: {e}")
                logger.info("Attempting to load with key mapping...")
                # Try with key mapping
                mapped_emb_dict = {}
                for k, v in emb_state_dict.items():
                    # Remove any prefix if present
                    new_key = k.replace('module.', '').replace('embedding_model.', '')
                    mapped_emb_dict[new_key] = v
                emb_model.load_state_dict(mapped_emb_dict, strict=False)
                logger.info("Loaded embedding model with key mapping")
        
        # Try to load BERT model
        bert_keys = [k for k in checkpoint.keys() if 'transformer' in k or 'fc' in k or 'layers' in k]
        
        if bert_keys:
            # Filter state dict for BERT model
            bert_state_dict = {k: v for k, v in checkpoint.items() 
                             if 'transformer' in k or ('fc' in k and 'cnn' not in k) or 'layers' in k}
            
            try:
                bert_model.load_state_dict(bert_state_dict, strict=False)
                logger.info("Loaded BERT model weights")
            except Exception as e:
                logger.warning(f"Could not load BERT model directly: {e}")
                logger.info("Attempting to load with key mapping...")
                mapped_bert_dict = {}
                for k, v in bert_state_dict.items():
                    new_key = k.replace('module.', '').replace('bert_model.', '')
                    mapped_bert_dict[new_key] = v
                bert_model.load_state_dict(mapped_bert_dict, strict=False)
                logger.info("Loaded BERT model with key mapping")
    
    logger.info("Model loading complete")
    return emb_model, bert_model

def save_models_for_finetuning(emb_model, bert_model, save_dir, task_name='Heartbeat_Classification'):
    """
    Save models in the format expected by Fine_tune_engine.
    
    Args:
        emb_model: ECGEmbeddingModel
        bert_model: ECGBERTModel
        save_dir: Directory to save models
        task_name: Task name for file naming
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save in the format expected by Fine_tune_engine
    emb_path = os.path.join(save_dir, 'emb_model_1_results.pth')
    bert_path = os.path.join(save_dir, 'bert_model_1_results.pth')
    
    torch.save(emb_model.state_dict(), emb_path)
    torch.save(bert_model.state_dict(), bert_path)
    
    logger.info(f"Saved models to {save_dir}")
    logger.info(f"  - Embedding model: {emb_path}")
    logger.info(f"  - BERT model: {bert_path}")

if __name__ == '__main__':
    # Example usage
    model_path = '../sf1.0_bs32_lr0.0005_ep500_ecgbert_model.pth'
    save_dir = '../pretrained_models'
    
    emb_model, bert_model = load_pretrained_models(model_path)
    save_models_for_finetuning(emb_model, bert_model, save_dir)


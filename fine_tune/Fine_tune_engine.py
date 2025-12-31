import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from fine_tune.models import ECGEmbeddingModel, ECGBERTModel
from tqdm import tqdm
import numpy as np

def load_pkl_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pkl_data(save_dir, file_name, save_pkl):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    with open(save_path, 'wb') as f:
        pickle.dump(save_pkl, f)

def load_batch_data(data_dir, batch_indices):
    batch_data = []
    for idx in batch_indices:
        file_path = os.path.join(data_dir, f'sentence_{idx}.pkl')
        batch_data.append(load_pkl_data(file_path))
    return batch_data

def pad_sequences(sequences, max_len=None):
    if not max_len:
        max_len = max(len(seq) for seq in sequences)
    padded_seqs = [torch.cat([seq, torch.zeros(max_len - len(seq), device=seq.device)]) for seq in sequences]
    return torch.stack(padded_seqs)

def get_batch_data(batch_num, batch_size, data_dir):
    batch_indices = range(batch_num * batch_size, (batch_num + 1) * batch_size)
    batch_data = load_batch_data(data_dir, batch_indices)
    
    tokens = [torch.tensor(data[0], device='cuda') for data in batch_data]
    signals = [torch.tensor(data[1], device='cuda') for data in batch_data]
    labels = [torch.tensor(data[2], device='cuda') for data in batch_data]

    tokens = pad_sequences(tokens)
    signals = pad_sequences(signals)
    labels = pad_sequences(labels)
    
    return tokens.float(), signals.float(), labels.float()

class CombinedModel(nn.Module):
    def __init__(self, bert_model, extra_model):
        super(CombinedModel, self).__init__()
        self.bert_model = bert_model
        self.extra_model = extra_model

    def forward(self, x):
        # Forward pass through bert_model transformer layers
        # x shape: [batch_size, seq_len, embedding_dim]
        # TransformerEncoderLayer expects [seq_len, batch_size, embedding_dim]
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, embedding_dim]
        
        # Process through transformer layers (without the fc layer)
        for layer in self.bert_model.layers:
            x = layer(x)
        
        # Permute back to [batch_size, seq_len, embedding_dim]
        x = x.permute(1, 0, 2)
        
        # Pass the output through extra_model
        x = self.extra_model(x)
        return x

def get_model(save_dir, extra_layers):

    vocab_size = 75  # wave 0~70 + cls + sep + mask , total 74
    embed_size = 32

    bert_model = ECGBERTModel(embed_size).cuda()
    embedding_model = ECGEmbeddingModel(vocab_size, embed_size).cuda()

    extra_model = nn.Sequential()
    for layer in extra_layers:
        extra_model.add_module(layer['name'], layer['module'].cuda())
    
    # Use CombinedModel instead of Sequential to properly handle BERT layers
    # CombinedModel processes through BERT transformer layers (without FC) then extra_model
    # This matches the structure used in fine_tune() function
    combined_model = CombinedModel(bert_model, extra_model)
    
    fine_tune_model_path = os.path.join(save_dir, 'pre_batch_fine_tune_model.pth')
    emb_model_path = os.path.join(save_dir, 'pre_batch_emb_model.pth')

    # Load saved model weights if they exist
    if os.path.exists(fine_tune_model_path):
        try:
            state_dict = torch.load(fine_tune_model_path, map_location='cuda')
            combined_model.load_state_dict(state_dict)
        except Exception as e:
            # If loading fails, the model will use randomly initialized weights
            # This is okay for the first evaluation before training
            pass
    
    if os.path.exists(emb_model_path):
        embedding_model.load_state_dict(torch.load(emb_model_path, map_location='cuda'))
    
    return combined_model, embedding_model


def save_model(fine_tune_model, embedding_model, epoch, batch_num, total_loss, save_dir, all_num_batches):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    fine_tune_file_name = f'pre_batch_fine_tune_model.pth'
    emb_file_name = f'pre_batch_emb_model.pth'
    
    if batch_num == all_num_batches-1 :
        fine_tune_file_name = f'fine_tune_model_{epoch+1}_results.pth'
        emb_file_name = f'emb_model_{epoch+1}_results.pth'
    
    fine_tune_model_path = os.path.join(save_dir, fine_tune_file_name)
    emb_model_path = os.path.join(save_dir, emb_file_name)
    torch.save(fine_tune_model.state_dict(), fine_tune_model_path)
    torch.save(embedding_model.state_dict(), emb_model_path)
    
    save_pkl_data(save_dir, f'pre_batch_trotal_loss.pkl', total_loss)

def fine_tune(emb_model, bert_model, experiment, train_data_dir, save_dir):

    extra_model = nn.Sequential()
    for layer in experiment["extra_layers"]:
        extra_model.add_module(layer['name'], layer['module'].cuda())
    fine_tune_model = CombinedModel(bert_model, extra_model)

    fine_tune_model.train()
    emb_model.eval()
    
    # Use appropriate loss function based on number of classes
    num_classes = experiment.get('num_classes', 5)  # Default to 5 for AAMI standard
    if num_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()  # For multi-class classification
    
    optimizer = optim.Adam(list(bert_model.parameters()), lr=experiment.get('lr', 0.001))

    train_num_batches = len(os.listdir(train_data_dir)) // experiment['batch_size']

    for epoch in range(experiment["epochs"]):
        total_loss = 0
        num_samples = 0
        
        with tqdm(total=train_num_batches, desc=f"Train Epoch {epoch+1}", unit="batch") as pbar:
            for batch_num in range(train_num_batches):

                org_tokens, org_signals, org_labels = get_batch_data(batch_num, experiment["batch_size"], train_data_dir)
                
                if batch_num > 0:
                    _, emb_model = get_model(save_dir, experiment["extra_layers"])
                    fine_tune_file_name = f'pre_batch_fine_tune_model.pth'
                    fine_tune_model_path = os.path.join(save_dir, fine_tune_file_name)
                    fine_tune_model.load_state_dict(torch.load(fine_tune_model_path))
                    emb_model.eval()
                    fine_tune_model.train()

                tokens = org_tokens[:, 1:-1]
                small_batch_seq_len = 3600  # Match positional encoding max_len
                
                for i in range(0, tokens.size(1), small_batch_seq_len):
                    
                    small_tokens = tokens[:, i:i+small_batch_seq_len]
                    small_signals = org_signals[:, i:i+small_batch_seq_len]
                    small_labels = org_labels[:, i:i+small_batch_seq_len]
                    
                    optimizer.zero_grad()
                
                    embeddings = emb_model(small_tokens, small_signals)
                    outputs = fine_tune_model(embeddings)
                    
                    # Handle different output shapes for binary vs multi-class
                    num_classes = experiment.get('num_classes', 5)
                    if num_classes == 1:
                        outputs = outputs.squeeze(-1)
                        outputs = torch.sigmoid(outputs)
                        loss = criterion(outputs, small_labels)
                    else:
                        # Multi-class: outputs shape should be [batch, seq_len, num_classes]
                        # Reshape for CrossEntropyLoss: [batch*seq_len, num_classes] and [batch*seq_len]
                        batch_size, seq_len, _ = outputs.shape
                        outputs_flat = outputs.view(-1, num_classes)
                        labels_flat = small_labels.view(-1).long()
                        loss = criterion(outputs_flat, labels_flat)

                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item() * small_labels.size(0)
                    num_samples += small_labels.size(0)

                save_model(fine_tune_model, emb_model, epoch, batch_num, total_loss/num_samples, save_dir, train_num_batches)

                pbar.update(1)

                torch.cuda.empty_cache()
        avg_loss = total_loss / num_samples
        logger.info(f'Epoch {epoch+1}/{experiment["epochs"]}, Train Loss: {avg_loss:.4f}')

def evaluate(experiment, val_data_dir, save_dir):
    combined_model, emb_model = get_model(save_dir, experiment["extra_layers"])
    
    combined_model.eval()
    emb_model.eval()
    
    val_num_batches = len(os.listdir(val_data_dir)) // experiment['batch_size']
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        with tqdm(total=val_num_batches, desc="Validation Batches", unit="batch") as pbar:
            for batch_num in range(val_num_batches):
                tokens, signals, labels = get_batch_data(batch_num, experiment["batch_size"], val_data_dir)

                tokens = tokens[:, 1:-1]  # [CLS]와 [SEP] 토큰 제거
                small_batch_seq_len = 3600  # Match positional encoding max_len
                
                batch_preds = []
                batch_labels = []
                
                for i in range(0, tokens.size(1), small_batch_seq_len):
                    small_tokens = tokens[:, i:i+small_batch_seq_len]
                    small_signals = signals[:, i:i+small_batch_seq_len]
                    small_labels = labels[:, i:i+small_batch_seq_len]
                    
                    embeddings = emb_model(small_tokens, small_signals)
                    outputs = combined_model(embeddings)
                    
                    num_classes = experiment.get('num_classes', 5)
                    if num_classes == 1:
                        outputs = outputs.squeeze(-1)
                        preds = torch.sigmoid(outputs)
                    else:
                        # Multi-class: get class predictions
                        preds = torch.softmax(outputs, dim=-1)
                        # Get predicted class (for evaluation, we'll use the probability of class 1+ for binary metrics)
                        # For multi-class, we'll compute accuracy per class
                        preds = preds  # Keep full probability distribution
                    
                    batch_preds.append(preds)
                    batch_labels.append(small_labels)
                
                # Concatenate chunks for this batch along sequence dimension
                batch_preds_concat = torch.cat(batch_preds, dim=1)  # [batch_size, total_seq_len, ...]
                batch_labels_concat = torch.cat(batch_labels, dim=1)  # [batch_size, total_seq_len]
                
                # Flatten to avoid shape mismatch when concatenating different batches
                # For binary: [batch_size, seq_len] -> [batch_size * seq_len]
                # For multi-class: [batch_size, seq_len, num_classes] -> [batch_size * seq_len, num_classes]
                num_classes = experiment.get('num_classes', 5)
                if num_classes == 1:
                    batch_preds_flat = batch_preds_concat.flatten()  # [batch_size * seq_len]
                    batch_labels_flat = batch_labels_concat.flatten()  # [batch_size * seq_len]
                else:
                    batch_size, seq_len, _ = batch_preds_concat.shape
                    batch_preds_flat = batch_preds_concat.view(-1, num_classes)  # [batch_size * seq_len, num_classes]
                    batch_labels_flat = batch_labels_concat.flatten()  # [batch_size * seq_len]
                
                all_preds.append(batch_preds_flat.cpu())
                all_labels.append(batch_labels_flat.cpu())
                
                pbar.update(1)

    # Now all tensors are flat, so we can concatenate along dim=0
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    num_classes = experiment.get('num_classes', 5)
    
    if num_classes == 1:
        # Binary classification
        # all_preds and all_labels are already flattened
        predicted_labels = (all_preds > 0.5).float()
        
        accuracy = (predicted_labels == all_labels).float().mean().item()
        precision = (predicted_labels * all_labels).sum().item() / (predicted_labels.sum().item() + 1e-8)
        recall = (predicted_labels * all_labels).sum().item() / (all_labels.sum().item() + 1e-8)
        specificity = ((1 - predicted_labels) * (1 - all_labels)).sum().item() / ((1 - all_labels).sum().item() + 1e-8)
    else:
        # Multi-class classification
        # all_preds is already [total_samples, num_classes], all_labels is [total_samples]
        all_preds_flat = all_preds  # Already in correct shape
        all_labels_flat = all_labels.long()  # Just convert to long
        
        # Get predicted classes
        predicted_labels = torch.argmax(all_preds_flat, dim=1)
        
        # Overall accuracy
        accuracy = (predicted_labels == all_labels_flat).float().mean().item()
        
        # Per-class metrics (macro-averaged)
        precision_per_class = []
        recall_per_class = []
        for cls in range(num_classes):
            cls_pred = (predicted_labels == cls).float()
            cls_true = (all_labels_flat == cls).float()
            tp = (cls_pred * cls_true).sum().item()
            fp = (cls_pred * (1 - cls_true)).sum().item()
            fn = ((1 - cls_pred) * cls_true).sum().item()
            
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            precision_per_class.append(prec)
            recall_per_class.append(rec)
        
        precision = np.mean(precision_per_class)
        recall = np.mean(recall_per_class)
        
        # Calculate specificity per class (true negative rate)
        specificity_per_class = []
        for cls in range(num_classes):
            cls_pred = (predicted_labels == cls).float()
            cls_true = (all_labels_flat == cls).float()
            tn = ((1 - cls_pred) * (1 - cls_true)).sum().item()
            fp = (cls_pred * (1 - cls_true)).sum().item()
            specificity_cls = tn / (tn + fp + 1e-8)
            specificity_per_class.append(specificity_cls)
        specificity = np.mean(specificity_per_class)
    
    
    logger.info(f'Validation Accuracy: {accuracy:.4f}')
    logger.info(f'Validation Precision: {precision:.4f}')
    logger.info(f'Validation Recall: {recall:.4f}')
    logger.info(f'Validation Specificity: {specificity:.4f}')
    
    return accuracy, precision, recall, specificity

import logging

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

def load_pretrained_weights_from_single_file(model_path, vocab_size=75, embed_size=32, device='cuda'):
    """
    Load embedding and BERT weights from a single pretrained model file.
    
    Args:
        model_path: Path to the single pretrained model file
        vocab_size: Vocabulary size
        embed_size: Embedding dimension
        device: Device to load on
    
    Returns:
        emb_model: ECGEmbeddingModel with loaded weights
        bert_model: ECGBERTModel with loaded weights
    """
    logger.info(f"Loading pretrained weights from single file: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pretrained model file not found: {model_path}")
    
    # Load the full model state dict
    full_state_dict = torch.load(model_path, map_location=device)
    
    # Initialize models
    emb_model = ECGEmbeddingModel(vocab_size, embed_size).to(device)
    bert_model = ECGBERTModel(embed_size).to(device)
    
    # Split weights based on key names
    # Embedding model keys: token_embedding, positional_embedding, cnn_embedding
    # BERT model keys: layers (transformer), fc
    
    emb_state_dict = {}
    bert_state_dict = {}
    
    for key, value in full_state_dict.items():
        # Remove any prefix (like 'module.' from DataParallel)
        clean_key = key.replace('module.', '')
        
        if 'token_embedding' in clean_key or 'positional_embedding' in clean_key or 'cnn_embedding' in clean_key:
            # This belongs to embedding model
            emb_state_dict[clean_key] = value
        elif 'transformer' in clean_key:
            # This belongs to BERT model
            # Map transformer.layers.X to layers.X format
            # transformer.layers.0.self_attn... -> layers.0.self_attn...
            new_key = clean_key.replace('transformer.layers.', 'layers.')
            bert_state_dict[new_key] = value
        elif ('fc' in clean_key and 'cnn' not in clean_key):
            # FC layer belongs to BERT model
            bert_state_dict[clean_key] = value
        elif 'layers' in clean_key:
            # Direct layers reference
            bert_state_dict[clean_key] = value
    
    # Load weights into models
    try:
        emb_model.load_state_dict(emb_state_dict, strict=False)
        logger.info(f"Loaded embedding model weights ({len(emb_state_dict)} parameters)")
    except Exception as e:
        logger.warning(f"Could not load embedding model strictly: {e}")
        emb_model.load_state_dict(emb_state_dict, strict=False)
    
    try:
        bert_model.load_state_dict(bert_state_dict, strict=False)
        logger.info(f"Loaded BERT model weights ({len(bert_state_dict)} parameters)")
    except Exception as e:
        logger.warning(f"Could not load BERT model strictly: {e}")
        bert_model.load_state_dict(bert_state_dict, strict=False)
    
    return emb_model, bert_model

def Fine_tune_engine(downstream_tasks, pre_train_model_dir, dir):
    
    for idx, downstream_task in enumerate(downstream_tasks):
        
        save_dir = os.path.join(dir, f'{downstream_task}/results')
        train_data_dir = os.path.join(dir, f'{downstream_task}/ECG_Sentence/train')
        val_data_dir = os.path.join(dir, f'{downstream_task}/ECG_Sentence/val')
    
        vocab_size = 75  # wave 0~70 + cls + sep + mask , total 74s
        embed_size = 32
        
        emb_model = ECGEmbeddingModel(vocab_size, embed_size).cuda()
        bert_model = ECGBERTModel(embed_size).cuda()
        
        # Try to load from separate files first, then fall back to single file
        emb_model_path = os.path.join(pre_train_model_dir, 'emb_model_1_results.pth')
        bert_model_path = os.path.join(pre_train_model_dir, 'bert_model_1_results.pth')
        single_model_path = os.path.join(pre_train_model_dir, 'sf1.0_bs32_lr0.0005_ep500_ecgbert_model.pth')
        
        if os.path.exists(emb_model_path) and os.path.exists(bert_model_path):
            # Load from separate files
            logger.info("Loading from separate embedding and BERT model files")
            state_dict = torch.load(emb_model_path, map_location='cuda')
            emb_model.load_state_dict(state_dict)
            
            state_dict = torch.load(bert_model_path, map_location='cuda')
            bert_model.load_state_dict(state_dict)
        elif os.path.exists(single_model_path):
            # Load from single file and split weights
            logger.info("Loading from single pretrained model file and splitting weights")
            emb_model, bert_model = load_pretrained_weights_from_single_file(
                single_model_path, vocab_size, embed_size, device='cuda'
            )
        else:
            # Try to find any .pth file in the directory
            pth_files = [f for f in os.listdir(pre_train_model_dir) if f.endswith('.pth')]
            if pth_files:
                model_file = os.path.join(pre_train_model_dir, pth_files[0])
                logger.info(f"Loading from found model file: {model_file}")
                emb_model, bert_model = load_pretrained_weights_from_single_file(
                    model_file, vocab_size, embed_size, device='cuda'
                )
            else:
                raise FileNotFoundError(
                    f"No pretrained model files found in {pre_train_model_dir}. "
                    f"Expected either:\n"
                    f"  - emb_model_1_results.pth and bert_model_1_results.pth, OR\n"
                    f"  - sf1.0_bs32_lr0.0005_ep500_ecgbert_model.pth (or any .pth file)"
                )
        
        experiments = [
            {
                "batch_size": 1,
                "lr": 0.001,
                "epochs": 13,
                "num_classes": 5,  # 5-class AAMI standard: N, S, V, F, Q
                "extra_layers": [
                    {'name': 'fc1', 'module': nn.Linear(embed_size, vocab_size)},
                    {'name': 'fc2', 'module': nn.Linear(vocab_size, 5)}  # 5 classes for AAMI standard
                ]
            }
        ]
        
        # label 해결
        
        logger.info(f"Running {downstream_task}")
        fine_tune(emb_model, bert_model, experiments[idx], train_data_dir, save_dir)
            
        # 저장된 모델 로드
        logger.info(f"{downstream_task} Fine Tuning Results")
        accuracy, precision, recall, specificity = evaluate(experiments[idx], val_data_dir, save_dir)

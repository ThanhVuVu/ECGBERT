# Fine-Tuning Process: Detailed Step-by-Step Guide with Data Shapes

## Overview
This document explains the complete fine-tuning process with exact data formats and shapes at each step.

---

## **INITIALIZATION PHASE**

### Step 0: Model Setup
**Location**: `Fine_tune_engine()` function (lines 382-430)

```python
vocab_size = 75      # Vocabulary: 0-70 (wave tokens) + 71 (CLS) + 72 (SEP) + 73 (MASK) + 74 (PAD)
embed_size = 32      # Embedding dimension
```

**Models Created**:
- `emb_model = ECGEmbeddingModel(vocab_size=75, embedding_dim=32)`
- `bert_model = ECGBERTModel(embedding_dim=32, num_layers=4, num_heads=8, dim_feedforward=64)`
- `extra_model = nn.Sequential(FC1: 32→75, FC2: 75→5)`
- `fine_tune_model = CombinedModel(bert_model, extra_model)`

**Model States**:
- `emb_model.eval()` - Frozen (not trained)
- `fine_tune_model.train()` - Trainable

**Training Configuration**:
```python
experiment = {
    "batch_size": 1,
    "lr": 0.001,
    "epochs": 13,
    "num_classes": 5,
    "extra_layers": [
        {'name': 'fc1', 'module': nn.Linear(32, 75)},
        {'name': 'fc2', 'module': nn.Linear(75, 5)}
    ]
}
```

---

## **TRAINING LOOP: EPOCH-BY-EPOCH**

### **EPOCH LOOP** (13 epochs)

```python
for epoch in range(13):  # 13 epochs
    for batch_num in range(train_num_batches):  # Process each batch
```

---

## **BATCH PROCESSING: STEP-BY-STEP**

### **STEP 1: Load Batch Data from Pickle Files**

**Function**: `get_batch_data(batch_num, batch_size, train_data_dir)`

#### 1.1 Load Individual Pickle Files
```python
batch_indices = range(batch_num * batch_size, (batch_num + 1) * batch_size)
# Example: batch_num=0, batch_size=1 → indices=[0]
# Example: batch_num=1, batch_size=1 → indices=[1]

batch_data = []
for idx in batch_indices:
    file_path = f'{train_data_dir}/sentence_{idx}.pkl'
    data = load_pkl_data(file_path)  # Load pickle file
    batch_data.append(data)
```

**Pickle File Format** (`sentence_{idx}.pkl`):
```python
data = [
    sentence,      # List: [71, token1, token2, ..., tokenN, 72]
                   # Length: ~8,000 (one token per wave segment)
    signal,        # NumPy array: [sig1, sig2, ..., sigN]
                   # Length: ~8,000 (aggregated per segment, mean value)
    labels         # NumPy array: [label1, label2, ..., labelN]
                   # Length: ~8,000 (aggregated per segment, majority vote)
]
```

**Example for one patient**:
```python
sentence = [71, 5, 12, 8, 2, 5, 15, 9, 1, ..., 72]  # ~8,000 tokens
signal = [0.2, 0.5, 0.8, 0.3, 0.1, 0.4, 0.9, 0.2, 0.0, ..., 0.4]  # ~8,000 values
labels = [0, 0, 1, 0, 0, 0, 1, 0, 0, ..., 1]  # ~8,000 labels
```

#### 1.2 Convert to Tensors
```python
tokens = [torch.tensor(data[0], device='cuda') for data in batch_data]
signals = [torch.tensor(data[1], device='cuda') for data in batch_data]
labels = [torch.tensor(data[2], device='cuda') for data in batch_data]
```

**Shape After Conversion** (batch_size=1):
```python
tokens[0].shape   = torch.Size([~8000])  # 1D tensor
signals[0].shape  = torch.Size([~8000])  # 1D tensor
labels[0].shape   = torch.Size([~8000])  # 1D tensor
```

#### 1.3 Pad Sequences to Same Length
```python
def pad_sequences(sequences, max_len=None):
    if not max_len:
        max_len = max(len(seq) for seq in sequences)  # Find longest sequence
    
    padded_seqs = [
        torch.cat([seq, torch.zeros(max_len - len(seq), device=seq.device)]) 
        for seq in sequences
    ]
    return torch.stack(padded_seqs)  # Stack into 2D tensor
```

**Example** (batch_size=1, but showing batch_size=3 for clarity):
```python
# Before padding:
tokens = [
    tensor([71, 5, 12, 8, 72]),           # Length: 5
    tensor([71, 5, 12, 8, 45, 23, 72]),  # Length: 7 (longest)
    tensor([71, 5, 72])                   # Length: 3
]

# After padding (max_len=7):
tokens = torch.tensor([
    [71, 5, 12, 8, 72, 0, 0],      # Padded with zeros
    [71, 5, 12, 8, 45, 23, 72],    # Already max length
    [71, 5, 72, 0, 0, 0, 0]        # Padded with zeros
])
```

**Shape After Padding**:
```python
tokens.shape  = torch.Size([batch_size, max_seq_len])
               = torch.Size([1, ~8000])  # Example: [1, 8234]
signals.shape = torch.Size([batch_size, max_seq_len])
               = torch.Size([1, ~8000])  # Example: [1, 8234]
labels.shape  = torch.Size([batch_size, max_seq_len])
               = torch.Size([1, ~8000])  # Example: [1, 8234]
```

#### 1.4 Return Batch Data
```python
return tokens.float(), signals.float(), labels.float()
```

**Final Output Shape**:
```python
org_tokens.shape  = [1, ~8000]  # [batch_size, seq_len]
org_signals.shape = [1, ~8000]  # [batch_size, seq_len]
org_labels.shape  = [1, ~8000]  # [batch_size, seq_len]
```

**Data Type**: `float32` (converted from int/long)

---

### **STEP 2: Remove CLS and SEP Tokens**

**Location**: Line 153

```python
tokens = org_tokens[:, 1:-1]  # Remove first (CLS=71) and last (SEP=72) tokens
```

**Shape Transformation**:
```python
# Before:
org_tokens.shape = [1, 8234]  # [CLS, token1, token2, ..., token8232, SEP]

# After:
tokens.shape = [1, 8232]  # [token1, token2, ..., token8232]
```

**Why**: CLS and SEP are special tokens, not wave tokens. We remove them for processing.

---

### **STEP 3: Sequence Chunking**

**Location**: Lines 154-160

**Purpose**: Split long sequences into chunks of 3600 (matching positional encoding max_len)

```python
small_batch_seq_len = 3600  # Chunk size

for i in range(0, tokens.size(1), small_batch_seq_len):
    # i = 0, 3600, 7200, ...
    small_tokens = tokens[:, i:i+small_batch_seq_len]
    small_signals = org_signals[:, i:i+small_batch_seq_len]
    small_labels = org_labels[:, i:i+small_batch_seq_len]
```

**Example** (seq_len=8232):
```python
# Chunk 1: i=0
small_tokens.shape  = [1, 3600]  # tokens[:, 0:3600]
small_signals.shape = [1, 3600]  # signals[:, 0:3600]
small_labels.shape  = [1, 3600]  # labels[:, 0:3600]

# Chunk 2: i=3600
small_tokens.shape  = [1, 3600]  # tokens[:, 3600:7200]
small_signals.shape = [1, 3600]  # signals[:, 3600:7200]
small_labels.shape  = [1, 3600]  # labels[:, 3600:7200]

# Chunk 3: i=7200
small_tokens.shape  = [1, 1032]  # tokens[:, 7200:8232] (last chunk, shorter)
small_signals.shape = [1, 1032]  # signals[:, 7200:8232]
small_labels.shape  = [1, 1032]  # labels[:, 7200:8232]
```

**Number of Chunks**: `ceil(seq_len / 3600)`
- Example: `ceil(8232 / 3600) = 3` chunks

---

### **STEP 4: Forward Pass Through Embedding Model**

**Function**: `emb_model(small_tokens, small_signals)`
**Location**: `ECGEmbeddingModel.forward()` (models.py, lines 126-147)

#### 4.1 Token Embedding
```python
tokens = tokens.long()  # Convert to int64
token_embedded = self.token_embedding(tokens)
```

**Shape Transformation**:
```python
# Input:
small_tokens.shape = [1, 3600]  # [batch_size, seq_len]
                   = [[5, 12, 8, 2, 5, 15, ...]]

# After Embedding:
token_embedded.shape = [1, 3600, 32]  # [batch_size, seq_len, embedding_dim]
                      # Each token ID → 32-dim vector
```

**Process**:
- `nn.Embedding(vocab_size=75, embedding_dim=32)`
- Lookup table: `token_id → [32-dim vector]`
- Example: `token_id=5 → [0.1, -0.3, 0.5, ..., 0.2]` (32 values)

#### 4.2 Positional Embedding
```python
token_embedded_transposed = token_embedded.permute(1, 0, 2)
# [1, 3600, 32] → [3600, 1, 32]

position_embedded = self.positional_embedding(token_embedded_transposed)
# PositionalEncoding adds position info

position_embedded = position_embedded.permute(1, 0, 2)
# [3600, 1, 32] → [1, 3600, 32]
```

**Shape Transformation**:
```python
# Input:
token_embedded.shape = [1, 3600, 32]

# After permute:
token_embedded_transposed.shape = [3600, 1, 32]

# Positional encoding adds position-dependent values:
# Position 0: [pe0_0, pe0_1, ..., pe0_31]
# Position 1: [pe1_0, pe1_1, ..., pe1_31]
# ...
# Position 3599: [pe3599_0, pe3599_1, ..., pe3599_31]

# After positional encoding:
position_embedded.shape = [3600, 1, 32]

# After permute back:
position_embedded.shape = [1, 3600, 32]
```

**Process**:
- Pre-computed positional encodings: `self.pe.shape = [3600, 1, 32]`
- Adds: `x = x + self.pe[:seq_len]`
- Each position gets unique encoding based on sine/cosine functions

#### 4.3 CNN Embedding (UNet)
```python
signals = signals.unsqueeze(1)  # [1, 3600] → [1, 1, 3600]
wave_features = self.cnn_embedding(signals)  # UNet processing
wave_features = wave_features.permute(0, 2, 1)  # [1, 1, 3600] → [1, 3600, 1]
```

**UNet Processing Details**:
```python
# Input:
signals.shape = [1, 3600]  # [batch_size, seq_len]
signals = signals.unsqueeze(1)  # [1, 1, 3600]  # [batch, channels, length]

# Encoder 1:
enc1 = Conv1d(1, 32, kernel=7, stride=2, padding=3)
# [1, 1, 3600] → [1, 32, 1800]  # Downsampled by 2

# Encoder 2:
enc2 = Conv1d(32, 64, kernel=5, stride=2, padding=2)
# [1, 32, 1800] → [1, 64, 900]  # Downsampled by 2

# Decoder 1:
dec1 = ConvTranspose1d(64, 32, kernel=4, stride=2, padding=1)
# [1, 64, 900] → [1, 32, 1800]  # Upsampled by 2

# Decoder 2 (with skip connection):
dec2 = ConvTranspose1d(32, 32, kernel=4, stride=2, padding=1)
# [1, 32, 1800] → [1, 32, 3600]  # Upsampled by 2
# dec2 = dec2 + enc1 (skip connection, with cropping if needed)

# Final Conv:
final = Conv1d(32, 32, kernel=3, padding=1)
# [1, 32, 3600] → [1, 32, 3600]

# After permute:
wave_features.shape = [1, 3600, 32]  # [batch_size, seq_len, embedding_dim]
```

**Shape Transformation**:
```python
# Input:
small_signals.shape = [1, 3600]

# After unsqueeze:
signals.shape = [1, 1, 3600]  # [batch, channels, length]

# After UNet:
wave_features.shape = [1, 32, 3600]  # [batch, embedding_dim, length]

# After permute:
wave_features.shape = [1, 3600, 32]  # [batch, length, embedding_dim]
```

#### 4.4 Length Alignment
```python
seq_len = token_embedded.size(1)  # 3600
wave_len = wave_features.size(1)   # 3600 (usually matches after UNet)

if wave_len < seq_len:
    wave_features = F.pad(wave_features, (0, 0, 0, seq_len - wave_len))
elif wave_len > seq_len:
    wave_features = wave_features[:, :seq_len, :]
```

**Purpose**: Ensure token and signal embeddings have same sequence length

#### 4.5 Combine Embeddings
```python
combined_embedding = token_embedded + position_embedded + wave_features
```

**Shape Transformation**:
```python
token_embedded.shape      = [1, 3600, 32]
position_embedded.shape   = [1, 3600, 32]
wave_features.shape       = [1, 3600, 32]
                          +
combined_embedding.shape  = [1, 3600, 32]  # Element-wise addition
```

**Process**: Element-wise addition of three 32-dim vectors per position
- Token embedding: Discrete wave type information
- Positional embedding: Temporal position information
- Wave features: Continuous signal information

**Output**:
```python
embeddings.shape = [1, 3600, 32]  # [batch_size, seq_len, embedding_dim]
```

---

### **STEP 5: Forward Pass Through BERT Model**

**Function**: `fine_tune_model(embeddings)`
**Location**: `CombinedModel.forward()` (lines 53-68)

#### 5.1 Transformer Encoder Layers
```python
# Input to CombinedModel:
x.shape = [1, 3600, 32]  # [batch_size, seq_len, embedding_dim]

# Permute for TransformerEncoderLayer:
x = x.permute(1, 0, 2)  # [3600, 1, 32]  # [seq_len, batch_size, embedding_dim]

# Process through 4 transformer layers:
for layer in self.bert_model.layers:  # 4 layers
    x = layer(x)  # TransformerEncoderLayer
    # Each layer: Self-attention + Feedforward
    # Shape maintained: [3600, 1, 32]

# Permute back:
x = x.permute(1, 0, 2)  # [1, 3600, 32]  # [batch_size, seq_len, embedding_dim]
```

**TransformerEncoderLayer Details** (4 layers):
```python
# Layer 1:
x.shape = [3600, 1, 32]  # Input
  ↓ Self-Attention (8 heads, 32-dim → 4-dim per head)
x.shape = [3600, 1, 32]  # Output
  ↓ Feedforward (32 → 64 → 32)
x.shape = [3600, 1, 32]  # Output

# Layer 2-4: Same process
```

**Shape After Transformer**:
```python
x.shape = [1, 3600, 32]  # [batch_size, seq_len, embedding_dim]
```

#### 5.2 Extra Classification Layers
```python
# Input:
x.shape = [1, 3600, 32]

# FC1: 32 → 75
x = nn.Linear(32, 75)(x)
x.shape = [1, 3600, 75]

# FC2: 75 → 5
x = nn.Linear(75, 5)(x)
x.shape = [1, 3600, 5]  # [batch_size, seq_len, num_classes]
```

**Output**:
```python
outputs.shape = [1, 3600, 5]  # [batch_size, seq_len, num_classes]
# Each position has 5 logits (one per class: N, S, V, F, Q)
```

---

### **STEP 6: Loss Calculation**

**Location**: Lines 167-179

#### 6.1 Multi-Class Classification (num_classes=5)
```python
batch_size, seq_len, _ = outputs.shape
# batch_size=1, seq_len=3600

outputs_flat = outputs.view(-1, num_classes)
# [1, 3600, 5] → [3600, 5]

labels_flat = small_labels.view(-1).long()
# [1, 3600] → [3600]

loss = criterion(outputs_flat, labels_flat)
# CrossEntropyLoss([3600, 5], [3600])
```

**Shape Transformation**:
```python
# Input:
outputs.shape = [1, 3600, 5]
small_labels.shape = [1, 3600]

# Flatten:
outputs_flat.shape = [3600, 5]  # [batch*seq_len, num_classes]
labels_flat.shape = [3600]     # [batch*seq_len]

# Loss calculation:
loss = CrossEntropyLoss(outputs_flat, labels_flat)
loss.shape = []  # Scalar (single value)
```

**Process**:
- For each of 3600 positions:
  - `outputs[i]` = [logit_N, logit_S, logit_V, logit_F, logit_Q]
  - `label[i]` = class index (0-4)
  - Compute cross-entropy loss
- Average across all 3600 positions

#### 6.2 Binary Classification (num_classes=1)
```python
outputs = outputs.squeeze(-1)  # [1, 3600, 1] → [1, 3600]
outputs = torch.sigmoid(outputs)  # Apply sigmoid
loss = criterion(outputs, small_labels)  # BCEWithLogitsLoss
```

---

### **STEP 7: Backward Pass and Optimization**

**Location**: Lines 181-182

```python
loss.backward()  # Compute gradients
optimizer.step()  # Update BERT model parameters
```

**What Gets Updated**:
- `bert_model.parameters()` - Transformer layers (4 layers)
- `extra_model.parameters()` - FC1 and FC2 layers
- `emb_model.parameters()` - NOT updated (frozen, `.eval()` mode)

**Gradient Flow**:
```
loss (scalar)
  ↓
outputs [1, 3600, 5]
  ↓
extra_model (FC2, FC1)
  ↓
bert_model (4 transformer layers)
  ↓
embeddings [1, 3600, 32]
  ↓
emb_model (frozen, no gradients)
```

---

### **STEP 8: Accumulate Loss and Continue**

**Location**: Lines 184-185

```python
total_loss += loss.item() * small_labels.size(0)
# loss.item() = scalar loss value
# small_labels.size(0) = 3600 (number of samples in chunk)

num_samples += small_labels.size(0)
# Track total number of samples processed
```

**After All Chunks Processed**:
```python
# Example: 3 chunks processed
total_loss = loss_chunk1 * 3600 + loss_chunk2 * 3600 + loss_chunk3 * 1032
num_samples = 3600 + 3600 + 1032 = 8232

avg_loss = total_loss / num_samples
```

---

### **STEP 9: Save Model (After Each Batch)**

**Location**: Line 187

```python
save_model(fine_tune_model, emb_model, epoch, batch_num, total_loss/num_samples, save_dir, train_num_batches)
```

**Saved Files**:
- `pre_batch_fine_tune_model.pth` - BERT + extra layers
- `pre_batch_emb_model.pth` - Embedding model (frozen)
- `pre_batch_trotal_loss.pkl` - Loss value

**Note**: Model is saved after EVERY batch (inefficient, but current implementation)

---

## **COMPLETE DATA FLOW SUMMARY**

### **Input Data (Pickle File)**
```python
sentence = [71, 5, 12, 8, 2, ..., 72]  # ~8,000 tokens (wave-level)
signal = [0.2, 0.5, 0.8, 0.3, ..., 0.4]  # ~8,000 values (aggregated per segment)
labels = [0, 0, 1, 0, 0, ..., 1]  # ~8,000 labels (aggregated per segment)
```

### **After Loading and Padding**
```python
org_tokens.shape  = [1, ~8000]  # [batch_size, seq_len]
org_signals.shape = [1, ~8000]  # [batch_size, seq_len]
org_labels.shape  = [1, ~8000]  # [batch_size, seq_len]
```

### **After Removing CLS/SEP**
```python
tokens.shape  = [1, ~8000]  # Removed first and last token
signals.shape = [1, ~8000]  # Unchanged
labels.shape  = [1, ~8000]  # Unchanged
```

### **After Chunking (Example: 3 chunks)**
```python
# Chunk 1:
small_tokens.shape  = [1, 3600]
small_signals.shape = [1, 3600]
small_labels.shape  = [1, 3600]

# Chunk 2:
small_tokens.shape  = [1, 3600]
small_signals.shape = [1, 3600]
small_labels.shape  = [1, 3600]

# Chunk 3:
small_tokens.shape  = [1, 1032]  # Last chunk (shorter)
small_signals.shape = [1, 1032]
small_labels.shape  = [1, 1032]
```

### **Through Embedding Model**
```python
# Input:
small_tokens.shape = [1, 3600]
small_signals.shape = [1, 3600]

# Token embedding:
token_embedded.shape = [1, 3600, 32]

# Positional embedding:
position_embedded.shape = [1, 3600, 32]

# CNN embedding:
wave_features.shape = [1, 3600, 32]

# Combined:
embeddings.shape = [1, 3600, 32]
```

### **Through BERT Model**
```python
# Input:
embeddings.shape = [1, 3600, 32]

# After transformer layers:
x.shape = [1, 3600, 32]

# After FC1:
x.shape = [1, 3600, 75]

# After FC2:
outputs.shape = [1, 3600, 5]
```

### **Loss Calculation**
```python
# Flatten:
outputs_flat.shape = [3600, 5]
labels_flat.shape = [3600]

# Loss:
loss = scalar value
```

---

## **KEY PARAMETERS AND SHAPES**

### **Model Architecture**
| Component | Input Shape | Output Shape | Parameters |
|-----------|------------|--------------|------------|
| Token Embedding | `[batch, seq_len]` | `[batch, seq_len, 32]` | 75 × 32 = 2,400 |
| Positional Encoding | `[seq_len, batch, 32]` | `[seq_len, batch, 32]` | Buffer (115K) |
| UNet CNN | `[batch, 1, seq_len]` | `[batch, 32, seq_len]` | ~50K |
| Transformer (4 layers) | `[seq_len, batch, 32]` | `[seq_len, batch, 32]` | ~100K |
| FC1 | `[batch, seq_len, 32]` | `[batch, seq_len, 75]` | 32 × 75 = 2,400 |
| FC2 | `[batch, seq_len, 75]` | `[batch, seq_len, 5]` | 75 × 5 = 375 |

### **Data Dimensions**
| Stage | Tokens | Signals | Labels |
|-------|--------|---------|--------|
| Pickle file | ~8,000 | ~8,000 | ~8,000 |
| After padding | max(~8,000) | max(~8,000) | max(~8,000) |
| After chunking | 3600 (or <3600) | 3600 (or <3600) | 3600 (or <3600) |
| After embedding | [1, 3600, 32] | - | - |
| After BERT | [1, 3600, 32] | - | - |
| After FC layers | [1, 3600, 5] | - | - |
| Loss input | - | - | [3600] |

### **Memory Usage (Per Chunk)**
- Tokens: `1 × 3600 × 4 bytes = 14.4 KB`
- Signals: `1 × 3600 × 4 bytes = 14.4 KB`
- Labels: `1 × 3600 × 4 bytes = 14.4 KB`
- Embeddings: `1 × 3600 × 32 × 4 bytes = 460.8 KB`
- Outputs: `1 × 3600 × 5 × 4 bytes = 72 KB`
- **Total per chunk**: ~600 KB (excluding model weights)

---

## **IMPORTANT NOTES**

1. **Wave-Level Tokens**: Each token represents one wave segment (P, QRS, T, or BG), not one sample
2. **Signal Aggregation**: Signal values are aggregated (mean) per segment to match token count
3. **Chunking**: Sequences longer than 3600 are split into chunks (due to positional encoding limit)
4. **Frozen Embedding**: `emb_model` is frozen (`.eval()`), only BERT layers are trained
5. **Multiple Optimizer Steps**: One optimizer step per chunk (not per batch)
6. **Model Saving**: Model saved after every batch (inefficient)

---

**End of Detailed Process Guide**


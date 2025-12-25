"""
Heartbeat Classification Preprocessing for MIT-BIH Database
Extracts beat-level labels from MIT-BIH annotations for heartbeat classification
"""

import wfdb
import numpy as np
import pywt
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
import pickle
import os
import logging

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

# Beat type mapping for classification
# Based on AAMI standard 5-category classification:
# Category N (Normal): Normal, Left/Right bundle branch block, Atrial escape, Nodal escape
# Category S (Supraventricular): Atrial premature, Aberrant atrial premature, Nodal premature, Supra-ventricular premature
# Category V (Ventricular): Premature ventricular contraction, Ventricular escape
# Category F (Fusion): Fusion of ventricular and normal
# Category Q (Unclassifiable/Paced): Paced, Fusion of paced and normal, Unclassifiable

BEAT_TYPE_MAP = {
    # Category N (Normal) - Label 0
    'N': 0,  # Normal beat
    'L': 0,  # Left bundle branch block
    'R': 0,  # Right bundle branch block
    'e': 0,  # Atrial escape
    'j': 0,  # Nodal (junctional) escape
    '+': 0,  # Rhythm change marker (treated as normal)
    '~': 0,  # Signal quality change (treated as normal)
    
    # Category S (Supraventricular) - Label 1
    'A': 1,  # Atrial premature beat
    'a': 1,  # Aberrated atrial premature beat
    'J': 1,  # Nodal (junctional) premature beat
    'S': 1,  # Supraventricular premature beat
    
    # Category V (Ventricular) - Label 2
    'V': 2,  # Premature ventricular contraction
    'E': 2,  # Ventricular escape beat
    
    # Category F (Fusion) - Label 3
    'F': 3,  # Fusion of ventricular and normal beat
    
    # Category Q (Unclassifiable/Paced) - Label 4
    '/': 4,  # Paced beat
    'f': 4,  # Fusion of paced and normal beat
    'Q': 4,  # Unclassifiable beat
}

def load_ecg_data_with_beat_labels(record_path, binary_classification=False):
    """
    Load ECG data and create beat-level labels from MIT-BIH annotations.
    
    Args:
        record_path: Path to the .dat file (without extension)
        binary_classification: If True, binary classification (Normal=0, Abnormal=1)
                             If False, 5-class AAMI classification:
                             - Category N (Normal) = 0
                             - Category S (Supraventricular) = 1
                             - Category V (Ventricular) = 2
                             - Category F (Fusion) = 3
                             - Category Q (Unclassifiable/Paced) = 4
    
    Returns:
        signal: ECG signal data
        fs: Sampling frequency
        labels: Beat-level labels (same length as signal)
    """
    record = wfdb.rdrecord(record_path[:-4], pn_dir='mitdb')
    annotation = wfdb.rdann(record_path[:-4], 'atr', pn_dir='mitdb')
    
    # Initialize labels array with zeros (default: normal)
    labels = np.zeros(len(record.p_signal), dtype=int)
    
    # Create beat-level labels based on annotations
    for i, (sample_idx, symbol) in enumerate(zip(annotation.sample, annotation.symbol)):
        # Determine the range for this beat
        if i + 1 < len(annotation.sample):
            end_idx = annotation.sample[i + 1]
        else:
            end_idx = len(labels)
        
        start_idx = sample_idx
        
        # Map beat type to label
        if binary_classification:
            # Binary: Normal (0) vs Abnormal (1)
            # Category N is Normal, all others are Abnormal
            if symbol in ['N', 'L', 'R', 'e', 'j', '+', '~']:
                label = 0  # Normal (Category N)
            else:
                label = 1  # Abnormal (Categories S, V, F, Q)
        else:
            # 5-class AAMI classification
            label = BEAT_TYPE_MAP.get(symbol, 0)  # Default to Category N (Normal) if unknown
        
        # Label the beat interval
        labels[start_idx:end_idx] = label
    
    return record.p_signal, record.fs, labels

def load_dataset(dataset_path, file_extension='.dat', binary_classification=True):
    """
    Load all ECG records from a directory.
    
    Args:
        dataset_path: Directory containing MIT-BIH .dat files
        file_extension: File extension to look for
        binary_classification: Binary or multi-class classification
    
    Returns:
        ecg_signals: List of (signal, fs) tuples
        patient_ids: List of patient/record IDs
        labels: List of label arrays
    """
    ecg_signals = []
    patient_ids = []
    labels = []
    
    for file_name in sorted(os.listdir(dataset_path)):
        if file_name.endswith(file_extension):
            file_path = os.path.join(dataset_path, file_name)
            record_id = file_name.replace(file_extension, '')
            
            try:
                ecg_data, fs, label = load_ecg_data_with_beat_labels(file_path, binary_classification)
                ecg_signals.append((ecg_data, fs))
                patient_ids.append(record_id)
                labels.append(label)
                logger.info(f"Loaded {record_id}: signal shape {ecg_data.shape}, labels shape {label.shape}")
            except Exception as e:
                logger.error(f"Error loading {file_name}: {e}")
                continue
    
    return ecg_signals, patient_ids, labels

def bandstop_filter(signal, fs, lowcut=50.0, highcut=60.0, order=2):
    """Remove power line interference (50-60 Hz)."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='bandstop', fs=fs)
    filtered_signal = np.array([filtfilt(b, a, signal[:, i]) for i in range(signal.shape[1])]).T
    return filtered_signal

def remove_baseline_wander(signal, fs, wavelet='db4', level=2):
    """Remove baseline wander using wavelet decomposition."""
    baseline_removed_signal = np.zeros_like(signal)
    
    def process_lead(lead_signal):
        coeffs = pywt.wavedec(lead_signal, wavelet, level=level)
        coeffs[0] = np.zeros_like(coeffs[0])
        baseline_wander_signal = pywt.waverec(coeffs, wavelet)
        
        if len(baseline_wander_signal) > len(lead_signal):
            baseline_wander_signal = baseline_wander_signal[:len(lead_signal)]
        elif len(baseline_wander_signal) < len(lead_signal):
            baseline_wander_signal = np.pad(
                baseline_wander_signal, 
                (0, len(lead_signal) - len(baseline_wander_signal)), 
                'constant'
            )
        
        return lead_signal - baseline_wander_signal
    
    for i in range(signal.shape[1]):
        baseline_removed_signal[:, i] = process_lead(signal[:, i])
    
    return baseline_removed_signal

def preprocess_ecg_signal(signal, fs):
    """Apply preprocessing pipeline to ECG signal."""
    filtered_signal = bandstop_filter(signal, fs)
    cleaned_signal = remove_baseline_wander(filtered_signal, fs)
    return cleaned_signal

def process_signals(signals, labels):
    """Process all signals and labels."""
    processed_signals = []
    processed_labels = []
    
    for idx, (signal, fs) in enumerate(signals):
        try:
            cleaned_signal = preprocess_ecg_signal(signal, fs)
            
            # Check for invalid values
            if np.any(np.isnan(cleaned_signal)) or np.any(np.isinf(cleaned_signal)):
                logger.warning(f"Skipping signal {idx} due to NaN/Inf values")
                continue
            
            processed_signals.append(cleaned_signal)
            processed_labels.append(labels[idx])
        except Exception as e:
            logger.error(f"Error processing signal {idx}: {e}")
            continue
    
    return processed_signals, processed_labels

def split_data(signals, patient_ids, labels, test_size=0.2, random_state=42):
    """Split data by patient ID to avoid data leakage."""
    unique_ids = list(set(patient_ids))
    train_ids, val_ids = train_test_split(unique_ids, test_size=test_size, random_state=random_state)
    
    train_signals = [signals[i] for i in range(len(signals)) if patient_ids[i] in train_ids]
    val_signals = [signals[i] for i in range(len(signals)) if patient_ids[i] in val_ids]
    train_labels = [labels[i] for i in range(len(labels)) if patient_ids[i] in train_ids]
    val_labels = [labels[i] for i in range(len(labels)) if patient_ids[i] in val_ids]
    
    return train_signals, val_signals, train_labels, val_labels

def save_pkl_data(save_dir, file_name, save_pkl):
    """Save data to pickle file."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    with open(save_path, 'wb') as f:
        pickle.dump(save_pkl, f)

def ecg_preprocess_for_heartbeat_classification(dataset_path, save_dir, downstream_task, 
                                                file_extension='.dat', binary_classification=False):
    """
    Main preprocessing function for heartbeat classification.
    
    Args:
        dataset_path: Path to directory containing MIT-BIH .dat files
        save_dir: Directory to save processed data
        downstream_task: Task name (e.g., 'Heartbeat_Classification')
        file_extension: File extension to look for
        binary_classification: Binary (True) or 5-class AAMI (False) classification
                              - Binary: Normal (0) vs Abnormal (1)
                              - 5-class: N(0), S(1), V(2), F(3), Q(4)
    """
    logger.info(f"Loading dataset from {dataset_path}")
    ecg_signals, patient_ids, labels = load_dataset(dataset_path, file_extension, binary_classification)
    
    logger.info(f"Loaded {len(ecg_signals)} records")
    logger.info(f"Splitting data into train/validation sets")
    train_signals, val_signals, train_labels, val_labels = split_data(
        ecg_signals, patient_ids, labels, test_size=0.2
    )
    
    logger.info(f"Processing {len(train_signals)} training signals")
    processed_train_signals, train_labels = process_signals(train_signals, train_labels)
    
    logger.info(f"Processing {len(val_signals)} validation signals")
    processed_val_signals, val_labels = process_signals(val_signals, val_labels)
    
    # Save processed data
    save_pkl_data(save_dir, f'{downstream_task}_train_processed_signals.pkl', processed_train_signals)
    save_pkl_data(save_dir, f'{downstream_task}_val_processed_signals.pkl', processed_val_signals)
    save_pkl_data(save_dir, f'{downstream_task}_train_labels.pkl', train_labels)
    save_pkl_data(save_dir, f'{downstream_task}_val_labels.pkl', val_labels)
    
    logger.info(f"Preprocessing complete. Saved to {save_dir}")
    logger.info(f"Train: {len(processed_train_signals)} signals")
    logger.info(f"Validation: {len(processed_val_signals)} signals")

def ECG_Heartbeat_Preprocessing(dataset_paths, base_dir, binary_classification=False):
    """
    Wrapper function for heartbeat classification preprocessing.
    
    Args:
        dataset_paths: List of [task_name, dataset_path, file_extension] tuples
        base_dir: Base directory for saving processed data
        binary_classification: Binary (True) or 5-class AAMI (False) classification
                              - Binary: Normal (0) vs Abnormal (1)
                              - 5-class: N(0), S(1), V(2), F(3), Q(4)
    """
    for downstream_task, dataset_path, file_extension in dataset_paths:
        save_dir = os.path.join(base_dir, f'{downstream_task}/ECG_Preprocessing')
        ecg_preprocess_for_heartbeat_classification(
            dataset_path, save_dir, downstream_task, file_extension, binary_classification
        )
        logger.info(f'{downstream_task} preprocessing done')


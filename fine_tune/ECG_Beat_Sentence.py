import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from itertools import groupby
from joblib import Parallel, delayed

def load_pkl_data(file_path):
    """
    Load data from pickle file, handling both standard pickle and joblib formats.
    """
    import joblib
    
    try:
        # Try joblib first (used for sklearn/tslearn models)
        return joblib.load(file_path)
    except Exception as e1:
        try:
            # Fallback to standard pickle
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e2:
            raise RuntimeError(f"Failed to load {file_path} with both joblib and pickle. "
                             f"Joblib error: {e1}. Pickle error: {e2}. "
                             f"File may be corrupted or in unsupported format.")

def save_pkl_data(save_dir, file_name, save_pkl):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    with open(save_path, 'wb') as f:
        pickle.dump(save_pkl, f)

def process_wave_type_segment(lead_signal_segments, preprocessed_signal_lead):
    """
    Extract wave segments from signal, handling edge cases.
    
    Args:
        lead_signal_segments: List of (start_idx, end_idx) tuples
        preprocessed_signal_lead: 1D array of signal values
    
    Returns:
        segments: List of signal segments (1D arrays)
    """
    signal_len = len(preprocessed_signal_lead)
    segments = []
    
    for st, end in lead_signal_segments:
        # Validate and clamp indices
        st = max(0, min(int(st), signal_len - 1))
        end = max(0, min(int(end), signal_len - 1))
        
        # Ensure st <= end
        if st > end:
            st, end = end, st
        
        # Extract segment (guaranteed to have at least one element)
        segment = preprocessed_signal_lead[st:end+1]
        segments.append(segment)
    
    return segments

def calculate_distance(signal, cluster_centers):
    """
    Calculate Euclidean distance between a signal and cluster centers.
    
    Args:
        signal: 1D array of signal values
        cluster_centers: 2D array of shape (n_clusters, n_features) or 3D from tslearn
    
    Returns:
        distances: 1D array of distances to each cluster center
    """
    # Ensure signal is 1D
    signal = np.array(signal).flatten()
    
    # Ensure cluster_centers is 2D
    cluster_centers = np.array(cluster_centers)
    if cluster_centers.ndim == 1:
        cluster_centers = cluster_centers.reshape(1, -1)
    elif cluster_centers.ndim == 3:
        # tslearn TimeSeriesKMeans: (n_clusters, n_timesteps, n_features) -> (n_clusters, n_timesteps * n_features)
        n_clusters, n_timesteps, n_features = cluster_centers.shape
        cluster_centers = cluster_centers.reshape(n_clusters, n_timesteps * n_features)
    elif cluster_centers.ndim > 2:
        # Flatten all dimensions except the first (cluster dimension)
        cluster_centers = cluster_centers.reshape(cluster_centers.shape[0], -1)
    
    signal_len = len(signal)
    cluster_len = cluster_centers.shape[1]
    
    # Pad or truncate signal to match cluster center length
    if signal_len != cluster_len:
        if signal_len < cluster_len:
            signal = np.pad(signal, (0, cluster_len - signal_len), 'constant')
        else:
            signal = signal[:cluster_len]
    
    # Calculate distances - signal should be (1, n_features), cluster_centers (n_clusters, n_features)
    distances = euclidean_distances(signal.reshape(1, -1), cluster_centers)
    return distances

def process_lead_signal(lead, signal_segments, preprocessed_signal, cluster_dir, wave_types):
    """
    Process lead signal and assign one token per wave segment (as per ECGBERT paper).
    
    Returns:
        wave_tokens: List of tokens (one per segment) in temporal order
        segment_info: List of (start_idx, end_idx, wave_type) tuples for signal alignment
    """
    wave_tokens = []  # List to store tokens (one per segment)
    segment_info = []  # Store segment info: (start_idx, end_idx, wave_type)
    
    # Map wave type to possible file name variations (handle case sensitivity)
    # Pretraining saves files as: P_cluster.pkl, QRS_cluster.pkl, T_cluster.pkl, BG_cluster.pkl
    wave_type_to_file = {
        'p': 'P_cluster.pkl',
        'qrs': 'QRS_cluster.pkl',
        't': 'T_cluster.pkl',
        'bg': 'BG_cluster.pkl'
    }
    
    for wave_type in wave_types:
        wave_type_preprocessed_signal = process_wave_type_segment(signal_segments[lead][wave_type][0], preprocessed_signal[:, lead])

        # Get the expected cluster file name
        expected_file = wave_type_to_file[wave_type]
        cluster_file = os.path.join(cluster_dir, expected_file)
        
        # If file doesn't exist, try to find it with case-insensitive search
        if not os.path.exists(cluster_file):
            if os.path.exists(cluster_dir):
                all_files = os.listdir(cluster_dir)
                # Find file matching pattern (case-insensitive)
                matching_files = [f for f in all_files if f.lower().endswith(f'{wave_type}_cluster.pkl'.lower())]
                if matching_files:
                    cluster_file = os.path.join(cluster_dir, matching_files[0])
                    # Use print if logger not available yet
                    try:
                        logger.info(f"Found cluster file: {matching_files[0]} (expected: {expected_file})")
                    except:
                        print(f"Found cluster file: {matching_files[0]} (expected: {expected_file})")
                else:
                    # List available files for debugging
                    cluster_files = [f for f in all_files if f.endswith('_cluster.pkl')]
                    raise FileNotFoundError(
                        f"Cluster file not found for wave type '{wave_type}'. "
                        f"Expected: {expected_file}\n"
                        f"Cluster directory: {cluster_dir}\n"
                        f"Available cluster files: {cluster_files}"
                    )
            else:
                raise FileNotFoundError(f"Cluster directory does not exist: {cluster_dir}")
        
        # Load the clustering model (saved with joblib.dump)
        try:
            cluster_model = load_pkl_data(cluster_file)
        except Exception as e:
            raise RuntimeError(f"Failed to load cluster file {cluster_file}: {e}. "
                             f"File may be corrupted or in wrong format.")
        
        # Extract cluster centers from the model
        # The model is a KMeans or TimeSeriesKMeans object saved with joblib
        if hasattr(cluster_model, 'cluster_centers_'):
            # Standard sklearn KMeans - shape: (n_clusters, n_features)
            cluster_centers = cluster_model.cluster_centers_
        elif hasattr(cluster_model, 'centroids_'):
            # tslearn TimeSeriesKMeans - shape: (n_clusters, n_timesteps, n_features) or (n_clusters, n_timesteps)
            cluster_centers = cluster_model.centroids_
            # If 3D, reshape to 2D: flatten time and feature dimensions
            if cluster_centers.ndim == 3:
                # Reshape from (n_clusters, n_timesteps, n_features) to (n_clusters, n_timesteps * n_features)
                n_clusters, n_timesteps, n_features = cluster_centers.shape
                cluster_centers = cluster_centers.reshape(n_clusters, n_timesteps * n_features)
            elif cluster_centers.ndim > 2:
                # Flatten all dimensions except the first (cluster dimension)
                cluster_centers = cluster_centers.reshape(cluster_centers.shape[0], -1)
        elif hasattr(cluster_model, 'cpu'):
            # PyTorch tensor (fallback)
            cluster_centers = cluster_model.cpu().numpy()
            if cluster_centers.ndim > 2:
                cluster_centers = cluster_centers.reshape(cluster_centers.shape[0], -1)
        elif isinstance(cluster_model, np.ndarray):
            # Already a numpy array (cluster centers directly)
            cluster_centers = cluster_model
            if cluster_centers.ndim > 2:
                cluster_centers = cluster_centers.reshape(cluster_centers.shape[0], -1)
        else:
            # Try to convert to numpy array
            cluster_centers = np.array(cluster_model)
            if cluster_centers.ndim > 2:
                cluster_centers = cluster_centers.reshape(cluster_centers.shape[0], -1)
        
        # Ensure it's a numpy array and 2D
        if not isinstance(cluster_centers, np.ndarray):
            cluster_centers = np.array(cluster_centers)
        
        # Final check: ensure 2D shape (n_clusters, n_features)
        if cluster_centers.ndim != 2:
            if cluster_centers.ndim == 1:
                cluster_centers = cluster_centers.reshape(1, -1)
            else:
                cluster_centers = cluster_centers.reshape(cluster_centers.shape[0], -1)

        # Process each wave segment and assign ONE token per segment
        for seg_idx, signal in enumerate(wave_type_preprocessed_signal):
            distances = calculate_distance(signal, cluster_centers)
            cluster_idx = np.argmin(distances) + {'p': 0, 'qrs': 12, 't': 31, 'bg': 45}[wave_type]

            idx_st, idx_end = signal_segments[lead][wave_type][0][seg_idx]
            
            # Store ONE token for this wave segment (not per sample)
            wave_tokens.append(int(cluster_idx))
            segment_info.append((idx_st, idx_end, wave_type))
    
    # Sort segments by start index to maintain temporal order
    sorted_indices = sorted(range(len(segment_info)), key=lambda i: segment_info[i][0])
    sorted_tokens = [wave_tokens[i] for i in sorted_indices]
    sorted_segment_info = [segment_info[i] for i in sorted_indices]
    
    return sorted_tokens, sorted_segment_info

def vocab_create_assignment(downstrem_task, processed_data_dir, seg_dir, cluster_dir, save_dir):
    wave_types = ['p', 'qrs', 't', 'bg']
    
    for suffix in ['train', 'val']:
        t_sentence_num = 0
        v_sentence_num = 0
        
        prefix = f'{downstrem_task}_{suffix}'
        
        # Load preprocessed signals
        processed_signals_file = os.path.join(processed_data_dir, f'{prefix}_processed_signals.pkl')
        if not os.path.exists(processed_signals_file):
            # List available files for debugging
            if os.path.exists(processed_data_dir):
                available_files = [f for f in os.listdir(processed_data_dir) if f.endswith('.pkl')]
                raise FileNotFoundError(
                    f"Processed signals file not found: {processed_signals_file}\n"
                    f"Available files in {processed_data_dir}: {available_files}"
                )
            else:
                raise FileNotFoundError(f"Processed data directory does not exist: {processed_data_dir}")
        
        preprocessed_signals = load_pkl_data(processed_signals_file)
                
        for idx, preprocessed_signal in enumerate(preprocessed_signals):
            # Load signal segments
            segments_file = os.path.join(seg_dir, f'{prefix}_{idx}_segments.pkl')
            if not os.path.exists(segments_file):
                # List available files for debugging
                if os.path.exists(seg_dir):
                    available_files = [f for f in os.listdir(seg_dir) if f.endswith('_segments.pkl')]
                    raise FileNotFoundError(
                        f"Segments file not found: {segments_file}\n"
                        f"Expected pattern: {prefix}_{idx}_segments.pkl\n"
                        f"Available segment files in {seg_dir}: {available_files}"
                    )
                else:
                    raise FileNotFoundError(f"Segmentation directory does not exist: {seg_dir}")
            
            signal_segments = load_pkl_data(segments_file)
            
            # Load labels
            labels_file = os.path.join(seg_dir, f'{prefix}_{idx}_label.pkl')
            if not os.path.exists(labels_file):
                # List available files for debugging
                if os.path.exists(seg_dir):
                    available_files = [f for f in os.listdir(seg_dir) if f.endswith('_label.pkl')]
                    raise FileNotFoundError(
                        f"Labels file not found: {labels_file}\n"
                        f"Expected pattern: {prefix}_{idx}_label.pkl\n"
                        f"Available label files in {seg_dir}: {available_files}"
                    )
            
            labels = load_pkl_data(labels_file)

            signal_vocabs = Parallel(n_jobs=-1)(
                delayed(process_lead_signal)(lead, signal_segments, preprocessed_signal, cluster_dir, wave_types) 
                for lead in range(len(signal_segments))
            )

            for lead, (wave_tokens, segment_info) in enumerate(signal_vocabs):
                # Create sentence: [CLS] + wave tokens + [SEP]
                # Each token represents one wave segment (as per ECGBERT paper)
                sentence = [71] + wave_tokens + [72]
                
                # Aggregate signal per segment to match token count
                # Each segment's signal is represented by its mean value
                sentence_signal = []
                signal_len = preprocessed_signal.shape[0]
                
                for idx_st, idx_end, _ in segment_info:
                    # Validate and clamp segment indices to valid range
                    idx_st = max(0, min(int(idx_st), signal_len - 1))
                    idx_end = max(0, min(int(idx_end), signal_len - 1))
                    
                    # Ensure idx_st <= idx_end
                    if idx_st > idx_end:
                        idx_st, idx_end = idx_end, idx_st
                    
                    # Extract segment signal (guaranteed to have at least one element after validation)
                    segment_signal = preprocessed_signal[idx_st:idx_end+1, lead]
                    
                    # Compute mean (should never be empty after validation, but check anyway)
                    if len(segment_signal) > 0:
                        segment_mean = float(np.mean(segment_signal))
                    else:
                        # Fallback: use value at idx_st (shouldn't happen after validation)
                        segment_mean = float(preprocessed_signal[idx_st, lead]) if idx_st < signal_len else 0.0
                    
                    sentence_signal.append(segment_mean)
                
                # Assign heartbeat-level labels to segments
                # IMPORTANT: Labels are per heartbeat, not per segment. 
                # All segments (P, QRS, T, BG) in the same heartbeat should have the SAME label.
                # 
                # Since labels are created at sample level with all samples in a heartbeat 
                # interval having the same label, we take the label from the first sample 
                # of each segment. This ensures all segments belonging to the same heartbeat 
                # get the same label (since they all start within the same heartbeat interval).
                sentence_label = []
                for idx_st, idx_end, _ in segment_info:
                    # Clamp indices to valid range
                    idx_st = max(0, min(idx_st, len(labels) - 1))
                    idx_end = max(0, min(idx_end, len(labels) - 1))
                    
                    # Get label from first sample of segment
                    # This represents the heartbeat this segment belongs to
                    heartbeat_label = labels[idx_st]
                    sentence_label.append(int(heartbeat_label))
                
                # Convert to numpy arrays
                sentence_signal = np.array(sentence_signal)
                sentence_label = np.array(sentence_label)
            
                data = [sentence, sentence_signal, sentence_label]
                if prefix.endswith('train'):
                    save_pkl_data(os.path.join(save_dir, 'train'), f'sentence_{t_sentence_num}.pkl', data)
                    t_sentence_num += 1
                else:
                    save_pkl_data(os.path.join(save_dir, 'val'), f'sentence_{v_sentence_num}.pkl', data)
                    v_sentence_num += 1

        logger.info(f'{prefix} Sentence Done')

import logging

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

def ECG_Beat_Sentence(downstrem_tasks, dir, cluster_dir):
    
    for downstrem_task in downstrem_tasks:
        processed_data_dir = os.path.join(dir, f'{downstrem_task}/ECG_Preprocessing')
        seg_dir = os.path.join(dir, f'{downstrem_task}/ECG_Segmentation')
        save_dir = os.path.join(dir, f'{downstrem_task}/ECG_Sentence')
        
        vocab_create_assignment(downstrem_task, processed_data_dir, seg_dir, cluster_dir, save_dir)

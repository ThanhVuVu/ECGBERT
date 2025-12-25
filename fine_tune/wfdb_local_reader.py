"""
Direct local file reader for MIT-BIH database files
Bypasses wfdb's URL construction to read files directly from disk
"""

import os
import numpy as np
import struct

def read_wfdb_record_local(record_path):
    """
    Read MIT-BIH record directly from local files without wfdb's URL construction.
    
    Args:
        record_path: Path to record (without extension) or full path to .dat file
    
    Returns:
        signal: ECG signal data (numpy array)
        fs: Sampling frequency
        sig_name: Signal names
    """
    # Remove extension if present
    if record_path.endswith('.dat') or record_path.endswith('.hea'):
        record_path = record_path[:-4]
    
    # Get directory and record name
    record_dir = os.path.dirname(record_path) if os.path.dirname(record_path) else '.'
    record_name = os.path.basename(record_path)
    
    # Read header file
    hea_file = os.path.join(record_dir, f"{record_name}.hea")
    if not os.path.exists(hea_file):
        raise FileNotFoundError(f"Header file not found: {hea_file}")
    
    with open(hea_file, 'r') as f:
        header_lines = f.readlines()
    
    # Parse header
    first_line = header_lines[0].strip().split()
    n_sig = int(first_line[1])
    fs = int(first_line[2])
    
    sig_name = []
    fmt = []
    gain = []
    baseline = []
    
    for i in range(1, n_sig + 1):
        line = header_lines[i].strip().split()
        sig_name.append(line[-1])  # Signal name is last field
        fmt.append(int(line[1]))
        gain.append(float(line[2]))
        baseline.append(int(line[3]))
    
    # Read data file
    dat_file = os.path.join(record_dir, f"{record_name}.dat")
    if not os.path.exists(dat_file):
        raise FileNotFoundError(f"Data file not found: {dat_file}")
    
    # Read binary data (format 212: 12-bit samples)
    with open(dat_file, 'rb') as f:
        data = f.read()
    
    # Parse format 212 (12-bit samples, 2 samples per 3 bytes)
    samples = []
    for i in range(0, len(data) - 2, 3):
        byte1 = data[i]
        byte2 = data[i + 1]
        byte3 = data[i + 2]
        
        # First sample: bits 0-11
        sample1 = ((byte2 & 0x0F) << 8) | byte1
        if sample1 > 2047:
            sample1 -= 4096
        
        # Second sample: bits 4-15
        sample2 = ((byte2 & 0xF0) >> 4) | (byte3 << 4)
        if sample2 > 2047:
            sample2 -= 4096
        
        samples.extend([sample1, sample2])
    
    # Convert to numpy array and reshape
    samples = np.array(samples[:n_sig * (len(samples) // n_sig)])
    signal = samples.reshape(-1, n_sig)
    
    # Convert to physical units
    for i in range(n_sig):
        signal[:, i] = (signal[:, i] - baseline[i]) / gain[i]
    
    return signal, fs, sig_name

def read_wfdb_annotation_local(record_path, extension='atr'):
    """
    Read MIT-BIH annotation file directly from local files.
    
    Args:
        record_path: Path to record (without extension)
        extension: Annotation file extension (default: 'atr')
    
    Returns:
        sample: Sample indices
        symbol: Annotation symbols
    """
    # Remove extension if present
    if record_path.endswith('.dat') or record_path.endswith('.hea') or record_path.endswith('.atr'):
        record_path = record_path.rsplit('.', 1)[0]
    
    # Get directory and record name
    record_dir = os.path.dirname(record_path) if os.path.dirname(record_path) else '.'
    record_name = os.path.basename(record_path)
    
    # Read annotation file
    ann_file = os.path.join(record_dir, f"{record_name}.{extension}")
    if not os.path.exists(ann_file):
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")
    
    # Use wfdb to read annotation (it handles this better than signals)
    # But force it to use local file
    import wfdb
    original_cwd = os.getcwd()
    try:
        os.chdir(record_dir)
        annotation = wfdb.rdann(record_name, extension, pn_dir='')
    finally:
        os.chdir(original_cwd)
    
    return annotation.sample, annotation.symbol


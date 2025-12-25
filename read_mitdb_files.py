"""
Script to read and display MIT-BIH Arrhythmia Database files
Displays header, signal data, and annotations in a readable format
"""

import os
import numpy as np
from wfdb import rdrecord, rdann

def display_header_info(record_name, db_dir='mitdb'):
    """Read and display header information from .hea file"""
    print("=" * 80)
    print(f"HEADER INFORMATION for record: {record_name}")
    print("=" * 80)
    
    try:
        record = rdrecord(record_name, pn_dir=db_dir)
        
        print(f"\nRecord Name: {record_name}")
        print(f"Number of Signals: {record.n_sig}")
        print(f"Sampling Frequency: {record.fs} Hz")
        print(f"Number of Samples: {record.sig_len}")
        print(f"Duration: {record.sig_len / record.fs:.2f} seconds ({record.sig_len / record.fs / 60:.2f} minutes)")
        
        print(f"\nSignal Details:")
        print("-" * 80)
        for i, sig_name in enumerate(record.sig_name):
            print(f"\nSignal {i+1}: {sig_name}")
            print(f"  Format: {record.fmt[i]}")
            print(f"  Gain: {record.adc_gain[i]}")
            print(f"  Baseline: {record.baseline[i]}")
            print(f"  Units: {record.units[i]}")
            print(f"  ADC Resolution: {record.adc_res[i]} bits")
            print(f"  ADC Zero: {record.adc_zero[i]}")
            print(f"  Initial Value: {record.init_value[i]}")
            if record.comments:
                print(f"  Comments: {record.comments}")
        
        if record.comments:
            print(f"\nAdditional Comments:")
            for comment in record.comments:
                print(f"  {comment}")
        
        return record
        
    except Exception as e:
        print(f"Error reading header: {e}")
        return None

def display_signal_data(record, num_samples=1000):
    """Display sample signal data"""
    print("\n" + "=" * 80)
    print(f"SIGNAL DATA (showing first {num_samples} samples)")
    print("=" * 80)
    
    if record is None:
        return
    
    signals = record.p_signal  # Physical signal values
    num_signals = signals.shape[1]
    
    # Limit the number of samples to display
    display_samples = min(num_samples, signals.shape[0])
    
    print(f"\nSignal shape: {signals.shape} (samples x channels)")
    print(f"Displaying first {display_samples} samples:\n")
    
    # Create header
    header = "Sample #"
    for i, sig_name in enumerate(record.sig_name):
        header += f"\t{sig_name}"
    print(header)
    print("-" * 80)
    
    # Display sample values
    for i in range(display_samples):
        row = f"{i:8d}"
        for j in range(num_signals):
            row += f"\t{signals[i, j]:10.4f}"
        print(row)
    
    if signals.shape[0] > display_samples:
        print(f"\n... ({signals.shape[0] - display_samples} more samples)")
    
    # Display statistics
    print("\n" + "=" * 80)
    print("SIGNAL STATISTICS")
    print("=" * 80)
    for i, sig_name in enumerate(record.sig_name):
        signal_data = signals[:, i]
        print(f"\n{sig_name}:")
        print(f"  Min: {np.min(signal_data):.4f} {record.units[i]}")
        print(f"  Max: {np.max(signal_data):.4f} {record.units[i]}")
        print(f"  Mean: {np.mean(signal_data):.4f} {record.units[i]}")
        print(f"  Std: {np.std(signal_data):.4f} {record.units[i]}")

def display_annotations(record_name, db_dir='mitdb', max_annotations=100):
    """Read and display annotations from .atr file"""
    print("\n" + "=" * 80)
    print(f"ANNOTATIONS for record: {record_name}")
    print("=" * 80)
    
    # Beat type descriptions
    beat_types = {
        'N': 'Normal beat',
        'L': 'Left bundle branch block beat',
        'R': 'Right bundle branch block beat',
        'A': 'Atrial premature beat',
        'a': 'Aberrated atrial premature beat',
        'J': 'Nodal (junctional) premature beat',
        'S': 'Supraventricular premature beat',
        'V': 'Premature ventricular contraction',
        'E': 'Ventricular escape beat',
        'F': 'Fusion of ventricular and normal beat',
        'Q': 'Unclassifiable beat',
        '/': 'Paced beat',
        'f': 'Fusion of paced and normal beat',
        '+': 'Rhythm change',
        '~': 'Signal quality change'
    }
    
    try:
        annotation = rdann(record_name, 'atr', pn_dir=db_dir)
        
        print(f"\nTotal Annotations: {len(annotation.sample)}")
        print(f"Annotation Type: {annotation.extension}")
        if hasattr(annotation, 'fs'):
            print(f"Sampling Frequency: {annotation.fs} Hz")
        
        # Display annotation symbols and their meanings
        print("\nAnnotation Symbols Distribution:")
        print("-" * 80)
        unique_symbols, counts = np.unique(annotation.symbol, return_counts=True)
        for symbol, count in zip(unique_symbols, counts):
            description = beat_types.get(symbol, 'Unknown beat type')
            print(f"  '{symbol}': {count:5d} occurrences - {description}")
        
        # Display first N annotations
        display_count = min(max_annotations, len(annotation.sample))
        fs = annotation.fs if hasattr(annotation, 'fs') else 360.0
        print(f"\nFirst {display_count} Annotations:")
        print("-" * 80)
        print(f"{'Index':<8} {'Sample':<10} {'Time':<12} {'Symbol':<8} {'Beat Type':<30}")
        print("-" * 80)
        
        for i in range(display_count):
            sample = annotation.sample[i]
            symbol = annotation.symbol[i]
            
            # Convert sample to time
            time_sec = sample / fs
            minutes = int(time_sec // 60)
            seconds = int(time_sec % 60)
            milliseconds = int((time_sec % 1) * 1000)
            time_str = f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
            
            beat_desc = beat_types.get(symbol, 'Unknown')
            
            print(f"{i:<8} {sample:<10} {time_str:<12} {symbol:<8} {beat_desc:<30}")
        
        if len(annotation.sample) > display_count:
            print(f"\n... ({len(annotation.sample) - display_count} more annotations)")
        
        # Annotation type distribution
        print("\n" + "=" * 80)
        print("ANNOTATION TYPE DISTRIBUTION")
        print("=" * 80)
        if hasattr(annotation, 'anntype'):
            unique_types = np.unique(annotation.anntype)
            for ann_type in unique_types:
                count = np.sum(annotation.anntype == ann_type)
                print(f"  {ann_type}: {count} occurrences")
        
        return annotation
        
    except Exception as e:
        print(f"Error reading annotations: {e}")
        return None

def main():
    """Main function to display all file contents"""
    record_name = '100'
    db_dir = 'mitdb'
    
    print("\n" + "=" * 80)
    print("MIT-BIH ARRHYTHMIA DATABASE FILE READER")
    print("=" * 80)
    print(f"\nReading record: {record_name}")
    print(f"Database directory: {db_dir}\n")
    
    # Check if files exist
    hea_file = os.path.join(db_dir, f"{record_name}.hea")
    dat_file = os.path.join(db_dir, f"{record_name}.dat")
    atr_file = os.path.join(db_dir, f"{record_name}.atr")
    
    print("Checking files...")
    print(f"  Header file (.hea): {'[OK] Found' if os.path.exists(hea_file) else '[X] Not found'}")
    print(f"  Data file (.dat): {'[OK] Found' if os.path.exists(dat_file) else '[X] Not found'}")
    print(f"  Annotation file (.atr): {'[OK] Found' if os.path.exists(atr_file) else '[X] Not found'}")
    print()
    
    # Read and display header
    record = display_header_info(record_name, db_dir)
    
    # Read and display signal data
    if record is not None:
        display_signal_data(record, num_samples=50)  # Show first 50 samples
    
    # Read and display annotations
    display_annotations(record_name, db_dir, max_annotations=50)  # Show first 50 annotations
    
    print("\n" + "=" * 80)
    print("END OF FILE DISPLAY")
    print("=" * 80)

if __name__ == "__main__":
    main()


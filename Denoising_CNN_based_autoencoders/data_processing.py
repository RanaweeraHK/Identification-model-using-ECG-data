import os
import numpy as np
import wfdb
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import neurokit2 as nk

dir_path = 'mit-bih-arrhythmia-database-1.0.0'

def normalize_signal(ecg_signal):
    scaler = MinMaxScaler()
    ecg_signal_normalized = scaler.fit_transform(ecg_signal.reshape(-1, 1)).flatten()
    return ecg_signal_normalized

def generate_baseline_wander_noise(ecg_signal, t):
    A = np.random.uniform(0, 0.15) * np.abs((np.max(ecg_signal) - np.min(ecg_signal)))
    w = 2 * np.pi * np.random.uniform(0.15, 0.3)
    phi = np.random.uniform(-np.pi, np.pi)
    return A * np.sin(w * t + phi).reshape(-1)

def generate_power_line_interference(ecg_signal, t):
    A = np.random.uniform(0, 0.5) * np.abs((np.max(ecg_signal) - np.min(ecg_signal)))
    w = 2 * np.pi * np.random.uniform(49.8, 50.2)
    phi = np.random.uniform(-np.pi, np.pi)
    return A * np.sin(w * t + phi).reshape(-1)

def generate_muscle_artefacts(ecg_signal, t):
    A = np.random.uniform(0, 0.1) * np.abs((np.max(ecg_signal) - np.min(ecg_signal)))
    w = 2 * np.pi * np.random.uniform(0, 10000)
    phi = np.random.uniform(-np.pi, np.pi)
    return A * np.sin(w * t + phi).reshape(-1)

def add_artifacts(ecg_signal, t):
    baseline_wander = generate_baseline_wander_noise(ecg_signal, t)
    power_line_interference = generate_power_line_interference(ecg_signal, t)
    muscle_artifacts = generate_muscle_artefacts(ecg_signal, t)
    noisy_signal = ecg_signal + baseline_wander + power_line_interference + muscle_artifacts
    return noisy_signal

def process_original_records(record_list, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    all_signals = []
    for record in record_list:
        signal, fields = wfdb.rdsamp(os.path.join(dir_path, record))
        ecg_signal = signal[:, 0]  
        ecg_signal = normalize_signal(ecg_signal)
        ecg_signal = nk.ecg_clean(ecg_signal, sampling_rate=360)
        t = np.arange(len(ecg_signal)) / fields['fs']  
        
        all_signals.append(ecg_signal.reshape(-1, 1))
        
        np.savetxt(os.path.join(output_dir, record + '_original.dat'), ecg_signal)
    
    return np.array(all_signals)

def process_noisy_records(record_list, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    all_signals = []
    for record in record_list:
        signal, fields = wfdb.rdsamp(os.path.join(dir_path, record))
        ecg_signal = signal[:, 0]  
        ecg_signal = normalize_signal(ecg_signal)
        ecg_signal = nk.ecg_clean(ecg_signal, sampling_rate=360)
        t = np.arange(len(ecg_signal)) / fields['fs']  
        
        noisy_signal = add_artifacts(ecg_signal, t)
        all_signals.append(noisy_signal.reshape(-1, 1))
        
        np.savetxt(os.path.join(output_dir, record + '_noisy.dat'), noisy_signal)
        
        plt.figure(figsize=(12, 6))
        plt.plot(t[:300], ecg_signal[:300], label='Original ECG')
        plt.plot(t[:300], noisy_signal[:300], label='Noisy ECG', alpha=0.7)
        plt.title(f'Noisy ECG Signal for {record}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.savefig(os.path.join(output_dir, record + '_noisy.png'))
        plt.close()
    
    return np.array(all_signals)
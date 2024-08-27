import numpy as np
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from data_processing import process_original_records, process_noisy_records
from sklearn.model_selection import train_test_split
from autoencoder import build_autoencoder

input_shape = (650000, 1)

dir_path = 'mit-bih-arrhythmia-database-1.0.0'

records = [f for f in os.listdir(dir_path) if f.endswith('.dat')]
records = [os.path.splitext(f)[0] for f in records]

train_records, test_records = train_test_split(records, test_size=0.3, random_state=42)
valid_records, test_records = train_test_split(test_records, test_size=0.5, random_state=42)


# original data
X_train = process_original_records(train_records, 'Dataset/original_data/train')
X_valid = process_original_records(valid_records, 'Dataset/original_data/valid')
X_test = process_original_records(test_records, 'Dataset/original_data/test')

# noisy data
X_train_noisy = process_noisy_records(train_records, 'Dataset/noisy_data/train')
X_valid_noisy = process_noisy_records(valid_records, 'Dataset/noisy_data/valid')
X_test_noisy = process_noisy_records(test_records, 'Dataset/noisy_data/test')

print("Dataset is created")

X_train = X_train.reshape(-1, 650000, 1)
X_valid  = X_valid.reshape(-1, 650000, 1)
X_test  = X_test.reshape(-1, 650000, 1)

X_train_noisy = X_train_noisy.reshape(-1,650000,1)
X_valid_noisy = X_valid_noisy.reshape(-1,650000,1)
X_test_noisy = X_test_noisy.reshape(-1,650000,1)

print("Dataset is reshaped")

input_shape = (650000, 1)


autoencoder = build_autoencoder(input_shape)
autoencoder.compile(optimizer='adam', loss='mse')

print("Model is compiled")
history = autoencoder.fit(
    X_train_noisy, X_train,
    epochs=1000,
    batch_size=8,
    validation_data=(X_valid_noisy, X_valid),
    verbose=1
)

print("Model is trained")
autoencoder.save('Model/autoencoder_cnn.h5')

print("Model is saved")

# Denoiser Model for ECG Signal

## 1. Autoencoder-based denoiser model

#### Model architecture

![alt text](<Images/Encoder (3).png>)

1. **Encoder**: 
    - A series of 1D convolutional layers followed by max-pooling layers.
    - Extracts features from the input ECG signal.

![Encoder](Images/image.png)

2. **Bottleneck**: 
    - A single 1D convolutional layer followed by max-pooling.
    - Compresses the encoded features into a lower-dimensional space.

![Bottleneck](Images/image-1.png)

3. **Decoder**: 
    - A series of 1D convolutional layers followed by upsampling layers.
    - Reconstructs the denoised signal from the compressed features.

![Decoder](Images/image-2.png)

This project includes a folder named Denoising_CNN_based_autoencoders, which contains all the necessary files for training the model. If you need to retrain the model, simply clone this folder and run the main.py script.

The trained model is already provided and can be found at Model\denoising_autoencoder_cnn.h5.


![alt text](Images/reconstructed.png>)
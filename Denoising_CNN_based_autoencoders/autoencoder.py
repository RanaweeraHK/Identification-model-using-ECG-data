import tensorflow as tf
from tensorflow.keras import layers, models

#Encoder
def build_encoder(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Conv1D(16, 3, padding='same', activation='relu')(x)
    encoded = layers.MaxPooling1D(2, padding='same')(x)
    return models.Model(inputs, encoded, name='encoder')

# Bottleneck
def build_bottleneck(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv1D(8, 3, padding='same', activation='relu')(inputs)
    bottleneck = layers.MaxPooling1D(2, padding='same')(x)
    
    return models.Model(inputs, bottleneck, name='bottleneck')

# Decoder
def build_decoder(encoded_shape):
    inputs = layers.Input(shape=encoded_shape)
    
    # Decoder layers
    x = layers.Conv1D(8, 3, padding='same', activation='relu')(inputs)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(16, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling1D(2)(x)
    
    # If necessary, use additional layers or cropping to adjust the final shape
    x = layers.Conv1D(1, 3, padding='same', activation='sigmoid')(x)
    
    # Ensure the final output shape matches the input shape
    # Calculate the required final upsampling or cropping to achieve the target shape
    final_shape = 650000
    x = layers.Lambda(lambda x: x[:, :final_shape, :])(x)  # Cropping if the output is larger
    
    return models.Model(inputs, x, name='decoder')



# Full Autoencoder Model
def build_autoencoder(input_shape):
    encoder = build_encoder(input_shape)
    bottleneck = build_bottleneck(encoder.output_shape[1:])
    decoder = build_decoder(bottleneck.output_shape[1:])
    
    inputs = layers.Input(shape=input_shape)
    encoded = encoder(inputs)
    bottleneck_output = bottleneck(encoded)
    decoded = decoder(bottleneck_output)
    
    return models.Model(inputs, decoded, name='autoencoder')
# Robust Person Identification model uisng Single Chnanel ECG data

In this project, we developed identification models using single channel ECG data. ECG signals can often be affected by noise from devices and changes from external factors, but our approach creates reliable models without needing any preprocessing or noise removal. This makes it faster and more practical for real-time applications.

## 1. Model overview

In our model, we designed a CNN to serve as the embedding generator, which learns meaningful representations from the ECG data. The model’s weights and parameters are continuously refined and updated using a Siamese network, optimized through contrastive loss. This combination allows the network to better distinguish between similar and different ECG patterns, enhancing identification accuracy.

### Training phase of the embedding generator model
![Weekly updates](https://github.com/user-attachments/assets/5f13622c-2c71-4f9e-ad86-f6407f0383db)

### Embedding Generator Model(CNN model) summary
![image](https://github.com/user-attachments/assets/7382e99c-712e-499c-80d5-82d12c845045)

### Datasets
1. [MIT-BIH Arrhythmia dataset](https://physionet.org/content/mitdb/1.0.0/)
2. [PTB Diagnostic dataset](https://physionet.org/content/ptbdb/1.0.0/)
3. [St-Petersburg dataset](https://physionet.org/content/incartdb/1.0.0/)

* The training process for the embedding generator model is available in `embedding_generator.ipynb`, and the trained model is saved as `embedding_generator.h5`.

For generating embeddings from the embedding generator model for person identification, we employed three classifiers,
1) K Nearest Neighbors
2) XGBoost
3) Neural Network.



- [Dashboard](https://github.com/ParameswaranSajeenthiran/ECGAnalysisDashbaord)
- [Dashboard Code](https://github.com/ParameswaranSajeenthiran/ECGAnalysisDashbaord)


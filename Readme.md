# Robust Person Identification model uisng Single Chnanel ECG data

In this project, we developed identification models using single channel ECG data. ECG signals can often be affected by noise from devices and changes from external factors, but our approach creates reliable models without needing any preprocessing or noise removal. This makes it faster and more practical for real-time applications.

##  Model overview

In our model, we designed a CNN to serve as the embedding generator, which learns meaningful representations from the ECG data. The model’s weights and parameters are continuously refined and updated using a Siamese network, optimized through contrastive loss. This combination allows the network to better distinguish between similar and different ECG patterns, enhancing identification accuracy.



### Notebooks
1. [Training & Testing ](https://github.com/RanaweeraHK/Identification-model-using-ECG-data/blob/main/Embedding_Generator_v20.ipynb)
2. [Testing with synthetic Noise Dataset ](https://github.com/RanaweeraHK/Identification-model-using-ECG-data/blob/main/Testing%20synthetically%20Distorted%20dataset.ipynb)




### Training phase of the embedding generator model
![Weekly updates](https://github.com/user-attachments/assets/5f13622c-2c71-4f9e-ad86-f6407f0383db)

### Embedding Generator Model(CNN model) summary
![image](https://github.com/user-attachments/assets/7382e99c-712e-499c-80d5-82d12c845045)

### Datasets
1. [MIT-BIH Arrhythmia dataset](https://physionet.org/content/mitdb/1.0.0/)
2. [PTB Diagnostic dataset](https://physionet.org/content/ptbdb/1.0.0/)
3. [St-Petersburg dataset](https://physionet.org/content/incartdb/1.0.0/)

* We used the MIT-BIH dataset for training. The training process for the embedding generator model is documented in embedding_generator.ipynb, and the trained model is saved as embedding_generator.h5.


For generating embeddings from the embedding generator model for person identification, we employed three classifiers,
1) K Nearest Neighbors
2) XGBoost
3) Neural Network
   
![image](https://github.com/user-attachments/assets/5bf9dc96-6880-4a9a-b7b2-e4540a1c247d)
---
   
* As part of an experiment, we combined each dataset and trained the model, which is detailed in [combined-dataset-training.ipynb](combined-dataset-training.ipynb).
* To create the embedding generator model, we utilized the raw, original datasets without any preprocessing or noise removal techniques. However, to ensure a fair comparison with state-of-the-art models, we subsequently applied their preprocessing methods and recorded the results, which can be found in [preprocessing-pipelines.ipynb](preprocessing-pipelines.ipynb)
* As part of our experiments, we also developed a denoiser model.[Denoiser](Denoising_CNN_based_autoencoders)

---
As a byproduct of our project, we developed a dashboard to display our experimental results. You can check it out here.
- [Dashboard](https://github.com/ParameswaranSajeenthiran/ECGAnalysisDashbaord)
- [Dashboard Code](https://github.com/ParameswaranSajeenthiran/ECGAnalysisDashbaord)

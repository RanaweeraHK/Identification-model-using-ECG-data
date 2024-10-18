# Robust Person Identification model uisng Single Chnanel ECG data

In this project, we developed identification models using single channel ECG data. ECG signals can often be affected by noise from devices and changes from external factors, but our approach creates reliable models without needing any preprocessing or noise removal. This makes it faster and more practical for real-time applications.

## 1. Model overview

In our model, we designed a CNN to serve as the embedding generator, which learns meaningful representations from the ECG data. The model’s weights and parameters are continuously refined and updated using a Siamese network, optimized through contrastive loss. This combination allows the network to better distinguish between similar and different ECG patterns, enhancing identification accuracy.

#### Training phase of the embedding genrator model
![Weekly updates](https://github.com/user-attachments/assets/5f13622c-2c71-4f9e-ad86-f6407f0383db)


- [Dashboard](https://github.com/ParameswaranSajeenthiran/ECGAnalysisDashbaord)
- [Dashboard Code](https://github.com/ParameswaranSajeenthiran/ECGAnalysisDashbaord)


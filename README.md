# Credit Card Anomaly Detection

## Overview

This project aims to detect credit card fraud using Deep Autoencoders. The dataset contains transactions over a two-day period with 492 frauds out of 284,807 transactions. Due to privacy concerns, all features, except for "Time" and "Amount," have undergone PCA transformations.

## Technologies Used

- Python
- Keras
- TensorBoard
- Deep Learning
- Data Visualization (matplotlib)

## Data Preprocessing

1. **Loading the Data**: The dataset, which can be downloaded from Kaggle, comprises credit card transactions spanning two days.
2. **Data Transformation**: Dropped the "Time" column and scaled the "Amount" column to unit variance using `StandardScaler`.

## Model Building

Constructed a Deep Autoencoder using Keras, which consists of:
- 4 fully connected layers with 14, 7, 7, and 29 neurons.
- First two layers act as the encoder, and the last two are the decoder.
- L1 regularization was employed during training.

The model was trained for 100 epochs with a batch size of 32 samples. The best-performing model was saved using Keras's `ModelCheckpoint`. Training progress can be monitored using TensorBoard.

## Fraud Prediction

The model's reconstruction error determines if a transaction is fraudulent or not. Transactions with errors exceeding a predefined threshold are labeled as fraud. This threshold was empirically set at 2.9 for this project.

## Visualization

A plot was created using matplotlib to visually depict the reconstruction error for both normal and fraudulent transactions. The threshold is also plotted for better visualization.

## Conclusion

The Deep Autoencoder successfully reconstructed what non-fraudulent transactions look like. Despite the model's simplicity, it achieved significant results in distinguishing between fraudulent and non-fraudulent transactions. However, there's room for improvement, especially in reducing false positives.

## Future Improvements

- Experiment with different threshold values to optimize the balance between false positives and true positives.
- Integrate more advanced visualization tools for better interpretability.
- Explore other deep learning architectures and compare performance.

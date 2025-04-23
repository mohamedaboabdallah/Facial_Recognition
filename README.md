
# Emotion Classification with ResNet-18 and FER-2013 Dataset

This document provides a detailed, step-by-step explanation of a deep learning pipeline implemented in PyTorch and TensorFlow for classifying emotions using the FER-2013 dataset. The model architecture is based on a pre-trained ResNet-18, with custom preprocessing, augmentation, loss function, and training strategies.

---

## 1. **Libraries and Technologies Used**

- **TensorFlow & Keras**: Used for initial data preprocessing and visualization.
- **PyTorch**: The primary deep learning framework used for model building, training, and evaluation.
- **OpenCV (cv2)**: Used for reading and preprocessing image data.
- **Matplotlib & Seaborn**: Used for plotting visualizations such as class distributions and confusion matrices.
- **Scikit-learn**: Used for computing class weights, splitting data, and performance metrics.
- **NumPy**: Core numerical computing for data manipulation.

---

## 2. **Data Preparation**

- The FER-2013 dataset is structured into labeled folders for each emotion (angry, happy, etc.).
- Grayscale images are resized to 48x48 pixels.
- Labels are mapped from string labels to integer indices.
- Training and test datasets are loaded separately, and pixel values are normalized to the range [0, 1].

---

## 3. **Validation Set Construction**

- The validation set is manually crafted to match the distribution of classes in the test set.
- Class-specific sample counts are used to create a balanced validation set from the training data.
- Remaining training data is used as the final training set.
- One-hot encoding is applied to all labels.

---

## 4. **Data Visualization**

- Class distributions for training, validation, and test sets are visualized using count plots.
- Example images from each class are displayed for visual inspection and sanity checking.

---

## 5. **Class Weights for Imbalance Handling**

- Class weights are computed using the `compute_class_weight` function to address class imbalance.
- These weights are later used in the loss function to give higher importance to underrepresented classes.

---

## 6. **Dataset Class in PyTorch**

- A custom `Dataset` class is defined to convert NumPy arrays into PyTorch-compatible format.
- Data augmentations such as horizontal flip, rotation, affine transformation, and color jitter are applied to training images.
- Evaluation transforms include resizing and normalization without augmentation.

---

## 7. **Model Architecture**

- A pre-trained **ResNet-18** model is used, modified to fit 7 emotion classes.
- The final fully connected layer is replaced with a dropout layer followed by a linear layer for classification.

---

## 8. **Loss Function and Optimization**

- A custom **Label Smoothing CrossEntropy Loss** is used to make the model less confident on its predictions and reduce overfitting.
- **Adam optimizer** is used with an initial learning rate of 0.001.
- A **ReduceLROnPlateau** scheduler monitors validation accuracy and reduces the learning rate when the metric plateaus.

---

## 9. **Training Strategy**

- The model is trained for a maximum of 40 epochs with early stopping enabled.
- Training includes logging of loss and accuracy for each epoch.
- If validation accuracy improves, the model is saved.
- If validation accuracy does not improve for 5 consecutive epochs, training is stopped early.

---

## 10. **Evaluation and Performance Metrics**

- After training, the model is evaluated on the test set.
- Predictions are compared with ground-truth labels.
- A confusion matrix is generated to visualize class-wise performance.
- A classification report (precision, recall, F1-score) is generated for detailed performance analysis.

---
# 📘 SVM Pipeline with CNN Feature Extraction and PCA

This project implements a classification pipeline using a Support Vector Machine (SVM) trained on features extracted from images via a CNN-based feature extractor. Dimensionality reduction was applied with PCA, and hyperparameters were tuned using Grid Search.

---

## 📦 1. Feature Extraction

- **Input Data**: `train_loader`, `val_loader`, and `test_loader` (PyTorch DataLoaders).
- **Method**: A pretrained CNN (e.g., ResNet) was used as a **feature extractor** to convert input images into high-dimensional feature vectors.
- **Output**: `X_train`, `X_val`, `X_test` along with corresponding labels.

---

## ⚖️ 2. Feature Scaling

- **Tool Used**: `StandardScaler` from `sklearn.preprocessing`.
- **Purpose**: Standardized features to have zero mean and unit variance, which helps SVM perform better.
- Applied:
  - `fit_transform()` on training set
  - `transform()` on validation and test sets

---

## 🔻 3. Dimensionality Reduction with PCA

- **Tool Used**: `PCA` from `sklearn.decomposition`.
- **Goal**: Reduce dimensionality while retaining **95%** of the variance.
- **Result**: PCA-transformed features (`X_train_pca`, `X_val_pca`, `X_test_pca`).

---

## 🔍 4. Hyperparameter Tuning with Grid Search

- **Model**: SVM (`SVC`) with a **linear kernel**.
- **Search Space**: `C ∈ [0.001, 0.01, 0.1, 1.0, 10.0]`
- **Method**: `GridSearchCV` with 5-fold cross-validation.
- **Best Parameter**: `C = 0.001`

---

## ✅ 5. Model Evaluation on Validation Set

- **Accuracy**: `91.91%`
- **Classification Report**:
  - High precision and recall for most classes
  - F1-score ranged between `0.87` to `0.97`
- **Tool**: `classification_report`, `confusion_matrix`, and `seaborn` heatmap for visualization.

---

## 🧪 6. Model Evaluation on Test Set

- **Accuracy**: `69.18%`
- **Classification Report**:
  - Class 3 was best predicted (F1 = 0.88)
  - Classes 2 and 4 showed poorer performance (~0.54 F1)
- **Possible Reasons for Drop**:
  - Domain shift between validation and test sets
  - Some overfitting, even with PCA and regularization

---

## 💾 7. Saving the Artifacts

- Model (`best_svm`): `svm_cnn_with_pca_grid.pkl`
- Scaler: `feature_scaler.pkl`
- PCA transformer: `pca_transform.pkl`
- Saved using `joblib.dump`

---

## 🔁 8. Additional Experiments

- **Tried Models**:
  - **ResNet18** as an alternative feature extractor
  - Applied both **Logistic Regression** and **SVM** on features extracted from both CNNs
- **Result**:
  - The combination of the selected CNN feature extractor (not ResNet18) + PCA + **SVM with `C=0.001`** yielded the **best performance**
  - ResNet18 and Logistic Regression combinations did not outperform this configuration on validation or test metrics

---

## 📝 Final Notes

- The pipeline demonstrates the power of combining deep feature extraction with classical ML techniques.
- PCA significantly reduced the feature space while maintaining most of the variance.
- Hyperparameter tuning was key to improving validation performance.
- Test results suggest further tuning or more robust features may be needed for real-world deployment.


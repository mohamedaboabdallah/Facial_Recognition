import cv2
import os
import torch
import joblib
import numpy as np
import time
from torchvision import models, transforms
from collections import Counter
import torch.nn as nn

# -------- Settings --------
NUM_CLASSES = 7
time_interval = 2.0  # seconds between prediction display updates

# -------- Paths --------
current_directory = os.getcwd()
model_dir = os.path.join(current_directory, 'models', 'resnet18_40epoch_svm')

# -------- Device --------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Load CNN Model for Feature Extraction --------
cnn_model = models.resnet18(pretrained=False)
cnn_model.fc = nn.Identity()  # Remove classifier head to extract features only
cnn_model.load_state_dict(torch.load(os.path.join(model_dir, 'resnet18_40epoch.pth'), map_location=DEVICE), strict=False)
cnn_model.to(DEVICE).eval()

# -------- Load Scaler, PCA, SVM --------
scaler = joblib.load(os.path.join(model_dir, "feature_scaler.pkl"))
pca = joblib.load(os.path.join(model_dir, "pca_transform.pkl"))
svm_model = joblib.load(os.path.join(model_dir, "svm_cnn_with_pca_grid.pkl"))

# -------- Preprocessing --------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Match CNN input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features_from_frame(frame, model):
    with torch.no_grad():
        input_tensor = transform(frame).unsqueeze(0).to(DEVICE)  # [1, 3, 224, 224]
        features = model(input_tensor)
        return features.view(-1).cpu().numpy()

# -------- Webcam & Prediction Loop --------
cap = cv2.VideoCapture(0)

predictions = []
window_start_time = time.time()
displayed_class = "..."

print("Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Extract features and predict
    try:
        features = extract_features_from_frame(frame, cnn_model)
        features_scaled = scaler.transform([features])
        features_pca = pca.transform(features_scaled)
        prediction = svm_model.predict(features_pca)[0]
        print("Predicted:", prediction)
        predictions.append(prediction)
    except Exception as e:
        print("Prediction error:", e)

    # Every 2 seconds, get most frequent prediction
    current_time = time.time()
    if current_time - window_start_time >= time_interval and predictions:
        most_common = Counter(predictions).most_common(1)[0][0]
        print("Most common class in last window:", most_common)
        displayed_class = most_common
        predictions = []  # reset
        window_start_time = current_time

    # Display prediction
    cv2.putText(frame, f"Prediction: {displayed_class}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Webcam Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

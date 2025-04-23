import cv2
import os
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
import joblib
import time
from collections import Counter
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------- Settings --------
NUM_CLASSES = 7
time_interval = 1.0

# -------- Paths --------
current_directory = os.getcwd()
model_dir = os.path.join(current_directory, 'models', 'resnet18_40epoch_svm')

# -------- Device --------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Load CNN Model with Dropout --------
class ResNetWithDropout(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.base_model = models.resnet18(pretrained=True)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, NUM_CLASSES)
        )

    def forward(self, x):
        return self.base_model(x)

cnn_model = ResNetWithDropout(dropout_rate=0.3)
cnn_model_path = os.path.join(model_dir, 'resnet18_40epoch.pth')
state_dict = torch.load(cnn_model_path, map_location=DEVICE)
cnn_model.load_state_dict(state_dict)
cnn_model.to(DEVICE).eval()

# -------- Feature Extractor --------
feature_extractor = nn.Sequential(*list(cnn_model.base_model.children())[:-1])
feature_extractor.to(DEVICE).eval()

# -------- Load SVM Model --------
svm_model = joblib.load(os.path.join(model_dir, "svm_cnn_with_pca_grid.pkl"))
scaler = joblib.load(os.path.join(model_dir, "feature_scaler.pkl"))
pca = joblib.load(os.path.join(model_dir, "pca_transform.pkl"))

# -------- Preprocessing Functions --------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_fer_style(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    resized = cv2.resize(gray, (48, 48))
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    pil_image = Image.fromarray(rgb)  # Convert to PIL
    return pil_image

def extract_features_from_frame(frame, model):
    with torch.no_grad():
        processed = preprocess_fer_style(frame)
        input_tensor = transform(processed).unsqueeze(0).to(DEVICE)
        features = model(input_tensor)
        return features.view(-1).cpu().numpy()

# -------- Webcam Loop --------
cap = cv2.VideoCapture(0)
predictions = []
window_start_time = time.time()
displayed_class = "..."

print("Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    try:
        features = extract_features_from_frame(frame, feature_extractor)
        features = features.reshape(1, -1)  # Reshape for model input
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)
        prediction = svm_model.predict(features_pca)
        prediction = int(prediction[0])
        predictions.append(prediction)
    except Exception as e:
        print(f"Prediction error: {e}")
        if 'features' in locals():
            print(f"Type of features: {type(features)}, Shape: {getattr(features, 'shape', None)}")

    current_time = time.time()
    if current_time - window_start_time >= time_interval and predictions:
        most_common = Counter(predictions).most_common(1)[0][0]
        displayed_class = most_common
        predictions = []
        window_start_time = current_time

    # Emotion Labels Map
    label_map_inv = {
        0: 'Angry', 1: 'Disgust', 2: 'Fear', 
        3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'
    }
    label_name = label_map_inv.get(displayed_class, str(displayed_class))
    cv2.putText(frame, f"Prediction: {label_name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Webcam Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

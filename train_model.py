import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("cleaned_smartphones.csv")

# -------------------------
# Features for Recommendation
# -------------------------
features = [
    "price",
    "5G_or_not",
    "battery_capacity",
    "fast_charging",
    "ram_capacity",
    "internal_memory",
    "refresh_rate",
    "primary_camera_rear",
    "primary_camera_front",
    "screen_size",
    "processor_brand",
    "os"
]

X = df[features].copy()

# -------------------------
# Encode Text Columns
# -------------------------
processor_encoder = LabelEncoder()
os_encoder = LabelEncoder()

X["processor_brand"] = processor_encoder.fit_transform(X["processor_brand"])
X["os"] = os_encoder.fit_transform(X["os"])

# -------------------------
# Scale Features
# -------------------------
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# -------------------------
# Train KNN Model
# -------------------------
knn = NearestNeighbors(
    n_neighbors=5,
    metric="euclidean"
)

knn.fit(X_scaled)

# -------------------------
# Save Everything
# -------------------------
joblib.dump(knn, "knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(processor_encoder, "processor_encoder.pkl")
joblib.dump(os_encoder, "os_encoder.pkl")

print("✅ Model trained successfully!")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors

# Load dataset
df = pd.read_csv("cleaned_smartphones.csv")

# Features used for KNN
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

# Encode categorical features
processor_encoder = LabelEncoder()
os_encoder = LabelEncoder()

X["processor_brand"] = processor_encoder.fit_transform(X["processor_brand"])
X["os"] = os_encoder.fit_transform(X["os"])

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature weights
weights = [5,2,3,2,4,4,2,3,1,1,3,2]
X_scaled = X_scaled * weights

# Train KNN
knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
knn.fit(X_scaled)

# -------------------------------
# User Input
# -------------------------------
user = pd.DataFrame([{
    "price":30000,
    "5G_or_not":1,
    "battery_capacity":5000,
    "fast_charging":67,
    "ram_capacity":8,
    "internal_memory":128,
    "refresh_rate":120,
    "primary_camera_rear":50,
    "primary_camera_front":16,
    "screen_size":6.6,
    "processor_brand":processor_encoder.transform(["snapdragon"])[0],
    "os":os_encoder.transform(["android"])[0]
}])

user_scaled = scaler.transform(user)
user_scaled = user_scaled * weights

# Find nearest neighbors
distances, indices = knn.kneighbors(user_scaled)

# -------------------------------
# Visualization
# -------------------------------
plt.figure(figsize=(10,6))

# All phones
plt.scatter(
    df["price"],
    df["ram_capacity"],
    color="lightgray",
    label="All Phones"
)

# Recommended phones
recommended = df.iloc[indices[0]]

plt.scatter(
    recommended["price"],
    recommended["ram_capacity"],
    color="green",
    s=120,
    label="Recommended Phones"
)

# User input
plt.scatter(
    30000,
    8,
    color="red",
    marker="*",
    s=300,
    label="User Input"
)

# Labels for recommended phones
for _, row in recommended.iterrows():
    plt.text(
        row["price"],
        row["ram_capacity"]+0.15,
        row["brand_name"],
        fontsize=8
    )

plt.xlabel("Price (₹)")
plt.ylabel("RAM (GB)")
plt.title("KNN Smartphone Recommendation Visualization")

plt.legend()

plt.grid(True)

plt.show()
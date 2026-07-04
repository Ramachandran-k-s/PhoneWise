import pandas as pd
import joblib

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("cleaned_smartphones.csv")

# -------------------------
# Load Trained Files
# -------------------------
knn = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")
processor_encoder = joblib.load("processor_encoder.pkl")
os_encoder = joblib.load("os_encoder.pkl")
weights = joblib.load("weights.pkl")


def recommend_phone(
    price,
    is5g,
    battery,
    fast_charging,
    ram,
    storage,
    refresh_rate,
    rear_camera,
    front_camera,
    screen_size,
    processor,
    os
):
    # Convert input to lowercase
    processor = processor.lower()
    os = os.lower()

    # Encode text inputs
    processor = processor_encoder.transform([processor])[0]
    os = os_encoder.transform([os])[0]

    # Create DataFrame with same column names used during training
    user = pd.DataFrame([{
        "price": price,
        "5G_or_not": is5g,
        "battery_capacity": battery,
        "fast_charging": fast_charging,
        "ram_capacity": ram,
        "internal_memory": storage,
        "refresh_rate": refresh_rate,
        "primary_camera_rear": rear_camera,
        "primary_camera_front": front_camera,
        "screen_size": screen_size,
        "processor_brand": processor,
        "os": os
    }])

    # Scale the input
    user_scaled = scaler.transform(user)

    # Apply feature weights
    user_scaled = user_scaled * weights

    # Find nearest phones
    distances, indices = knn.kneighbors(user_scaled)

    # Get recommendations
    recommendations = df.iloc[indices[0]]

    # Return selected columns
    return recommendations[
        [
            "brand_name",
            "model",
            "price",
            "avg_rating",
            "ram_capacity",
            "internal_memory",
            "battery_capacity",
            "processor_brand",
            "primary_camera_rear",
            "5G_or_not"
        ]
    ]


# -------------------------
# Test Recommendation
# -------------------------
phones = recommend_phone(
    price=30000,
    is5g=1,
    battery=5000,
    fast_charging=67,
    ram=8,
    storage=128,
    refresh_rate=120,
    rear_camera=50,
    front_camera=16,
    screen_size=6.6,
    processor="snapdragon",
    os="android"
)

print("\nTop 5 Recommended Phones:\n")
print(phones.to_string(index=False))
import pandas as pd
import joblib

# Load dataset
df = pd.read_csv("cleaned_smartphones.csv")

# Load trained files
knn = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")
processor_encoder = joblib.load("processor_encoder.pkl")
os_encoder = joblib.load("os_encoder.pkl")


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

    # Convert user input to lowercase
    processor = processor.lower()
    os = os.lower()

    # Encode text inputs
    processor = processor_encoder.transform([processor])[0]
    os = os_encoder.transform([os])[0]

    # Create input row
    user = [[
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
    ]]

    # Scale input
    user_scaled = scaler.transform(user)

    # Find nearest phones
    distances, indices = knn.kneighbors(user_scaled)

    # Display results
    recommendations = df.iloc[indices[0]]

    return recommendations[[
        "brand_name",
        "model",
        "price",
        "avg_rating"
    ]]


# ------------------------
# Test Recommendation
# ------------------------

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

print(phones)
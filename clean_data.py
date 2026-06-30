import pandas as pd

# Load dataset
df = pd.read_csv("smartphones.csv")

# Fill missing numerical values with median
df["avg_rating"] = df["avg_rating"].fillna(df["avg_rating"].median())
df["num_cores"] = df["num_cores"].fillna(df["num_cores"].median())
df["processor_speed"] = df["processor_speed"].fillna(df["processor_speed"].median())
df["battery_capacity"] = df["battery_capacity"].fillna(df["battery_capacity"].median())
df["primary_camera_front"] = df["primary_camera_front"].fillna(df["primary_camera_front"].median())

# Fill missing categorical values with mode
df["processor_brand"] = df["processor_brand"].fillna(df["processor_brand"].mode()[0])
df["os"] = df["os"].fillna(df["os"].mode()[0])

# Phones without fast charging get 0
df["fast_charging"] = df["fast_charging"].fillna(0)

# Save cleaned dataset
df.to_csv("cleaned_smartphones.csv", index=False)

print("Dataset cleaned successfully!")
print(df.isnull().sum())
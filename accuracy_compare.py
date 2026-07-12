import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv("cleaned_smartphones.csv")

# Create target: High-rated phone (1) or not (0)
df["target"] = (df["avg_rating"] >= 8.0).astype(int)

# Features
features = [
    "price",
    "5G_or_not",
    "battery_capacity",
    "fast_charging",
    "ram_capacity",
    "internal_memory",
    "screen_size",
    "refresh_rate",
    "primary_camera_rear",
    "primary_camera_front",
    "processor_brand",
    "os"
]

X = df[features]
y = df["target"]

# Encode categorical features
le_processor = LabelEncoder()
le_os = LabelEncoder()

X["processor_brand"] = le_processor.fit_transform(X["processor_brand"])
X["os"] = le_os.fit_transform(X["os"])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
models = {
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC()
}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"{name}: {acc*100:.2f}%")
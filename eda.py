import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("cleaned_smartphones.csv")

# -----------------------------
# 1. Top 10 Smartphone Brands
# -----------------------------
top_brands = df["brand_name"].value_counts().head(10)

plt.figure(figsize=(10,5))
top_brands.plot(kind="bar")
plt.title("Top 10 Smartphone Brands")
plt.xlabel("Brand")
plt.ylabel("Number of Phones")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("top_brands.png")
plt.show()

# -----------------------------
# 2. Price Distribution
# -----------------------------
plt.figure(figsize=(8,5))
plt.hist(df["price"], bins=20)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("price_distribution.png")
plt.show()

# -----------------------------
# 3. RAM vs Price
# -----------------------------
plt.figure(figsize=(8,5))
plt.scatter(df["ram_capacity"], df["price"])
plt.title("RAM vs Price")
plt.xlabel("RAM (GB)")
plt.ylabel("Price")
plt.tight_layout()
plt.savefig("ram_vs_price.png")
plt.show()

# -----------------------------
# 4. Battery Capacity Distribution
# -----------------------------
plt.figure(figsize=(8,5))
plt.hist(df["battery_capacity"], bins=15)
plt.title("Battery Capacity")
plt.xlabel("Battery (mAh)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("battery_distribution.png")
plt.show()

print("EDA Completed!")
import pandas as pd

df = pd.read_csv("cleaned_smartphones.csv")

print(df["processor_brand"].unique())
print()
print(df["os"].unique())
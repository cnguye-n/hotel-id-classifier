import pandas as pd

df = pd.read_csv("full_train_set.csv")
sampled = df.sample(n=50000, random_state=42)
sampled.to_csv("50k_train_set.csv", index=False)

print("Created train_set.csv with 50k rows")

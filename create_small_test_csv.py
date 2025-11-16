import pandas as pd

df = pd.read_csv("full_test_set.csv")
sampled = df.sample(n=5000, random_state=42)   # choose any number you want
sampled.to_csv("5k_test_set.csv", index=False)

print("Created 5k_test_set.csv with 5k rows")

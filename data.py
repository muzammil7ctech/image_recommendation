import pandas as pd
df = pd.read_csv("data.csv")
df = df["product_link"]

df = df.apply(path)
df.to_csv("check_function.csv")
print(df)
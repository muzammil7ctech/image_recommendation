
#  ------------  load_csv function check ----------------------- #
import csv
import pandas as pd

data = []

# Open the CSV file and read its contents
with open("productsandimages.csv", newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    for row in reader:
        data.append(row)

df = pd.DataFrame(data)

df = df.loc[0, ['product_id', 'product_thumbnail']]

print(df)

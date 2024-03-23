from Extract_code.lib.libraries import *

def load_csv(path):
    data = []

    # Open the CSV file and read its contents
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            data.append(row)

    df = pd.DataFrame(data)
    print(df.columns)
    print(df.head(5))
    print("#------------------------working load_csv function----------------------------#")
    
    return df


from Extract_code.lib.libraries import *

def load_csv(path):
    df = pd.read_csv(path)

    print(df.columns)
    print(df.head(5))
    print("#------------------------working load_csv function----------------------------#")
    
    return df


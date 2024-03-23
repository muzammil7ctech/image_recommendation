from Extract_code.lib.libraries import *

def filter_column(df):
    print(df.head(5))
    print("Columns in DataFrame:")
    print(df.columns)

    # Select specific columns after DataFrame creation
    df = df[['product_id', 'product_thumbnail']]

    print("Columns after filtering:")
    print(df.columns)
    return df
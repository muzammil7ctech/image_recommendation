from Extract_code.lib.libraries import *

def filter_column(df,column = None):

    df =df[column]
    
    df = df.drop_duplicates()


    print("--------------------------<< filter_column function work >>--------------------------------")


    return df
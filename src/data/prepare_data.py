import pandas as pd

def prepare_data(df: pd.DataFrame):

    df = df.drop(columns=['id'])
    df = df.rename(columns={'Class': 'label'})

    return df
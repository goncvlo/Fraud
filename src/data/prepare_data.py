import pandas as pd

def prepare_data(df: pd.DataFrame):

    df = df.rename(columns={'Class': 'label'})
    df.columns = [col.lower() for col in df.columns]

    return df
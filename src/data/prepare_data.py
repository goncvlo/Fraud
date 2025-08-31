import pandas as pd

def prepare_data(df: pd.DataFrame):

    df = df.rename(columns={'Class': 'label'})
    df["label"] = df["label"].astype("float")
    df.columns = [col.lower() for col in df.columns]
    for col in ['time', 'amount']:
        df[col] = (df[col]-df[col].mean())/df[col].std()

    return df
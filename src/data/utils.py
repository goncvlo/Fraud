import pandas as pd
from sklearn.utils import resample

def resample_data(df: pd.DataFrame, pos_share: float):
    """Resamples obs. based on pos_share param - share of pos. labels. Labels are 1 and 0."""

    # resample positive and negative observations
    df_pos, df_neg = [
        resample(
            df[df['target'] == i]
            , replace=False
            , n_samples=int(
                (pos_share*i+(1-pos_share)*(1-i)) * df[df['target'] == i].shape[0]
                )
            , random_state=42
            )
        for i in [1, 0]
    ]

    # combine positive and negative sampled observations
    df_sampled = pd.concat([df_pos, df_neg])

    return df_sampled

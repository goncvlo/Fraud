import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

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

def train_splits(features: pd.DataFrame, labels: pd.Series, config: dict):
    """Split train set into multiple sets using stratified sampling."""

    train_sets = {}
    train_sets[0] = pd.concat([features, labels], axis=1)

    for i in range(len(config['train_sizes'])):
        
        X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(
            train_sets[0].iloc[:,:-1], train_sets[0].iloc[:,-1]
            , test_size=config['train_sizes'][i]
            , random_state=123
            , shuffle=True
            , stratify=train_sets[0].iloc[:,-1]
        )

        train_sets[i+1] = pd.concat([X_train_1, y_train_1], axis=1)
        train_sets[0] = pd.concat([X_train_2, y_train_2], axis=1)
    
    train_sets[len(config["train_sizes"])+1] = train_sets[0]
    train_sets.pop(0)

    return train_sets

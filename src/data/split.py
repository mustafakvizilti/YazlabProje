import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def split_and_scale_batadal(df):
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    possible_targets = ['ATTACK', 'ATT_FLAG', 'attack', 'label', 'Attack']
    target = next((col for col in possible_targets if col in df.columns), None)
    
    if target is None:
        raise ValueError(f"Hedef sutun bulunamadi! Mevcut sutunlar: {df.columns.tolist()}")

    features = [col for col in df.columns if col not in ['DATETIME', 'datetime', target]]

    scaler = StandardScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    val_df[features] = scaler.transform(val_df[features])
    test_df[features] = scaler.transform(test_df[features])

    pca = PCA(n_components=1)
    train_df['PC1'] = pca.fit_transform(train_df[features])
    val_df['PC1'] = pca.transform(val_df[features])
    test_df['PC1'] = pca.transform(test_df[features])

    return train_df, val_df, test_df, features, target

def get_skab_splits(df):
    features = [col for col in df.columns if col not in ['datetime', 'anomaly', 'changepoint', 'source_group', 'source_file']]
    target = 'anomaly'
    groups = df['source_file'].values

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(df, df[target], groups))

    return df, features, target, splits
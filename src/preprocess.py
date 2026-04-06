import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, OneHotEncoder, TargetEncoder
from sklearn.model_selection import train_test_split
import pickle
from typing import Tuple

def load_data(trans_path: str, id_path: str) -> pd.DataFrame:
    df_trans = pd.read_csv(trans_path)
    df_id = pd.read_csv(id_path)
    merged = df_trans.merge(df_id, on="TransactionID", how="left")
    assert merged.shape[0] == df_trans.shape[0]
    assert df_trans.shape[0] == len(pd.unique(df_trans["TransactionID"]))
    return merged

def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    df['TransactionAmt'] = np.log1p(df['TransactionAmt'])
    df['hour_of_day'] = (df['TransactionDT'] // 3600) % 24
    null_pct = df.isnull().sum() / len(df)
    majorly_null_cols = list(null_pct[null_pct > 0.9].index)
    for col in majorly_null_cols:
        df[f"{col}_present"] = df[col].notna().astype(int)
    return df, majorly_null_cols

def drop_cols(df: pd.DataFrame, cols_to_drop: list) -> pd.DataFrame:
    return df.drop(cols_to_drop, axis="columns")

def fit_scaler(X_train: pd.DataFrame, num_cols: list):
    scaler = RobustScaler()
    scaler.fit(X_train[num_cols])
    return scaler

def apply_scaler(X: pd.DataFrame, scaler: RobustScaler, num_cols: list) -> pd.DataFrame:
    scaled = scaler.transform(X[num_cols])
    scaled_cols = scaler.get_feature_names_out(num_cols)
    scaled_df = pd.DataFrame(scaled, columns=scaled_cols, index=X.index)
    X = X.drop(columns=num_cols)
    return pd.concat([X, scaled_df], axis=1)

def fit_ohe(X_train: pd.DataFrame, low_cardinality_cols: list):
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe.fit(X_train[low_cardinality_cols])
    return ohe

def apply_ohe(X: pd.DataFrame, ohe: OneHotEncoder, low_cardinality_cols: list) -> pd.DataFrame:
    encoded = ohe.transform(X[low_cardinality_cols])
    encoded_cols = ohe.get_feature_names_out(low_cardinality_cols)
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=X.index)
    X = X.drop(columns=low_cardinality_cols)
    return pd.concat([X, encoded_df], axis=1)

def fit_target_encoder(X_train: pd.DataFrame, y_train: pd.Series, high_cardinality_cols: list):
    te = TargetEncoder(random_state=42)
    te.fit(X_train[high_cardinality_cols], y_train)
    return te

def apply_target_encoder(X: pd.DataFrame, te: TargetEncoder, high_cardinality_cols: list) -> pd.DataFrame:
    encoded = te.transform(X[high_cardinality_cols])
    encoded_cols = te.get_feature_names_out(high_cardinality_cols)
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=X.index)
    X = X.drop(columns=high_cardinality_cols)
    return pd.concat([X, encoded_df], axis=1)

if __name__ == "__main__":
    df = load_data(
        trans_path='../data/raw/train_transaction.csv',
        id_path='../data/raw/train_identity.csv'
    )

    df, majorly_null_cols = build_features(df)

    cols_to_drop = ["TransactionDT", "TransactionID", "DeviceInfo"] + majorly_null_cols
    df = drop_cols(df, cols_to_drop)

    X_train, X_temp, y_train, y_temp = train_test_split(
        df.drop('isFraud', axis=1), df['isFraud'],
        test_size=0.05, stratify=df['isFraud'], random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5, stratify=y_temp, random_state=42
    )

    # Scale
    numeric_cols = [c for c in X_train.select_dtypes(exclude='object').columns if not c.endswith("_present")]
    scaler = fit_scaler(X_train, numeric_cols)
    X_train = apply_scaler(X_train, scaler, numeric_cols)
    X_val   = apply_scaler(X_val,   scaler, numeric_cols)
    X_test  = apply_scaler(X_test,  scaler, numeric_cols)

    # OHE
    low_cardinality_cols = [c for c in df.select_dtypes(include='object').columns if df[c].nunique() < 10]
    ohe = fit_ohe(X_train, low_cardinality_cols)
    X_train = apply_ohe(X_train, ohe, low_cardinality_cols)
    X_val   = apply_ohe(X_val,   ohe, low_cardinality_cols)
    X_test  = apply_ohe(X_test,  ohe, low_cardinality_cols)

    # Target encode
    high_cardinality_cols = [c for c in df.select_dtypes(include='object').columns if df[c].nunique() >= 10]
    te = fit_target_encoder(X_train, y_train, high_cardinality_cols)
    X_train = apply_target_encoder(X_train, te, high_cardinality_cols)
    X_val   = apply_target_encoder(X_val,   te, high_cardinality_cols)
    X_test  = apply_target_encoder(X_test,  te, high_cardinality_cols)

    # Save
    for name, obj in [('X_train', X_train), ('X_val', X_val), ('X_test', X_test),
                      ('y_train', y_train), ('y_val', y_val), ('y_test', y_test)]:
        with open(f'{name}.pkl', 'wb') as f:
            pickle.dump(obj, f)
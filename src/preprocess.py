import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, OneHotEncoder, TargetEncoder
from sklearn.model_selection import train_test_split
import joblib
from typing import Tuple

def load_data(trans_path: str, id_path: str) -> pd.DataFrame:
    """Load and merge transaction + identity files on TransactionID."""
    df_trans = pd.read_csv(trans_path)
    df_id = pd.read_csv(id_path)
    merged = df_trans.merge(df_id, on = "TransactionID", how = "left")
    assert merged.shape[0] == df_trans.shape[0], "Merged changed row count - check for duplicate key values"
    assert df_trans.shape[0] == len(pd.unique(df_trans["TransactionID"])), "Trans df contains duplicates in TransactionID (key column)"
    return merged



def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Feature engineering and Transformations : RETURNS transformed df and list of cols to drop"""

    #log transform Transaction AMT
    df['TransactionAmt'] = np.log1p(df['TransactionAmt'])

    #create hour_of_day from TransactionDT
    df['hour_of_day'] = (df['TransactionDT']//3600)%24

    #Presence flags for >90% Null cols
    majorlynullcols = []
    nullpercentage = df.isnull().sum()/len(df)
    majorlynullcols = nullpercentage[nullpercentage>0.9].index
    majorlynullcols = list(majorlynullcols)

    for col in majorlynullcols:
        df[f"{col}_present"] = df[col].notna().astype(int)
    return (df, majorlynullcols)




def encode_features(df_train:pd.DataFrame,df_test:pd.DataFrame, low_cardinality_cols:list) -> Tuple[pd.DataFrame,pd.DataFrame]:
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = ohe.fit_transform(df_train[low_cardinality_cols])
    encoded_cols = ohe.get_feature_names_out(low_cardinality_cols)

    test_encoded = ohe.transform(df_test[low_cardinality_cols])

    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df_train.index)
    test_encoded_df = pd.DataFrame(test_encoded, columns=encoded_cols, index=df_test.index)

    df_train = df_train.drop(columns=low_cardinality_cols)
    df_test = df_test.drop(columns=low_cardinality_cols)

    df_train = pd.concat([df_train, encoded_df], axis=1)
    df_test = pd.concat([df_test, test_encoded_df], axis=1)
    return (df_train,df_test)

def target_encode_features(df_train:pd.DataFrame,df_test:pd.DataFrame, y_train:pd.DataFrame,high_cardinality_cols:list) -> Tuple[pd.DataFrame,pd.DataFrame]:
    te = TargetEncoder(random_state=42)
    encoded = te.fit_transform(df_train[high_cardinality_cols], y_train)
    encoded_cols = te.get_feature_names_out(high_cardinality_cols)

    test_encoded = te.transform(df_test[high_cardinality_cols])

    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df_train.index)
    test_encoded_df = pd.DataFrame(test_encoded, columns=encoded_cols, index=df_test.index)

    df_train = df_train.drop(columns=high_cardinality_cols)
    df_test = df_test.drop(columns=high_cardinality_cols)

    df_train = pd.concat([df_train, encoded_df], axis=1)
    df_test = pd.concat([df_test, test_encoded_df], axis=1)
    return (df_train,df_test)


def drop_cols(df: pd.DataFrame, cols_to_drop: list) -> pd.DataFrame:
    """Drop specified columns from the dataframe."""
    return df.drop(cols_to_drop, axis = "columns")


def scale_features(X_train, X_test, num_cols: list):
    rscaler = RobustScaler()
    train_fitted = rscaler.fit_transform(X_train[num_cols])
    test_fitted = rscaler.transform(X_test[num_cols])
    scaled_cols = rscaler.get_feature_names_out(num_cols)

    train_fitted = pd.DataFrame(train_fitted, columns = scaled_cols, index = X_train.index)
    test_fitted = pd.DataFrame(test_fitted, columns = scaled_cols, index = X_test.index)

    X_train = X_train.drop(columns=num_cols)
    X_test = X_test.drop(columns=num_cols)

    X_train = pd.concat([X_train, train_fitted], axis=1)
    X_test = pd.concat([X_test, test_fitted], axis=1)
    return (X_train,X_test)

if __name__ == "__main__":
    df = load_data(
        trans_path='../data/raw/train_transaction.csv',
        id_path='../data/raw/train_identity.csv'
    )
    
    df,listofcolstodrop = build_features(df)
    
    cols_to_drop = ["TransactionDT","TransactionID", "DeviceInfo"]
    cols_to_drop.extend(listofcolstodrop)

    df = drop_cols(df, cols_to_drop)
    


    #categorical cols (low cardinality <10 cols get OHE, high cardinality cols get target encoded (to capture fraud signal per category) easy for model to learn in high cardinality/sparse cols like this)
    # OHE for sparse cols (high cardinality) doesnt work efficiently bcz model has to learn patterns where each category only has very few training examples

    X_train, X_test, y_train, y_test = train_test_split(df.drop('isFraud', axis = 1), df['isFraud'], test_size = 0.2, random_state =42)

    numeric_cols = X_train.select_dtypes(exclude='object').columns.tolist()
    numeric_cols = [x for x in numeric_cols if not x.endswith("_present")]
    X_train,X_test = scale_features(X_train,X_test,numeric_cols)

    low_cardinality_cols = [col for col in df.select_dtypes(include='object').columns.tolist() if df[col].nunique() < 10]
    X_train,X_test = encode_features(X_train, X_test,low_cardinality_cols) 
    high_cardinality_cols = [col for col in df.select_dtypes(include='object').columns.tolist() if df[col].nunique() >= 10]
    X_train,X_test = target_encode_features(X_train,X_test, y_train, high_cardinality_cols)

    print(X_train.shape)
    print(X_test.shape)
    print(X_train.select_dtypes(include='object').columns.tolist())
    



    



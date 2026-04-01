import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib


def load_data(trans_path: str, id_path: str) -> pd.DataFrame:
    """Load and merge transaction + identity files on TransactionID."""
    df_trans = pd.read_csv(trans_path)
    df_id = pd.read_csv(id_path)
    print("df_trans dim:", df_trans.shape)
    print("df_id dim:", df_id.shape)

    merged = df_trans.merge(df_id, on = "TransactionID", how = "left")
    print("merged dim:", merged.shape)
    assert merged.shape[0] == df_trans.shape[0], "Merged changed row count - check for duplicate key values"
    assert df_trans.shape[0] != len(pd.unique(df_trans["TransactionID"])), "Trans df contains duplicates in TransactionID (key column)"
    return merged



def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering and Transformations"""

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

    #categorical cols (low cardinality <10 cols get OHE, high cardinality cols get target encoded (to capture fraud signal per category) easy for model to learn in high cardinality/sparse cols like this)
    # OHE for sparse cols (high cardinality) doesnt work efficiently bcz model has to learn patterns where each category only has very few training examples


    categ_cols = df.select_dtypes(include='object').columns.tolist()
    

def drop_cols(df: pd.DataFrame, cols_to_drop: list) -> pd.DataFrame:
    """Drop specified columns from the dataframe."""
    return df.drop(cols_to_drop, axis = "columns")









def scale_features(X_train, X_test, num_cols: list, scaler_path: str):
    """Fit RobustScaler on train, transform both splits, save scaler artifact."""
    pass


if __name__ == "__main__":
    df = load_data(
        trans_path='../data/raw/train_transaction.csv',
        id_path='../data/raw/train_identity.csv'
    )

    cols_to_drop = ["TransactionDT"]

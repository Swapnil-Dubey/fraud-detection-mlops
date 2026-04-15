import pandas as pd
import requests

df = pd.read_csv('../data/raw/train_transaction.csv')
df_id = pd.read_csv('../data/raw/train_identity.csv')
merged = df.merge(df_id, on='TransactionID', how='left')

row = {k: None if pd.isna(v) else v for k, v in merged.iloc[0].to_dict().items() if k != 'isFraud'}
response = requests.post('http://localhost:8000/predict', json=row)
print(response.json())

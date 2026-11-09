import pandas as pd
file_path="dataset/processed/medquad_processed.csv"
df=pd.read_csv(file_path)
print("Dataset shape:",df.shape)
print("\ncolumns:")
print(df.columns.tolist())
print('\nMissing values:')
print(df.isnull().sum())
print('\nDuplicate questions:',df["question"].duplicated().sum())
print("\nFirst five records:")
print(df.head())
print("\nQuestion types:")
print(df["question_type"].value_counts().head(10))
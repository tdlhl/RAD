import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('./final_50_file.csv')

label_columns = df.columns[2:]

train_df = pd.DataFrame()
test_df = pd.DataFrame()

for label in label_columns:
    positive_samples = df[df[label] == 1]
    
    test_sample = positive_samples.sample(n=1)
    test_df = pd.concat([test_df, test_sample])
    
    df = df.drop(test_sample.index)

train_df, temp_test_df = train_test_split(df, test_size=0.2, random_state=42)

test_df = pd.concat([test_df, temp_test_df])

headers = test_df.columns.tolist()
for i in headers:
    print(f"\"{i}\",")

train_df.to_csv('./skincap_50_train_set.csv', index=False)
test_df.to_csv('./skincap_50_test_set.csv', index=False)
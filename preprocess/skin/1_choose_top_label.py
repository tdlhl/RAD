import pandas as pd

df = pd.read_csv('output_file.csv')

disease_counts = df['disease'].value_counts()

print('len(disease_counts)=', len(disease_counts))

top_50_diseases = disease_counts.head(50).index

df_filtered = df[df['disease'].isin(top_50_diseases)]

print('len(df_filtered)=', len(df_filtered))

df_filtered.to_csv('./filtered_50_file.csv', index=False)

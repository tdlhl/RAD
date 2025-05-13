import pandas as pd

df = pd.read_csv('./filtered_50_file.csv')

disease_dummies = pd.get_dummies(df['disease']).astype(int)

df_final = pd.concat([df[['skincap_file_path', 'caption_zh_polish_en']], disease_dummies], axis=1)

df_final.to_csv('./final_50_file.csv', index=False)

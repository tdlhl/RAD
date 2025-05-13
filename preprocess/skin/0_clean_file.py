import pandas as pd

df = pd.read_csv('skincap_v240623.csv')

df = df[['skincap_file_path', 'disease', 'caption_zh_polish_en']]

df.to_csv('output_file.csv', index=False)
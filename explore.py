import pandas as pd
trans_df = pd.read_csv('../data/rus_tyv_parallel_50k.tsv', sep="\t")


df_train = trans_df[trans_df.split=='train'].copy() # 49000 items
df_dev = trans_df[trans_df.split=='dev'].copy()     # 500 items
df_test = trans_df[trans_df.split=='test'].copy()   # 500 items
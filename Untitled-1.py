# %%
import pandas as pd 
import numpy as np 

# %%
df = pd.read_csv(r"D:\vs_MT\IV_data\New_kd_data\kd_2025\KD_2025.csv", encoding='utf-8', encoding_errors='ignore')

# %%
df.head()

# %%
pd.set_option('display.max_columns', None)
df.head()

# %%
df.describe()

# %%
df.isna().sum()


# %%
df.info()

# %%
df.duplicated().sum()

# %%
df.describe(include= "object")

# %%
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# %%
numeric_df.corr()['aktiv'].plot(kind='bar')

# %%
df.head()

# %%
df['end_date'] = pd.to_datetime(df['datum_s'], format='%d.%m.%Y')
df['end_year'] = df['end_date'].dt.year
df['end_month'] = df['end_date'].dt.month
df['end_day'] = df['end_date'].dt.day


# %%
#import pandas as pd
#pd.set_option('display.max_columns', None)  # Show all columns
#print(df.head())  # Now all columns will be visible

# %%
df.vtrweg.unique()

# %%
df.spartek.replace({'VK':1, 'KH':2, 'TK':3},inplace=True)

# %%
df.replace({'V':1, 'M':2, 'S':3, 'D':4},inplace=True)

# %%
df[['gfeld_c','gfeld_n']]= df['gfeld'].str.split('/',expand=True)

# %%
df['ort_id'] = pd.factorize(df['ort'])[0]+1
print(df[['ort','ort_id']])

# %%
df.head()

# %%
df.to_csv('1212')

# %%




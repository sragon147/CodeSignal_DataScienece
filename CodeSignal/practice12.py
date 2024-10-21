import pandas as pd
import numpy as np

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

# #  number of rows and columns
# print(df.shape)

# # datatypes
# print(df.dtypes)

# how many columns under each dtype
print(df.dtypes.value_counts())

# summary statistics
df_stats = df.describe()

# # numpy array 
# df_arr = df.values

# # list
# df_list = df.values.tolist()
# Solution
# Get Manufacturer with highest price
df.loc[df.Price == np.max(df.Price), ['Manufacturer', 'Model', 'Type']]

# Get Row and Column number
row, col = np.where(df.values == np.max(df.Price))

# Get the value
df.iat[row[0], col[0]]
df.iloc[row[0], col[0]]

# Input
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

# Solution
# Step 1:
# df=df.rename(columns = {'Type':'CarType'})

# # Step 2:
# df.columns = df.columns.map(lambda x: x.replace('.', '_'))
# print(df.columns)
# nan_count_per_row = df.isnull().sum(axis=1)
# print(nan_count_per_row)

# max_nan_count = nan_count_per_row.max()
# rows_with_max_nan = nan_count_per_row[nan_count_per_row == max_nan_count]
# print(rows_with_max_nan)

# df_out = df[['Min.Price', 'Max.Price']]
# mean = df_out.mean()
# df_out.fillna(mean, inplace=True)
# print(df_out.isnull().sum())

# df_min = df[['Min.Price']]
# df_max = df[['Max.Price']]
# mean = df_min.mean()
# median = df_max.median()
# df_min.fillna(mean, inplace=True)
# df_max.fillna(median, inplace=True)
# df = pd.concat([df_min, df_max], axis=1) 
# print(df.loc[df.index % 20 == 0][['Manufacturer', 'Model', 'Type']])
# print(df.iloc[::20, :][['Manufacturer', 'Model', 'Type']])

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv', usecols=[0,1,2,3,5])

df[['Manufacturer', 'Model', 'Type']] = df[['Manufacturer', 'Model', 'Type']].fillna('missing')
df.index = df.Manufacturer + '_' + df.Model + '_' + df.Type
print(df.index.is_unique)
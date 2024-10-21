import pandas as pd
import numpy as np

# ser = pd.Series([1,10,3,np.nan], index=pd.to_datetime(['2000-01-01', '2000-01-03', '2000-01-06', '2000-01-08']))
# print(ser.resample('D').ffill())

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')
df['medv'] = df['medv'].apply(lambda x: 'High' if x > 25 else 'Low')
print(df.head())
# df2 = pd.DataFrame()
# for chunk in df:
#     print(chunk.shape)
#     df2 = df2.append(chunk.iloc[0,:])
L = pd.Series(range(15))
print(L.values.reshape(3,5))
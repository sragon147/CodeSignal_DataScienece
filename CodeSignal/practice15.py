import pandas as pd
import numpy as np

# df = pd.DataFrame(np.random.randint(1, 30, 30).reshape(10,-1), columns=list('abc'))
# sort_df=pd.DataFrame()
# sort_df=df.sort_values(by='a', ascending=False)
# print(sort_df)
# fifth_largest_index = sort_df.index[4]
# print(fifth_largest_index)
ser = pd.Series(np.random.randint(1, 100, 15))
mean = ser.mean()
ser_mean = pd.Series()
ser_mean = ser.where(ser>mean).dropna()
ser_mean.sort_values(ascending=False, inplace=True)
print(ser_mean)
print(ser_mean.index[1])

rooms_ix, bedrooms_ix, population_ix, households_ix = [
    housing.columns.get_loc(c) for c in col_names] # get the column indices
import pandas as pd
import numpy as np
mylist = list('abcedfghijklmnopqrstuvwxyz')
myarr = np.arange(26)
mydict = dict(zip(mylist, myarr))

ser1 = pd.Series(mylist)
ser2 = pd.Series(myarr)
ser3 = pd.Series(mydict)

ser3.name = 'alphabets'

df = ser3.to_frame().reset_index()
df = pd.concat([ser1, ser2], axis=1)
print(df)

df = pd.DataFrame({'col1': ser1, 'col2': ser2})
print(df)
# ~ = not -> isin = is in
result = ser1[~ser1.isin(ser2)]

ser_u = pd.Series(np.union1d(ser1, ser2))  # union
ser_i = pd.Series(np.intersect1d(ser1, ser2))  # intersect
ser_u[~ser_u.isin(ser_i)]
print(df.head())

df = pd.DataFrame(ser.values.reshape(7,5))
np.argwhere(ser % 3==0)

# Input
ser = pd.Series(list('abcdefghijklmnopqrstuvwxyz'))
pos = [0, 4, 8, 14, 20]

# Solution
ser.take(pos) 
# = ser[ser.index.isin(pos)] 

# Output
# Vertical
ser1.append(ser2)
# = pd.concat([ser1, ser2], axis=0)

# Horizontal
df = pd.concat([ser1, ser2], axis=1)


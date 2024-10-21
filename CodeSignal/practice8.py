import pandas as pd
import numpy as np

fruit = pd.Series(np.random.choice(['apple', 'banana', 'carrot'], 10))
weights = pd.Series(np.linspace(1, 10, 10))
print(weights.tolist())
print(fruit.tolist())
# solution2
# weights.groupby(fruit).mean()

df = pd.DataFrame({'fruit': fruit, 'weight': weights})
df = df.groupby('fruit')['weight'].mean().reset_index()
print(df)

p = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
q = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
df = pd.DataFrame({'p': p, 'q': q})
# row1 = df.iloc[0]
# row2 = df.iloc[1]
# distance = np.linalg.norm(row1 - row2)
# Solution 
# sum((p - q)**2)**.5
distance = np.linalg.norm(p-q)
print("Euclidean Distance (using NumPy):", distance)

ser = pd.Series([2, 10, 3, 4, 9, 10, 2, 7, 3])
# find peaks (local maxima) in a numeric series
peaks = ser[(ser.shift(1) < ser) & (ser.shift(-1) < ser)]

my_str = 'dbc deb abed gade'

ser = pd.Series(list(my_str))
freq = ser.value_counts().sort_values()
least_freq = freq.dropna().index[0]
ser_filled = ser.replace(' ', least_freq)
print("".join(ser_filled))

import numpy as np
import pandas as pd
# Input
state = np.random.RandomState(100)
ser = pd.Series(state.normal(10, 5, 25))

# Solution
print(np.percentile(ser, q=[0, 25, 50, 75, 100]))
print(np.max(ser))
print(ser.value_counts)

print(ser.describe())

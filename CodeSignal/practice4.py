import pandas as pd
import numpy as np
# Input
ser = pd.Series(np.take(list('abcdefgh'), np.random.randint(8, size=30)))

# Solution
ser.value_counts()

# f    8
# g    7
# b    6
# c    4
# a    2
# e    2
# h    1
# dtype: int64
import pandas as pd
import numpy as np

df = pd.DataFrame(np.arange(20).reshape(-1, 5), columns=list('abcde'))
print(df.loc[df['a']>10])

# Input
df = pd.DataFrame(np.random.random(4), columns=['random'])

# Solution
out = df.style.format({    'random': '{0:.2%}'.format,})
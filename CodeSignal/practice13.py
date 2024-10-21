import pandas as pd
import numpy as np

# Example DataFrame
data = {
    'A': [10, 25, 30, 40, 50],
    'B': [15, 30, 35, 45, 55],
    'C': [12, 22, 32, 42, 52]
}
df = pd.DataFrame(data)

# Threshold value
threshold = 30

# Apply condition and replace columns 'A' and 'B' where 'A' > threshold
mask = df['A'] > threshold
df.loc[mask, ['A', 'B']] = np.array([10, 20])

# Display the modified DataFrame
print(df)

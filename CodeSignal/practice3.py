import numpy as np
import pandas as pd

# Create a sample DataFrame using pandas (to use numpy calculations later)
data = {'A': [1, 2, 3, 4, 5],
        'B': [5, 4, 3, 2, 1],
        'C': [2, 3, 4, 5, 6]}
df = pd.DataFrame(data)

# Convert DataFrame to numpy array
array = df.to_numpy()

# Calculate statistics manually using numpy
mean = np.mean(array, axis=0)
std = np.std(array, axis=0, ddof=1)  # ddof=1 for sample standard deviation
min_val = np.min(array, axis=0)
max_val = np.max(array, axis=0)
percentile_25 = np.percentile(array, 25, axis=0)
percentile_50 = np.percentile(array, 50, axis=0)
percentile_75 = np.percentile(array, 75, axis=0)
# Calculate statistics using pandas functions
mean = df.mean()
std = df.std()
min_val = df.min()
max_val = df.max()
percentile_25 = df.quantile(0.25)
percentile_50 = df.median()  # median is the 50th percentile
percentile_75 = df.quantile(0.75)

# Create a summary DataFrame
summary = pd.DataFrame({
    'mean': mean,
    'std': std,
    'min': min_val,
    '25%': percentile_25,
    '50%': percentile_50,
    '75%': percentile_75,
    'max': max_val
}, index=df.columns)

print(summary)
# mean squared error
# Input
truth = pd.Series(range(10))
pred = pd.Series(range(10)) + np.random.random(10)

# Solution
np.mean((truth-pred)**2)
from sklearn.metrics import mean_squared_error

# Example actual and predicted values
actual_values = [3, -0.5, 2, 7]
predicted_values = [2.5, 0.0, 2, 8]

# Calculate MSE using scikit-learn
mse = mean_squared_error(actual_values, predicted_values)
print("Mean Squared Error:", mse)
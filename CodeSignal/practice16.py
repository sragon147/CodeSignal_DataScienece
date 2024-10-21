import pandas as pd

# Example DataFrame
data = {
    'Age': [22, 38, None, 35, None, 40, 20, None],
    'Sex': ['male', 'female', 'male', 'male', 'female', 'female', 'male', 'female']
}
df = pd.DataFrame(data)

# Calculate mean age per sex
mean_age_by_sex = df.groupby('Sex')['Age'].transform('mean')

# Fill NaN values in Age column based on mean age by sex
df['Age'] = df['Age'].fillna(mean_age_by_sex)

print(df)
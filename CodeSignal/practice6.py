# Input
ser = pd.Series(['how', 'to', 'kick', 'ass?'])

# Solution 1
ser.map(lambda x: x.title())

# Solution 2
ser.map(lambda x: x[0].upper() + x[1:])

# Solution 3
pd.Series([i.title() for i in ser])

# Solution
ser.map(lambda x: len(x))

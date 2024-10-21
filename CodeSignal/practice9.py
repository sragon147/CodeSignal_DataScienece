import pandas as pd
import numpy as np

ser = pd.Series(['01 Jan 2010', '02-02-2011', '20120303', '2013/04/04', '2014-05-05', '2015-06-06T12:20'])
dates = pd.to_datetime(ser)

# Extract day of month
day_of_month = dates.dt.day

# Extract week number (ISO week date)
week_number = dates.dt.isocalendar().week

# Extract day of year
day_of_year = dates.dt.dayofyear

# Extract day of week (Monday=0, Sunday=6)
day_of_week = dates.dt.dayofweek

# Create a DataFrame to display the results
df = pd.DataFrame({
    'Date': dates,
    'Day of Month': day_of_month,
    'Week Number': week_number,
    'Day of Year': day_of_year,
    'Day of Week': day_of_week
})

emails = pd.Series(['buying books at amazom.com', 'rameses@egypt.com', 'matt@t.co', 'narendra@modi.com'])


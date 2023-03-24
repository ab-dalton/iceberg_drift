# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 11:38:09 2023

@author: adalt043
"""

import pandas as pd

# -----------------------------------------------------------------------------
# Create dataframe that subtracts tracks within Talbot Inlet from complete dataframe
# -----------------------------------------------------------------------------

df_all = pd.read_csv("D:/Abby/paper_2/Iceberg Beacon Database-20211026T184427Z-001/Iceberg Beacon Database/iceberg_beacon_database_17032023_all.csv")

# Convert to datetime
df_all["datetime_data"] = pd.to_datetime(
    df_all["datetime_data"].astype(str), format="%Y-%m-%d %H:%M:%S"
)

#Create year, month, day columns
df_all['day'] = df_all['datetime_data'].dt.day
df_all['month'] = df_all['datetime_data'].dt.month
df_all['year'] = df_all['datetime_data'].dt.year

df_talbot = pd.read_csv("D:/Abby/paper_2/Iceberg Beacon Database-20211026T184427Z-001/Iceberg Beacon Database/iceberg_beacon_database_17032023_talbot.csv")

# Convert to datetime
df_talbot["datetime_data"] = pd.to_datetime(
    df_talbot["datetime_data"].astype(str), format="%Y-%m-%d %H:%M:%S"
)

#Create year, month, day columns
df_talbot['day'] = df_talbot['datetime_data'].dt.day
df_talbot['month'] = df_talbot['datetime_data'].dt.month
df_talbot['year'] = df_talbot['datetime_data'].dt.year

df = pd.concat([df_all,df_talbot]).drop_duplicates(keep=False)

df['beacon_id'].nunique()

df_id = df.groupby('beacon_id').first()

df_id.to_csv("D:/Abby/paper_2/Iceberg Beacon Database-20211026T184427Z-001/Iceberg Beacon Database/iceberg_beacon_database_17032023_no_talbot_deployments.csv")



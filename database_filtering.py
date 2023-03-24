# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 16:39:57 2022

@author: adalt043
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 10:45:19 2022

@author: adalt043
"""

import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyproj
import seaborn as sns


# -----------------------------------------------------------------------------
# Load database
# -----------------------------------------------------------------------------

# Load most recent Iceberg Beacon Database output file
df_raw = pd.read_csv('D:/Abby/paper_2/Iceberg Beacon Database-20211026T184427Z-001/Iceberg Beacon Database/iceberg_beacon_database_20210622.csv', index_col=False)

# Convert to datetime
df_raw["datetime_data"] = pd.to_datetime(df_raw["datetime_data"].astype(str), format="%Y-%m-%d %H:%M:%S")

# -----------------------------------------------------------------------------
# Filter out repeated lat/lon transmissions from calib beacons
# -----------------------------------------------------------------------------

# CALIB_IRIDIUM (include 3,6,9, etc.)
calib_hours = [1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23]

df_calib = df_raw.loc[~((df_raw['beacon_type']=='CALIB_IRIDIUM') & (df_raw['datetime_data'].dt.hour.isin(calib_hours)))]

# Oceanetic & ROCKSTAR (include hourly transmission m = 0) 
df_ocean = df_calib.loc[~((df_calib['beacon_type']=='OCEANETIC') & (df_calib['datetime_data'].dt.minute > 0))]
df_rockstar = df_ocean.loc[~((df_ocean['beacon_type']=='ROCKSTAR') & (df_ocean['datetime_data'].dt.minute > 0))]

# SVP (include 1,4,7, etc.)
svp_hours = [0,2,3,5,6,8,9,11,12,14,15,17,18,20,21,23]

df = df_rockstar.loc[~((df_rockstar['beacon_type']=='SVP') & (df_rockstar['datetime_data'].dt.hour.isin(svp_hours)))]

# -----------------------------------------------------------------------------
# Calculate speed and distance
# -----------------------------------------------------------------------------
from pyproj import Proj

# Initialize pyproj with appropriate ellipsoid
geodesic = pyproj.Geod(ellps="WGS84")

# Create empty dataframe
df2 = pd.DataFrame()
    
for label, group in df.groupby(["beacon_type", "beacon_id"]):

    # Calculate forward azimuth and Great Circle distance between successive beacon positions
    group["azimuth_obs"], back_azimuth, group["distance"] = geodesic.inv(
        group["longitude"].shift().tolist(),
        group["latitude"].shift().tolist(),
        group["longitude"].tolist(),
        group["latitude"].tolist(),
    )

    # Convert azimuth from (-180° to 180°) to (0° to 360°)
    group["azimuth_obs"] = (group["azimuth_obs"] + 360) % 360
    
    # Convert distance to kilometres
    group["distance"] = group["distance"] / 1000.0

    # Calculate speed in m/s
    group["speed_ms"] = (group["distance"] * 1000) / group[
        "datetime_data"
    ].diff().dt.seconds

    # Calculate the length of the iceberg track
    duration = (group["datetime_data"].max() - group["datetime_data"].min()).days

    # Calculate cumulative distance of the iceberg track
    distance = group["distance"].sum()

    
    # Append calculated data into new dataframe
    df2 = df2.append(group, ignore_index=False)


# -----------------------------------------------------------------------------
# Subset the data based on location
# -----------------------------------------------------------------------------

# Subset the database according to latitude & longitude

# Baffin Bay (whole study area)
#df3 = df2[(df2['latitude'] >= 50) & (df2['latitude']<=80.5) & (df2['longitude']>=-85) & (df2['longitude']<=-50)]

# Talbot Inlet
df3 = df2[(df2['latitude'] >= 77.5) & (df2['latitude']<=78) & (df2['longitude']>=-78.5) & (df2['longitude']<=-76)]


# -----------------------------------------------------------------------------
# Subset dataframe based length of time between transmissions 
# -----------------------------------------------------------------------------

# Calculate transmission interval between successive beacon positions in hours
df3["timedelta"] = (df3.groupby('beacon_id')['datetime_data'].diff())

#Convert timedelta to seconds
df3['timedelta_s'] = df3['timedelta'].dt.total_seconds()

#Convert timedelta to hours
df3['timedelta_h'] = df3['timedelta_s'] / 3600

#Subset dataframe based on transmission interval (less than 12h between transmissions)
df4 = df3[(df3["timedelta_h"] <= 6) & (df3["timedelta_h"] >= 0)]


# -----------------------------------------------------------------------------
# Filter out speed outliers (>= 0.01 m/s & =< 2 m/s) 
# -----------------------------------------------------------------------------

df5 = df4[(df4["speed_ms"] <= 2.4) & (df4["speed_ms"] >= 0.01)]


# -----------------------------------------------------------------------------
# Subset dataframe based on transmission origin (>60N)
# -----------------------------------------------------------------------------

lat_60 = df5[df5['latitude'] >= 60].groupby('beacon_id').head(1)

df6 = df5[df5.set_index(['beacon_id']).index.isin(lat_60.set_index(['beacon_id']).index)]


# -----------------------------------------------------------------------------
# Loop to calculate durations of each beacon
# -----------------------------------------------------------------------------

# Specify column names
column_names = ["beacon_id", "duration"]

# Create temporary list
temporary_list = []

# For loop used to iterate through the database and for each unique beacon ID
for label, group in df6.groupby(["beacon_id"]):
    
    # Calculate the length of the iceberg track (in days)
    duration = (group["datetime_data"].max() - group["datetime_data"].min()).days
    
    # Append data to list
    temporary_list.append([label, duration])
    
# Create dataframe
df7 = pd.DataFrame(temporary_list, columns=column_names)


# -----------------------------------------------------------------------------
# Subset dataframe based on duration
# -----------------------------------------------------------------------------

# Subset the database 
df8 = df7[df7["duration"] >= 10]

df9 = df6[df6.set_index(['beacon_id']).index.isin(df8.set_index(['beacon_id']).index)]

# -----------------------------------------------------------------------------
# Subset dataframe based on beacon type (!= CALIB_ARGOS)
# -----------------------------------------------------------------------------

df10 = df9[(df9['beacon_type'] != "CALIB_ARGOS") & (df9['beacon_type'] != "BIO")]


# Create dataframe based on variables for corr plot
df11 = df10[['beacon_id','beacon_type','datetime_data','datetime_transmit','latitude','longitude','distance','azimuth_obs','speed_ms','timedelta','timedelta_s','timedelta_h']].copy()



# -----------------------------------------------------------------------------
# Subset dataframe based time period (2011-2019)
# -----------------------------------------------------------------------------

df12 = df11[(df11['datetime_data'] > "2011-01-01") & (df11['datetime_data'] < "2020-01-01")]

df12['beacon_id'].nunique()


# -----------------------------------------------------------------------------
# Save new filtered dataset as CSV
# -----------------------------------------------------------------------------

# Save filtered dataframe as CSV to use in other scripts
df12.to_csv("D:/Abby/paper_2/Iceberg Beacon Database-20211026T184427Z-001/Iceberg Beacon Database/iceberg_beacon_database_17032023_talbot.csv")


























# # -----------------------------------------------------------------------------
# # Filter speed outliers
# # -----------------------------------------------------------------------------

# # Load most recent Iceberg Beacon Database output file
# df_outliers = pd.read_csv("C:/Users/CRYO/Documents/paper_2/working_files/Iceberg Beacon Database-20211026T184427Z-001/Iceberg Beacon Database/iceberg_beacon_database_09272022_clean_TALBOT.csv", index_col=False)

# # Convert to datetime
# df_outliers["datetime_data"] = pd.to_datetime(df_outliers["datetime_data"].astype(str), format="%Y-%m-%d %H:%M:%S")

# df_clean = df_outliers[~df_outliers.index.isin([57165, 58165, 56128, 55889, 55965, 55011, 49777, 3813, 38119])]


# df_clean.to_csv("D:/Abby/paper_2/Iceberg Beacon Database-20211026T184427Z-001/Iceberg Beacon Database/iceberg_beacon_database_.csv")
























# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Add Natural Earth coastline
coast = cfeature.NaturalEarthFeature("physical", "land", "10m",
                                     edgecolor="black",
                                     facecolor="lightgray",
                                     lw=0.75)
# Global plot parameters
plt.rc("legend",fancybox=False, framealpha=1, edgecolor="black")

# Seaborn configuration
sns.set_theme(style="ticks")
sns.set_context("paper") # talk, paper, poster

# Set figure DPI
dpi = 500

# Initialize pyproj with appropriate ellipsoid
geodesic = pyproj.Geod(ellps='WGS84')

# Path for output figures
path_figures = "D:/Abby/paper_2/plots/drift_tracks/test/"

# -----------------------------------------------------------------------------
# Create map plots
# -----------------------------------------------------------------------------

# Individual beacon track plots
for label, group in df7.groupby(['beacon_type','beacon_id']):

    # Calculate the length of the iceberg track    
    duration = (group['datetime_data'].max() - group['datetime_data'].min()).days
    
    # Calculate cumulative distance of the iceberg track
    distance = group['distance'].sum() / 1000
    
    plt.figure(figsize=(12,12))
    ax = plt.axes(projection=ccrs.Orthographic((
        (group['longitude'].min() + group['longitude'].max()) / 2),
        (group['latitude'].min() + group['latitude'].max()) / 2))
    ax.coastlines(resolution='10m')
    ax.gridlines(draw_labels=True, color='black', alpha=0.5, linestyle='-')
    ax.plot(group['longitude'], group['latitude'], 
            marker='o', ms=3, fillstyle='none',
            linestyle='', lw=2,
            color='red',
            transform=ccrs.PlateCarree())
    plt.title("%s %s\n%s to %s\n%s days %.2f km" % (label[0], 
                                                    label[1], 
                                                    group['datetime_data'].min(), 
                                                    group['datetime_data'].max(), 
                                                    duration, 
                                                    distance.sum()), loc='left')
    # Save figure
    plt.savefig(path_figures + "%s.png" % label[1], dpi=dpi, transparent=False, bbox_inches='tight')
    
    


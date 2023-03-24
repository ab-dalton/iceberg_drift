#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 12:31:58 2022

@author: abby
"""

import pandas as pd
import xarray as xr
import numpy as np


# -----------------------------------------------------------------------------
# Resources
# -----------------------------------------------------------------------------

# https://stackoverflow.com/questions/42125653/extracting-nearest-lat-lon-and-time-value-from-netcdf-using-xarray?noredirect=1&lq=1

# https://stackoverflow.com/questions/58758480/xarray-select-nearest-lat-lon-with-multi-dimension-coordinates

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

# Path to ERA5 data
era5_path = "D:/Abby/paper_2/era5/"

# Path to GLORYS data
glorys_path = "D:/Abby/paper_2/GLORYS/"

# # Path to NSIDC sea ice data
seaice_path = "D:/Abby/paper_2/sea_ice/netcdf_daily/"


# -----------------------------------------------------------------------------
# Load iceberg tracks
# -----------------------------------------------------------------------------

df = pd.read_csv(
    'D:/Abby/paper_2/Iceberg Beacon Database-20211026T184427Z-001/Iceberg Beacon Database/iceberg_beacon_database_17032023_no_talbot.csv',
    index_col=False,
)


# -----------------------------------------------------------------------------
# Prepare Data
# -----------------------------------------------------------------------------

# Set bounding box 
dimensions = [83, -83, 53, -60] 

# Create start and end dates
start_date = "2011-01-01 00:00:00"
end_date = "2020-01-01 00:00:00"


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------

# plt.figure(figsize=(10,10))
# ax = plt.axes(projection=ccrs.LambertConformal(-70, 60))
# era5.isel(time=0).plot.pcolormesh(ax=ax, cmap='coolwarm', transform=ccrs.PlateCarree());
# ax.coastlines(resolution='50m')
# ax.gridlines(draw_labels=True)
# ax.set_extent([-80, -40, 33, 70])


# -----------------------------------------------------------------------------
# Select data
# -----------------------------------------------------------------------------

# Create arrays of latitude and longitude coordinates from iceberg database
times = xr.DataArray(df["datetime_data"], dims='z')
lats = xr.DataArray(df["latitude"], dims='z')
lons = xr.DataArray(df["longitude"], dims='z')


# -----------------------------------------------------------------------------
# Load ERA5
# -----------------------------------------------------------------------------

# Load ERA5 data, slice by coordinates and time and drop unnecessary variable
drop_vars = ['t2m']
era5 = xr.open_mfdataset(era5_path + '*.nc', combine="by_coords", drop_variables=drop_vars).sel(
    #latitude=slice(dimensions[2], dimensions[0]),
    #longitude=slice(dimensions[1], dimensions[3]),
    time=slice(start_date, end_date))

era5.u10.attrs

# Array
era5_data = era5.sel(time=times, latitude = lats, longitude = lons, method = "nearest")

# Point
# era5.u10.sel(time="2008-01-01 00:29:00", latitude = 50, longitude = -70, method = "nearest").values
# data.u10.values

# Add data to original dataframe
df["era5_time"] = era5_data.time
df["era5_u10"] = era5_data.u10
df["era5_v10"] = era5_data.v10

# Prepare data

# # define x and y for quiver plot
# era5_x = df['longitude'].values
# era5_y = df['latitude'].values

# 10 m eastward wind velocity (m/s)
era5_u = df['era5_u10']

# 10 m northward wind velocity (m/s)
era5_v = df['era5_v10']

# Calculate wind speed (m/s)
df['era5_speed'] = np.sqrt(era5_u**2 + era5_v**2)

# Calculate wind direction (radians) HEADING
df['era5_direction'] = np.arctan2(era5_v, era5_u)

# Convert radians to degrees and (-180° to 180°) to (0° to 360°) Heading "to" direction
df['era5_direction_deg'] = np.mod(90-(np.rad2deg(df['era5_direction'])),360)

# Convert radians to degrees and (-180° to 180°) to (0° to 360°) Meteorological "from" direction
# df['era5_direction_deg'] = np.mod(270-np.rad2deg(df['era5_direction']),360)


# -----------------------------------------------------------------------------
# Load GLORYS ocean
# -----------------------------------------------------------------------------

# Load GLORYS data, slice by coordinates and time and drop uncessary variable
glorys = xr.open_mfdataset(glorys_path + '*.nc', combine="by_coords").sel(
    #latitude=slice(dimensions[2], dimensions[0]),
    #longitude=slice(dimensions[1], dimensions[3]),
    time=slice(start_date, end_date))



#depth_avg=slice(0.494,109.7293)

# Array
glorys_data = glorys.sel(time=times, latitude = lats, longitude = lons, depth = 0.494, method = "nearest")

# Add data to original dataframe
df["glorys_time"] = glorys_data.time
df["glorys_u"] = glorys_data.uo
df["glorys_v"] = glorys_data.vo

# Define x and y for quiver plot
glorys_x = df['longitude'].values
glorys_y = df['latitude'].values

# 10 m eastward ocean velocity (m/s)
glorys_u = df['glorys_u']

# 10 m northward ocean velocity (m/s)
glorys_v = df['glorys_v']

# Calculate ocean speed (m/s)
df['glorys_speed_ms'] = np.sqrt(glorys_u**2 + glorys_v**2)

# Calculate ocean direction (radians) 
df['glorys_direction'] = np.arctan2(glorys_v, glorys_u)

# Convert radians to degrees and (-180° to 180°) to (0° to 360°) Heading "to" direction
df['glorys_direction_deg'] = np.mod(90-(np.rad2deg(df['glorys_direction'])),360)

# Convert radians to degrees and (-180° to 180°) to (0° to 360°) Meteorological "from" direction
# df['glorys_direction_deg'] = np.mod(270-np.rad2deg(df['glorys_direction']),360)


# -----------------------------------------------------------------------------
# Load NSIDC Sea Ice
# -----------------------------------------------------------------------------

# Load NSIDC data, slice by coordinates and time and drop uncessary variable
drop_vars = ['crs', 'number_of_observations']
seaice = xr.open_mfdataset(seaice_path + '*.nc', combine="by_coords", drop_variables=drop_vars).sel(
    time=slice(start_date, end_date))

# Convert to datetime
datetimeindex = seaice.indexes['time'].to_datetimeindex()
seaice['time'] = datetimeindex

# Array
seaice_data = seaice.sel(time=times, y = lats, x = lons, method = "nearest")

# Add data to original dataframe
df["seaice_time"] = seaice_data.time
df["seaice_u"] = seaice_data.u
df["seaice_v"] = seaice_data.v

# Calculate speed and direction

# define x and y for quiver plot
seaice_x = df['longitude'].values
seaice_y = df['latitude'].values

# 10 m eastward sea ice velocity 
seaice_u = df['seaice_u']

# 10 m northward sea ice velocity 
seaice_v = df['seaice_v']

# Convert longitude from degrees to radians
df['lon_rad'] = df['longitude']*(np.pi/180)

# Compute East and North using NSIDC rotation matrix
seaice_east = (df['seaice_u'] * np.cos(df['lon_rad'])) + (df['seaice_v'] * np.sin(df['lon_rad']))
seaice_north = (-df['seaice_u'] * np.sin(df['lon_rad'])) + (df['seaice_v'] * np.cos(df['lon_rad']))

df['seaice_east'] = seaice_east
df['seaice_north'] = seaice_north

# Calculate sea ice speed (cm/s)
df['seaice_speed'] = np.sqrt(seaice_east**2 + seaice_north**2)

# Convert sea ice speed to m/s
df['seaice_speed_ms'] = df['seaice_speed'] /100

# Calculate sea ice direction (radians)
df['seaice_direction'] = np.arctan2(seaice_north, seaice_east)

# Convert radians to degrees and (-180° to 180°) to (0° to 360°) Heading "to" direction
df['seaice_direction_deg'] = np.mod(90-np.rad2deg(df['seaice_direction']),360)

# Convert radians to degrees and (-180° to 180°) to (0° to 360°) Meteorological "from" direction
# df['seaice_direction_deg'] = np.mod(270-np.rad2deg(df['seaice_direction']),360)





# -----------------------------------------------------------------------------
# Save dataframe as csv
# -----------------------------------------------------------------------------

df.to_csv("D:/Abby/paper_2/Iceberg Beacon Database-20211026T184427Z-001/Iceberg Beacon Database/iceberg_beacon_database_env_variables_22032023_notalbot.csv")




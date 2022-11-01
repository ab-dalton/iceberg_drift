# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 11:05:20 2022

@author: adalt043
"""
 # test


# -----------------------------------------------------------------------------
# Load libraries
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shapely
from shapely.geometry import Point

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

# Abby
path_data = ""
path_figures = 'D:/Abby/paper_2/Iceberg Beacon Database-20211026T184427Z-001/Iceberg Beacon Database/plots/'

# -----------------------------------------------------------------------------
# Library configuration
# -----------------------------------------------------------------------------

# Add Natural Earth coastline
coast = cfeature.NaturalEarthFeature(
    "physical", "land", "10m", edgecolor="black", facecolor="lightgray", lw=0.75
)

# Add Natural Earth coastline
coastline = cfeature.NaturalEarthFeature(
    "physical", "coastline", "10m", edgecolor="black", facecolor="none", lw=0.75
)

# Configure Seaborn styles
sns.set_theme(style="ticks")
sns.set_context("paper")  # Options: talk, paper, poster

# Optional legend box outline
# plt.rc("legend", fancybox=False, framealpha=1, edgecolor="k")

# Set colour palette
# colour = ["red", "lime", "blue", "magenta", "cyan", "yellow"]
# sns.set_palette(colour)
sns.set_palette("turbo")

# -----------------------------------------------------------------------------
# Create grid
# Projection: NAD83 Statistics Canada Lambert (EPSG 3347)
# -----------------------------------------------------------------------------

# Set grid extents - utm mercator values to set corners in m
xmin = 6000000
ymin = 1900000
xmax = 9500000
ymax = 5000000

# Cell size
cell_size = 50000  # cell size in m needs to be divisible by extents above

# Create the cells in a loop
grid_cells = []
for x0 in np.arange(xmin, xmax + cell_size, cell_size):
    for y0 in np.arange(ymin, ymax + cell_size, cell_size):
        # Bounds
        x1 = x0 - cell_size
        y1 = y0 + cell_size
        grid_cells.append(shapely.geometry.box(x0, y0, x1, y1))
# Set grid projection
grid = gpd.GeoDataFrame(grid_cells, columns=["geometry"], crs="epsg:3347")

# Create grid coordinates (cheap centroid)
grid["coords"] = grid["geometry"].apply(lambda x: x.representative_point().coords[:])
grid["coords"] = [coords[0] for coords in grid["coords"]]

# Optional: Output grid to shapefile
#grid.to_file("D:/Abby/paper_2/Iceberg Beacon Database-20211026T184427Z-001/grid_cell_analysis/100_km_grid.shp")

# -----------------------------------------------------------------------------
# Plot grid
# -----------------------------------------------------------------------------

# Plot grid
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
grid.plot(
    edgecolor="black",
    color="white",
    ax=ax,
    legend=True,
)

# Observe distortion of grid with WGS84 (EPSG 4326) projection
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
grid.to_crs(4326).plot(
    edgecolor="black",
    color="white",
    ax=ax,
    legend=True,
)

# ----------------------------------------------------------------------------
# Load iceberg beacon database
# ----------------------------------------------------------------------------

# Load database
df = pd.read_csv("D:/Abby/paper_2/Iceberg Beacon Database-20211026T184427Z-001/Iceberg Beacon Database/iceberg_beacon_database_filtered_03312022.csv",
    index_col=None,
)

# Convert to datetime
df["datetime_data"] = pd.to_datetime(
    df["datetime_data"].astype(str), format="%Y-%m-%d %H:%M:%S"
)

#Create year, month, day columns
df['day'] = df['datetime_data'].dt.day
df['month'] = df['datetime_data'].dt.month
df['year'] = df['datetime_data'].dt.year

# Function to associate month numbers with season names
seasons = {12: 'Winter',
           1: 'Winter',
           2:'Winter',
           3: 'Spring',
           4: 'Spring',
           5:'Spring',
           6:'Summer',
           7:'Summer',
           8:'Summer',
           9:'Fall',
           10:'Fall',
           11:'Fall'}

df['Season'] = df['month'].apply(lambda x: seasons[x])

# ----------------------------------------------------------------------------
# Create geodataframe
# ----------------------------------------------------------------------------

# Create GeoDataFrame
# gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon_model, df.lat_model), crs='epsg:4326') # Depracated?
geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
gdf = GeoDataFrame(df, crs="epsg:4326", geometry=geometry)

# Reproject data to EPSG 3347
gdf = gdf.to_crs(epsg=3347)

# Spatial join grid with points
spatial_joined = gpd.sjoin(gdf, grid, how="left", predicate="within")

# ----------------------------------------------------------------------------
# Calculate grid cell statistics
# ----------------------------------------------------------------------------

#Select Season (optional)
joined = spatial_joined.loc[spatial_joined['Season'] == "Spring"]

#Print # of unique beacon IDs in each season
beacon_count = joined["beacon_id"].nunique()

# Summarize the stats for each attribute in the point layer - speed
stats_speed = joined.groupby(["index_right"])["speed_ms"].agg(
    ["median"]
)
stats_speed.rename(columns={'median':'median_speed'}, inplace=True)

# Summarize the stats for each attribute in the point layer - residence time
stats_res_time = joined.groupby(["index_right"])["speed_ms"].agg(
    ["min", "max"]
)

stats_long_res_time = joined.groupby(["index_right", "beacon_id"])["datetime_data"].agg(
    ["min", "max"]
)

stats_long_res_time["duration"] = stats_long_res_time['max'] - stats_long_res_time['min']

res_time = stats_long_res_time.groupby(["index_right"])["duration"].agg('mean')

stats_res_time['diff'] = res_time

stats_res_time['diff_days'] = (stats_res_time["diff"]).dt.days

stats_res_time['diff_days'] = (stats_res_time['diff'] / np.timedelta64(1 , 'h')) / 24

# Summarize stats for each attribute in the point layer - heading
stats_dir = joined.groupby(["index_right"])["azimuth_obs"].agg(
    ["mean"]
)

# Convert azimuth angle to x y coordinates for plotting vectors
stats_dir["u_drift"] = 1 * np.sin(np.radians(stats_dir['mean']))
stats_dir["v_drift"] = 1 * np.cos(np.radians(stats_dir['mean']))

# Merge dataframes to add statistics to the polygon layer
merged = pd.merge(grid, stats_speed, left_index=True, right_index=True, how="outer")
merged = pd.merge(merged, stats_res_time, left_index=True, right_index=True, how="outer")
merged = pd.merge(merged, stats_dir, left_index=True, right_index=True, how="outer")
merged = merged.dropna()

# -----------------------------------------------------------------------------
# Plot map - grid cells
# -----------------------------------------------------------------------------

# Get max values for colorbar normalization
#norm_max = np.nanmax(stats_long["mean"])

# Normalize colourbar
# Note: Can be set to norm_max or specified manually
norm_speed = mpl.colors.Normalize(vmin=0, vmax=1.5)
norm_res = mpl.colors.Normalize(vmin=0, vmax=200)
cmap = cm.get_cmap("turbo", 100)

# Set extents all of Canada
# extents = [-95, -60, 40, 85]

# Zoom to Baffin Island
# extents = [-80, -60, 60, 75]

# Zoom to Baffin Bay
extents = [-83, -50, 50, 83]

# Set figure DPI
dpi = 500

# Set map projection
proj = ccrs.epsg(3347)

# Prepare data
merged["latitude"] = merged.centroid.y
merged["longitude"] = merged.centroid.x

# Quiver location - centre of grids
x = merged['longitude'].values
y = merged['latitude'].values

# Quiver direction
u = merged['u_drift'].values
v = merged['v_drift'].values

# Plot figures (N = 11,6) (S = 13.5)
fig, axs = plt.subplots(
    2, 2, figsize=(14, 12), constrained_layout=True, subplot_kw={"projection": proj},
)
fig.suptitle('Spring n = %i' % beacon_count, fontsize=16) # Change according to season selected above

# Drift tracks

axs[0, 0].add_feature(coast)
axs[0, 0].set_extent([-80, -60, 60, 75])
axs[0, 0].set(box_aspect=1)
axs[0, 0].set_title('Iceberg Tracks', fontsize = 12)
axs[0, 0].scatter(
    x = 'longitude',
    y = 'latitude', 
    data = joined,
    marker='.',
    s = 2,
    color='red',
    transform=ccrs.PlateCarree()
)
gl_1 = axs[0, 0].gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    color="black",
    alpha=0.25,
    linestyle="dotted",
)
gl_1.top_labels = False
gl_1.right_labels = False
gl_1.rotate_labels = False
        
# Speed

axs[0, 1].add_feature(coast)
axs[0, 1].set_extent([-80, -60, 60, 75])
axs[0, 1].set(box_aspect=1)
axs[0, 1].set_title('Median Speed (m/s)', fontsize = 12)
p2 = merged.plot(
    column="median_speed",
    cmap=cmap,
    norm=norm_speed,
    edgecolor="black",
    lw=0.5, legend=False,
    ax=axs[0 ,1]
)
gl_2 = axs[0, 1].gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    color="black",
    alpha=0.25,
    linestyle="dotted",
)
gl_2.top_labels = False
gl_2.right_labels = False
gl_2.rotate_labels = False

fig.colorbar(cm.ScalarMappable(norm=norm_speed, cmap=cmap), ax=axs[0, 1], shrink=0.8)

# Direction

axs[1, 0].add_feature(coast)
axs[1, 0].set_extent([-80, -60, 60, 75])
axs[1, 0].set(box_aspect=1)
axs[1, 0].set_title('Mean Drift Direction', fontsize = 12)

gl_3 = axs[1, 0].gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    color="black",
    alpha=0.25,
    linestyle="dotted",
)
gl_3.top_labels = False
gl_3.right_labels = False
gl_3.rotate_labels = False
axs[1, 0].quiver(x, y, u, v, color="blue", linewidth = 0.5)

# Residence Time

axs[1, 1].add_feature(coast)
axs[1, 1].set_extent([-80, -60, 60, 75])
axs[1, 1].set(box_aspect=1)
axs[1, 1].set_title('Mean Residence Time (days)', fontsize = 12)
p3 = merged.plot(
    column="diff_days",
    cmap=cmap,
    norm=norm_res,
    edgecolor="black",
    lw=0.5, legend=False,
    ax=axs[1 ,1]
)
gl_4 = axs[1, 1].gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    color="black",
    alpha=0.25,
    linestyle="dotted",
)
gl_4.top_labels = False
gl_4.right_labels = False
gl_4.rotate_labels = False

fig.colorbar(cm.ScalarMappable(norm=norm_res, cmap=cmap), ax=axs[1, 1], shrink=0.8)









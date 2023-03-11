# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 12:00:04 2022

@author: adalt043
"""

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
path_figures = 'D:/Abby/paper_2/plots/seasonal_panels/'

# -----------------------------------------------------------------------------
# Library configuration
# -----------------------------------------------------------------------------

# Add Natural Earth coastline
coast = cfeature.NaturalEarthFeature(
    "physical", "land", "10m", edgecolor="darkgrey", facecolor="lightgray", lw=0.75
)

# Configure Seaborn styles
sns.set_theme(style="ticks")
sns.set_context("paper")  # Options: talk, paper, poster
sns.set_palette("turbo")

# -----------------------------------------------------------------------------
# Create grid
# Projection: NAD83 Statistics Canada Lambert (EPSG 3347)
# -----------------------------------------------------------------------------

# Set grid extents - utm mercator values to set corners in m
# Baffin Bay
xmin = 6000000
ymin = 1900000
xmax = 9500000
ymax = 5000000

# Cell size
# Baffin Bay
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

# Observe distortion of grid with WGS84 (EPSG 4326) projection -- 
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
df = pd.read_csv("D:/Abby/paper_2/Iceberg Beacon Database-20211026T184427Z-001/Iceberg Beacon Database/iceberg_beacon_database_filtered_08312022_clean.csv",
    index_col=None,
)

# Convert to datetime
df["datetime_data"] = pd.to_datetime(
    df["datetime_data"].astype(str), format="%Y-%m-%d %H:%M:%S")

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

# Load RGI
# rgi = gpd.read_file("D:/Abby/paper_2/rgi/rgi60_Arctic_glaciers_3995_simple150.gpkg")
# rgi['geometry'] = rgi.buffer(0) #clean errors

# rgi_gis = gpd.read_file("C:/Users/adalt043/Downloads/Greenland_Ice_Sheet.gpkg")


# ----------------------------------------------------------------------------
# Create geodataframe
# ----------------------------------------------------------------------------

# Create GeoDataFrame
geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
gdf = GeoDataFrame(df, crs="epsg:4326", geometry=geometry)

# Reproject data to EPSG 3347
gdf = gdf.to_crs(epsg=3347)

# Spatial join grid with points (call this spatial_joined if selecting season)
spatial_joined = gpd.sjoin(gdf, grid, how="left", predicate="within")

spatial_joined['beacon_id'].nunique()

# ----------------------------------------------------------------------------
# Calculate grid cell statistics
# ----------------------------------------------------------------------------

# Select season (optional)

joined = spatial_joined.loc[spatial_joined['Season'] == "Winter"]

joined['datetime_data'].min()
joined['datetime_data'].max()

# Print # of unique beacon IDs in each season
beacon_count = joined["beacon_id"].nunique()

joined['beacon_type'].nunique()

# Summarize the stats for each attribute in the point layer - speed
stats_speed = joined.groupby(["index_right"])["speed_ms"].agg(
    ["median", 'max', 'std']
)
stats_speed.rename(columns={'median':'median_speed'}, inplace=True)
stats_speed.rename(columns={'max':'max_speed'}, inplace=True)
stats_speed.rename(columns={'std':'std_speed'}, inplace=True)

# Filter speeds out that are > 2 m/s
#stats_speed = stats_speed[(stats_speed['max_speed'] <= 2)]

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


# Group by grid cells and count number of observations in each group
n_obs = joined.groupby("index_right").size()

# Add the counts as a new column in the grid dataframe
stats_speed["n_obs"] = n_obs.values

# Merge dataframes to add statistics to the polygon layer
merged = pd.merge(grid, stats_speed, left_index=True, right_index=True, how="outer")
merged = pd.merge(merged, stats_res_time, left_index=True, right_index=True, how="outer")
merged = pd.merge(merged, stats_dir, left_index=True, right_index=True, how="outer")
merged = merged.dropna()

stats_res_time['diff_days'].max()
stats_speed['median_speed'].max()

# -----------------------------------------------------------------------------
# Prepare quiver plot data
# -----------------------------------------------------------------------------

# Get centroids of cells
merged["latitude"] = merged.centroid.y
merged["longitude"] = merged.centroid.x

# Quiver location - centre of grids
x = merged['longitude'].values
y = merged['latitude'].values

# Quiver direction
u = merged['u_drift'].values
v = merged['v_drift'].values

# -----------------------------------------------------------------------------
# Plot map - grid cells
# -----------------------------------------------------------------------------

# Get max values for colorbar normalization
#norm_max = np.nanmax(stats_long["mean"])

# Normalize colourbar
# Note: Can be set to norm_max or specified manually

# Speed - median
norm_speed = mpl.colors.Normalize(vmin=0, vmax=1.2)
cmap_speed = cm.get_cmap("plasma_r", 12)

# Speed - std
norm_std = mpl.colors.Normalize(vmin=0, vmax=0.5)
cmap_std = cm.get_cmap("plasma_r", 12)

# Res Time
cmap_res = cm.get_cmap("plasma_r",20)
norm_res = mpl.colors.Normalize(vmin=0, vmax=100)

# Number of Observations
norm_obs = mpl.colors.Normalize(vmin=0, vmax=1000)
cmap_obs = cm.get_cmap("plasma_r", 20)

# ## Winter
# norm_res = mpl.colors.Normalize(vmin=0, vmax=275) 

# ## Spring
# norm_res = mpl.colors.Normalize(vmin=0, vmax=400)

# ## Summer
# norm_res = mpl.colors.Normalize(vmin=0, vmax=200)

# ## Fall    
# norm_res = mpl.colors.Normalize(vmin=0, vmax=550)

# Zoom to Baffin Bay
extents = [-83, -60, 55, 83]

# Set figure DPI
dpi = 300

# Set map projection
proj = ccrs.epsg(3347)

# Plot figures (N = 11,6) (S = 13.5)
fig, axs = plt.subplots(
    2,3, figsize=(18, 18), constrained_layout=True, subplot_kw={"projection": proj},
)
params = {'mathtext.default': 'regular' }   
plt.rcParams.update(params)
font = {'size'   : 12,
        'weight' : 'normal'}
mpl.rc('font', **font)


## Drift tracks

axs[0, 0].add_feature(coast)
axs[0, 0].set_extent(extents)
axs[0, 0].set(box_aspect=1) # 1
axs[0,0].annotate('A', (1, 1),
                    xytext=(-5,-5),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    ha='right', va='top',
                    fontsize=14,
                    weight='bold')
axs[0, 0].scatter(
    x ='longitude',
    y = 'latitude', 
    data = joined,
    marker='.',
    s = 2,
    # color='firebrick',
    c = joined.beacon_id.astype('category').cat.codes,
    cmap=cmap_speed,
    transform=ccrs.PlateCarree()
)
axs[0,0].set_facecolor('#D6EAF8')
for k, spine in axs[0,0].spines.items():  #ax.spines is a dictionary
    spine.set_zorder(10)
gl_1 = axs[0, 0].gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    color="black",
    alpha=0.25,
    linestyle="dotted",
    zorder=5
)
gl_1.top_labels = False
gl_1.right_labels = False
gl_1.rotate_labels = False
# rgi.plot(color='white',
#           ax=axs[0,0],
#           edgecolor='none',
#           transform=ccrs.epsg('3995'),
#           zorder=4)
# rgi_gis.plot(color='white',
#              ax=ax,
#              edgecolor='none',
#              transform=ccrs.epsg('3995'),
#              zorder=3)
        
## Number of Observations

axs[0, 1].add_feature(coast)
axs[0, 1].set_extent(extents)
axs[0, 1].set(box_aspect=1)
axs[0, 1].annotate('B', (1, 1),
                    xytext=(-5,-5),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    ha='right', va='top',
                    fontsize=14,
                    weight='bold')
axs[0,1].set_facecolor('#D6EAF8')
p2 = merged.plot(
    column="n_obs",
    cmap=cmap_obs,
    norm=norm_obs,
    edgecolor="black",
    lw=0.5, legend=False,
    ax=axs[0 ,1]
)
for k, spine in axs[0,1].spines.items():  #ax.spines is a dictionary
    spine.set_zorder(10)
gl_2 = axs[0, 1].gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    color="black",
    alpha=0.25,
    linestyle="dotted",
    zorder=5
)
gl_2.top_labels = False
gl_2.right_labels = False
gl_2.rotate_labels = False
# rgi.plot(color='white',
#           ax=axs[0,1],
#           edgecolor='none',
#           transform=ccrs.epsg('3995'),
#           zorder=4)
cb = fig.colorbar(cm.ScalarMappable(norm=norm_obs, cmap=cmap_obs),
                  ax=axs[0, 1],
                  shrink=0.52,
                  orientation='vertical') 
cb.ax.tick_params(labelsize=12)
cb.ax.set_ylabel('Number of Observations',fontsize=14,rotation=90)

## Speed - median

axs[0, 2].add_feature(coast)
axs[0, 2].set_extent(extents)
axs[0, 2].annotate('C', (1, 1),
                    xytext=(-5,-5),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    ha='right', va='top',
                    fontsize=14,
                    weight='bold')
axs[0, 2].set(box_aspect=1)
p3 = merged.plot(
    column="median_speed",
    cmap=cmap_speed,
    norm=norm_speed,
    edgecolor="black",
    lw=0.5, legend=False,
    ax=axs[0, 2]
)
axs[0, 2].set_facecolor('#D6EAF8')
gl_4 = axs[0, 2].gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    color="black",
    alpha=0.25,
    linestyle="dotted",
    zorder=5
)
for k, spine in axs[0, 2].spines.items():  #ax.spines is a dictionary
    spine.set_zorder(10)
gl_4.top_labels = False
gl_4.right_labels = False
gl_4.rotate_labels = False
# rgi.plot(color='white',
#                 ax=axs[1,1],
#                 edgecolor='none', 
#                 transform=ccrs.epsg('3995'),
#                 zorder=4)
cb = fig.colorbar(cm.ScalarMappable(norm=norm_speed, cmap=cmap_speed),
                  ax=axs[0, 2],
                  shrink=0.52,
                  orientation='vertical') 
cb.ax.tick_params(labelsize=12)
cb.ax.set_ylabel('Speed (m $s^{-1}$)',fontsize=14,rotation=90)

## Speed - std

axs[1, 2].add_feature(coast)
axs[1, 2].set_extent(extents)
axs[1, 2].annotate('F', (1, 1),
                    xytext=(-5,-5),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    ha='right', va='top',
                    fontsize=14,
                    weight='bold')
axs[1, 2].set(box_aspect=1)
p3 = merged.plot(
    column="std_speed",
    cmap=cmap_std,
    norm=norm_std,
    edgecolor="black",
    lw=0.5, legend=False,
    ax=axs[1 ,2]
)
axs[1, 2].set_facecolor('#D6EAF8')
gl_4 = axs[1, 2].gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    color="black",
    alpha=0.25,
    linestyle="dotted",
    zorder=5
)
for k, spine in axs[1,2].spines.items():  #ax.spines is a dictionary
    spine.set_zorder(10)
gl_4.top_labels = False
gl_4.right_labels = False
gl_4.rotate_labels = False
# rgi.plot(color='white',
#                 ax=axs[1,1],
#                 edgecolor='none', 
#                 transform=ccrs.epsg('3995'),
#                 zorder=4)
cb = fig.colorbar(cm.ScalarMappable(norm=norm_std, cmap=cmap_std),
                  ax=axs[1, 2],
                  shrink=0.52,
                  orientation='vertical') 
cb.ax.tick_params(labelsize=12)
cb.ax.set_ylabel('Speed (m $s^{-1}$)',fontsize=14,rotation=90)


## Direction 

axs[1, 0].add_feature(coast)
axs[1, 0].set_extent(extents)
axs[1, 0].annotate('D', (1, 1),
                    xytext=(-5,-5),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    ha='right', va='top',
                    fontsize=14,
                    weight='bold')
axs[1, 0].set(box_aspect=1)
gl_3 = axs[1, 0].gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    color="black",
    alpha=0.25,
    linestyle="dotted",
    zorder=5
)
axs[1,0].set_facecolor('#D6EAF8')
for k, spine in axs[1, 0].spines.items():  #ax.spines is a dictionary
    spine.set_zorder(10)
gl_3.top_labels = False
gl_3.right_labels = False
gl_3.rotate_labels = False
axs[1, 0].quiver(x, y, u, v, 
                 color="indigo",
                 linewidth = 0.5)
# rgi.plot(color='white',
#           ax=axs[1,0],
#           edgecolor='none',
#           transform=ccrs.epsg('3995'),
#           zorder=4)

## Residence Time


axs[1, 1].add_feature(coast)
axs[1, 1].set_extent(extents)
axs[1, 1].annotate('E', (1, 1),
                    xytext=(-5,-5),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    ha='right', va='top',
                    fontsize=14,
                    weight='bold')
axs[1, 1].set(box_aspect=1)
p3 = merged.plot(
    column="diff_days",
    cmap=cmap_res,
    norm=norm_res,
    edgecolor="black",
    lw=0.5, legend=False,
    ax=axs[1 ,1]
)
axs[1,1].set_facecolor('#D6EAF8')
gl_4 = axs[1, 1].gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    color="black",
    alpha=0.25,
    linestyle="dotted",
    zorder=5
)
for k, spine in axs[1,1].spines.items():  #ax.spines is a dictionary
    spine.set_zorder(10)
gl_4.top_labels = False
gl_4.right_labels = False
gl_4.rotate_labels = False
# rgi.plot(color='white',
#                 ax=axs[1,1],
#                 edgecolor='none', 
#                 transform=ccrs.epsg('3995'),
#                 zorder=4)
cb = fig.colorbar(cm.ScalarMappable(norm=norm_res, cmap=cmap_res),
                  ax=axs[1, 1],
                  shrink=0.52,
                  orientation='vertical') 
cb.ax.tick_params(labelsize=12)
cb.ax.set_ylabel('Residence Time (days)',fontsize=14,rotation=90)


# Save figure
fig.savefig(
    path_figures + "winter.png",
    dpi=dpi,
    transparent=False,
    bbox_inches="tight",
)
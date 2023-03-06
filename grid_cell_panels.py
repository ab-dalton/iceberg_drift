# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 11:05:20 2022

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
path_figures = 'D:/Abby/paper_2/plots/seasonal_panels/talbot/'

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
# # Baffin Bay
xmin = 6000000
ymin = 1900000
xmax = 9500000
ymax = 5000000

# Talbot Inlet
xmin = 6511830
ymin = 4589610
xmax = 6576320
ymax = 4652510

# # Smith Sound
# xmin = 6517500
# ymin = 4299300
# xmax = 6691800
# ymax = 4752700


# Cell size
# Baffin Bay
# cell_size = 50000  # cell size in m needs to be divisible by extents above

# Talbot Inlet
cell_size = 2000  # cell size in m needs to be divisible by extents above

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
rgi = gpd.read_file("D:/Abby/paper_2/rgi/rgi60_Arctic_glaciers_3995_simple150.gpkg")
rgi['geometry'] = rgi.buffer(0) #clean errors
rgi_talbot = rgi[(rgi['CenLat'] >= 77.4) & (rgi['CenLat']<=78.5) & (rgi['CenLon']>=-80) & (rgi['CenLon']<=-76)]


# ----------------------------------------------------------------------------
# Create geodataframe
# ----------------------------------------------------------------------------

# Create GeoDataFrame
geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
gdf = GeoDataFrame(df, crs="epsg:4326", geometry=geometry)

# Reproject data to EPSG 3347
gdf = gdf.to_crs(epsg=3347)

# Spatial join grid with points (call this spatial_joined if selecting season)
joined = gpd.sjoin(gdf, grid, how="left", predicate="within")

# spatial_joined['beacon_id'].nunique()

# ----------------------------------------------------------------------------
# Calculate grid cell statistics
# ----------------------------------------------------------------------------

# Select season (optional)
# joined = spatial_joined.loc[spatial_joined['Season'] == "Winter"]

joined['datetime_data'].min()
joined['datetime_data'].max()

# Print # of unique beacon IDs in each season
beacon_count = joined["beacon_id"].nunique()

joined['beacon_type'].nunique()

# Summarize the stats for each attribute in the point layer - speed
stats_speed = joined.groupby(["index_right"])["speed_ms"].agg(
    ["median", 'max']
)
stats_speed.rename(columns={'median':'median_speed'}, inplace=True)
stats_speed.rename(columns={'max':'max_speed'}, inplace=True)

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

# Merge dataframes to add statistics to the polygon layer
merged = pd.merge(grid, stats_speed, left_index=True, right_index=True, how="outer")
merged = pd.merge(merged, stats_res_time, left_index=True, right_index=True, how="outer")
merged = pd.merge(merged, stats_dir, left_index=True, right_index=True, how="outer")
merged = merged.dropna()

stats_res_time['diff_days'].max()
stats_speed['median_speed'].max()

plt.hist(stats_res_time['diff_days'])

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
norm_speed = mpl.colors.Normalize(vmin=0, vmax=0.5)
norm_res = mpl.colors.Normalize(vmin=0, vmax=200) # change based on residence time avg
cmap = cm.get_cmap("plasma", 100)
speed_cmap = cm.get_cmap("plasma_r", 10)
res_cmap = cm.get_cmap("plasma_r", 20)

# Zoom to Baffin Bay
# extents = [-83, -60, 55, 83]

# Zoom to Talbot Inlet
extents = [-78.5, -76.2, 77.7, 78]
# extents = [-78.5, -76, 77.7, 78]

# Zoom to Smith Sound
# extents = [-79, -70, 75, 78]

# Set figure DPI
dpi = 300

# Set map projection
proj = ccrs.epsg(3347)

# Plot figures (N = 11,6) (S = 13.5)
fig, axs = plt.subplots(
    2, 2, figsize=(12.5, 8.5), constrained_layout=True, subplot_kw={"projection": proj},   #(12.5, 12) for baffin
)
params = {'mathtext.default': 'regular' }   
plt.rcParams.update(params)
font = {'size'   : 12,
        'weight' : 'normal'}
mpl.rc('font', **font)

# fig.suptitle('Winter n = %i' % beacon_count, fontsize=16) # Change according to season selected above
# fig.suptitle('Talbot Inlet', fontsize=16) # Change according to season selected above

# Drift tracks

axs[0, 0].add_feature(coast)
axs[0, 0].set_extent(extents)
# axs[0, 0].set(box_aspect=1) # 1
# axs[0, 0].set_title('A',y=1, pad=-13, fontsize = 12, loc='right')
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
    cmap=cmap,
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
rgi_talbot.plot(color='white',ax=axs[0,0],edgecolor='lightgrey', transform=ccrs.epsg('3995'),zorder=4)
        
# Speed

axs[0, 1].add_feature(coast)
axs[0, 1].set_extent(extents)
# axs[0, 1].set(box_aspect=1)
# axs[0, 1].set_title('Median Speed (m/s)', fontsize = 12)
axs[0,1].annotate('B', (1, 1),
                    xytext=(-5,-5),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    ha='right', va='top',
                    fontsize=14,
                    weight='bold')
axs[0,1].set_facecolor('#D6EAF8')
p2 = merged.plot(
    column="median_speed",
    cmap=speed_cmap,
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
    zorder=5
)
for k, spine in axs[0,1].spines.items():  #ax.spines is a dictionary
    spine.set_zorder(10)
gl_2.top_labels = False
gl_2.right_labels = False
gl_2.rotate_labels = False
rgi_talbot.plot(color='white',ax=axs[0,1],edgecolor='lightgrey', transform=ccrs.epsg('3995'),zorder=4)
cb = fig.colorbar(cm.ScalarMappable(norm=norm_speed, cmap=speed_cmap), ax=axs[0, 1], shrink=0.8, orientation='vertical') #shrink = 0.8 for Baffin Bay plots when box_aspect=1, 0.9 for talbot
cb.ax.tick_params(labelsize=12)
cb.ax.set_ylabel('Speed (m $s^{-1}$)',fontsize=14,rotation=90)

# Direction

axs[1, 0].add_feature(coast)
axs[1, 0].set_extent(extents)
axs[1,0].annotate('C', (1, 1),
                    xytext=(-5,-5),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    ha='right', va='top',
                    fontsize=14,
                    weight='bold')
# axs[1, 0].set(box_aspect=1)
# axs[1, 0].set_title('Mean Drift Direction', fontsize = 12)

gl_3 = axs[1, 0].gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    color="black",
    alpha=0.25,
    linestyle="dotted",
    zorder=5
)
axs[1,0].set_facecolor('#D6EAF8')
for k, spine in axs[1,0].spines.items():  #ax.spines is a dictionary
    spine.set_zorder(10)
gl_3.top_labels = False
gl_3.right_labels = False
gl_3.rotate_labels = False
axs[1, 0].quiver(x, y, u, v, color="indigo", linewidth = 0.5)
rgi_talbot.plot(color='white',ax=axs[1,0],edgecolor='lightgrey', transform=ccrs.epsg('3995'),zorder=4)

# Residence Time

axs[1, 1].add_feature(coast)
axs[1, 1].set_extent(extents)
axs[1,1].annotate('D', (1, 1),
                    xytext=(-5,-5),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    ha='right', va='top',
                    fontsize=14,
                    weight='bold')
# axs[1, 1].set(box_aspect=1)
# axs[1, 1].set_title('Mean Residence Time (days)', fontsize = 12)
p3 = merged.plot(
    column="diff_days",
    cmap=res_cmap,
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
rgi_talbot.plot(color='white',ax=axs[1,1],edgecolor='lightgrey', transform=ccrs.epsg('3995'),zorder=4)

cb = fig.colorbar(cm.ScalarMappable(norm=norm_res, cmap=res_cmap), ax=axs[1, 1], shrink=0.8, orientation='vertical') #shrink = 0.8 for Baffin Bay plots when box_aspect=1, 0.9 for talbot
cb.ax.tick_params(labelsize=12)
cb.ax.set_ylabel('Residence Time (days)',fontsize=14,rotation=90)


# Save figure
fig.savefig(
    path_figures + "talbot.png",
    dpi=dpi,
    transparent=False,
    bbox_inches="tight",
)
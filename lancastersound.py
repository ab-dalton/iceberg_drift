# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:30:57 2022

@author: CRYO
"""


import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyproj
import seaborn as sns
from matplotlib import cm
import matplotlib as mpl
import geopandas as gpd
import xarray as xr
import numpy as np


# -----------------------------------------------------------------------------
# Load database
# -----------------------------------------------------------------------------

# Load most recent Iceberg Beacon Database output file
df = pd.read_csv("D:/Abby/paper_2/Iceberg Beacon Database-20211026T184427Z-001/Iceberg Beacon Database/iceberg_beacon_database_env_variables_09122022.csv", index_col=False)

# Load RGI
rgi = gpd.read_file("D:/Abby/paper_2/rgi/rgi60_Arctic_glaciers_3995_simple150.gpkg")
rgi['geometry'] = rgi.buffer(0) #clean errors
rgi_lancaster = rgi[(rgi['CenLat'] >= 71) & (rgi['CenLat']<=77) & (rgi['CenLon']>=-85) & (rgi['CenLon']<=-71)]

# Load bathymetry
ds = xr.open_dataset("D:/Abby/paper_2/bathymetry/IBCAO_v4_2_200m.nc").sel(x=slice(-2065753,-1194086),y=slice(-733880,387072))

# Get min z values
ds.z.values.min()

# Select underwater values only
ds0 = ds.where(ds.z.values <= 0) 


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Add Natural Earth coastline
# Add Natural Earth coastline
coast = cfeature.NaturalEarthFeature(
    "physical", "land", "10m", edgecolor="lightgrey", facecolor="lightgray", lw=0.75
)

# Seaborn configuration
sns.set_theme(style="ticks")
sns.set_context("paper") # talk, paper, poster

# Set figure DPI
dpi = 500

# Initialize pyproj with appropriate ellipsoid
geodesic = pyproj.Geod(ellps='WGS84')

# Path for output figures
path_figures = "D:/Abby/paper_2/plots/lancaster/"

cmap = cm.get_cmap('plasma_r',15)

# -----------------------------------------------------------------------------
# Subset data
# -----------------------------------------------------------------------------

# Lancaster Sound
extents = [-72,-85,72,76]

norm_speed = mpl.colors.Normalize(vmin=0, vmax=1.5)

df2 = df[(df['latitude'] >= 71) & (df['latitude']<=77) & (df['longitude']>=-85) & (df['longitude']<=-71)]

df3 = df2[(df2['beacon_id'] == "2017_300234060692710") |
          (df2['beacon_id'] == "2016_300234063515450") |
          (df2['beacon_id'] == "2017_300234062328750") |
          (df2['beacon_id'] == "2017_300234062327750")]

       # (df2['beacon_id'] == "2017_300234060177480") ] 
       # (df2['beacon_id'] == "2011_300234010959690") |
       # (df2['beacon_id'] == "2013_300234011242410") ]
       
df3['iceberg_ocean_diff'] = df3['speed_ms'] - df3['glorys_speed_ms']
df3['iceberg_seaice_diff'] = df3['speed_ms'] - df3['seaice_speed_ms']


# Median speed differences
df3['iceberg_seaice_diff'].median()
df3['iceberg_ocean_diff'].median()
          

# df3.to_csv("D:/Abby/paper_2/Iceberg Beacon Database-20211026T184427Z-001/Iceberg Beacon Database/iceberg_beacon_database_lancaster_sound.csv")

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------

plt.figure(figsize=(12,12))
params = {'mathtext.default': 'regular' }   
plt.rcParams.update(params)
font = {'size'   : 12,
        'weight' : 'normal'}
mpl.rc('font', **font)
ax = plt.axes(projection=ccrs.Orthographic((
    (df3['longitude'].min() + df3['longitude'].max()) / 2),
    (df3['latitude'].min() + df3['latitude'].max()) / 2))
gl = ax.gridlines(draw_labels=True, color='grey', alpha=0.5, linestyle='dotted')
gl.bottom_labels=False   # suppress top labels
gl.right_labels=False # suppress right labels
ax.add_feature(coast,zorder=1)
ax.set_extent(extents)
ax.coastlines(resolution='10m',zorder=2)
ax.scatter(df3['longitude'], df3['latitude'], 
         marker='o',          
         s = 10,
         c=df3['speed_ms'],
         cmap=cmap,
         norm=norm_speed,
         transform=ccrs.PlateCarree(),zorder=3)
ax.set_facecolor('#D6EAF8')
for k, spine in ax.spines.items():  #ax.spines is a dictionary
    spine.set_zorder(10)
ax.annotate(
    text="Bylot Island",
    xy=(-80, -130),
    xycoords="data",
    fontsize=11,
    weight="bold",
    textcoords="offset points",zorder=5
)
ax.annotate(
    text="Devon Island",
    xy=(-210, 130),
    xycoords="data",
    fontsize=11,
    weight="bold",
    textcoords="offset points",zorder=5
)
ax.annotate(
    text="Baffin Island",
    xy=(-240, -210),
    xycoords="data",
    fontsize=11,
    weight="bold",
    textcoords="offset points",zorder=5
)
ax.annotate(
    text="A",
    xy=(-90, 50),
    xycoords="data",
    fontsize=11,
    weight="bold",
    textcoords="offset points",
    zorder=5
)
ax.annotate(
    text="B",
    xy=(-85, 30),
    xycoords="data",
    fontsize=11,
    weight="bold",
    textcoords="offset points",
    zorder=5,
)
ax.annotate(
    text="C",
    xy=(-20, 40),
    xycoords="data",
    fontsize=11,
    weight="bold",
    textcoords="offset points",
    zorder=5,
)
ax.annotate(
    text="D",
    xy=(40, 60),
    xycoords="data",
    fontsize=11,
    weight="bold",
    textcoords="offset points",
    zorder=5,
)
rgi_lancaster.plot(color='white',ax=ax,edgecolor='none', transform=ccrs.epsg('3995'),zorder=4)
cb = plt.colorbar(cm.ScalarMappable(norm=norm_speed,cmap=cmap), ax=ax, shrink=0.47,orientation='horizontal', pad=0.05)
cb.ax.tick_params(labelsize=12)
cb.ax.set_xlabel('Speed (m $s^{-1}$)',fontsize=12, fontdict=dict(weight='normal'))
cs = plt.contour(ds0.x, ds0.y, ds0.z, 15, zorder=1, cmap="Greys", linewidths=0.5, alpha=0.25, transform=ccrs.NorthPolarStereo(true_scale_latitude=75)) # Specifiy true scale latitude to avoid misalignment
ax.clabel(cs, cs.levels, inline=True,fontsize=8, zorder=1) # REQUIRES TWEAKING

#Save figure
plt.savefig(path_figures + "Figure_14_test.png", dpi=dpi, transparent=False, bbox_inches='tight')

# -----------------------------------------------------------------------------
# Tides
# -----------------------------------------------------------------------------

# Load tide database
tides =  pd.read_csv('D:/Abby/paper_2/tides/webtide_lancaster_sound_2017.csv', index_col=False)

# Convert to datetime
tides["date"] = pd.to_datetime(
    tides["date"].astype(str), format="%d/%m/%Y %H:%M")

# 10 m eastward ocean velocity (m/s)
tides_u = tides['u_current']

# 10 m northward ocean velocity (m/s)
tides_v = tides['v_current']

# Calculate ocean speed (m/s)
tides['tide_speed_ms'] = np.sqrt(tides_u**2 + tides_v**2)


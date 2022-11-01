# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 15:33:26 2022

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


# -----------------------------------------------------------------------------
# Load database
# -----------------------------------------------------------------------------

# Load most recent Iceberg Beacon Database output file
df = pd.read_csv("D:/Abby/paper_2/Iceberg Beacon Database-20211026T184427Z-001/Iceberg Beacon Database/iceberg_beacon_database_filtered_08312022_clean.csv", index_col=False)


# Load RGI
rgi = gpd.read_file("C:/Users/adalt043/Downloads/rgi60_Arctic_glaciers_3995_simple150.gpkg")
rgi['geometry'] = rgi.buffer(0) #clean errors
rgi_nares = rgi[(rgi['CenLat'] >= 75.5) & (rgi['CenLat']<=81) & (rgi['CenLon']>=-82) & (rgi['CenLon']<=-60)]

rgi_gis = gpd.read_file("C:/Users/adalt043/Downloads/Greenland_Ice_Sheet.gpkg")

# Load bathymetry
ds = xr.open_dataset("D:/Abby/paper_2/bathymetry/IBCAO_v4_2_200m.nc").sel(x=slice(-1718268,-411542),y=slice(-761491,721012))

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
path_figures = "D:/Abby/paper_2/plots/nares/"

cmap = cm.get_cmap('plasma_r')

# Extents

# Nares Strait
extents = [-66,-78,75.5,80.5]

norm_speed = mpl.colors.Normalize(vmin=0, vmax=2.5)

df2 = df[(df['latitude'] >= 75.5) & (df['latitude']<=81) & (df['longitude']>=-78) & (df['longitude']<=-65)]

df3 = df2[(df2['beacon_id'] == "2013_300234011242410") | (df2['beacon_id'] == "2016_300234061768060")]

plt.figure(figsize=(12,12), zorder=5)
params = {'mathtext.default': 'regular' }   
plt.rcParams.update(params)
font = {'size'   : 12,
        'weight' : 'normal'}
mpl.rc('font', **font)
ax = plt.axes(projection=ccrs.Orthographic((
    (df3['longitude'].min() + df3['longitude'].max()) / 2),
    (df3['latitude'].min() + df3['latitude'].max()) / 2), zorder=5)
ax.add_feature(coast)
ax.set_extent(extents)
ax.coastlines(resolution='10m', zorder=5)
ax.gridlines(draw_labels=True, color='grey', alpha=0.5, linestyle='-', zorder=5)
ax.set_facecolor('#D6EAF8')
for k, spine in ax.spines.items():  #ax.spines is a dictionary
    spine.set_zorder(10)
ax.scatter(df3['longitude'], df3['latitude'], 
         marker='o',          
         s = 10,
         c=df3['speed_ms'],
         cmap=cmap,
         norm=norm_speed,
         transform=ccrs.PlateCarree(),
         zorder=3
)
ax.annotate(
    text="Ellesmere Island",
    xy=(-110, 230),
    xycoords="data",
    fontsize=11,
    weight="bold",
    textcoords="offset points",zorder=5
)
ax.annotate(
    text="Greenland",
    xy=(70, 45),
    xycoords="data",
    fontsize=11,
    weight="bold",
    textcoords="offset points",zorder=5
)
rgi_nares.plot(color='white',ax=ax,edgecolor='none', transform=ccrs.epsg('3995'),zorder=4)
# rgi_gis.plot(color='white',ax=ax,edgecolor='none', transform=ccrs.epsg('3995'),zorder=3)
cb = plt.colorbar(cm.ScalarMappable(norm=norm_speed,cmap=cmap), ax=ax, shrink=0.47,orientation='horizontal', pad=0.05)
cb.ax.tick_params(labelsize=12)
cb.ax.set_xlabel('Speed (m $s^{-1}$)',fontsize=12, fontdict=dict(weight='normal'))
cs = plt.contour(ds0.x, ds0.y, ds0.z, 15, zorder=1, cmap="Greys", linewidths=0.5, alpha=0.25, transform=ccrs.NorthPolarStereo(true_scale_latitude=75)) # Specifiy true scale latitude to avoid misalignment
ax.clabel(cs, cs.levels, inline=True,fontsize=8, zorder=1) # REQUIRES TWEAKING

#Save figure
plt.savefig(path_figures + "8060_2410_tracks_bathy.png", dpi=dpi, transparent=False, bbox_inches='tight')




























# # -----------------------------------------------------------------------------
# # Create map plots
# # -----------------------------------------------------------------------------

# # Individual beacon track plots
# for label, group in df.groupby(['beacon_type','beacon_id']):

#     # Calculate the length of the iceberg track    
#     duration = (group['datetime_data'].max() - group['datetime_data'].min()).days
    
#     # Calculate cumulative distance of the iceberg track
#     distance = group['distance'].sum() / 1000
    
#     plt.figure(figsize=(12,12))
#     ax = plt.axes(projection=ccrs.Orthographic((
#         (group['longitude'].min() + group['longitude'].max()) / 2),
#         (group['latitude'].min() + group['latitude'].max()) / 2))
#     ax.coastlines(resolution='10m')
#     ax.gridlines(draw_labels=True, color='black', alpha=0.5, linestyle='-')
#     ax.plot(group['longitude'], group['latitude'], 
#             marker='o', ms=3, fillstyle='none',
#             linestyle='', lw=2,
#             color='red',
#             transform=ccrs.PlateCarree())
#     plt.title("%s %s\n%s to %s\n%s days %.2f km" % (label[0], 
#                                                     label[1], 
#                                                     group['datetime_data'].min(), 
#                                                     group['datetime_data'].max(), 
#                                                     duration, 
#                                                     distance.sum()), loc='left')
#     # Save figure
#     plt.savefig(path_figures + "%s.png" % label[1], dpi=dpi, transparent=False, bbox_inches='tight')
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
import numpy as np


# -----------------------------------------------------------------------------
# Load database
# -----------------------------------------------------------------------------

# Load most recent Iceberg Beacon Database output file
df = pd.read_csv("D:/Abby/paper_2/Iceberg Beacon Database-20211026T184427Z-001/Iceberg Beacon Database/iceberg_beacon_database_env_variables_22032023_notalbot.csv", index_col=False)

df["datetime_data"] = pd.to_datetime(
    df["datetime_data"].astype(str), format="%Y-%m-%d %H:%M:%S")

# Load tide data
tides =  pd.read_csv('D:/Abby/paper_2/tides/webtide_nares_201314.csv', index_col=False)

# Load RGI
rgi = gpd.read_file("D:/Abby/paper_2/rgi/rgi60_Arctic_glaciers_3995_simple150.gpkg")
rgi['geometry'] = rgi.buffer(0) #clean errors
rgi_nares = rgi[(rgi['CenLat'] >= 75.5) & (rgi['CenLat']<=81) & (rgi['CenLon']>=-82) & (rgi['CenLon']<=-60)]

# rgi_gis = gpd.read_file("C:/Users/adalt043/Downloads/Greenland_Ice_Sheet.gpkg")

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

cmap = cm.get_cmap('plasma_r', 25)

# Extents

# Nares Strait
extents = [-66,-78,75.5,80.5]

norm_speed = mpl.colors.Normalize(vmin=0, vmax=2.5)

df2 = df[(df['latitude'] >= 75.5) & (df['latitude']<=81) & (df['longitude']>=-78) & (df['longitude']<=-65)]

df3 = df2[(df2['beacon_id'] == "2016_300234061768060")]# | (df2['beacon_id'] == "2016_300234061768060")]

# Calculating speed differences

df3['iceberg_ocean_diff'] = df3['speed_ms'] - df3['glorys_speed_ms']
df3['iceberg_seaice_diff'] = df3['speed_ms'] - df3['seaice_speed_ms']


# Median speed differences

df3['iceberg_seaice_diff'].median()
df3['iceberg_ocean_diff'].median()


# Speed Plot

plt.hist(df3['iceberg_seaice_diff'], 50)
plt.axvline(0.08, color='black')
plt.axvline(0.14, color='red')

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
for k, spine in ax.spines.items():  # this puts figure spine on top of layers for clean edge
    spine.set_zorder(10)
ax.scatter(df3['longitude'], df3['latitude'], 
          marker='o',          
          s = 10,
          c=df3['iceberg_ocean_diff'],
          cmap=cmap,
          norm=norm_speed,
          transform=ccrs.PlateCarree(),
          zorder=3
)
ax.scatter(-71.817, 79.258, marker='^', color='black', zorder=7, transform=ccrs.PlateCarree())
ax.scatter(-67.357, 79.836, marker='^', color='blue', zorder=7, transform=ccrs.PlateCarree())
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
ax.annotate(
    text="A",
    xy=(-110, -110),
    xycoords="data",
    fontsize=11,
    weight="bold",
    textcoords="offset points",
    zorder=5
)
ax.annotate(
    text="B",
    xy=(-65, -120),
    xycoords="data",
    fontsize=11,
    weight="bold",
    textcoords="offset points",
    zorder=5,
    color='blue'
)
rgi_nares.plot(color='white',ax=ax,edgecolor='none', transform=ccrs.epsg('3995'),zorder=4)
# rgi_gis.plot(color='white',ax=ax,edgecolor='none', transform=ccrs.epsg('3995'),zorder=3)
cb = plt.colorbar(cm.ScalarMappable(norm=norm_speed,cmap=cmap), ax=ax, shrink=0.47,orientation='horizontal', pad=0.05)
cb.ax.tick_params(labelsize=12)
cb.ax.set_xlabel('Speed (m $s^{-1}$)',fontsize=12, fontdict=dict(weight='normal'))
cs = plt.contour(ds0.x, ds0.y, ds0.z, 15, zorder=1, cmap="Greys", linewidths=0.5, alpha=0.25, transform=ccrs.NorthPolarStereo(true_scale_latitude=75)) # Specifiy true scale latitude to avoid misalignment
ax.clabel(cs, cs.levels, inline=True,fontsize=8, zorder=1) 

#Save figure
plt.savefig(path_figures + "Figure_9_test.png", dpi=dpi, transparent=False, bbox_inches='tight')



# -----------------------------------------------------------------------------
# Tides
# -----------------------------------------------------------------------------

# Convert to datetime
tides["date"] = pd.to_datetime(
    tides["date"].astype(str), format="%d/%m/%Y %H:%M")

# 10 m eastward ocean velocity (m/s)
tides_u = tides['u_current']

# 10 m northward ocean velocity (m/s)
tides_v = tides['v_current']

# Calculate ocean speed (m/s)
tides['tide_speed_ms'] = np.sqrt(tides_u**2 + tides_v**2)

# Calculate tide direction (radians)
tides['tide_direction'] = np.arctan2(tides_v, tides_u)

tides['tide_direction_heading'].plot()

# Convert radians to degrees and (-180° to 180°) to (0° to 360°) HEADING
tides['tide_direction_heading'] = np.mod(90 - (np.rad2deg(tides['tide_direction'])),360)


# -----------------------------------------------------------------------------
# Hans Island wind plot
# -----------------------------------------------------------------------------

# 2014-09-25 to 2014-10-10

start_date = '2014-09-25'
end_date = '2014-10-10'

hans = pd.read_csv("D:/Abby/paper_2/hans/aws02Weather_2014.csv")

df2 = df.loc[(df["beacon_id"] == '2013_300234011242410') & (df['datetime_data'] >= start_date) & (df['datetime_data'] <= end_date)]

# Merge hans island data and iceberg data to do linear regression

hans.index = hans['Date']
df2.index = df2['datetime_data']

hans = hans.dropna()
df2.dropna()

merge = pd.merge_asof(
    left=hans,
    right=df2,
    right_index=True,
    left_index=True,
    direction='nearest')


merge['ResWindDir'].median()

# Convert to datetime
hans["Date"] = pd.to_datetime(hans["Date"].astype(str), format="%d/%m/%Y %H:%M")

plt.figure(figsize=(6,4))
ax = merge.plot(x='datetime_data',
             y='speed_ms',
             label="Iceberg",
             color='orange')
merge.plot(x = 'datetime_data',
            y = 'MeanWindS',
            label='Hans Island AWS',
            color='royalblue',
            ax=ax)
merge.plot(x='datetime_data',
             y='glorys_speed_ms',
             label='GLORYS',
             color='purple',
             ax=ax)
merge.plot(x='datetime_data',
             y='era5_speed',
             label='ERA5',
             color='mediumvioletred',
             ax=ax)
plt.xticks(rotation=45)
plt.margins(x=0.01, y=0)
plt.axhline(0.02, color='black')
ax.set(xlabel='Date', ylabel='Speed (m $s^{-1}$)')
plt.legend(frameon=False)
plt.text(("2014-09-26"), 15, "$ R ^{2}$ = 0.56", color='royalblue')
plt.text(("2014-09-26"), 13.8, "$ R ^{2}$ = 0.48", color='purple')
plt.text(("2014-09-26"), 12.7, "$ R ^{2}$ = 0.66", color='mediumvioletred')


plt.savefig(path_figures + "hans_island_2410.png", dpi=300, transparent=False, bbox_inches='tight')




from scipy import stats
from sklearn.metrics import r2_score


# Linear

x = merge['era5_speed']
y = merge['speed_ms']

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)


print("slope:", slope,
      "\nintercept:", intercept,
      "\nr value:", r_value**2,
      "\np value:", p_value,
      "\np standard error:", std_err)

# Multiple Regression

mymodel = np.poly1d(np.polyfit(x,y, 2))

myline = np.linspace(1, 29, 100)

print(r2_score(y, mymodel(x)))


## Wind plot for beacon 8060

import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%d')



start_date = "2016-12-14"
end_date = "2016-12-21"

df2 = df.loc[(df["beacon_id"] == '2016_300234061768060') & (df['datetime_data'] >= start_date) & (df['datetime_data'] <= end_date)]

plt.figure(figsize=(6,4))
ax = df2.plot(x='datetime_data',
             y='speed_ms',
             label="Iceberg",
             color='orange')
df2.plot(x='datetime_data',
             y='glorys_speed_ms',
             label='GLORYS',
             color='purple',
             ax=ax)
df2.plot(x='datetime_data',
             y='era5_speed',
             label='ERA5',
             color='mediumvioletred',
             ax=ax)
plt.margins(x=0.01, y=0)
plt.axhline(0.02, color='black')
ax.set(xlabel='Date', ylabel='Speed (m $s^{-1}$)')
plt.legend(frameon=False)
plt.text(("2016-12-15"), 13.8, "$ R ^{2}$ = 0.48", color='purple')
plt.text(("2016-12-15"), 12.7, "$ R ^{2}$ = 0.66", color='mediumvioletred')
plt.autofmt_xdate()



plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=48))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.tick_params(axis="x", which="major", rotation=45)

x = merge['era5_speed']
y = merge['speed_ms']

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

print("slope:", slope,
      "\nintercept:", intercept,
      "\nr value:", r_value**2,
      "\np value:", p_value,
      "\np standard error:", std_err)



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
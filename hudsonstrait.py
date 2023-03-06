# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:14:18 2022

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
import numpy as np
import xarray as xr
import matplotlib.dates as mdates
from scipy import stats

# -----------------------------------------------------------------------------
# Load database
# -----------------------------------------------------------------------------

# Load most recent Iceberg Beacon Database output file
df = pd.read_csv("D:/Abby/paper_2/Iceberg Beacon Database-20211026T184427Z-001/Iceberg Beacon Database/iceberg_beacon_database_env_variables_09122022.csv", index_col=False)

# Load tide data
tides =  pd.read_csv('D:/Abby/paper_2/tides/webtide_acadiacove_2019.csv', index_col=False)

# Convert to datetime
df["datetime_data"] = pd.to_datetime(
    df["datetime_data"].astype(str), format="%Y-%m-%d %H:%M:%S")

#Create year, month, day columns
df['day'] = df['datetime_data'].dt.day
df['month'] = df['datetime_data'].dt.month
df['year'] = df['datetime_data'].dt.year

# # Convert to datetime
# tides["Date"] = pd.to_datetime(
#     tides["Date"].astype(str), format="%Y-%m-%d %H:%M")

# #Create year, month, day columns
# tides['day'] = tides['Date'].dt.day
# tides['month'] = tides['Date'].dt.month
# tides['year'] = tides['Date'].dt.year


# Load RGI data
rgi = gpd.read_file("D:/Abby/paper_2/rgi/rgi60_Arctic_glaciers_3995_simple150.gpkg")
rgi['geometry'] = rgi.buffer(0) #clean errors
rgi_hudson = rgi[(rgi['CenLat'] >= 60) & (rgi['CenLat']<=64) & (rgi['CenLon']>=-70) & (rgi['CenLon']<=-60)]


# Load bathymetry
ds = xr.open_dataset("D:/Abby/paper_2/bathymetry/gebco_2022_hudson_strait.nc")#.sel(x=slice(-4025221,-1640666),y=slice(-858599,-2277228))

# Get min z values
ds.elevation.values.min()

# Select underwater values only
ds0 = ds.where(ds.elevation.values <= 0) 

x = ds0.variables['lon'][:]
y = ds0.variables['lat'][:]
z = ds0.variables['elevation'][:]

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


# -----------------------------------------------------------------------------
# Plot speed tracks
# -----------------------------------------------------------------------------

# Set figure DPI
dpi = 500

# Initialize pyproj with appropriate ellipsoid
geodesic = pyproj.Geod(ellps='WGS84')

# Path for output figures
path_figures = "D:/Abby/paper_2/plots/hudson/"

cmap = cm.get_cmap('plasma_r', 20)

# Extents
# Hudson Strait
extents = [-60,-66.5,60,63]

norm_speed = mpl.colors.Normalize(vmin=0, vmax=2)

df2 = df[(df['latitude'] >= 60) & (df['latitude']<=64) & (df['longitude']>=-70) & (df['longitude']<=-60)]

df3 = df2[(df2['beacon_id'] == "2018_300434063415110") | (df2['beacon_id'] == "2013_300234011240410")| (df2['beacon_id'] == "2013_300234011241410")]

df3['iceberg_ocean_diff'] = df3['speed_ms'] - df3['glorys_speed_ms']
df3['iceberg_seaice_diff'] = df3['speed_ms'] - df3['seaice_speed_ms']

# Median speed differences

df3['iceberg_seaice_diff'].median()
df3['iceberg_ocean_diff'].median()



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
         c=df3['iceberg_wind_diff'],
         cmap=cmap,
         norm=norm_speed,
         transform=ccrs.PlateCarree(),
         zorder=3)
#ax.scatter(-64.917, 61.350, marker='^', color='red', zorder=7, transform=ccrs.PlateCarree())
ax.scatter(-64, 61, marker='^', color='red',s=20, zorder=7, transform=ccrs.PlateCarree())
ax.set_facecolor('#D6EAF8')
for k, spine in ax.spines.items():  #ax.spines is a dictionary
    spine.set_zorder(10)
ax.annotate(
    text="Baffin\nIsland",
    xy=(-360, 35),
    xycoords="data",
    fontsize=11,
    weight="bold",
    textcoords="offset points",
    zorder=5
)
ax.annotate(
    text="Resolution\n     Island",
    xy=(-258, -80),
    xycoords="data",
    fontsize=11,
    weight="bold",
    textcoords="offset points",
    zorder=5
)
ax.annotate(
    text="A",
    xy=(-190, -150),
    xycoords="data",
    fontsize=11,
    weight="bold",
    textcoords="offset points",
    zorder=5
)
ax.annotate(
    text="B",
    xy=(-110, -150),
    xycoords="data",
    fontsize=11,
    weight="bold",
    textcoords="offset points",
    zorder=5,
)
ax.annotate(
    text="C",
    xy=(0, -105),
    xycoords="data",
    fontsize=11,
    weight="bold",
    textcoords="offset points",
    zorder=5,
)
rgi_hudson.plot(color='white',ax=ax,edgecolor='none', transform=ccrs.epsg('3995'),zorder=4)
cb = plt.colorbar(cm.ScalarMappable(norm=norm_speed,cmap=cmap), ax=ax, shrink=0.47,orientation='horizontal', pad=0.05)
cb.ax.tick_params(labelsize=12)
cb.ax.set_xlabel('Iceberg-Ocean Speed (m $s^{-1}$)',fontsize=12, fontdict=dict(weight='normal'))
cs = plt.contour(x,y,z, 15, zorder=1, cmap="Greys", linewidths=0.5, alpha=0.25, transform=ccrs.PlateCarree()) # Specifiy true scale latitude to avoid misalignment
ax.clabel(cs, cs.levels, inline=True,fontsize=8, zorder=1) # REQUIRES TWEAKING

#Save figure
plt.savefig(path_figures + "Figure_18.png", dpi=dpi, transparent=False, bbox_inches='tight')





# -----------------------------------------------------------------------------
# Plot tides with speed
# -----------------------------------------------------------------------------

# Plot daily amplitude with max daily speed

# tides_stats = tides.groupby(by=[tides.month, tides.day]).agg({'predictions(m)':[np.ptp]})

# tides_amp =  pd.read_csv('D:/Abby/paper_2/tides/acadia_cove_0410_0503_2019_amplitude.csv', index_col=False)

# Convert to datetime
tides["date"] = pd.to_datetime(
    tides["date"].astype(str), format="%d/%m/%Y %H:%M")

# 10 m eastward ocean velocity (m/s)
tides_u = tides['u_current']

# 10 m northward ocean velocity (m/s)
tides_v = tides['v_current']

# Calculate ocean speed (m/s)
tides['tide_speed_ms'] = np.sqrt(tides_u**2 + tides_v**2)

# params = {'mathtext.default': 'regular' }   
# plt.rcParams.update(params)
# font = {'size'   : 12,
#         'weight' : 'normal'}
# mpl.rc('font', **font)
# fig, ax = plt.subplots(1,1, figsize=(8,4), constrained_layout=True)
# tides.plot(x="date", y="tide_speed_ms",ax=ax, legend=False)
# ax2 = ax.twinx()
# tides.plot(x="date", y="speed_ms", ax=ax2, legend=False, color="r")
# ax.figure.legend(['Tide', 'Iceberg'],bbox_to_anchor=(0.2, 1), bbox_transform=ax.transAxes)
# ax.set_ylabel('Speed (m $s^{-1}$)')
# ax2.set_ylabel('Tidal Current (m $s^{-1}$)')
# ax.set_xlabel('Date')
# plt.gca().xaxis.set_minor_locator(mdates.DayLocator(interval=1))
# plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter(''))
# plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=24))
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
# ax.tick_params(axis="x", which="major", rotation=45)
# ax.annotate('A', (1, 1),
#                     xytext=(-5,-5),
#                     xycoords='axes fraction',
#                     textcoords='offset points',
#                     ha='right', va='top',
#                     fontsize=12,
#                     weight='bold')



params = {'mathtext.default': 'regular' }   
plt.rcParams.update(params)
font = {'size'   : 12,
        'weight' : 'normal'}
mpl.rc('font', **font)

fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
plt.plot(tides['date'], tides['tide_speed_ms'])
plt.plot(tides['date'], tides['speed_ms'], color='r')
# ax2 = ax.twinx()
# ax2.scatter(tides['date'], tides['seaice_speed_ms'], color='g', marker='.')
ax.figure.legend(['Iceberg','Tide', 'Sea Ice'],bbox_to_anchor=(0.2, 1), bbox_transform=ax.transAxes)
ax.set_ylabel('Speed (m $s^{-1}$)')
ax.set_xlabel('Date')
# plt.gca().xaxis.set_minor_locator(mdates.DayLocator(interval=1))
#plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter(''))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=24))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.tick_params(axis="x", which="major", rotation=45)
plt.margins(x=0, y=0)


plt.savefig(path_figures + "5110_tides_webtide.png", dpi=300, transparent=False, bbox_inches='tight')

# Plot linear regression between daily amplitude and max daily speed

x = tides['tide_speed_ms']
y = tides['speed_ms']

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

print("slope:", slope,
      "\nintercept:", intercept,
      "\nr^2 value:", r_value**2,
      "\np value:", p_value,
      "\np standard error:", std_err)

ax = sns.lmplot(data = tides,
           x = 'tide_speed_ms',
           y = 'speed_ms',
           ci=None,
           legend=False,
           scatter_kws={"s": 12, 'linewidth':0, 'alpha' : 0.7, 'color' : 'black'},
           line_kws={"color": 'black'})
ax.set_axis_labels('Daily Tidal Amplitude (m)','Maximum Iceberg Speed (m $s^{-1}$)')
# plt.text(x=2.8, y=1.65, s='$R^2 = $' + str("{:.2f}".format(r_value**2)), fontsize=12)
# plt.text(x=7, y=1.65, s='B', fontsize=12, weight='bold')

plt.savefig(path_figures + "amplitude_vs_maxspeed.png", dpi=300, transparent=False, bbox_inches='tight')

























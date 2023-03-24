# -*- coding: utf-8 -*-
"""
Created on Tue May 31 09:41:19 2022

@author: adalt043
"""

# -----------------------------------------------------------------------------
# Load libraries
# -----------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.feature as cfeature
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

# Abby
path_data = ""
path_figures = 'D:/Abby/paper_2/plots/wind_speed_ratios/'

dpi = 300

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
sns.set_palette("turbo")


# ----------------------------------------------------------------------------
# Prepare data
# ----------------------------------------------------------------------------

# Load database
df = pd.read_csv("D:/Abby/paper_2/Iceberg Beacon Database-20211026T184427Z-001/Iceberg Beacon Database/iceberg_beacon_database_env_variables_22032023_notalbot.csv",
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

lat_band = []
for row in df['latitude']:
    if ((row >=50) and (row <55)):
        lat_band.append('50-55')
    elif ((row >=55) and (row <60)):
        lat_band.append('55-60')
    elif ((row >=60) and (row <65)):
        lat_band.append('60-65')
    elif ((row >=65) and (row <70)):
        lat_band.append('65-70')
    elif ((row >=70) and (row <75)):
        lat_band.append('70-75')
    elif ((row >=75) and (row <80)):
        lat_band.append('75-80')
    elif ((row >=80) and (row <85)):
        lat_band.append('80-85')
    else:
        print('error')

df['Latitude Band'] = lat_band


# ----------------------------------------------------------------------------
# Filter speed outliers
# ----------------------------------------------------------------------------

# Filter nan values from GLORYS dataset
# df = df[df['glorys_speed_ms'].notna()]

# Filter based on variables (e.g. speed, season, beacon id)
# Nares
df_filt = df.loc[(df['beacon_id'] == "2013_300234011242410") |
                 (df['beacon_id'] == "2016_300234061768060")]

# Lancaster
df_filt = df.loc[(df['beacon_id'] == '2017_300234060692710') |
                 (df['beacon_id'] == '2016_300234063515450') |
                 (df['beacon_id'] == '2017_300234062328750') |
                 (df['beacon_id'] == '2017_300234062327750')]

# Davis
df_filt = df.loc[(df['beacon_id'] == "2018_300434063415110") |
                 (df['beacon_id'] == "2013_300234011240410")|
                 (df['beacon_id'] == "2013_300234011241410")]


# df_filt = df_filt.set_index('datetime_data').sort_index()
# df_filt = df_filt["2013-08-25 00:00:00":"2014-09-30 23:00:00"]

# Filter by Latitude band - change based on location
df_filt = df_filt.loc[(df['Latitude Band'] == '60-65')]


# ----------------------------------------------------------------------------
# Filter speed outliers - ERA5
# ----------------------------------------------------------------------------

# Create dataframe based on variables for corr plot
df2 = df[['beacon_id', 'datetime_data', 'latitude', 'longitude', 'distance', 'speed_ms', 'azimuth_obs',
          'era5_speed', 'era5_direction_deg', 'glorys_speed_ms', 'glorys_direction_deg','Latitude Band']].copy()

df2 = df2.dropna()

# df2.to_csv("D:/Abby/paper_2/Iceberg Beacon Database-20211026T184427Z-001/Iceberg Beacon Database/iceberg_beacon_database_env_variables_filtered.csv")

# Compute the correlation matrix
corr = df2.corr()

# Exclude duplicate correlations by masking uper right values
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set background color / chart style
sns.set_style(style = 'white')

# Set up  matplotlib figure
f, ax = plt.subplots(figsize=(12, 12))

# Add diverging colormap
cmap = sns.diverging_palette(10, 250, as_cmap=True)

# Draw correlation plot
sns.heatmap(corr, mask=mask, cmap=cmap, 
        square=True,
        linewidths=.5, cbar_kws={"shrink": .5}, ax=ax, annot=True).set(title='')


# -----------------------------------------------------------------------------
# Plot variables in scatterplot
# -----------------------------------------------------------------------------

# Calculate % of iceberg speed vs ERA5 speed
df3 = df2.loc[(df2["era5_speed"]>1)]
# df3 = df2.loc[(df2["glorys_speed_ms"]>0.05)]
# df3 = df2.loc[(df2["seaice_speed_ms"]>0.01)]


df3['iceberg_wind_ratio'] = (df3['speed_ms'] / df3['era5_speed']) * 100
# df3['iceberg_ocean_ratio'] = (df3['speed_ms'] / df3['glorys_speed_ms'])
# df3['iceberg_seaice_ratio'] = (df3['speed_ms'] / df3['seaice_speed_ms'])

#df3 = df3.loc[(df3["iceberg_wind_ratio"] <= 2)]


df3['iceberg_wind_ratio'].std()

# Plot polynomial regression

x = df3['era5_speed']
y = df3['iceberg_wind_ratio']

mymodel_1 = np.poly1d(np.polyfit(x,y, 1))
mymodel_2 = np.poly1d(np.polyfit(x,y, 2))
mymodel_3 = np.poly1d(np.polyfit(x,y, 3))





myline = np.linspace(1, 29, 100)

print(r2_score(y, mymodel_3(x)))


params = {'mathtext.default': 'regular' }   
plt.rcParams.update(params)
plt.figure(figsize=(6,4))
ax = sns.lmplot(data = df3,
           x = 'era5_speed',
           y = 'iceberg_wind_ratio',
           # hue = 'Latitude Band', # can be latitude_band, beacon_id etc
           palette = 'plasma',
           ci=None,
           legend=False,
           fit_reg=False,  # optional
           markers='o',
           scatter_kws={"s": 5, 'linewidth':0, 'alpha' : 0.7},
           line_kws={'color': 'red'})#.set(title='$R^2 = $' + str("{:.5f}".format(r_value**2)))
plt.margins(x=0.01, y=0)
plt.axhline(2, color='black', linestyle='dashed')
plt.plot(myline, mymodel_1(myline), color='green', label="1st degree:  $ R^{2}$= 0.07")
plt.plot(myline, mymodel_2(myline), color='red', label="2nd degree: $ R^{2}$= 0.10")
plt.plot(myline, mymodel_3(myline), color='purple', label="3rd degree:  $ R^{2}$= 0.12")
# plt.annotate("degree 1: $ R ^{2}$ = 0.07", (27,100), ha='right', va='top')
# plt.annotate("degree 2: $ R ^{2}$ = 0.10", (27,95), ha='right', va='top')
# plt.annotate("degree 3: $ R ^{2}$ = 0.12", (27,90), ha='right', va='top')
plt.legend(frameon=False)
ax.set_axis_labels('ERA5 Wind Speed (m $ s ^{-1}$)',"Speed Ratio % (Iceberg Speed/Wind Speed)")


plt.savefig(
    path_figures + "wind_iceberg_ratio_notalbot_123.png",
    dpi=dpi,
    transparent=False,
    bbox_inches="tight"
)


## Plot xy scatter plots

x = df3['era5_speed']
y = df3['iceberg_wind_ratio']

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

print("slope:", slope,
      "\nintercept:", intercept,
      "\nr^2 value:", r_value**2,
      "\np value:", p_value,
      "\np standard error:", std_err)


plt.figure(figsize=(6,4))
ax = sns.lmplot(data = df2,
           x = 'era5_speed',
           y = 'speed_ms',
           hue = 'Latitude Band', # can be latitude_band, beacon_id etc
           palette = 'plasma',
           ci=None,
           legend=False,
           # fit_reg=False,  # optional
           markers='o',
           scatter_kws={"s": 6, 'linewidth':0, 'alpha' : 0.7})#.set(title='$R^2 = $' + str("{:.5f}".format(r_value**2)))
# plt.axhline(2)
ax.add_legend(title="Latitude Band ($^\circ$N)", loc="upper left", bbox_to_anchor = (0.12, 0.97), label_order = ['60-65', '65-70','70-75','75-80', '80-85']) #'50-55', '55-60', '60-65', '65-70', '70-75','75-80', '80-85'
plt.margins(x=0, y=0)
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)
plt.text(16.2,1.3, "B", fontsize=12)
ax.set_axis_labels('ERA5 Speed (m $ s ^{-1}$)',"Iceberg Speed (m $ s ^{-1}$)")

# plt.xlim([0,0.01])
# plt.ylim([0,0])

# Save figure
plt.savefig(
    path_figures + "Figure_13b_8060.png",
    dpi=dpi,
    transparent=False,
    bbox_inches="tight",
)













# # # -----------------------------------------------------------------------------
# # # Plot pcolormesh at time 0
# # # -----------------------------------------------------------------------------

# norm = mpl.colors.Normalize(vmin=0, vmax=100)
# cmap = cm.get_cmap("turbo", 100)

# # Smith Sound
# #extents = [-79, -70, 75, 78]

# # Baffin Bay
# # extents = [-83, -60, 55, 83]

# # Set the figure size, projection, and extent
# fig = plt.figure(figsize=(10,6))
# ax = plt.axes(projection=ccrs.Orthographic(-78.5,74))
# ax.set_extent([-83, -60, 55, 83])
# ax.set(box_aspect = 1)
# ax.coastlines(resolution="10m")
# gl = ax.gridlines(draw_labels=True, 
#                   color='black', 
#                   alpha=0.5, 
#                   norm=norm,
#                   linestyle='dotted', 
#                   x_inline=False, 
#                   y_inline=False)
# gl.rotate_labels = False
# plt.scatter(x, y, c=z, cmap=cmap, transform=ccrs.PlateCarree(), s=1.5)
# plt.title('Iceberg Speed vs Wind Speed %', size=14)
# cb = plt.colorbar(ax=ax, orientation='vertical', pad = 0.1)
# cb.set_label('%', size = 12, rotation = 0, labelpad = 15)


# # Save figure
# fig.savefig(
#     path_figures + "iceberg_speed.png",
#     dpi=dpi,
#     transparent=False,
#     bbox_inches="tight",
# )




















